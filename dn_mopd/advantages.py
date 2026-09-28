# SPDX-License-Identifier: Apache-2.0
"""On-policy distillation advantages and their per-domain scaling.

Two log-ratios appear in DN-MOPD and they differ only in which student log-prob they use:

* the *scale estimate* ``r = teacher_logp - rollout_logp`` uses the log-probs cached by the rollout engine (vLLM,
  SGLang, ...) when the responses were sampled; it feeds :class:`dn_mopd.DomainNormalizer`;
* the *advantage* ``A = teacher_logp - old_logp`` uses the trainer's own recomputed pre-update log-probs (the same
  ``old_logp`` the PPO ratio divides by); it enters the loss.

Both are computed by :func:`reverse_kl_advantages`. A trainer without separate rollout log-probs can pass its
``old_logp`` for both (the paper used the rollout engine's values for the scale estimate).
"""
from __future__ import annotations

from typing import Hashable, List, Mapping, Optional, Sequence

import torch

__all__ = ["reverse_kl_advantages", "scale_advantages", "scale_advantages_batched"]


def reverse_kl_advantages(teacher_log_probs: torch.Tensor, student_log_probs: torch.Tensor,
                          valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-token reverse-KL OPD advantage ``log p_T(y_t | h_t) - log pi(y_t | h_t)``, detached.

    Works on any shape (one response ``(T,)`` or a padded batch ``(B, T)``). Positions where ``valid`` is false get
    0 (padding, or tokens the teacher failed to score), whatever the inputs hold there.
    """
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(f"teacher/student log-prob shapes differ: {tuple(teacher_log_probs.shape)} vs "
                         f"{tuple(student_log_probs.shape)}")
    adv = (teacher_log_probs - student_log_probs).detach()
    if valid is None:
        return adv
    if valid.shape != adv.shape:
        raise ValueError(f"valid mask shape {tuple(valid.shape)} != log-prob shape {tuple(adv.shape)}")
    return torch.where(valid.to(device=adv.device, dtype=torch.bool), adv, torch.zeros_like(adv))


def scale_advantages(advantages: Sequence[torch.Tensor], domains: Sequence[Hashable],
                     weights: Mapping[Hashable, float]) -> List[torch.Tensor]:
    """Multiply every advantage of a response by its domain's multiplier and detach (signs are preserved)."""
    if len(advantages) != len(domains):
        raise ValueError("one domain label per response is required")
    return [(a * float(weights[d])).detach() for a, d in zip(advantages, domains)]


def scale_advantages_batched(advantages: torch.Tensor, domain_ids: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
    """Scale a batch of advantages ``(B, ...)`` by ``weights[domain_ids]`` and detach.

    Args:
        advantages: ``(B, T)`` (or any ``(B, ...)``) advantages.
        domain_ids: ``(B,)`` domain index per response.
        weights: ``(D,)`` per-domain multipliers, e.g. from :meth:`DomainNormalizer.multipliers_batched`.
    """
    if domain_ids.shape != advantages.shape[:1]:
        raise ValueError(f"domain_ids must be ({advantages.shape[0]},), got shape {tuple(domain_ids.shape)}")
    if weights.dim() != 1:
        raise ValueError("weights must be a 1-D per-domain tensor")
    w = weights.to(device=advantages.device, dtype=advantages.dtype)[domain_ids.to(advantages.device).long()]
    return (advantages * w.reshape((-1,) + (1,) * (advantages.dim() - 1))).detach()
