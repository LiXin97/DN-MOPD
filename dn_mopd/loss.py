# SPDX-License-Identifier: Apache-2.0
"""Clipped policy-gradient OPD loss with the paper's reduction.

Per token: ``ratio = exp(logp_new - logp_old)`` and ``loss_t = -min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A)``
(PPO's surrogate with the distillation advantage ``A``). Reduction: the mean over the valid tokens of each response,
then the mean over responses (every response weighs the same, whatever its length). DN-MOPD leaves this loss
unchanged; only ``A`` is rescaled per domain.

With micro-batches or data-parallel shards, pass ``num_responses`` = the number of responses in the whole optimizer
batch so that the per-shard losses add up to the batch mean (the paper's trainer divided the summed per-response
means by the global batch size).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["clipped_opd_loss", "clipped_opd_loss_batched"]


def _check_eps(eps_low: float, eps_high: float) -> None:
    if not (0.0 <= eps_low < 1.0 and eps_high >= 0.0):
        raise ValueError(f"need 0 <= eps_low < 1 and eps_high >= 0, got {eps_low}, {eps_high}")


def clipped_opd_loss(logp_new: Sequence[torch.Tensor], logp_old: Sequence[torch.Tensor],
                     advantages: Sequence[torch.Tensor], masks: Sequence[torch.Tensor], eps_low: float = 0.2,
                     eps_high: float = 0.2, num_responses: Optional[int] = None) -> torch.Tensor:
    """Loss over a list of responses (one 1-D tensor per response for each argument).

    ``logp_old`` is the actor-recomputed pre-update log-prob (the one the advantage was formed with); ``advantages``
    are treated as constants. Returns a scalar: token mean within each response, then the mean over responses (or the
    sum over responses divided by ``num_responses``).
    """
    _check_eps(eps_low, eps_high)
    if not (len(logp_new) == len(logp_old) == len(advantages) == len(masks)) or not logp_new:
        raise ValueError("need the same, non-zero number of responses in every argument")
    per_resp = []
    for lp, lo, a, m in zip(logp_new, logp_old, advantages, masks):
        ratio = torch.exp(lp - lo.detach())
        a = a.detach()
        term = -torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
        m = m.float()
        per_resp.append((term * m).sum() / m.sum().clamp_min(1.0))
    stacked = torch.stack(per_resp)
    return stacked.mean() if num_responses is None else stacked.sum() / num_responses


def clipped_opd_loss_batched(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                             mask: torch.Tensor, eps_low: float = 0.2, eps_high: float = 0.2,
                             num_responses: Optional[int] = None) -> torch.Tensor:
    """Same loss for padded ``(B, T)`` tensors. Masked positions contribute nothing, not even NaN gradients."""
    _check_eps(eps_low, eps_high)
    if not (logp_new.shape == logp_old.shape == advantages.shape == mask.shape) or logp_new.dim() != 2:
        raise ValueError("logp_new, logp_old, advantages and mask must all be (B, T)")
    if logp_new.shape[0] == 0:
        raise ValueError("empty batch")
    valid = mask.to(dtype=torch.bool)
    zero = torch.zeros((), dtype=logp_new.dtype, device=logp_new.device)
    ratio = torch.exp(torch.where(valid, logp_new - logp_old.detach(), zero))
    a = torch.where(valid, advantages.detach().to(logp_new.dtype), zero)
    term = -torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
    m = valid.to(term.dtype)
    per_resp = (term * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return per_resp.mean() if num_responses is None else per_resp.sum() / num_responses
