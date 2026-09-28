# SPDX-License-Identifier: Apache-2.0
"""DN-MOPD: domain-normalized multi-teacher on-policy distillation.

A framework-agnostic implementation of the per-domain advantage multipliers of DN-MOPD, the reverse-KL OPD
advantage and the clipped OPD loss with the paper's reduction. It depends only on ``torch``.

Three lines in a generic OPD loop (once per rollout batch, padded ``(B, T)`` tensors)::

    weights, diag = normalizer.multipliers_batched(teacher_logp - rollout_logp, mask, domain_ids)   # (1)
    adv = scale_advantages_batched(reverse_kl_advantages(teacher_logp, old_logp, mask), domain_ids, weights)  # (2)
    loss = clipped_opd_loss_batched(new_logp, old_logp, adv, mask)                                   # (3)

See ``dn_mopd/examples/toy_opd_loop.py`` for a runnable CPU example.
"""
from .advantages import reverse_kl_advantages, scale_advantages, scale_advantages_batched
from .distributed import all_reduce_statistics, distributed_world_size, merge_statistics
from .loss import clipped_opd_loss, clipped_opd_loss_batched
from .normalizer import (
    DEFAULT_BOUNDS,
    MODES,
    DNDiagnostics,
    DomainDiagnostics,
    DomainNormalizer,
    batch_statistics,
    domain_multipliers,
    domain_multipliers_batched,
    list_statistics,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_BOUNDS",
    "MODES",
    "DNDiagnostics",
    "DomainDiagnostics",
    "DomainNormalizer",
    "all_reduce_statistics",
    "batch_statistics",
    "clipped_opd_loss",
    "clipped_opd_loss_batched",
    "distributed_world_size",
    "domain_multipliers",
    "domain_multipliers_batched",
    "list_statistics",
    "merge_statistics",
    "reverse_kl_advantages",
    "scale_advantages",
    "scale_advantages_batched",
]
