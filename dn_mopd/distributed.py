# SPDX-License-Identifier: Apache-2.0
"""Reduction of per-domain sufficient statistics across data-parallel ranks.

The DN-MOPD multiplier needs the population standard deviation of the rollout log-ratio over the *global* rollout
batch. A rank that holds only a shard of the batch computes its local statistics (per domain: responses, valid tokens,
sum, sum of squares; see :func:`dn_mopd.normalizer.batch_statistics`) and sums them over all ranks. Sums are exact
to float64 rounding, so the result equals the single-process computation on the concatenated batch.

Two ways to reduce are provided:

* :func:`all_reduce_statistics` uses ``torch.distributed`` when a process group is initialized and returns the
  input unchanged otherwise (single process);
* :func:`merge_statistics` sums statistics that the caller has gathered by other means (for example a Ray driver
  that collects one tensor per worker). It is plain tensor arithmetic.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist

__all__ = ["distributed_world_size", "all_reduce_statistics", "merge_statistics"]


def distributed_world_size(group: Optional[Any] = None) -> int:
    """World size of ``group`` if ``torch.distributed`` is initialized, else 1."""
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group)


def all_reduce_statistics(stats: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
    """Sum ``stats`` over the ranks of ``group`` (a new tensor; the input is not modified).

    Every rank must call this with a tensor of the same shape, in the same order relative to other collectives.
    Without an initialized process group (or with world size 1) the input is returned as is. With the NCCL backend a
    CPU tensor is moved to the current CUDA device for the collective and moved back afterwards.
    """
    if distributed_world_size(group) == 1:
        return stats
    buf = stats.detach().clone()
    if dist.get_backend(group) == "nccl" and buf.device.type != "cuda":
        buf = buf.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    return buf.to(stats.device)


def merge_statistics(stats: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum per-shard statistics tensors of identical shape (pure-tensor reduction, no process group needed)."""
    items = list(stats)
    if not items:
        raise ValueError("merge_statistics needs at least one statistics tensor")
    shape = items[0].shape
    if any(t.shape != shape for t in items):
        raise ValueError(f"statistics shapes differ: {[tuple(t.shape) for t in items]}")
    device = items[0].device
    return torch.stack([t.to(device=device, dtype=torch.float64) for t in items]).sum(0)
