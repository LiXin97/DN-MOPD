# SPDX-License-Identifier: Apache-2.0
"""Sharded statistics reduce to the single-process result (gloo on CPU with 2 processes, and pure-tensor reductions)."""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import dn_dist_worker
from dn_mopd import (
    DomainNormalizer,
    all_reduce_statistics,
    batch_statistics,
    domain_multipliers_batched,
    merge_statistics,
)

DOMAINS = dn_dist_worker.DOMAINS


def _full_batch(seed=0, batch=40, max_len=16):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 2, (batch,), generator=g)
    ids[torch.tensor([1, 5, 9, 14, 20])] = 2                  # ifeval only in the first shard
    spread = torch.tensor([0.3, 1.0, 4.0])[ids]
    lengths = torch.randint(0, max_len + 1, (batch,), generator=g)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    r = spread[:, None] * torch.randn(batch, max_len, generator=g) + 0.2
    r = torch.where(mask, r, torch.full_like(r, float("nan")))
    return r, mask, ids


def _expected(r, mask, ids):
    w, diag = domain_multipliers_batched(r, mask, ids, DOMAINS)
    return {"weights": w, "global_std": diag.global_std,
            "tokens": {d: v.tokens for d, v in diag.domains.items()},
            "responses": {d: v.responses for d, v in diag.domains.items()},
            "std": {d: v.std for d, v in diag.domains.items()},
            "present": [d for d, v in diag.domains.items() if v.responses > 0]}


@pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()), reason="gloo backend unavailable")
def test_two_process_gloo_equals_single_process(tmp_path):
    r, mask, ids = _full_batch()
    cut = 25                                                  # uneven shards; shard 1 has no ifeval response
    shards = [{"r": r[:cut], "mask": mask[:cut], "ids": ids[:cut]},
              {"r": r[cut:], "mask": mask[cut:], "ids": ids[cut:]}]
    assert int((shards[1]["ids"] == 2).sum()) == 0
    mp.spawn(dn_dist_worker.check_rank, args=(2, str(tmp_path / "pg_init"), shards, _expected(r, mask, ids)),
             nprocs=2, join=True)


def test_callable_sync_with_merge_statistics_equals_single_process():
    r, mask, ids = _full_batch(seed=1)
    cut = 17
    other = batch_statistics(r[cut:], mask[cut:], ids[cut:], len(DOMAINS))
    sync = lambda local: merge_statistics([local, other])  # noqa: E731  (what a framework collective would return)
    w, diag = DomainNormalizer(DOMAINS, sync=sync).multipliers_batched(r[:cut], mask[:cut], ids[:cut])
    w_full, diag_full = domain_multipliers_batched(r, mask, ids, DOMAINS)
    assert torch.allclose(w, w_full, rtol=1e-6)
    assert diag.global_std == pytest.approx(diag_full.global_std, rel=1e-12)
    assert diag.global_tokens == diag_full.global_tokens == int(mask.sum())


def test_replicated_full_batch_on_every_rank_gives_the_same_multipliers():
    """Summing identical statistics W times leaves means and variances unchanged (only counts scale)."""
    r, mask, ids = _full_batch(seed=2)
    local = batch_statistics(r, mask, ids, len(DOMAINS))
    w_rep, _ = DomainNormalizer(DOMAINS, sync=lambda s: merge_statistics([s] * 4)).multipliers_batched(r, mask, ids)
    w_one, _ = DomainNormalizer(DOMAINS).multipliers_batched(r, mask, ids)
    assert torch.allclose(w_rep, w_one, rtol=1e-6)
    assert torch.equal(merge_statistics([local, local]), 2 * local)


def test_without_process_group_all_reduce_is_identity():
    assert not dist.is_initialized()
    stats = torch.randn(4, 3, dtype=torch.float64)
    assert all_reduce_statistics(stats) is stats
    r, mask, ids = _full_batch(seed=3)
    w_sync, _ = DomainNormalizer(DOMAINS, sync=True).multipliers_batched(r, mask, ids)
    w_local, _ = DomainNormalizer(DOMAINS).multipliers_batched(r, mask, ids)
    assert torch.equal(w_sync, w_local)


def test_merge_statistics_validates_shapes():
    with pytest.raises(ValueError):
        merge_statistics([])
    with pytest.raises(ValueError):
        merge_statistics([torch.zeros(4, 3), torch.zeros(4, 2)])
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, sync=lambda s: s[:, :2]).multipliers_batched(*_full_batch(seed=4))
