# SPDX-License-Identifier: Apache-2.0
"""Worker for test_dn_distributed.py: one gloo rank computing DN multipliers on its shard with sync=True.

Kept in its own module (not a test file) so that spawned processes import only this and dn_mopd.
"""
import datetime
from typing import Dict, List

import torch
import torch.distributed as dist

from dn_mopd import DomainNormalizer, domain_multipliers_batched

DOMAINS = ("math", "code", "ifeval")


def check_rank(rank: int, world: int, init_file: str, shards: List[Dict[str, torch.Tensor]],
               expected: Dict[str, object]) -> None:
    """Asserts inside the child; mp.spawn re-raises a failure in the parent."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    try:
        shard = shards[rank]
        # padded API, functional form
        w, diag = domain_multipliers_batched(shard["r"], shard["mask"], shard["ids"], DOMAINS, sync=True)
        assert torch.allclose(w, expected["weights"], rtol=1e-6, atol=0), (w, expected["weights"])
        assert abs(diag.global_std - expected["global_std"]) <= 1e-12 * expected["global_std"]
        for d in DOMAINS:
            assert diag.domains[d].tokens == expected["tokens"][d]
            assert diag.domains[d].responses == expected["responses"][d]
            assert abs(diag.domains[d].std - expected["std"][d]) <= 1e-12 * max(expected["std"][d], 1.0)
        # list API with registered domains
        pairs = [(DOMAINS[int(i)], shard["r"][b][shard["mask"][b]]) for b, i in enumerate(shard["ids"])]
        wl, _ = DomainNormalizer(DOMAINS, sync=True).multipliers(pairs)
        assert set(wl) == set(expected["present"])
        for i, d in enumerate(DOMAINS):
            if d in wl:
                assert abs(wl[d] - float(expected["weights"][i])) <= 1e-6 * float(expected["weights"][i])
        # frozen mode stays identical across ranks over two steps
        frozen = DomainNormalizer(DOMAINS, mode="frozen", sync=True)
        w0, _ = frozen.multipliers_batched(shard["r"], shard["mask"], shard["ids"])
        w1, _ = frozen.multipliers_batched(shard["r"] * 3.0, shard["mask"], shard["ids"].flip(0))
        assert torch.equal(w0, w1)
        gathered = [torch.zeros(3) for _ in range(world)]
        dist.all_gather(gathered, w0)
        assert all(torch.equal(g, gathered[0]) for g in gathered)
    finally:
        dist.destroy_process_group()
