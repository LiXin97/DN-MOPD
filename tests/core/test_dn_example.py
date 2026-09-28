# SPDX-License-Identifier: Apache-2.0
"""The CPU toy example keeps running and behaves as documented."""
import math

import pytest
import torch

from dn_mopd.examples import toy_opd_loop


@pytest.fixture(autouse=True)
def _one_thread():
    """The toy tensors are tiny: on many-core hosts intra-op threading makes each step seconds instead of ms."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("mode", ["dn", "observe", "frozen"])
def test_toy_loop_runs(mode):
    history = toy_opd_loop.run(steps=3, mode=mode, verbose=False)
    assert len(history) == 3
    assert all(math.isfinite(v) for row in history for v in row.values())
    weights = [[row[f"w/{d}"] for d in toy_opd_loop.DOMAINS] for row in history]
    if mode == "observe":
        assert all(w == 1.0 for ws in weights for w in ws)
    if mode == "frozen":
        assert weights[0] == weights[1] == weights[2]
    if mode == "dn":
        assert all(0.25 <= w <= 4.0 for ws in weights for w in ws)


def test_toy_loop_reduces_reverse_kl():
    history = toy_opd_loop.run(steps=30, verbose=False)
    first = sum(history[0][f"rkl/{d}"] for d in toy_opd_loop.DOMAINS)
    last = sum(history[-1][f"rkl/{d}"] for d in toy_opd_loop.DOMAINS)
    assert last < first
