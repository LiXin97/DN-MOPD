# SPDX-License-Identifier: Apache-2.0
"""The supplement's unit tests, ported to the package API, and equivalence with the supplement's reference code."""
import pytest
import torch

import dn_reference_supplement as ref
from dn_mopd import clipped_opd_loss, domain_multipliers, scale_advantages


def _batch(spreads, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(d, s * torch.randn(n, generator=g)) for d, s in spreads]


def _random_pairs(seed, n_resp=60):
    """Responses of three domains with different spreads and offsets, variable lengths, some empty."""
    g = torch.Generator().manual_seed(seed)
    spread = {"math": 0.3, "code": 1.0, "ifeval": 6.0}
    offset = {"math": -0.2, "code": 0.1, "ifeval": -1.0}
    pairs = []
    for i in range(n_resp):
        d = ("math", "code", "ifeval")[int(torch.randint(0, 3, (1,), generator=g))]
        length = int(torch.randint(0, 40, (1,), generator=g))
        pairs.append((d, offset[d] + spread[d] * torch.randn(length, generator=g)))
    return pairs


# ------------------------------------------------------------------------------------ supplement tests (ported)
def test_unclipped_domains_match_pooled_spread():
    pairs = _batch([("math", 0.5), ("code", 1.0), ("code", 1.1), ("math", 0.45)])
    w, info = domain_multipliers(pairs)
    for d in ("math", "code"):
        v = torch.cat([x for dd, x in pairs if dd == d])
        if not info.domains[d].clipped:
            assert abs(float((w[d] * v).std(unbiased=False)) - info.global_std) < 1e-5


def test_clipping_bounds():
    # IF supplies few tokens with a large spread (as in the Qwen3.5 traces), math a small spread
    g = torch.Generator().manual_seed(1)
    pairs = [("math", 0.01 * torch.randn(64, generator=g)), ("code", torch.randn(64, generator=g)),
             ("ifeval", 50.0 * torch.randn(4, generator=g))]
    w, info = domain_multipliers(pairs)
    assert w["math"] == 4.0 and info.domains["math"].clipped
    assert w["ifeval"] == 0.25 and info.domains["ifeval"].clipped


def test_degenerate_std_gives_one():
    w, _ = domain_multipliers([("math", torch.zeros(10)), ("code", torch.randn(10))])
    assert w["math"] == 1.0
    w, _ = domain_multipliers([("ifeval", torch.tensor([0.3]))])
    assert w["ifeval"] == 1.0


def test_scaling_preserves_sign_and_label_is_all_ones():
    adv = [torch.tensor([0.5, -1.0, 2.0]), torch.tensor([-0.2, 0.1])]
    out = scale_advantages(adv, ["math", "ifeval"], {"math": 2.0, "ifeval": 0.25})
    assert torch.equal(torch.sign(out[0]), torch.sign(adv[0])) and torch.equal(torch.sign(out[1]), torch.sign(adv[1]))
    same = scale_advantages(adv, ["math", "ifeval"], {"math": 1.0, "ifeval": 1.0})
    assert all(torch.equal(a, b) for a, b in zip(same, adv))


def test_loss_reduction_is_response_mean():
    lp = [torch.zeros(3), torch.zeros(1)]
    adv = [torch.ones(3), 2 * torch.ones(1)]
    m = [torch.ones(3), torch.ones(1)]
    loss = clipped_opd_loss(lp, lp, adv, m)
    assert abs(float(loss) - (-(1.0 + 2.0) / 2)) < 1e-6


# ------------------------------------------------------------------------------------ equivalence with the reference
@pytest.mark.parametrize("seed", range(6))
def test_multipliers_match_reference(seed):
    pairs = _random_pairs(seed)
    w, info = domain_multipliers(pairs)
    rw, rinfo = ref.domain_multipliers(pairs)
    assert set(w) == set(rw)
    assert info.global_std == pytest.approx(rinfo["global_std"], rel=1e-5)
    for d in rw:
        assert w[d] == pytest.approx(rw[d], rel=1e-5)
        got = info.domains[d]
        assert got.clipped == rinfo[d]["clipped"]
        assert got.tokens == rinfo[d]["tokens"]
        assert got.std == pytest.approx(rinfo[d]["std"], rel=1e-5)
        assert got.raw_factor == pytest.approx(rinfo[d]["raw_factor"], rel=1e-5)


def test_multipliers_match_reference_with_clipping_on_both_sides():
    g = torch.Generator().manual_seed(3)
    pairs = [("math", 0.02 * torch.randn(200, generator=g)), ("code", torch.randn(300, generator=g)),
             ("ifeval", 8.0 * torch.randn(12, generator=g))]
    w, info = domain_multipliers(pairs)
    rw, rinfo = ref.domain_multipliers(pairs)
    assert w["math"] == rw["math"] == 4.0 and w["ifeval"] == rw["ifeval"] == 0.25
    assert w["code"] == pytest.approx(rw["code"], rel=1e-6)
    assert info.domains["math"].clipped and info.domains["ifeval"].clipped and not info.domains["code"].clipped


def test_scale_and_loss_match_reference():
    g = torch.Generator().manual_seed(7)
    lengths = [5, 1, 9, 3]
    lp_new = [torch.randn(n, generator=g).mul(0.3).requires_grad_() for n in lengths]
    lp_old = [x.detach() + 0.2 * torch.randn(x.shape[0], generator=g) for x in lp_new]
    adv = [torch.randn(n, generator=g) for n in lengths]
    masks = [(torch.rand(n, generator=g) > 0.2).float() for n in lengths]
    domains = ["math", "code", "ifeval", "math"]
    weights = {"math": 2.1, "code": 0.8, "ifeval": 0.25}
    scaled, rscaled = scale_advantages(adv, domains, weights), ref.scale_advantages(adv, domains, weights)
    assert all(torch.equal(a, b) for a, b in zip(scaled, rscaled))
    loss = clipped_opd_loss(lp_new, lp_old, scaled, masks)
    rloss = ref.clipped_opd_loss(lp_new, lp_old, rscaled, masks)
    assert torch.equal(loss, rloss)
    grads = torch.autograd.grad(loss, lp_new)
    rgrads = torch.autograd.grad(rloss, lp_new)
    assert all(torch.allclose(a, b) for a, b in zip(grads, rgrads))
