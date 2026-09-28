# SPDX-License-Identifier: Apache-2.0
"""Modes (dn / observe / fixed / frozen), bounds, state and diagnostics of DomainNormalizer."""
import json
import math

import pytest
import torch

from dn_mopd import DomainNormalizer, domain_multipliers

DOMAINS = ("math", "code", "ifeval")


def _pairs(spreads, n=200, seed=0, domains=DOMAINS):
    g = torch.Generator().manual_seed(seed)
    return [(d, s * torch.randn(n, generator=g)) for d, s in zip(domains, spreads)]


def test_dn_mode_applies_measured_factor():
    w, diag = DomainNormalizer(DOMAINS).multipliers(_pairs([0.5, 1.0, 3.0]))
    for d in DOMAINS:
        assert w[d] == diag.domains[d].factor == diag.domains[d].applied
    assert w["math"] > 1.0 > w["ifeval"]


def test_observe_mode_is_label_but_measures():
    pairs = _pairs([0.5, 1.0, 3.0])
    w_dn, diag_dn = DomainNormalizer(DOMAINS).multipliers(pairs)
    w_obs, diag_obs = DomainNormalizer(DOMAINS, mode="observe").multipliers(pairs)
    assert all(v == 1.0 for v in w_obs.values())
    for d in DOMAINS:
        assert diag_obs.domains[d].factor == diag_dn.domains[d].factor == w_dn[d]


def test_fixed_mode_applies_constants_and_validates():
    fixed = {"math": 2.0, "code": 1.0, "ifeval": 0.25}
    w, diag = DomainNormalizer(DOMAINS, mode="fixed", fixed=fixed).multipliers(_pairs([0.5, 1.0, 3.0]))
    assert w == fixed
    assert diag.domains["math"].factor != 2.0                       # the DN statistic is still measured
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, mode="fixed", fixed={"math": 2.0, "code": 1.0})          # ifeval missing
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, mode="fixed", fixed={"math": 5.0, "code": 1.0, "ifeval": 1.0})  # outside bounds
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, mode="dn", fixed=fixed)
    with pytest.raises(ValueError):                                 # discovered domain without a constant
        DomainNormalizer(mode="fixed", fixed={"math": 2.0}).multipliers(_pairs([1.0, 2.0], domains=("math", "code")))


def test_fixed_mode_reproduces_paper_frozen_update0_control():
    calib_4b = {"math": 2.009, "code": 0.811, "ifeval": 0.341}   # the paper's frozen update-0 multipliers, 4B
    w, _ = DomainNormalizer(DOMAINS, mode="fixed", fixed=calib_4b).multipliers(_pairs([0.5, 1.0, 3.0]))
    assert w == calib_4b


def test_frozen_mode_holds_first_measurement():
    norm = DomainNormalizer(DOMAINS, mode="frozen")
    w0, _ = norm.multipliers(_pairs([0.5, 1.0, 3.0], seed=0))
    w1, d1 = norm.multipliers(_pairs([2.0, 1.0, 0.2], seed=1))       # very different batch
    assert w1 == w0
    assert d1.domains["math"].factor != w0["math"]                    # still measured and reported
    assert norm.state_dict()["frozen_step"] == {d: 0 for d in DOMAINS}


def test_frozen_mode_freezes_a_domain_at_its_first_measurable_batch():
    norm = DomainNormalizer(DOMAINS, mode="frozen")
    g = torch.Generator().manual_seed(0)
    w0, d0 = norm.multipliers([("math", 0.5 * torch.randn(50, generator=g)), ("code", torch.randn(50, generator=g)),
                               ("ifeval", torch.tensor([0.7]))])       # ifeval: one token, not measurable
    assert w0["ifeval"] == 1.0 and d0.domains["ifeval"].tokens == 1
    w1, _ = norm.multipliers(_pairs([0.5, 1.0, 3.0], seed=2))
    assert w1["math"] == w0["math"] and w1["code"] == w0["code"]
    assert w1["ifeval"] < 1.0
    w2, _ = norm.multipliers(_pairs([0.5, 1.0, 0.1], seed=3))
    assert w2["ifeval"] == w1["ifeval"]
    assert norm.state_dict()["frozen_step"] == {"math": 0, "code": 0, "ifeval": 1}


def test_state_dict_round_trip_resumes_frozen_values():
    a = DomainNormalizer(DOMAINS, mode="frozen")
    w0, _ = a.multipliers(_pairs([0.5, 1.0, 3.0]))
    state = json.loads(json.dumps(a.state_dict()))                      # survives a JSON checkpoint
    b = DomainNormalizer(DOMAINS, mode="frozen")
    b.load_state_dict(state)
    w1, d1 = b.multipliers(_pairs([3.0, 1.0, 0.5], seed=9))
    assert w1 == w0 and d1.step == 1
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, mode="dn").load_state_dict(state)


@pytest.mark.parametrize("bounds", [(0.5, 2.0), (1.0, 1.0), (0.0, math.inf)])
def test_configurable_bounds(bounds):
    lo, hi = bounds
    w, diag = DomainNormalizer(DOMAINS, bounds=bounds).multipliers(_pairs([0.01, 1.0, 50.0]))
    for d in DOMAINS:
        raw = diag.domains[d].raw_factor
        assert w[d] == min(hi, max(lo, raw))
        assert diag.domains[d].clipped == (w[d] != raw)
    ref, _ = domain_multipliers(_pairs([0.01, 1.0, 50.0]), lower=lo, upper=hi)
    assert ref == w


@pytest.mark.parametrize("bounds", [(2.0, 1.0), (-0.1, 4.0), (float("nan"), 4.0)])
def test_invalid_bounds_raise(bounds):
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, bounds=bounds)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        DomainNormalizer(DOMAINS, mode="per_domain_scale")
    with pytest.raises(ValueError):
        DomainNormalizer(("math", "math"))
    with pytest.raises(ValueError):
        DomainNormalizer(sync=True)                                    # sync needs explicit domains
    with pytest.raises(TypeError):
        DomainNormalizer(DOMAINS, sync="yes")
    with pytest.raises(ValueError):
        DomainNormalizer(("math",)).multipliers([("code", torch.randn(4))])


def test_diagnostics_serialize_and_step_counts():
    norm = DomainNormalizer(DOMAINS)
    _, d0 = norm.multipliers(_pairs([0.5, 1.0, 3.0]))
    _, d1 = norm.multipliers(_pairs([0.5, 1.0, 3.0], seed=1))
    assert (d0.step, d1.step, norm.step) == (0, 1, 2)
    blob = json.loads(json.dumps(d0.to_dict()))
    assert set(blob["domains"]) == set(DOMAINS) and blob["mode"] == "dn"
    metrics = d0.to_metrics()
    assert metrics["dn/math/applied"] == d0.domains["math"].applied
    assert metrics["dn/global_tokens"] == 600.0
