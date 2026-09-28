# SPDX-License-Identifier: Apache-2.0
"""Pure functions of Uni_OPD_utils.mopd_hook.core: teacher targets and DN-MOPD multipliers."""
import math

import pytest
import torch

from Uni_OPD_utils.mopd_hook import core


def rows3(length=4):
    return {"a": torch.tensor([-0.1, -0.2, -0.3, -0.4][:length]),
            "b": torch.tensor([-1.0, -0.5, -0.1, -2.0][:length]),
            "c": torch.tensor([-0.3, -0.3, -0.3, -0.3][:length])}


def test_validated_rows_accepts_complete_scores():
    rows = core.validated_rows(rows3(), ["a", "b", "c"], 4)
    assert sorted(rows) == ["a", "b", "c"] and all(r.dtype == torch.float32 for r in rows.values())


@pytest.mark.parametrize("bad", [
    lambda r: r.pop("c"),                                              # a teacher missing
    lambda r: r.__setitem__("d", torch.zeros(4)),                      # an unregistered teacher
    lambda r: r.__setitem__("a", torch.zeros(3)),                      # wrong length
    lambda r: r.__setitem__("a", torch.tensor([0.0, float("nan"), 0.0, 0.0])),
    lambda r: r.__setitem__("b", torch.tensor([-1.0, core.SENTINEL, -1.0, -1.0])),   # a failed teacher position
])
def test_validated_rows_refuses_unusable_scores(bad):
    rows = rows3()
    bad(rows)
    with pytest.raises(core.ScoreError):
        core.validated_rows(rows, ["a", "b", "c"], 4)


def test_response_mask():
    assert core.response_mask(None, 3).tolist() == [True, True, True]
    assert core.response_mask([1, 0, 1], 3).tolist() == [True, False, True]
    with pytest.raises(core.ScoreError):
        core.response_mask([1, 1], 3)
    with pytest.raises(core.ScoreError):
        core.response_mask([0, 0, 0], 3)


def test_mean_scores_and_argmin_with_tie_break():
    rows = rows3()
    scores = core.mean_scores(rows, torch.ones(4, dtype=torch.bool))
    assert scores["b"] == pytest.approx(-0.9)
    assert core.argmin_name(scores) == "b"
    assert core.argmin_name({"z": -1.0, "a": -1.0}) == "a"            # exact tie: lexicographically first
    with pytest.raises(core.ScoreError):
        core.argmin_name({"a": float("nan")})


def test_pool_is_the_uniform_arithmetic_mixture():
    rows = rows3()
    pool = core.pool_logp(rows)
    manual = torch.log(sum(torch.exp(r) for r in rows.values()) / 3)
    assert torch.allclose(pool, manual, atol=1e-6)
    assert torch.allclose(pool, core.mixture_logp(rows, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), atol=1e-6)
    assert torch.equal(core.mixture_logp(rows, {"a": 0.0, "b": 1.0, "c": 0.0}), rows["b"])


def test_teacher_log_probs_modes():
    rows = rows3()
    scores = core.mean_scores(rows, torch.ones(4, dtype=torch.bool))
    new, name = core.teacher_log_probs("label", rows, scores, label_teacher="c")
    assert name == "c" and torch.equal(new, rows["c"])
    new, name = core.teacher_log_probs("dynamic", rows, scores)
    assert name == "b" and torch.equal(new, rows["b"])
    new, name = core.teacher_log_probs("pool", rows, scores)
    assert name == "pool" and torch.allclose(new, core.pool_logp(rows))
    with pytest.raises(core.ScoreError):
        core.teacher_log_probs("label", rows, scores, label_teacher="missing")
    with pytest.raises(ValueError):
        core.teacher_log_probs("sgv0", rows, scores)


def _pairs(spec):
    return [(d, torch.tensor(v, dtype=torch.float32)) for d, v in spec]


def test_dn_multipliers_are_the_pooled_over_domain_std_ratio():
    pairs = _pairs([("math", [0.0, 0.2, -0.2]), ("math", [0.1]), ("code", [1.0, -1.0, 0.5, -0.5]),
                    ("ifeval", [3.0, -3.0])])
    w, info = core.dn_multipliers(pairs)
    allv = torch.cat([v for _, v in pairs])
    g = float(allv.std(unbiased=False))
    assert info["global_std"] == pytest.approx(g)
    for d in ("math", "code", "ifeval"):
        t = torch.cat([v for dd, v in pairs if dd == d])
        sd = float(t.std(unbiased=False))
        raw = g / sd
        assert info["by_domain"][d]["std"] == pytest.approx(sd)
        assert info["by_domain"][d]["raw_factor"] == pytest.approx(raw)
        assert w[d] == pytest.approx(min(4.0, max(0.25, raw)))
        assert info["by_domain"][d]["tokens"] == t.numel()
    assert w["math"] > 1.0 > w["ifeval"]                                # low-spread domain up, high-spread down


def test_dn_multipliers_clip_to_quarter_and_four():
    pairs = _pairs([("math", [1e-3, -1e-3] * 10), ("code", [1.0, -1.0] * 10), ("ifeval", [50.0, -50.0] * 10)])
    w, info = core.dn_multipliers(pairs)
    assert w["math"] == 4.0 and info["by_domain"]["math"]["clipped"] and info["by_domain"]["math"]["raw_factor"] > 4
    assert w["code"] == pytest.approx(min(4.0, max(0.25, info["by_domain"]["code"]["raw_factor"])))
    assert w["ifeval"] == pytest.approx(info["by_domain"]["ifeval"]["raw_factor"]) or w["ifeval"] == 0.25
    # a few very wide tokens among many narrow ones: sigma_all / sigma_ifeval ~ 0.14 -> clipped to 0.25
    pairs = _pairs([("math", [1.0, -1.0] * 50), ("ifeval", [100.0, -100.0])])
    w, info = core.dn_multipliers(pairs)
    assert info["by_domain"]["ifeval"]["raw_factor"] < 0.25
    assert w["ifeval"] == 0.25 and info["by_domain"]["ifeval"]["clipped"]
    assert (core.DN_CLIP_LOWER, core.DN_CLIP_UPPER) == (0.25, 4.0)


@pytest.mark.parametrize("spec,expect", [
    ([("math", [0.5])], {"math": 1.0}),                                         # one token: no std
    ([("math", [0.3, 0.3, 0.3]), ("code", [1.0, -1.0])], {"math": 1.0}),       # zero domain std
    ([("math", [0.2, 0.2]), ("code", [0.2, 0.2])], {"math": 1.0, "code": 1.0}),  # zero pooled std
    ([], {}),                                                                   # empty batch
])
def test_dn_multipliers_degenerate_statistics_give_one(spec, expect):
    w, _ = core.dn_multipliers(_pairs(spec))
    for d, v in expect.items():
        assert w[d] == v


def test_dn_multipliers_single_domain_is_one():
    w, _ = core.dn_multipliers(_pairs([("math", [0.1, -0.4, 0.9]), ("math", [2.0, -1.0])]))
    assert w == {"math": pytest.approx(1.0)}


def test_dn_multipliers_match_the_standalone_package_when_installed():
    dn = pytest.importorskip("dn_mopd")
    fn = getattr(dn, "domain_multipliers", None)
    if fn is None:
        pytest.skip("dn_mopd exposes no domain_multipliers()")
    pairs = _pairs([("math", [0.0, 0.2, -0.2, 0.05]), ("code", [1.0, -1.0, 0.5]), ("ifeval", [3.0, -3.0, 0.1])])
    ours, _ = core.dn_multipliers(pairs)
    try:
        theirs = fn(pairs)
    except TypeError:
        pytest.skip("dn_mopd.domain_multipliers has another signature")
    theirs = theirs[0] if isinstance(theirs, tuple) else theirs
    for d in ours:
        assert float(theirs[d]) == pytest.approx(ours[d], rel=1e-6)
    assert math.isfinite(sum(ours.values()))
