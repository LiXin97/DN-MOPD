# SPDX-License-Identifier: Apache-2.0
"""Uni_OPD_utils.mopd_hook.hook.MopdHook on synthetic rollout batches: configs, targets, routing, DN-MOPD and its
controls, annealed injection, audit files and panels."""
import json

import pytest
import torch

from conftest import LABEL_MAP, N, RBS, TEACHERS, Args, make_batch
from Uni_OPD_utils.mopd_hook import core


def run(hook_module, path, samples, args=None):
    hook = hook_module.MopdHook(path)
    new, rows = hook.apply(args or Args(), samples)
    return hook, new, rows


# ------------------------------------------------------------------------------------------------ configuration
@pytest.mark.parametrize("fields,msg", [
    ({"mode": "sgv0"}, "unknown hook mode"),
    ({"n_samples": 0}, "n_samples"),
    ({"teachers": ["a", "a"]}, "distinct"),
    ({"label_map": {"math": "math_teacher"}}, "default"),
    ({"label_map": {"default": "nobody"}}, "unregistered"),
    ({"adv_norm": "per_token"}, "unknown adv_norm"),
    ({"adv_norm": "fixed_domain_scale"}, "adv_norm_fixed"),
    ({"adv_norm": "fixed_domain_scale", "adv_norm_fixed": {"math": 5.0, "code": 1.0, "ifeval": 1.0}}, "adv_norm_fixed"),
    ({"adv_norm": "fixed_domain_scale", "adv_norm_fixed": {"math": 2.0, "code": 1.0}}, "adv_norm_fixed"),
    ({"adv_norm_fixed": {"math": 2.0, "code": 1.0, "ifeval": 1.0}}, "fixed_domain_scale only"),
    ({"mode": "pool", "adv_norm": "per_domain_scale"}, "label mode only"),
    ({"inject_adv_const": 1.0}, "without an injection bank"),
])
def test_config_refuses_unregistered_settings(hook_module, write_config, fields, msg):
    with pytest.raises(ValueError, match=msg):
        hook_module.MopdHook(write_config(**fields))


def test_server_map_env_takes_precedence(hook_module, write_config, tmp_path, monkeypatch):
    m = tmp_path / "map.json"
    m.write_text(json.dumps({k: "math_teacher" for k in LABEL_MAP}))
    monkeypatch.setenv("OPD_TEACHER_SERVER_MAP", str(m))
    hook = hook_module.MopdHook(write_config())
    assert hook.label_teacher("code") == "math_teacher"


def test_apply_checks_estimator_and_slot_count(hook_module, write_config):
    hook = hook_module.MopdHook(write_config())
    with pytest.raises(ValueError, match="on_policy_distillation"):
        hook.apply(Args(estimator="grpo"), make_batch())
    with pytest.raises(ValueError, match="n-samples-per-prompt"):
        hook.apply(Args(n=4), make_batch())


# ------------------------------------------------------------------------------------------------ targets
def test_label_mode_equals_the_legacy_routed_scores_and_routes_by_domain(hook_module, write_config):
    samples = make_batch()
    hook, new, rows = run(hook_module, write_config(), samples)
    for s, t, r in zip(samples, new, rows):
        dom = s.metadata["domain"]
        assert r["applied"] == LABEL_MAP[dom] == r["label_teacher"]
        assert torch.equal(t, s.per_teacher_log_probs[LABEL_MAP[dom]])
        assert torch.equal(s.teacher_log_probs, t)
        assert not hasattr(s, "dn_adv_scale") and not hasattr(s, "inject_adv_const")    # Label: nothing attached


def test_label_mode_refuses_a_target_that_differs_from_the_legacy_route(hook_module, write_config):
    samples = make_batch()
    samples[3].teacher_log_probs = samples[3].teacher_log_probs + 0.01
    with pytest.raises(RuntimeError, match="differs from the legacy"):
        run(hook_module, write_config(), samples)


def test_pool_and_dynamic_targets(hook_module, write_config):
    samples = make_batch()
    _, new, rows = run(hook_module, write_config(mode="pool"), samples)
    for s, t, r in zip(samples, new, rows):
        assert torch.allclose(t, core.pool_logp(s.per_teacher_log_probs))
        assert r["applied"] == "pool"
    samples = make_batch(seed=1)
    _, new, rows = run(hook_module, write_config(mode="dynamic"), samples)
    for s, t, r in zip(samples, new, rows):
        means = {k: float(v.mean()) for k, v in s.per_teacher_log_probs.items()}
        best = min(sorted(means), key=lambda k: (means[k], k))
        assert r["applied"] == best and torch.equal(t, s.per_teacher_log_probs[best])


def test_a_failed_teacher_score_stops_the_batch(hook_module, write_config):
    samples = make_batch()
    samples[0].per_teacher_log_probs["ifeval_teacher"][0] = core.SENTINEL
    with pytest.raises(core.ScoreError):
        run(hook_module, write_config(mode="pool"), samples)


def test_a_sample_is_processed_once(hook_module, write_config):
    samples = make_batch()
    hook, _, _ = run(hook_module, write_config(), samples)
    with pytest.raises(RuntimeError, match="twice"):
        hook.apply(Args(), samples)


# ------------------------------------------------------------------------------------------------ DN-MOPD and controls
def _expected_factors(samples, new):
    pairs = [(s.metadata["domain"], (t - torch.tensor(s.rollout_log_probs))[torch.tensor(s.loss_mask).bool()])
             for s, t in zip(samples, new)]
    return core.dn_multipliers(pairs)[0]


def test_dn_mopd_sets_the_domain_multiplier_on_every_sample(hook_module, write_config):
    samples = make_batch()
    hook, new, rows = run(hook_module, write_config(adv_norm="per_domain_scale"), samples)
    want = _expected_factors(samples, new)
    assert set(want) == {"math", "code", "ifeval"}
    assert want["math"] > 1.0 > want["ifeval"]                     # the batch's spreads are 0.3 / 1.0 / 3.0
    for s, r in zip(samples, rows):
        d = s.metadata["domain"]
        assert s.dn_adv_scale == pytest.approx(want[d])
        assert r["dnorm_factor"] == r["dnorm_measured_factor"] == pytest.approx(want[d])
        assert 0.25 <= s.dn_adv_scale <= 4.0
    assert hook._dnorm_last["mode"] == "per_domain_scale"
    assert hook._dnorm_last["applied"] == pytest.approx(want)


def test_dn_mopd_reads_the_masked_tokens_only(hook_module, write_config):
    samples = make_batch()
    for s in samples:
        s.loss_mask = [1] * (s.response_length - 1) + [0]
    _, new, _ = run(hook_module, write_config(adv_norm="per_domain_scale"), samples)
    want = _expected_factors(samples, new)
    assert all(s.dn_adv_scale == pytest.approx(want[s.metadata["domain"]]) for s in samples)


def test_fixed_weights_apply_the_constants_and_record_the_statistic(hook_module, write_config):
    fixed = {"math": 2.0, "code": 1.0, "ifeval": 0.25}
    samples = make_batch()
    hook, new, rows = run(hook_module, write_config(adv_norm="fixed_domain_scale", adv_norm_fixed=fixed), samples)
    measured = _expected_factors(samples, new)
    for s, r in zip(samples, rows):
        d = s.metadata["domain"]
        assert s.dn_adv_scale == fixed[d] == r["dnorm_factor"]
        assert r["dnorm_measured_factor"] == pytest.approx(measured[d])


def test_observe_applies_one_and_records_the_statistic(hook_module, write_config):
    samples = make_batch()
    _, new, rows = run(hook_module, write_config(adv_norm="observe_domain_scale"), samples)
    measured = _expected_factors(samples, new)
    for s, r in zip(samples, rows):
        assert s.dn_adv_scale == 1.0 == r["dnorm_factor"]
        assert r["dnorm_measured_factor"] == pytest.approx(measured[s.metadata["domain"]])


def test_frozen_update0_uses_the_size_multipliers_from_the_recipe(hook_module, tmp_path):
    import recipe

    for size, want in {"9b": {"math": 1.768, "code": 0.888, "ifeval": 0.25},
                       "4b": {"math": 2.009, "code": 0.811, "ifeval": 0.341},
                       "2b": {"math": 2.102, "code": 0.822, "ifeval": 0.431}}.items():
        cfg = recipe.hook_config(method="frozen_update0", size=size, seed=42, updates=80,
                                 audit_dir=str(tmp_path / size))
        assert cfg["adv_norm"] == "fixed_domain_scale" and cfg["adv_norm_fixed"] == want
        path = tmp_path / f"{size}.json"
        path.write_text(json.dumps(dict(cfg, n_samples=N)))
        samples = make_batch()
        hook = hook_module.MopdHook(path)
        hook.apply(Args(), samples)
        assert {s.metadata["domain"]: s.dn_adv_scale for s in samples} == want


def test_label_hook_observe_and_legacy_label_train_on_the_same_target(hook_module, write_config):
    a, b = make_batch(seed=3), make_batch(seed=3)
    _, new_label, _ = run(hook_module, write_config(), a)
    _, new_obs, _ = run(hook_module, write_config(adv_norm="observe_domain_scale"), b)
    for x, y, s in zip(new_label, new_obs, b):
        assert torch.equal(x, y) and torch.equal(y, s.per_teacher_log_probs[LABEL_MAP[s.metadata["domain"]]])


def test_dn_needs_rollout_log_probs(hook_module, write_config):
    samples = make_batch()
    samples[2].rollout_log_probs = None
    with pytest.raises(RuntimeError, match="rollout_log_probs"):
        run(hook_module, write_config(adv_norm="per_domain_scale"), samples)


# ------------------------------------------------------------------------------------------------ annealed injection
def make_bank(tmp_path, samples, covered=(0, 2, 3, 5)):
    from Uni_OPD_utils.mopd_hook.hook import prompt_key

    entries = {}
    by_group = {}
    for s in samples:
        by_group.setdefault(s.group_index % RBS, s)
    for gi in covered:
        s = by_group[gi]
        plen = len(s.tokens) - s.response_length
        entries[prompt_key(s.prompt)] = {
            "domain": s.metadata["domain"], "teacher": "t",
            "prompt_ids": torch.tensor(s.tokens[:plen], dtype=torch.int32),
            "response_ids": torch.tensor([5000 + gi, 5001, 5002], dtype=torch.int32),
            "teacher_logp": torch.tensor([-0.5, -0.25, -0.1], dtype=torch.float32)}
    path = tmp_path / "bank.pt"
    torch.save({"meta": {}, "entries": entries}, path)
    return path


def test_annealed_injection_replaces_one_sample_per_covered_group(hook_module, write_config, tmp_path):
    base = make_batch()
    bank = make_bank(tmp_path, base)
    path = write_config(inject_bank_path=str(bank), inject_adv_const=1.0, inject_anneal_steps=40)
    hook = hook_module.MopdHook(path)
    for step, const in ((0, 1.0), (20, 0.5), (39, 1.0 - 39 / 40), (40, 0.0), (41, 0.0)):
        samples = make_batch(step=step, correct=lambda gi, j: j != 2 if gi % 2 else True)
        new, rows = hook.apply(Args(), samples)
        injected = [s for s in samples if s.inject_adv_const > 0]
        if const == 0.0:
            assert not injected and all(not r["injected"] for r in rows)
            continue
        assert len(injected) == 4                                # one per covered group (groups 0, 2, 3, 5)
        for s in injected:
            gi = s.group_index % RBS
            assert s.inject_adv_const == pytest.approx(const)
            assert s.tokens[-3:] == [5000 + gi, 5001, 5002] and s.response_length == 3
            assert s.loss_mask == [1, 1, 1] and s.response_correct is True
            assert torch.equal(s.teacher_log_probs, torch.tensor([-0.5, -0.25, -0.1]))
            # the lowest-index verifier-WRONG sample (slot 2 in odd groups), else the lowest-index sample
            assert s.index % N == (2 if gi % 2 else 0)
        assert all(s.inject_adv_const == 0.0 for s in samples if s not in injected)
        assert not any(hasattr(s, "dn_adv_scale") for s in samples)


def test_injection_refuses_missing_verdicts(hook_module, write_config, tmp_path):
    bank = make_bank(tmp_path, make_batch())
    hook = hook_module.MopdHook(write_config(inject_bank_path=str(bank), inject_adv_const=1.0, inject_anneal_steps=40))
    samples = make_batch()
    samples[9].response_correct = None
    with pytest.raises(RuntimeError, match="no verified reward"):
        hook.apply(Args(), samples)


def test_injection_skips_a_prompt_token_mismatch(hook_module, write_config, tmp_path):
    base = make_batch()
    bank = make_bank(tmp_path, base, covered=(0,))
    samples = make_batch()
    for s in samples:
        if s.group_index % RBS == 0:
            s.tokens = [7] + s.tokens[1:]
    hook = hook_module.MopdHook(write_config(inject_bank_path=str(bank), inject_adv_const=1.0, inject_anneal_steps=40))
    hook.apply(Args(), samples)
    assert not any(s.inject_adv_const > 0 for s in samples)
    assert hook.inject_stats["prompt_token_mismatch"] == 1


def test_injection_checks_the_bank_hash_and_mode(hook_module, write_config, tmp_path):
    bank = make_bank(tmp_path, make_batch())
    with pytest.raises(ValueError, match="hash"):
        hook_module.MopdHook(write_config(inject_bank_path=str(bank), inject_bank_sha256="0" * 64,
                                          inject_adv_const=1.0, inject_anneal_steps=40))
    with pytest.raises(ValueError, match="label mode"):
        hook_module.MopdHook(write_config(mode="pool", inject_bank_path=str(bank), inject_adv_const=1.0,
                                          inject_anneal_steps=40))
    with pytest.raises(ValueError, match="separate"):
        hook_module.MopdHook(write_config(adv_norm="per_domain_scale", inject_bank_path=str(bank),
                                          inject_adv_const=1.0, inject_anneal_steps=40))


# ------------------------------------------------------------------------------------------------ audit and panel
def test_finish_writes_the_audit_and_prints_the_panel(hook_module, write_config, tmp_path, capsys):
    samples = make_batch()
    hook, new, rows = run(hook_module, write_config(adv_norm="per_domain_scale"), samples)
    panel = hook.finish(rows)
    out = capsys.readouterr().out
    line = [x for x in out.splitlines() if x.startswith(hook_module.PANEL_PREFIX)][-1]
    printed = json.loads(line[len(hook_module.PANEL_PREFIX) + 1:])
    assert printed["dnorm"]["applied"] == pytest.approx(panel["dnorm"]["applied"])
    assert set(printed["dnorm"]["by_domain"]) == {"math", "code", "ifeval"}
    assert printed["label_agreement"] == 1.0 and printed["n"] == N * RBS
    files = list((tmp_path / "audit").glob("samples_*.json"))
    assert len(files) == 1
    audit = json.loads(files[0].read_text())
    assert audit["mode"] == "label" and len(audit["rows"]) == N * RBS
    assert all("dnorm_factor" in r for r in audit["rows"])


def test_resume_restores_the_switch_memory(hook_module, write_config, capsys):
    path = write_config(mode="dynamic")
    hook, _, rows = run(hook_module, path, make_batch(step=0))
    hook.finish(rows)
    resumed = hook_module.MopdHook(path)
    resumed.apply(Args(), make_batch(step=1))
    assert "MOPD_HOOK_RESTORED files=1" in capsys.readouterr().out
    assert len(resumed.last_applied) == N * RBS


def test_entry_points_read_the_config_from_the_environment(hook_module, write_config, monkeypatch):
    monkeypatch.setattr(hook_module, "_STATE", None)
    monkeypatch.delenv(hook_module.CONFIG_ENV, raising=False)
    with pytest.raises(RuntimeError, match=hook_module.CONFIG_ENV):
        hook_module._state()
    monkeypatch.setenv(hook_module.CONFIG_ENV, str(write_config(adv_norm="observe_domain_scale")))
    state = hook_module._state()
    assert state.mode == "label" and state.adv_norm == "observe_domain_scale" and state.names == sorted(TEACHERS)
    monkeypatch.setattr(hook_module, "_STATE", None)
