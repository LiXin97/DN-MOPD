# SPDX-License-Identifier: Apache-2.0
"""The per-sample DN-MOPD multiplier from the hook to the loss: the rollout manager carries dn_adv_scale /
inject_adv_const into train_data and into every data-parallel shard, and the OPD advantages are scaled (or replaced)
before the unchanged PPO-clipped policy loss."""
from types import SimpleNamespace

import pytest
import torch

from conftest import Args, make_batch


def _rollout_manager_shim(monkeypatch, post_process):
    """An uninitialised RolloutManager with only what _convert_samples_to_train_data needs."""
    import miles.ray.rollout as rollout

    cls = getattr(rollout.RolloutManager, "__ray_actor_class__", None) or \
        rollout.RolloutManager.__ray_metadata__.modified_class
    shim = cls.__new__(cls)
    shim.args = SimpleNamespace(advantage_estimator="on_policy_distillation", balance_data=False)
    shim.custom_convert_samples_to_train_data_func = None
    shim.custom_reward_post_process_func = post_process
    monkeypatch.setattr(rollout.ray, "put", lambda x: x)
    return rollout, shim


def _hooked_samples(hook_module, write_config, **fields):
    samples = make_batch()
    hook = hook_module.MopdHook(write_config(**fields))
    new, _ = hook.apply(Args(), samples)
    return samples, new


def test_train_data_carries_the_multiplier_and_every_dp_shard_gets_it(hook_module, write_config, monkeypatch):
    samples, new = _hooked_samples(hook_module, write_config, adv_norm="per_domain_scale")
    rollout, shim = _rollout_manager_shim(monkeypatch, lambda args, s: (new, new))
    data = shim._convert_samples_to_train_data(samples)
    assert data["dn_adv_scale"] == [s.dn_adv_scale for s in samples]
    assert "inject_adv_const" not in data
    assert all(torch.equal(a, b) for a, b in zip(data["teacher_log_probs"], new))
    refs = shim._split_train_data_by_dp(data, 3)
    seen = {}
    for ref in refs:
        shard = ref.inner                                           # Box(ray.put(...)); ray.put is the identity here
        part = list(shard["partition"])
        assert shard["dn_adv_scale"] == [data["dn_adv_scale"][i] for i in part]
        seen.update({i: v for i, v in zip(part, shard["dn_adv_scale"])})
    assert len(seen) == len(samples)


def test_label_train_data_has_no_multiplier(hook_module, write_config, monkeypatch):
    samples, new = _hooked_samples(hook_module, write_config)
    _, shim = _rollout_manager_shim(monkeypatch, lambda args, s: (new, new))
    data = shim._convert_samples_to_train_data(samples)
    assert "dn_adv_scale" not in data and "inject_adv_const" not in data


def _opd_rollout_data(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = [3 + i % 4 for i in range(n)]
    teacher = [-torch.rand(L, generator=g) * 2 for L in lengths]
    teacher[1][0] = -100.0                                          # a failed teacher position (sentinel)
    student = [-torch.rand(L, generator=g) * 2 for L in lengths]
    return {"log_probs": student, "teacher_log_probs": [torch.cat([torch.zeros(5), t]) for t in teacher],
            "response_lengths": lengths, "loss_masks": [torch.ones(L) for L in lengths],
            "total_lengths": [L + 5 for L in lengths], "rewards": [0.0] * n}, teacher, student


def _args():
    return SimpleNamespace(use_rollout_logprobs=False, kl_coef=0.0, advantage_estimator="on_policy_distillation",
                           use_opd_margin_shift=False, use_opd_margin_mask_greedy=False, normalize_advantages=False)


def test_opd_advantages_are_scaled_by_the_per_sample_multiplier(capsys):
    from miles.backends.training_utils.loss import compute_advantages_and_returns

    ps = SimpleNamespace(cp_size=1, cp_rank=0)
    base, teacher, student = _opd_rollout_data()
    compute_advantages_and_returns(_args(), ps, base)
    label_adv = base["advantages"]
    for a, t, s in zip(label_adv, teacher, student):
        want = (t - s) * (t != -100.0).float()
        assert torch.equal(a, want)
    assert "MOPD_DN_APPLIED" not in capsys.readouterr().out

    w = [2.0, 0.25, 4.0, 1.0, 0.5, 1.5]
    scaled, _, _ = _opd_rollout_data()
    scaled["dn_adv_scale"] = w
    compute_advantages_and_returns(_args(), ps, scaled)
    for a, b, wi in zip(scaled["advantages"], label_adv, w):
        assert torch.equal(a, b * wi)
    assert scaled["advantages"][1][0] == 0.0                       # the sentinel position stays 0
    assert scaled["returns"] is scaled["advantages"]
    assert "MOPD_DN_APPLIED samples=6 min=0.25 max=4" in capsys.readouterr().out

    ones, _, _ = _opd_rollout_data()
    ones["dn_adv_scale"] = [1.0] * 6                               # observe: identical to Label
    compute_advantages_and_returns(_args(), ps, ones)
    assert all(torch.equal(a, b) for a, b in zip(ones["advantages"], label_adv))


def test_injected_samples_take_the_constant_advantage():
    from miles.backends.training_utils.loss import compute_advantages_and_returns

    data, _, _ = _opd_rollout_data()
    ref, _, _ = _opd_rollout_data()
    compute_advantages_and_returns(_args(), SimpleNamespace(cp_size=1, cp_rank=0), ref)
    data["inject_adv_const"] = [0.0, 0.5, 0.0, 0.0, 1.0, 0.0]
    compute_advantages_and_returns(_args(), SimpleNamespace(cp_size=1, cp_rank=0), data)
    for i, (a, b) in enumerate(zip(data["advantages"], ref["advantages"])):
        c = data["inject_adv_const"][i]
        assert torch.equal(a, torch.full_like(b, c) if c > 0 else b)


def test_length_mismatch_is_refused():
    from miles.backends.training_utils.loss import apply_dn_adv_scale, apply_inject_adv_const

    adv = [torch.ones(2), torch.ones(3)]
    with pytest.raises(RuntimeError, match="dn_adv_scale"):
        apply_dn_adv_scale(adv, [1.0])
    with pytest.raises(RuntimeError, match="inject_adv_const"):
        apply_inject_adv_const(adv, [0.0, 0.0, 0.0])
    assert apply_dn_adv_scale(adv, None) is adv and apply_inject_adv_const(adv, None) is adv


def test_scaled_advantages_enter_the_unchanged_clipped_loss():
    """PPO-clip with A~ = w_d * (teacher - old student) equals the reference clipped OPD loss, per token."""
    from miles.utils.ppo_utils import compute_policy_loss

    g = torch.Generator().manual_seed(7)
    L = 11
    old = -torch.rand(L, generator=g)
    new = old + (torch.rand(L, generator=g) - 0.5) * 0.8          # some ratios outside [0.8, 1.2]
    teacher = -torch.rand(L, generator=g) * 2
    for w in (0.25, 1.0, 2.102):
        adv = (teacher - old) * w
        pg_loss, clipfrac = compute_policy_loss(old - new, adv, 0.2, 0.2)
        ratio = torch.exp(new - old)
        ref = -torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
        assert torch.allclose(pg_loss, ref, atol=1e-6)
        # the gradient direction of every token is that of Label's (same sign), its size w times Label's
        adv1 = teacher - old
        assert torch.equal(torch.sign(adv), torch.sign(adv1))
    assert float(clipfrac.mean()) > 0.0
