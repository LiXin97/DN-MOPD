# SPDX-License-Identifier: Apache-2.0
"""Clipped OPD loss: reduction, clipping, normalization over shards, sign preservation, Label equivalence."""
import pytest
import torch

from dn_mopd import (
    DomainNormalizer,
    clipped_opd_loss,
    clipped_opd_loss_batched,
    reverse_kl_advantages,
    scale_advantages_batched,
)

DOMAINS = ("math", "code", "ifeval")


def test_every_response_weighs_the_same_whatever_its_length():
    # a 100-token response with advantage 1 and a 1-token response with advantage 3: mean of means = 2
    lp = torch.zeros(2, 100)
    adv = torch.zeros(2, 100)
    adv[0] = 1.0
    adv[1, 0] = 3.0
    mask = torch.zeros(2, 100, dtype=torch.bool)
    mask[0] = True
    mask[1, 0] = True
    assert float(clipped_opd_loss_batched(lp, lp, adv, mask)) == pytest.approx(-2.0)


def test_num_responses_normalizes_over_shards():
    g = torch.Generator().manual_seed(0)
    new = 0.1 * torch.randn(8, 6, generator=g)
    old = new + 0.05 * torch.randn(8, 6, generator=g)
    adv = torch.randn(8, 6, generator=g)
    mask = torch.rand(8, 6, generator=g) > 0.3
    full = clipped_opd_loss_batched(new, old, adv, mask)
    shards = sum(clipped_opd_loss_batched(new[s], old[s], adv[s], mask[s], num_responses=8)
                 for s in (slice(0, 3), slice(3, 8)))
    assert torch.allclose(full, shards, atol=1e-7)
    listed = clipped_opd_loss(list(new[:3]), list(old[:3]), list(adv[:3]), list(mask[:3]), num_responses=8)
    assert torch.allclose(listed, clipped_opd_loss_batched(new[:3], old[:3], adv[:3], mask[:3], num_responses=8))


@pytest.mark.parametrize("adv_sign, log_ratio, clipped", [
    (+1.0, 0.5, True),     # A > 0, ratio > 1 + eps: clipped, no gradient
    (+1.0, -0.5, False),   # A > 0, ratio < 1 - eps: not clipped
    (-1.0, -0.5, True),    # A < 0, ratio < 1 - eps: clipped, no gradient
    (-1.0, 0.5, False),    # A < 0, ratio > 1 + eps: not clipped
    (+1.0, 0.1, False),    # inside the trust region
])
def test_ppo_clipping(adv_sign, log_ratio, clipped):
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), log_ratio, requires_grad=True)
    adv = torch.full((1, 1), adv_sign)
    loss = clipped_opd_loss_batched(new, old, adv, torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    ratio = torch.exp(torch.tensor(log_ratio))
    if clipped:
        assert float(new.grad) == 0.0
        bound = 1.2 if log_ratio > 0 else 0.8
        assert float(loss.detach()) == pytest.approx(-bound * adv_sign)
    else:
        assert float(new.grad) == pytest.approx(float(-adv_sign * ratio))
        assert float(loss.detach()) == pytest.approx(float(-adv_sign * ratio))


@pytest.mark.parametrize("eps_high, clipped", [(0.2, True), (0.28, False)])
def test_asymmetric_clip_range(eps_high, clipped):
    new = torch.full((1, 1), 0.24, requires_grad=True)             # ratio exp(0.24) = 1.271
    loss = clipped_opd_loss_batched(new, torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
                                    eps_low=0.2, eps_high=eps_high)
    loss.backward()
    assert (float(new.grad) == 0.0) == clipped


def test_invalid_clip_range_raises():
    x = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        clipped_opd_loss_batched(x, x, x, torch.ones(1, 1), eps_low=1.5)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 3, (12,), generator=g)
    teacher = -torch.rand(12, 9, generator=g) * torch.tensor([4.0, 1.0, 0.2])[ids][:, None]
    rollout = -torch.rand(12, 9, generator=g)
    old = rollout + 0.01 * torch.randn(12, 9, generator=g)
    new = (old + 0.1 * torch.randn(12, 9, generator=g)).requires_grad_()
    mask = torch.rand(12, 9, generator=g) > 0.2
    return ids, teacher, rollout, old, new, mask


def test_scaling_preserves_advantage_signs_and_gradient_directions():
    ids, teacher, rollout, old, new, mask = _batch()
    weights, _ = DomainNormalizer(DOMAINS).multipliers_batched(teacher - rollout, mask, ids)
    adv = reverse_kl_advantages(teacher, old, mask)
    scaled = scale_advantages_batched(adv, ids, weights)
    assert torch.equal(torch.sign(scaled), torch.sign(adv))
    (g_plain,) = torch.autograd.grad(clipped_opd_loss_batched(new, old, adv, mask), new)
    (g_dn,) = torch.autograd.grad(clipped_opd_loss_batched(new, old, scaled, mask), new)
    both = (g_plain != 0) & (g_dn != 0)
    assert torch.equal(torch.sign(g_plain[both]), torch.sign(g_dn[both]))


def test_label_equivalence_unit_weights_leave_the_update_unchanged():
    ids, teacher, rollout, old, new, mask = _batch(1)
    adv = reverse_kl_advantages(teacher, old, mask)
    weights, diag = DomainNormalizer(DOMAINS, mode="observe").multipliers_batched(teacher - rollout, mask, ids)
    assert torch.equal(weights, torch.ones(3)) and any(v.factor != 1.0 for v in diag.domains.values())
    assert torch.equal(scale_advantages_batched(adv, ids, weights), adv)
    assert torch.equal(clipped_opd_loss_batched(new, old, scale_advantages_batched(adv, ids, weights), mask),
                       clipped_opd_loss_batched(new, old, adv, mask))


def test_dn_loss_equals_loss_on_prescaled_advantages():
    ids, teacher, rollout, old, new, mask = _batch(2)
    weights, _ = DomainNormalizer(DOMAINS).multipliers_batched(teacher - rollout, mask, ids)
    adv = reverse_kl_advantages(teacher, old, mask)
    manual = adv * weights[ids][:, None]
    assert torch.allclose(clipped_opd_loss_batched(new, old, scale_advantages_batched(adv, ids, weights), mask),
                          clipped_opd_loss_batched(new, old, manual, mask))
