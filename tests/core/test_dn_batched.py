# SPDX-License-Identifier: Apache-2.0
"""The padded (B, T) API agrees with the per-response list API."""
import pytest
import torch

from dn_mopd import (
    DomainNormalizer,
    batch_statistics,
    clipped_opd_loss,
    clipped_opd_loss_batched,
    domain_multipliers,
    domain_multipliers_batched,
    list_statistics,
    reverse_kl_advantages,
    scale_advantages,
    scale_advantages_batched,
)

DOMAINS = ("math", "code", "ifeval")


def _padded(seed, batch=32, max_len=24, garbage=float("nan")):
    """Padded batch with per-domain spreads; masked positions hold `garbage`."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 3, (batch,), generator=g)
    spread = torch.tensor([0.4, 1.0, 5.0])[ids]
    lengths = torch.randint(0, max_len + 1, (batch,), generator=g)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    r = spread[:, None] * torch.randn(batch, max_len, generator=g) - 0.1
    r = torch.where(mask, r, torch.full_like(r, garbage))
    return r, mask, ids


def _as_pairs(r, mask, ids):
    return [(DOMAINS[int(i)], r[b][mask[b]]) for b, i in enumerate(ids)]


@pytest.mark.parametrize("seed", range(4))
def test_batched_equals_list(seed):
    r, mask, ids = _padded(seed)
    w_b, diag_b = domain_multipliers_batched(r, mask, ids, DOMAINS)
    w_l, diag_l = domain_multipliers(_as_pairs(r, mask, ids))
    assert w_b.dtype == torch.float32 and w_b.shape == (3,)
    for i, d in enumerate(DOMAINS):
        if d in w_l:
            assert float(w_b[i]) == pytest.approx(w_l[d], rel=1e-6)
            assert diag_b.domains[d].tokens == diag_l.domains[d].tokens
            assert diag_b.domains[d].std == pytest.approx(diag_l.domains[d].std, rel=1e-12)
    assert diag_b.global_std == pytest.approx(diag_l.global_std, rel=1e-12)


def test_statistics_layouts_agree_and_ignore_padding():
    r, mask, ids = _padded(11, garbage=float("inf"))
    sb = batch_statistics(r, mask, ids, 3)
    sl = list_statistics(_as_pairs(r, mask, ids), DOMAINS)
    assert sb.dtype == torch.float64 and sb.shape == (4, 3)
    assert torch.allclose(sb, sl, rtol=1e-12, atol=1e-12)
    assert sb[0].sum() == r.shape[0] and sb[1].sum() == mask.sum()


def test_scale_batched_equals_list():
    r, mask, ids = _padded(2, garbage=0.0)
    weights = torch.tensor([2.0, 0.9, 0.25])
    out = scale_advantages_batched(r, ids, weights)
    ref = scale_advantages(list(r), [DOMAINS[int(i)] for i in ids], dict(zip(DOMAINS, weights.tolist())))
    assert torch.equal(out, torch.stack(ref))
    assert not out.requires_grad


def test_loss_batched_equals_list_value_and_gradient():
    g = torch.Generator().manual_seed(5)
    b, t = 6, 10
    mask = torch.arange(t)[None, :] < torch.tensor([10, 1, 4, 7, 0, 3])[:, None]
    new = (0.3 * torch.randn(b, t, generator=g)).requires_grad_()
    old = new.detach() + 0.3 * torch.randn(b, t, generator=g)
    adv = torch.randn(b, t, generator=g)
    loss_b = clipped_opd_loss_batched(new, old, adv, mask)
    loss_l = clipped_opd_loss(list(new), list(old), list(adv), list(mask))
    assert torch.allclose(loss_b, loss_l, rtol=1e-6, atol=1e-7)
    (gb,) = torch.autograd.grad(loss_b, new)
    (gl,) = torch.autograd.grad(loss_l, new)
    assert torch.allclose(gb, gl, rtol=1e-5, atol=1e-7)


def test_loss_batched_ignores_nan_padding_including_gradients():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    new = torch.tensor([[-0.5, -1.0, float("nan")], [-0.2, float("nan"), float("nan")]], requires_grad=True)
    old = torch.tensor([[-0.6, -0.9, float("nan")], [-0.3, float("inf"), float("nan")]])
    adv = torch.tensor([[1.0, -1.0, float("nan")], [0.5, float("nan"), 3.0]])
    loss = clipped_opd_loss_batched(new, old, adv, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all()
    assert new.grad[0, 2] == 0 and new.grad[1, 1] == 0


def test_reverse_kl_advantages_masks_and_detaches():
    t = torch.tensor([[-1.0, -2.0, float("nan")]])
    s = torch.tensor([[-1.5, -1.0, -3.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    a = reverse_kl_advantages(t, s, valid)
    assert torch.equal(a, torch.tensor([[0.5, -1.0, 0.0]])) and not a.requires_grad


def test_normalizer_batched_requires_domains_and_valid_inputs():
    r, mask, ids = _padded(0)
    with pytest.raises(ValueError):
        DomainNormalizer().multipliers_batched(r, mask, ids)
    with pytest.raises(ValueError):
        domain_multipliers_batched(r, mask, ids + 3, DOMAINS)            # domain id out of range
    with pytest.raises(ValueError):
        domain_multipliers_batched(r, mask[:, :-1], ids, DOMAINS)        # shape mismatch
    bad = r.clone()
    bad[mask] = float("nan")
    with pytest.raises(ValueError):
        domain_multipliers_batched(bad, mask, ids, DOMAINS)              # non-finite value at a valid token
