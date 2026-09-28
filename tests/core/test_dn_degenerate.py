# SPDX-License-Identifier: Apache-2.0
"""Degenerate batches: no tokens, one token, zero spread, missing domains, large offsets."""
import pytest
import torch

import dn_reference_supplement as ref
from dn_mopd import DomainNormalizer, domain_multipliers, domain_multipliers_batched

DOMAINS = ("math", "code", "ifeval")


def test_empty_batch():
    w, diag = domain_multipliers([])
    assert w == {} and diag.global_std == 0.0 and diag.global_tokens == 0


def test_response_with_zero_tokens_counts_as_present_with_multiplier_one():
    g = torch.Generator().manual_seed(0)
    pairs = [("math", torch.randn(30, generator=g)), ("code", torch.zeros(0)), ("code", torch.zeros(0))]
    w, diag = domain_multipliers(pairs)
    rw, rinfo = ref.domain_multipliers(pairs)
    assert w == rw == {"math": 1.0, "code": 1.0}            # math alone: sigma_math == sigma_all
    assert diag.domains["code"].responses == 2 and diag.domains["code"].tokens == 0


def test_single_token_domain_and_single_token_batch():
    g = torch.Generator().manual_seed(1)
    w, diag = domain_multipliers([("math", torch.randn(40, generator=g)), ("ifeval", torch.tensor([2.5]))])
    assert w["ifeval"] == 1.0 and diag.domains["ifeval"].std == 0.0
    w, diag = domain_multipliers([("ifeval", torch.tensor([2.5]))])
    assert w == {"ifeval": 1.0} and diag.global_std == 0.0


@pytest.mark.parametrize("value", [0.0, 0.3, -7.25, 1e4])
def test_constant_domain_has_zero_std(value):
    g = torch.Generator().manual_seed(2)
    pairs = [("math", torch.full((500,), value)), ("code", torch.randn(100, generator=g))]
    w, diag = domain_multipliers(pairs)
    assert diag.domains["math"].std == 0.0 and w["math"] == 1.0
    rw = ref.domain_multipliers(pairs)[0]                      # the float32 reference agrees (torch.std is 0 too)
    assert rw["math"] == 1.0 and w["code"] == pytest.approx(rw["code"], rel=1e-6)


def test_all_tokens_equal_gives_all_ones():
    w, diag = domain_multipliers([("math", torch.full((5,), 0.5)), ("code", torch.full((7,), 0.5))])
    assert diag.global_std == 0.0 and w == {"math": 1.0, "code": 1.0}


def test_missing_domain_in_padded_batch():
    g = torch.Generator().manual_seed(3)
    r = torch.randn(8, 6, generator=g)
    r[4:] *= 3.0
    mask = torch.ones(8, 6, dtype=torch.bool)
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])                  # no ifeval response
    w, diag = domain_multipliers_batched(r, mask, ids, DOMAINS)
    assert float(w[2]) == 1.0 and diag.domains["ifeval"].responses == 0 and diag.domains["ifeval"].tokens == 0
    w_obs, _ = DomainNormalizer(DOMAINS, mode="observe").multipliers_batched(r, mask, ids)
    assert torch.equal(w_obs, torch.ones(3))
    # the list API reports only the domains that are present
    w_list, _ = DomainNormalizer(DOMAINS).multipliers([(DOMAINS[int(i)], r[b]) for b, i in enumerate(ids)])
    assert set(w_list) == {"math", "code"}


def test_fully_masked_batch():
    r = torch.randn(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    w, diag = domain_multipliers_batched(r, mask, torch.tensor([0, 1, 2, 0]), DOMAINS)
    assert torch.equal(w, torch.ones(3)) and diag.global_tokens == 0


def test_large_offset_keeps_float64_accuracy():
    """E[r^2] - E[r]^2 in float64 stays accurate when |mean| >> std (log-ratios are near zero-mean in practice)."""
    g = torch.Generator().manual_seed(4)
    a = 50.0 + 0.05 * torch.randn(20000, generator=g, dtype=torch.float64)
    b = -50.0 + 0.2 * torch.randn(20000, generator=g, dtype=torch.float64)
    w, diag = domain_multipliers([("math", a), ("code", b)])
    assert diag.domains["math"].std == pytest.approx(float(a.std(unbiased=False)), rel=1e-6)
    assert diag.domains["code"].std == pytest.approx(float(b.std(unbiased=False)), rel=1e-6)
    assert diag.global_std == pytest.approx(float(torch.cat([a, b]).std(unbiased=False)), rel=1e-9)
    assert w == {"math": 4.0, "code": 4.0}                        # sigma_all ~ 50 dominates both domains
