# SPDX-License-Identifier: Apache-2.0
# Test oracle: the reference implementation shipped with the paper's supplementary material (code/dn_mopd.py),
# copied verbatim below this header. The dn_mopd package must reproduce it; do not edit the copy.
"""Reference implementation of the DN-MOPD advantage scaling (Section 2 and App. D.1 of the paper).

DN-MOPD keeps label routing: every prompt x with domain label d is scored by the specialist T_d, and the per-token
distillation advantage of a sampled token is A_t = log p_{T_d}(y_t | h_t) - log pi_u(y_t | h_t). DN-MOPD adds one
operation per rollout batch:

    r_{i,t}   = teacher log-prob - cached rollout log-prob, on valid response tokens (the scale estimate)
    sigma_d   = population std of r over the valid tokens of domain d
    sigma_all = population std of r over all valid tokens of the batch, pooled across domains
    w_d       = clip(sigma_all / sigma_d, 0.25, 4)          (w_d = 1 if either std is zero or has < 2 observations)
    A~_{i,t}  = stopgrad(w_{d_i} * A_{i,t})                  (A uses the actor-recomputed pre-update log-prob)

The scaled advantage then enters the unchanged clipped policy-gradient (PPO-style) OPD loss. Setting every w_d to one
recovers MOPD with label routing ("Label"). Only torch is required.
"""
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

LOWER, UPPER = 0.25, 4.0


def domain_multipliers(log_ratios: Iterable[Tuple[str, torch.Tensor]], lower: float = LOWER,
                       upper: float = UPPER) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Per-domain multipliers from (domain, 1-D tensor of valid-token rollout log-ratios) pairs, one pair per response.

    Returns ({domain: w_d}, diagnostics). Standard deviations use the population convention; the pooled statistic is
    computed after concatenating all domains' tokens (it is not an average of the domain standard deviations).
    """
    pairs = [(d, v.detach().float().reshape(-1)) for d, v in log_ratios]
    by: Dict[str, List[torch.Tensor]] = {}
    for d, v in pairs:
        by.setdefault(d, []).append(v)
    allv = torch.cat([v for _, v in pairs]) if pairs else torch.zeros(0)
    g = float(allv.std(unbiased=False)) if allv.numel() > 1 else 0.0
    weights, info = {}, {'global_std': g}
    for d in sorted(by):
        t = torch.cat(by[d])
        sd = float(t.std(unbiased=False)) if t.numel() > 1 else 0.0
        raw = g / sd if (sd > 0.0 and g > 0.0) else 1.0
        w = min(upper, max(lower, raw))
        weights[d] = w
        info[d] = {'std': sd, 'raw_factor': raw, 'factor': w, 'clipped': w != raw, 'tokens': int(t.numel())}
    return weights, info


def scale_advantages(advantages: Sequence[torch.Tensor], domains: Sequence[str],
                     weights: Dict[str, float]) -> List[torch.Tensor]:
    """Multiply every advantage of a response by its domain's multiplier and detach (signs are preserved)."""
    if len(advantages) != len(domains):
        raise ValueError('one domain label per response is required')
    return [(a * float(weights[d])).detach() for a, d in zip(advantages, domains)]


def clipped_opd_loss(logp_new: Sequence[torch.Tensor], logp_old: Sequence[torch.Tensor],
                     advantages: Sequence[torch.Tensor], masks: Sequence[torch.Tensor],
                     eps_low: float = 0.2, eps_high: float = 0.2) -> torch.Tensor:
    """Clipped policy-gradient OPD loss with the Label reduction: mean over valid tokens within each response, then
    mean over responses. logp_old is the actor-recomputed pre-update log-prob used for the advantage and the ratio."""
    per_resp = []
    for lp, lo, a, m in zip(logp_new, logp_old, advantages, masks):
        ratio = torch.exp(lp - lo)
        term = -torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
        m = m.float()
        per_resp.append((term * m).sum() / m.sum().clamp_min(1.0))
    return torch.stack(per_resp).mean()
