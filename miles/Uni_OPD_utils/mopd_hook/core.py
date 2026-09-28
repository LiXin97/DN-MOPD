# SPDX-License-Identifier: Apache-2.0
"""Pure functions of the multi-teacher OPD hook (no I/O, no trainer imports; CPU-testable).

Teacher rules. Every function below works on the per-teacher sampled-token log-probabilities of ONE student response.
Only the target handed to the unchanged on_policy_distillation surrogate is produced here; everything is
stop-gradient data. There is no top-k, union support, floor or renormalisation on this path. A missing, misaligned,
sentinel or non-finite teacher score raises: nothing falls back to the label, to uniform weights or to any teacher.

    label    the teacher named by the prompt's domain label (Label / MOPD with label routing)
    pool     log of the arithmetic mixture of all teachers with weights 1/K (uniform pool)
    dynamic  the teacher with the lowest mean sampled-token log-prob on this response (dynamic router)

DN-MOPD multipliers (`dn_multipliers`). From (domain, token log-ratio) pairs of one rollout batch, where the log-ratio
is (label-teacher log-prob - cached rollout log-prob) on the valid response tokens:
    sigma_d   = population std of the domain's tokens,  sigma_all = population std of all tokens of the batch
    w_d       = clip(sigma_all / sigma_d, 0.25, 4)      (w_d = 1 when either std is 0 or has < 2 observations)
"""
from __future__ import annotations

import math

import torch

MODES = ("label", "pool", "dynamic")
SENTINEL = -100.0  # == post_process_rewards.TEACHER_LOGP_FAILED_SENTINEL (duplicated so this file imports nothing heavy)
DN_CLIP_LOWER, DN_CLIP_UPPER = 0.25, 4.0


class ScoreError(RuntimeError):
    """A teacher score is unusable. The batch must not train."""


def validated_rows(per_teacher, names, length):
    """{name: float32 tensor[length]} for exactly `names`; raises on anything that is not a complete real score."""
    if not isinstance(per_teacher, dict) or set(per_teacher) != set(names):
        raise ScoreError(f"teacher set {sorted(per_teacher or {})} != registered {sorted(names)}")
    rows = {}
    for name in sorted(names):
        row = torch.as_tensor(per_teacher[name], dtype=torch.float32)
        if tuple(row.shape) != (int(length),) or length <= 0:
            raise ScoreError(f"{name}: score shape {tuple(row.shape)} != response length {length}")
        if not bool(torch.isfinite(row).all()) or bool((row == SENTINEL).any()):
            raise ScoreError(f"{name}: failed/non-finite sampled-token score")
        rows[name] = row
    return rows


def response_mask(loss_mask, length):
    """Boolean mask of the response tokens that carry loss (all tokens when loss_mask is None)."""
    if loss_mask is None:
        return torch.ones(int(length), dtype=torch.bool)
    mask = torch.as_tensor(loss_mask).to(torch.bool)
    if tuple(mask.shape) != (int(length),):
        raise ScoreError(f"response mask shape {tuple(mask.shape)} != response length {length}")
    if not bool(mask.any()):
        raise ScoreError("empty response mask")
    return mask


def mean_scores(rows, mask):
    """s_k = mean over response-mask tokens of teacher k's sampled-token log-prob, float32 accumulation."""
    return {name: float(row[mask].to(torch.float32).mean().item()) for name, row in rows.items()}


def argmin_name(scores):
    """Minimum-likelihood teacher; exact ties go to the lexicographically first teacher name."""
    if not scores or any(not math.isfinite(v) for v in scores.values()):
        raise ScoreError("non-finite selection score")
    return min(sorted(scores), key=lambda name: (scores[name], name))


def pool_logp(rows):
    """log q_pool(y_t|h) = logsumexp_k log p_Tk(y_t|h) - log K: the arithmetic mixture with fixed weights 1/K."""
    stacked = torch.stack([rows[name] for name in sorted(rows)], dim=0).to(torch.float32)
    return torch.logsumexp(stacked, dim=0) - math.log(stacked.shape[0])


def mixture_logp(rows, weights):
    """General fixed-weight arithmetic mixture (one-hot weights reduce to one teacher; uniform weights to pool_logp)."""
    names = sorted(rows)
    w = torch.tensor([float(weights[n]) for n in names], dtype=torch.float32)
    if bool((w < 0).any()) or abs(float(w.sum()) - 1.0) > 1e-6:
        raise ScoreError("mixture weights must be a distribution")
    stacked = torch.stack([rows[n] for n in names], dim=0).to(torch.float32)
    logw = torch.where(w > 0, torch.log(w), torch.full_like(w, -float("inf")))
    return torch.logsumexp(stacked + logw[:, None], dim=0)


def teacher_log_probs(mode, rows, scores, *, label_teacher=None):
    """(teacher_log_probs float32[length], applied teacher name or 'pool')."""
    if mode == "pool":
        return pool_logp(rows), "pool"
    if mode == "dynamic":
        name = argmin_name(scores)
    elif mode == "label":
        name = label_teacher
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if name not in rows:
        raise ScoreError(f"{mode}: applied teacher {name!r} is not a registered scored teacher")
    return rows[name].clone(), name


def dn_multipliers(pairs, lower=DN_CLIP_LOWER, upper=DN_CLIP_UPPER):
    """DN-MOPD multipliers of one rollout batch.

    pairs: iterable of (domain, 1-D float tensor of token log-ratios), one pair per response.
    Returns ({domain: w_d}, {"global_std": sigma_all, "by_domain": {domain: {std, raw_factor, factor, clipped,
    tokens}}}). Population std (unbiased=False); sigma_all pools the tokens of every domain; a degenerate std (zero or
    fewer than two tokens) gives w_d = 1.
    """
    pairs = list(pairs)
    by = {}
    for d, v in pairs:
        by.setdefault(d, []).append(v)
    allv = torch.cat([v for _, v in pairs]) if pairs else torch.zeros(0)
    g = float(allv.std(unbiased=False)) if allv.numel() > 1 else 0.0
    out, info = {}, {}
    for d in sorted(by):
        t = torch.cat(by[d])
        sd = float(t.std(unbiased=False)) if t.numel() > 1 else 0.0
        raw = (g / sd) if (sd > 0.0 and g > 0.0) else 1.0
        f = min(upper, max(lower, raw))
        out[d] = f
        info[d] = {"std": sd, "raw_factor": raw, "factor": f, "clipped": f != raw, "tokens": int(t.numel())}
    return out, {"global_std": g, "by_domain": info}
