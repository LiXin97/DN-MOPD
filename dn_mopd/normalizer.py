# SPDX-License-Identifier: Apache-2.0
"""Per-domain advantage multipliers of DN-MOPD (domain-normalized multi-teacher on-policy distillation).

DN-MOPD keeps label routing: a prompt with domain label d is scored by the specialist teacher T_d, and the per-token
distillation advantage of a sampled token is ``A_t = log p_{T_d}(y_t | h_t) - log pi(y_t | h_t)``. Once per rollout
batch it measures the spread of the rollout-side log-ratio in every domain and rescales that domain's advantages:

    r_{i,t}   = teacher log-prob - rollout log-prob, on the valid response tokens of response i
    sigma_d   = population std of r over the valid tokens of all responses of domain d
    sigma_all = population std of r over all valid tokens of the batch (pooled across domains, not an average)
    w_d       = clip(sigma_all / sigma_d, lower, upper)        (paper: lower = 0.25, upper = 4)
    w_d       = clip(1, lower, upper) when sigma_d or sigma_all is zero or has fewer than two tokens
    A~_{i,t}  = stopgrad(w_{d_i} * A_{i,t})

Setting every ``w_d`` to 1 recovers label-routed MOPD ("Label").

Numerics. The statistics are accumulated in float64 as per-domain sufficient statistics (responses, valid tokens,
sum, sum of squares), so that a sharded batch can be reduced exactly across data-parallel ranks. The variance is
``E[r^2] - E[r]^2`` in float64 and is treated as zero when it is below ``1e-12 * E[r^2]`` (a cancellation guard: a
domain whose spread is below 1e-6 of its RMS is constant for every practical purpose). The paper's trainer computed
``torch.std(unbiased=False)`` on float32 tensors in one process; the two agree to about 1e-6 relative, far below the
precision the multiplier is used at (it scales bf16/fp32 advantages).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .distributed import all_reduce_statistics

__all__ = [
    "DEFAULT_BOUNDS",
    "MODES",
    "DomainDiagnostics",
    "DNDiagnostics",
    "DomainNormalizer",
    "batch_statistics",
    "list_statistics",
    "domain_multipliers",
    "domain_multipliers_batched",
]

DEFAULT_BOUNDS: Tuple[float, float] = (0.25, 4.0)
MODES: Tuple[str, ...] = ("dn", "fixed", "observe", "frozen")
STAT_ROWS: Tuple[str, ...] = ("responses", "tokens", "sum", "sumsq")
_ZERO_VAR_RTOL = 1e-12

SyncSpec = Union[bool, Callable[[torch.Tensor], torch.Tensor]]


# ----------------------------------------------------------------------------------------------- statistics
def batch_statistics(log_ratio: torch.Tensor, mask: torch.Tensor, domain_ids: torch.Tensor,
                     num_domains: int) -> torch.Tensor:
    """Per-domain sufficient statistics of a padded batch.

    Args:
        log_ratio: ``(B, T)`` rollout log-ratio ``teacher_logp - rollout_logp``. Values at masked positions are
            ignored (they may be arbitrary, including NaN).
        mask: ``(B, T)`` bool/0-1 tensor, true on valid response tokens.
        domain_ids: ``(B,)`` integer domain index of every response, in ``[0, num_domains)``.
        num_domains: number of domains ``D``.

    Returns:
        A float64 tensor of shape ``(4, D)`` on ``log_ratio.device`` whose rows are the number of responses, the number
        of valid tokens, the sum of ``r`` and the sum of ``r**2`` per domain.
    """
    if log_ratio.dim() != 2:
        raise ValueError(f"log_ratio must be (B, T), got shape {tuple(log_ratio.shape)}")
    if mask.shape != log_ratio.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != log_ratio shape {tuple(log_ratio.shape)}")
    batch = log_ratio.shape[0]
    if domain_ids.shape != (batch,):
        raise ValueError(f"domain_ids must be ({batch},), got shape {tuple(domain_ids.shape)}")
    if num_domains < 1:
        raise ValueError("num_domains must be positive")
    device = log_ratio.device
    ids = domain_ids.to(device=device, dtype=torch.long)
    if batch and (int(ids.min()) < 0 or int(ids.max()) >= num_domains):
        raise ValueError(f"domain_ids must lie in [0, {num_domains})")
    valid = mask.to(device=device, dtype=torch.bool)
    x = torch.where(valid, log_ratio.detach().to(torch.float64), torch.zeros((), dtype=torch.float64, device=device))
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite log-ratio at a valid token")
    per_response = torch.stack([
        torch.ones(batch, dtype=torch.float64, device=device),
        valid.sum(-1).to(torch.float64),
        x.sum(-1),
        (x * x).sum(-1),
    ])
    out = torch.zeros(len(STAT_ROWS), num_domains, dtype=torch.float64, device=device)
    out.index_add_(1, ids, per_response)
    return out


def list_statistics(log_ratios: Iterable[Tuple[Hashable, torch.Tensor]], domains: Sequence[Hashable]) -> torch.Tensor:
    """Per-domain sufficient statistics from ``(domain, 1-D tensor of valid-token log-ratios)`` pairs.

    One pair per response; the tensor holds only that response's valid tokens (it may be empty). Returns a float64
    ``(4, len(domains))`` tensor laid out as in :func:`batch_statistics`.
    """
    index = {d: i for i, d in enumerate(domains)}
    pairs = list(log_ratios)
    device = pairs[0][1].device if pairs else torch.device("cpu")
    out = torch.zeros(len(STAT_ROWS), len(domains), dtype=torch.float64, device=device)
    if not pairs:
        return out
    ids: List[int] = []
    counts: List[float] = []
    sums: List[torch.Tensor] = []
    sumsqs: List[torch.Tensor] = []
    for d, v in pairs:
        if d not in index:
            raise ValueError(f"domain {d!r} is not one of the registered domains {list(domains)}")
        x = v.detach().reshape(-1).to(device=device, dtype=torch.float64)
        ids.append(index[d])
        counts.append(float(x.numel()))
        sums.append(x.sum())
        sumsqs.append((x * x).sum())
    per_response = torch.stack([torch.ones(len(pairs), dtype=torch.float64, device=device),
                                torch.tensor(counts, dtype=torch.float64, device=device),
                                torch.stack(sums), torch.stack(sumsqs)])
    if not bool(torch.isfinite(per_response).all()):
        raise ValueError("non-finite log-ratio in a response")
    out.index_add_(1, torch.tensor(ids, dtype=torch.long, device=device), per_response)
    return out


def _population_std(n: float, s: float, q: float) -> float:
    """Population std from (count, sum, sum of squares); 0.0 below two observations or for a constant sample."""
    if n < 2:
        return 0.0
    mean_sq = q / n
    var = mean_sq - (s / n) ** 2
    if var <= _ZERO_VAR_RTOL * mean_sq:
        return 0.0
    return math.sqrt(var)


# ----------------------------------------------------------------------------------------------- diagnostics
@dataclass(frozen=True)
class DomainDiagnostics:
    """What the normalizer measured and applied for one domain on one batch.

    Attributes:
        responses: responses of this domain in the (global) batch.
        tokens: valid tokens of this domain.
        std: population std ``sigma_d`` of the rollout log-ratio (0.0 if degenerate).
        raw_factor: ``sigma_all / sigma_d`` (1.0 if either std is degenerate).
        factor: ``clip(raw_factor, lower, upper)``, the DN-MOPD multiplier measured on this batch.
        clipped: whether the clip changed ``raw_factor``.
        applied: the multiplier actually applied (equal to ``factor`` in ``dn`` mode).
    """

    responses: int
    tokens: int
    std: float
    raw_factor: float
    factor: float
    clipped: bool
    applied: float


@dataclass(frozen=True)
class DNDiagnostics:
    """Per-batch record of the DN statistic (the paper logged the same fields every update)."""

    mode: str
    step: int
    global_std: float
    global_tokens: int
    domains: Dict[Hashable, DomainDiagnostics]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable nested dict."""
        return {"mode": self.mode, "step": self.step, "global_std": self.global_std,
                "global_tokens": self.global_tokens,
                "domains": {str(d): asdict(v) for d, v in self.domains.items()}}

    def to_metrics(self, prefix: str = "dn") -> Dict[str, float]:
        """Flat ``{name: float}`` dict for scalar loggers, e.g. ``dn/math/applied``."""
        out = {f"{prefix}/global_std": self.global_std, f"{prefix}/global_tokens": float(self.global_tokens)}
        for d, v in self.domains.items():
            for k in ("tokens", "std", "raw_factor", "factor", "applied"):
                out[f"{prefix}/{d}/{k}"] = float(getattr(v, k))
            out[f"{prefix}/{d}/clipped"] = float(v.clipped)
        return out


# ----------------------------------------------------------------------------------------------- normalizer
class DomainNormalizer:
    """Computes DN-MOPD per-domain advantage multipliers once per rollout batch.

    Args:
        domains: ordered domain names. Required for :meth:`multipliers_batched` (``domain_ids`` index into it) and
            whenever ``sync`` is enabled (every rank must lay the statistics out identically). The list API may
            omit it; domains are then discovered from each batch.
        mode: ``"dn"`` applies the multiplier measured on the current batch (DN-MOPD); ``"observe"`` measures and
            reports it but applies 1.0 (Label); ``"fixed"`` applies the constants in ``fixed``; ``"frozen"`` applies,
            for every domain, the multiplier measured on the first batch in which that domain was measurable
            (non-degenerate), and the measured value until then. In every mode the statistic is computed and reported.
        bounds: ``(lower, upper)`` clip of the multiplier, default ``(0.25, 4.0)``; ``0 <= lower <= upper``
            (``upper`` may be ``math.inf``).
        fixed: ``{domain: multiplier}`` for ``mode="fixed"``, each within ``bounds``; must cover ``domains`` if given.
        sync: ``False`` (default) uses the local batch only. ``True`` sums the statistics over the ranks of
            ``process_group`` with ``torch.distributed`` (a no-op when no process group is initialized). A callable
            receives the local float64 ``(4, D)`` statistics tensor and must return the global one (use this for
            framework-specific collectives, or :func:`dn_mopd.distributed.merge_statistics`).
        process_group: process group for ``sync=True`` (default: the world group).

    Call :meth:`multipliers` or :meth:`multipliers_batched` exactly once per rollout batch, on every rank when
    ``sync`` is enabled. The state of ``frozen`` mode is in :meth:`state_dict`; save it with the checkpoint.
    """

    def __init__(self, domains: Optional[Sequence[Hashable]] = None, *, mode: str = "dn",
                 bounds: Tuple[float, float] = DEFAULT_BOUNDS, fixed: Optional[Mapping[Hashable, float]] = None,
                 sync: SyncSpec = False, process_group: Optional[Any] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        lower, upper = (float(b) for b in bounds)
        if math.isnan(lower) or math.isnan(upper) or not 0.0 <= lower <= upper:
            raise ValueError(f"bounds must satisfy 0 <= lower <= upper, got {bounds!r}")
        if domains is not None:
            domains = tuple(domains)
            if not domains or len(set(domains)) != len(domains):
                raise ValueError(f"domains must be non-empty and unique, got {domains!r}")
        if not (isinstance(sync, bool) or callable(sync)):
            raise TypeError("sync must be a bool or a callable")
        if sync is not False and domains is None:
            raise ValueError("sync needs an explicit, identical `domains` list on every rank")
        if mode == "fixed":
            if not fixed:
                raise ValueError("mode='fixed' needs a non-empty `fixed` mapping")
            for d, v in fixed.items():
                v = float(v)
                if not lower <= v <= upper:
                    raise ValueError(f"fixed multiplier {d!r}={v} lies outside bounds [{lower}, {upper}]")
            if domains is not None and set(domains) - set(fixed):
                raise ValueError(f"fixed multipliers missing for domains {sorted(map(str, set(domains) - set(fixed)))}")
        elif fixed is not None:
            raise ValueError("`fixed` is only used with mode='fixed'")
        self.mode = mode
        self.lower, self.upper = lower, upper
        self._domains: Optional[Tuple[Hashable, ...]] = domains
        self._fixed: Dict[Hashable, float] = {d: float(v) for d, v in (fixed or {}).items()}
        self._sync = sync
        self._group = process_group
        self.step = 0
        self._frozen: Dict[Hashable, float] = {}
        self._frozen_step: Dict[Hashable, int] = {}

    @property
    def domains(self) -> Optional[Tuple[Hashable, ...]]:
        """The registered domain order (``None`` if domains are discovered per batch)."""
        return self._domains

    # --------------------------------------------------------------------------------------------- public API
    def multipliers(self, log_ratios: Iterable[Tuple[Hashable, torch.Tensor]]
                    ) -> Tuple[Dict[Hashable, float], DNDiagnostics]:
        """Multipliers from ``(domain, 1-D valid-token log-ratio tensor)`` pairs, one pair per response.

        Returns ``({domain: applied multiplier}, diagnostics)``; the dict holds every domain with at least one response
        in the (global) batch.
        """
        pairs = list(log_ratios)
        domains = self._domains if self._domains is not None else tuple(sorted({d for d, _ in pairs}, key=str))
        stats = self._reduce(list_statistics(pairs, domains))
        applied, diag = self._finalize(stats, domains)
        return {d: applied[i] for i, d in enumerate(domains) if diag.domains[d].responses > 0}, diag

    def multipliers_batched(self, log_ratio: torch.Tensor, mask: torch.Tensor, domain_ids: torch.Tensor
                            ) -> Tuple[torch.Tensor, DNDiagnostics]:
        """Multipliers for a padded batch.

        Args:
            log_ratio: ``(B, T)`` ``teacher_logp - rollout_logp``; masked positions are ignored.
            mask: ``(B, T)`` valid-token mask.
            domain_ids: ``(B,)`` index into ``domains``.

        Returns:
            ``(weights, diagnostics)`` where ``weights`` is a float32 ``(D,)`` tensor on ``log_ratio.device`` with the
            applied multiplier of every registered domain (``weights[domain_ids]`` is the per-response multiplier).
            A domain absent from the batch gets the degenerate value ``clip(1, lower, upper)`` in ``dn`` mode (and in
            ``frozen`` mode until it has been frozen), 1.0 in ``observe`` mode and its constant in ``fixed`` mode.
        """
        if self._domains is None:
            raise ValueError("multipliers_batched needs `domains` at construction")
        stats = self._reduce(batch_statistics(log_ratio, mask, domain_ids, len(self._domains)))
        applied, diag = self._finalize(stats, self._domains)
        return torch.tensor(applied, dtype=torch.float32, device=log_ratio.device), diag

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state: the step counter and, in ``frozen`` mode, the frozen multipliers."""
        return {"mode": self.mode, "step": self.step, "frozen": dict(self._frozen),
                "frozen_step": dict(self._frozen_step)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` output (for example after resuming from a checkpoint)."""
        if state.get("mode") != self.mode:
            raise ValueError(f"state was saved in mode {state.get('mode')!r}, this normalizer is {self.mode!r}")
        self.step = int(state["step"])
        self._frozen = {d: float(v) for d, v in state.get("frozen", {}).items()}
        self._frozen_step = {d: int(v) for d, v in state.get("frozen_step", {}).items()}

    # --------------------------------------------------------------------------------------------- internals
    def _reduce(self, stats: torch.Tensor) -> torch.Tensor:
        if self._sync is False:
            return stats
        if self._sync is True:
            return all_reduce_statistics(stats, self._group)
        out = self._sync(stats)
        if not isinstance(out, torch.Tensor) or out.shape != stats.shape:
            raise ValueError("the sync callable must return a tensor shaped like its input")
        return out.to(dtype=torch.float64)

    def _finalize(self, stats: torch.Tensor, domains: Sequence[Hashable]) -> Tuple[List[float], DNDiagnostics]:
        responses, tokens, sums, sumsqs = stats.detach().to("cpu").tolist()
        global_std = _population_std(sum(tokens), sum(sums), sum(sumsqs))
        applied: List[float] = []
        per: Dict[Hashable, DomainDiagnostics] = {}
        for i, d in enumerate(domains):
            std = _population_std(tokens[i], sums[i], sumsqs[i])
            measurable = std > 0.0 and global_std > 0.0
            raw = global_std / std if measurable else 1.0
            factor = min(self.upper, max(self.lower, raw))
            w = self._applied(d, factor, measurable)
            applied.append(w)
            per[d] = DomainDiagnostics(responses=int(responses[i]), tokens=int(tokens[i]), std=std, raw_factor=raw,
                                       factor=factor, clipped=factor != raw, applied=w)
        diag = DNDiagnostics(mode=self.mode, step=self.step, global_std=global_std,
                             global_tokens=int(sum(tokens)), domains=per)
        self.step += 1
        return applied, diag

    def _applied(self, domain: Hashable, factor: float, measurable: bool) -> float:
        if self.mode == "dn":
            return factor
        if self.mode == "observe":
            return 1.0
        if self.mode == "fixed":
            if domain not in self._fixed:
                raise ValueError(f"mode='fixed': no registered multiplier for domain {domain!r}")
            return self._fixed[domain]
        # frozen
        if domain in self._frozen:
            return self._frozen[domain]
        if measurable:
            self._frozen[domain] = factor
            self._frozen_step[domain] = self.step
        return factor


# ----------------------------------------------------------------------------------------------- functional API
def domain_multipliers(log_ratios: Iterable[Tuple[Hashable, torch.Tensor]], lower: float = DEFAULT_BOUNDS[0],
                       upper: float = DEFAULT_BOUNDS[1]) -> Tuple[Dict[Hashable, float], DNDiagnostics]:
    """DN-MOPD multipliers of one batch from ``(domain, 1-D valid-token log-ratio tensor)`` pairs (one per response).

    Stateless shortcut for ``DomainNormalizer(mode="dn", bounds=(lower, upper)).multipliers(log_ratios)``.
    """
    return DomainNormalizer(mode="dn", bounds=(lower, upper)).multipliers(log_ratios)


def domain_multipliers_batched(log_ratio: torch.Tensor, mask: torch.Tensor, domain_ids: torch.Tensor,
                               domains: Sequence[Hashable], lower: float = DEFAULT_BOUNDS[0],
                               upper: float = DEFAULT_BOUNDS[1], sync: SyncSpec = False,
                               process_group: Optional[Any] = None) -> Tuple[torch.Tensor, DNDiagnostics]:
    """DN-MOPD multipliers of one padded batch; see :meth:`DomainNormalizer.multipliers_batched`."""
    return DomainNormalizer(domains, mode="dn", bounds=(lower, upper), sync=sync,
                            process_group=process_group).multipliers_batched(log_ratio, mask, domain_ids)
