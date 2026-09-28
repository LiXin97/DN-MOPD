# SPDX-License-Identifier: Apache-2.0
"""Multi-teacher OPD hook: label / uniform pool / dynamic router targets, DN-MOPD and its controls, and annealed
teacher-trajectory injection, on the unchanged sampled-token reverse-KL (on_policy_distillation) loss of miles.

Wiring (the recipes set all of this; see recipes/qwen3.5/train_student.sh):
    MOPD_HOOK_CONFIG=<run config JSON>          (exported before `ray start`, so the rollout workers inherit it)
    --advantage-estimator on_policy_distillation
    --custom-rm-path                  Uni_OPD_utils.mopd_hook.hook.get_reward
    --custom-reward-post-process-path Uni_OPD_utils.mopd_hook.hook.post_process_rewards

Every registered teacher scores the sampled tokens of every response (OPD_reward.get_reward._get_reward_fanout). The
legacy reward fields (and hence sample.teacher_log_probs) still come from the label-routed teacher; this hook then
replaces sample.teacher_log_probs by the configured target and, for DN-MOPD / injection, attaches per-sample values
that the rollout manager carries into the training batch and the loss applies (miles/backends/training_utils/loss.py):
    dn_adv_scale      the domain multiplier w_d (DN-MOPD, fixed weights, frozen update-0 multipliers, observe = 1.0)
    inject_adv_const  the constant advantage c(t) of an injected teacher trajectory (0.0 = not injected)

Run config (JSON):
    mode                "label" | "pool" | "dynamic"
    n_samples           responses per prompt (must equal --n-samples-per-prompt)
    teachers            registered teacher names (the keys of the teacher server list)
    label_map           {"default": T, <domain label>: T, ...}; OPD_TEACHER_SERVER_MAP, when set, takes precedence
    audit_dir           optional directory for the per-batch audit files (samples_<first>_<last>_<ns>.json)
    extra_score_rounds  repair rounds for a teacher whose five scoring attempts all failed (default 3)
    adv_norm            optional, label mode only:
                          "per_domain_scale"      DN-MOPD: w_d = clip(sigma_all / sigma_d, 0.25, 4) per batch
                          "fixed_domain_scale"    w_d = adv_norm_fixed[d] (the DN statistic is still recorded)
                          "observe_domain_scale"  w_d = 1 (Label update; the DN statistic is recorded, never applied)
    adv_norm_fixed      {"math": w, "code": w, "ifeval": w}, each in [0.25, 4] (fixed_domain_scale only)
    inject_bank_path    optional, label mode only: annealed injection. In every prompt group whose prompt the bank
                        holds, the lowest-index verifier-wrong sample (else the lowest-index sample) is replaced by the
                        bank's verified-correct label-teacher trajectory, whose advantage is the constant
                        c(t) = inject_adv_const * max(0, 1 - t / inject_anneal_steps) on every token at update t.
    inject_bank_sha256  optional integrity check of the bank file
    inject_adv_const    c0 in (0, 2]
    inject_anneal_steps A >= 1 (no injection from t = A on)

Each batch prints one "MOPD_HOOK_PANEL {json}" line (routing shares, DN statistics, injection counts).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import torch

from Uni_OPD_utils.mopd_hook import core

logger = logging.getLogger(__name__)
PANEL_PREFIX = "MOPD_HOOK_PANEL"
CONFIG_ENV = "MOPD_HOOK_CONFIG"
ADV_NORM_MODES = ("per_domain_scale", "fixed_domain_scale", "observe_domain_scale")
DOMAINS = ("math", "code", "ifeval")
IDENTITY = {
    "label": "Label-routed MOPD (sampled-token reverse KL against the domain-label teacher)",
    "pool": "Uniform-pool MOPD (sampled-token reverse KL against the arithmetic 1/K mixture of the teachers)",
    "dynamic": "Dynamic-router MOPD (sampled-token reverse KL against the per-response minimum-likelihood teacher)",
}
_STATE = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_key(prompt):
    """Key of a prompt in the injection bank: sha256 of the rendered prompt (as JSON, sorted keys, compact)."""
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def strict_verdict(sample, what):
    """True for a verified-correct response (True, or a numeric score >= 1.0), False for a wrong one; raises when the
    verifier gave no verdict (None), because a choice that reads correctness must not guess."""
    rc = getattr(sample, "response_correct", None)
    if rc is None or not isinstance(rc, (bool, int, float)):
        raise RuntimeError(f"{what}: sample {sample.index} has no verified reward (response_correct={rc!r})"
                           "; the verifier or code judge failed")
    return (rc is True) or (isinstance(rc, (int, float)) and not isinstance(rc, bool) and float(rc) >= 1.0)


class MopdHook:
    def __init__(self, config_path, server_map_path=None):
        self.config_path = Path(config_path)
        self.cfg = cfg = json.loads(self.config_path.read_text())
        self.mode = cfg.get("mode")
        if self.mode not in core.MODES:
            raise ValueError(f"unknown hook mode {self.mode!r}; expected one of {core.MODES}")
        self.n = cfg.get("n_samples")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n_samples must be a positive integer, got {self.n!r}")
        names = cfg.get("teachers")
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError(f"teachers must be a non-empty list of distinct names, got {names!r}")
        self.names = sorted(names)
        if self.mode in ("pool", "dynamic") and len(self.names) < 2:
            raise ValueError(f"{self.mode} needs at least two teachers")
        map_path = server_map_path or os.environ.get("OPD_TEACHER_SERVER_MAP")
        self.label_map = json.loads(Path(map_path).read_text()) if map_path else dict(cfg.get("label_map") or {})
        if not set(self.label_map.values()) <= set(self.names) or "default" not in self.label_map:
            raise ValueError("the teacher label map names an unregistered teacher or lacks a 'default' key")
        self.audit_dir = Path(cfg["audit_dir"]) if cfg.get("audit_dir") else None
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.extra_rounds = int(cfg.get("extra_score_rounds", 3))
        self.last_applied = {}
        self.restored = False

        # ---- annealed teacher-trajectory injection (label mode only; off without a bank)
        self.inject, self.inject_stats = None, Counter()
        self.inject_const, self.inject_anneal, self._anneal_step, self._step_const = None, None, None, None
        bank_path = cfg.get("inject_bank_path")
        inject_keys = ("inject_bank_sha256", "inject_adv_const", "inject_anneal_steps")
        if bank_path:
            if self.mode != "label":
                raise ValueError(f"{self.mode}: teacher-trajectory injection is defined for the label mode only")
            want = cfg.get("inject_bank_sha256")
            if want and sha256(bank_path) != want:
                raise ValueError("injection bank hash mismatch")
            c = cfg.get("inject_adv_const")
            if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 < float(c) <= 2.0:
                raise ValueError(f"inject_adv_const {c!r} is outside (0, 2]")
            an = cfg.get("inject_anneal_steps")
            if isinstance(an, bool) or not isinstance(an, int) or not 1 <= an <= 100000:
                raise ValueError(f"inject_anneal_steps {an!r} is not a positive step count")
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            entries = bank["entries"]
            if not all(isinstance(e, dict) and {"prompt_ids", "domain", "response_ids", "teacher_logp"} <= set(e)
                       for e in entries.values()):
                raise ValueError("injection bank entries need prompt_ids, domain, response_ids and teacher_logp")
            self.inject, self.inject_const, self.inject_anneal = entries, float(c), an
            print(f"MOPD_HOOK_INJECT_BANK entries={len(entries)} const={self.inject_const} "
                  f"anneal_steps={self.inject_anneal}", flush=True)
        elif any(k in cfg for k in inject_keys):
            raise ValueError("injection fields without an injection bank")

        # ---- DN-MOPD and its controls (label mode only)
        self.adv_norm = cfg.get("adv_norm")
        if self.adv_norm is not None and self.adv_norm not in ADV_NORM_MODES:
            raise ValueError(f"unknown adv_norm {self.adv_norm!r}; expected one of {ADV_NORM_MODES}")
        self.adv_norm_fixed = None
        if self.adv_norm == "fixed_domain_scale":
            fx = cfg.get("adv_norm_fixed")
            if (not isinstance(fx, dict) or sorted(fx) != sorted(DOMAINS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not core.DN_CLIP_LOWER <= float(v) <= core.DN_CLIP_UPPER for v in fx.values())):
                raise ValueError(f"fixed_domain_scale needs adv_norm_fixed = {{math, code, ifeval}} in "
                                 f"[{core.DN_CLIP_LOWER}, {core.DN_CLIP_UPPER}]: {fx!r}")
            self.adv_norm_fixed = {k: float(v) for k, v in fx.items()}
        elif "adv_norm_fixed" in cfg:
            raise ValueError("adv_norm_fixed belongs to adv_norm = fixed_domain_scale only")
        if self.adv_norm is not None:
            if self.mode != "label":
                raise ValueError("adv_norm is defined for the label mode only")
            if self.inject is not None:
                raise ValueError("adv_norm and teacher-trajectory injection are separate methods")
        self._dnorm_last = None

    # ------------------------------------------------------------------ helpers
    def _restore(self, first_index):
        """After a resume, rebuild the (prompt, slot) -> last routed teacher memory from this run's audit files. Rows
        at or after the resumed batch belong to steps being redone and are ignored; later files win."""
        files = sorted(self.audit_dir.glob("samples_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        latest = {}
        for f in files:
            value = json.loads(f.read_text())
            if value["mode"] != self.mode:
                raise ValueError(f"audit directory holds another mode's rows: {f}")
            for r in value["rows"]:
                if r["index"] < first_index:
                    latest[r["index"]] = r
        for index in sorted(latest):
            r = latest[index]
            self.last_applied[(r["prompt_sha256"], r["slot"])] = r["applied"] if self.mode != "pool" else r["current_argmin"]
        print(f"MOPD_HOOK_RESTORED files={len(files)} rows_before_index_{first_index}={len(latest)}", flush=True)

    def label_teacher(self, alias):
        key = alias if alias not in (None, "", "default") else "default"
        return self.label_map.get(key, key)

    def _select_injections(self, samples, batch_size):
        """{sample index: 0} of the samples to replace this step. Anneal: t = lowest index // batch_size,
        c(t) = const * max(0, 1 - t / A); from c(t) = 0 on nothing is injected and no verdict is read. Otherwise, in
        every group whose prompt the bank holds, the first of [verifier-wrong samples by index] + [the remaining
        samples by index] is replaced; a missing verdict anywhere in the group refuses the batch."""
        lo = min(s.index for s in samples)
        if lo % batch_size:
            raise ValueError(f"batch starts at sample {lo}, not a multiple of {batch_size}")
        t = lo // batch_size
        self._anneal_step = t
        self._step_const = self.inject_const * max(0.0, 1.0 - t / self.inject_anneal)
        if self._step_const <= 0.0:
            self.inject_stats["anneal_off"] += 1
            return {}
        groups = {}
        for s in samples:
            groups.setdefault(s.group_index, []).append(s)
        chosen = {}
        for gi in sorted(groups):
            grp = sorted(groups[gi], key=lambda s: s.index)
            wrong = [s for s in grp if not strict_verdict(s, "injection")]
            entry = self.inject.get(prompt_key(grp[0].prompt))
            if entry is None:
                dom = (grp[0].metadata or {}).get("domain") or "unknown"
                self.inject_stats[f"groups_not_covered_{dom}"] += 1
                continue
            dom = entry["domain"]
            self.inject_stats[f"dense_into_wrong_group_{dom}" if wrong else f"dense_into_all_correct_group_{dom}"] += 1
            wrong_ids = {s.index for s in wrong}
            order = wrong + [s for s in grp if s.index not in wrong_ids]
            chosen[order[0].index] = 0
        return chosen

    def _inject(self, sample, entry, key, length):
        """Replace the sample's response by the bank trajectory; returns (teacher row, mask, length) or None when the
        prompt tokens do not match the bank entry."""
        from miles.utils.types import Sample as _Sample

        plen = len(sample.tokens) - length
        if plen <= 0 or list(sample.tokens[:plen]) != entry["prompt_ids"].tolist():
            self.inject_stats["prompt_token_mismatch"] += 1
            return None
        if entry["domain"] != (sample.metadata or {}).get("domain"):
            raise ValueError(f"injection bank domain {entry['domain']} differs from the prompt's domain")
        rids = entry["response_ids"].tolist()
        tlp = entry["teacher_logp"].to(torch.float32).clone()
        if tlp.shape[0] != len(rids) or not bool(torch.isfinite(tlp).all()) or bool((tlp > 0).any()):
            raise core.ScoreError("injection bank row unusable")
        sample.tokens = list(sample.tokens[:plen]) + rids
        sample.response_length = length = len(rids)
        sample.loss_mask = [1] * length
        # diagnostic only (no --use-rollout-logprobs, no TIS): the PPO ratio and the advantage use the actor-recomputed
        # pre-update log-probs of these tokens
        sample.rollout_log_probs = tlp.tolist()
        sample.response = f"[injected teacher trajectory {key[:12]}]"
        sample.status = _Sample.Status.COMPLETED
        sample.response_correct = True
        self.inject_stats["injected"] += 1
        return tlp, torch.ones(length, dtype=torch.bool), length

    # ------------------------------------------------------------------ the batch
    def apply(self, args, samples):
        if getattr(args, "advantage_estimator", None) != "on_policy_distillation":
            raise ValueError("the MOPD hook belongs to the on_policy_distillation estimator only")
        if args.n_samples_per_prompt != self.n:
            raise ValueError(f"--n-samples-per-prompt {args.n_samples_per_prompt} != the config's n_samples {self.n}")
        if not self.restored:
            if self.audit_dir is not None:
                self._restore(min(s.index for s in samples))
            self.restored = True
        rows_out, new_lists = [], []
        # the samples to replace are chosen BEFORE any per-sample work
        inject_ids = {}
        if self.inject is not None:
            inject_ids = self._select_injections(samples, int(args.rollout_batch_size) * self.n)
        dnorm_pairs = [] if self.adv_norm is not None else None
        for sample in samples:
            if getattr(sample, "mopd_hook_applied", False):
                raise RuntimeError("MOPD hook applied twice to one sample")
            if sample.index is None or sample.group_index is None or sample.index // self.n != sample.group_index:
                raise ValueError("response-slot indexing no longer matches the data source")
            length = int(sample.response_length)
            rows = core.validated_rows(getattr(sample, "per_teacher_log_probs", None), self.names, length)
            mask = core.response_mask(sample.loss_mask, length)
            scores = core.mean_scores(rows, mask)
            current = core.argmin_name(scores)
            key, slot = prompt_key(sample.prompt), sample.index % self.n
            label = self.label_teacher(getattr(sample, "teacher_model_name", None))
            new, applied = core.teacher_log_probs(self.mode, rows, scores, label_teacher=label)
            if self.mode == "label":
                # the hook's label target must be exactly the legacy routed teacher's scores
                legacy = torch.as_tensor(sample.teacher_log_probs, dtype=torch.float32)
                if not torch.equal(new, legacy):
                    raise RuntimeError("label-mode hook differs from the legacy routed teacher_log_probs")
            if tuple(new.shape) != (length,) or not bool(torch.isfinite(new).all()) or bool((new == core.SENTINEL).any()):
                raise core.ScoreError("invalid replacement teacher_log_probs (would be silently masked or mis-shaped)")
            injected, original_length = False, length
            if sample.index in inject_ids:
                out = self._inject(sample, self.inject[key], key, length)
                if out is not None:
                    new, mask, length = out
                    injected = True
            seen = self.last_applied.get((key, slot))
            tracked = applied if self.mode != "pool" else current
            self.last_applied[(key, slot)] = tracked
            sample.teacher_log_probs = new
            sample.mopd_hook_applied = True
            if self.inject is not None:     # 0.0 = this sample's advantage is unchanged
                sample.inject_adv_const = self._step_const if injected else 0.0
            if dnorm_pairs is not None:     # rollout-side log-ratio for the per-domain statistic
                rlp = getattr(sample, "rollout_log_probs", None)
                if rlp is None:
                    raise RuntimeError(f"adv_norm: sample {sample.index} carries no rollout_log_probs")
                srow = torch.as_tensor(rlp, dtype=torch.float32)
                if tuple(srow.shape) != (length,) or not bool(torch.isfinite(srow).all()):
                    raise RuntimeError(f"adv_norm: sample {sample.index} rollout_log_probs unusable")
                dnorm_pairs.append(((sample.metadata or {}).get("domain") or "unknown",
                                    (new.to(torch.float32) - srow)[mask]))
            status = getattr(sample, "status", None)
            row = {
                "prompt_sha256": key, "slot": slot, "index": sample.index, "group_index": sample.group_index,
                "applied": applied, "current_argmin": current, "label_teacher": label,
                "teacher_mean_logp": scores, "response_length": length,
                "truncated": bool(getattr(status, "name", str(status)) == "TRUNCATED"),
                "domain": (sample.metadata or {}).get("domain"),
                "rollout_delta_mean": _rollout_delta_mean(sample, new, mask),
                "repeat": seen is not None, "switched": bool(seen is not None and seen != tracked),
            }
            if self.inject is not None:
                row.update(injected=injected, original_response_length=original_length,
                           anneal_step=self._anneal_step, adv_const=self._step_const if injected else None)
            rows_out.append(row)
            new_lists.append(new)
        if dnorm_pairs is not None:
            factors, self._dnorm_last = core.dn_multipliers(dnorm_pairs)
            applied = {}                    # the multiplier actually applied, per domain
            for d, f in factors.items():
                if self.adv_norm == "fixed_domain_scale":
                    if d not in self.adv_norm_fixed:
                        raise RuntimeError(f"adv_norm fixed_domain_scale: no multiplier registered for domain {d!r}")
                    applied[d] = self.adv_norm_fixed[d]
                elif self.adv_norm == "observe_domain_scale":
                    applied[d] = 1.0
                else:
                    applied[d] = f
            for sample, row in zip(samples, rows_out):
                d = row["domain"] or "unknown"
                sample.dn_adv_scale = applied[d]
                row["dnorm_factor"] = applied[d]
                row["dnorm_measured_factor"] = factors[d]
            self._dnorm_last["mode"] = self.adv_norm
            self._dnorm_last["applied"] = applied
        return new_lists, rows_out

    def finish(self, rows):
        if not rows:
            raise ValueError("empty audited batch")
        lo, hi = min(r["index"] for r in rows), max(r["index"] for r in rows)
        if self.audit_dir is not None:
            atomic_json(self.audit_dir / f"samples_{lo:08d}_{hi:08d}_{time.time_ns()}.json", {
                "mode": self.mode, "identity": IDENTITY[self.mode], "config_sha256": sha256(self.config_path),
                "rows": rows,
            })
        n = len(rows)
        routed = [r["applied"] if self.mode != "pool" else r["current_argmin"] for r in rows]
        by_domain = {}
        for r, name in zip(rows, routed):
            by_domain.setdefault(r["domain"] or "unknown", Counter())[name] += 1
        repeats = [r for r in rows if r["repeat"]]
        panel = {
            "mode": self.mode, "n": n, "iteration_first_index": lo,
            "share": {k: v / n for k, v in sorted(Counter(routed).items())},
            "share_by_domain": {d: {k: v / sum(c.values()) for k, v in sorted(c.items())} for d, c in sorted(by_domain.items())},
            "label_agreement": sum(name == r["label_teacher"] for r, name in zip(rows, routed)) / n,
            "shadow_argmin_disagreement": sum(r["applied"] != r["current_argmin"] for r in rows) / n if self.mode != "pool" else None,
            "repeat_prompts": len(repeats), "repeat_switch_rate": (sum(r["switched"] for r in repeats) / len(repeats)) if repeats else None,
            "mean_response_length": sum(r["response_length"] for r in rows) / n,
            "truncation_rate": sum(r["truncated"] for r in rows) / n,
            "routed_field": "applied teacher" if self.mode != "pool" else "shadow argmin (pool trains on the mixture)",
            "rollout_delta_by_domain": _delta_by_domain(rows),
            "inject": ({"injected_share": sum(bool(r.get("injected")) for r in rows) / n,
                        "injected_by_domain": {d: sum(bool(r.get("injected")) for r in rows if (r["domain"] or "unknown") == d)
                                               for d in sorted({r["domain"] or "unknown" for r in rows})},
                        "counts": dict(self.inject_stats), "anneal_step": self._anneal_step,
                        "step_const": self._step_const} if self.inject is not None else None),
            **({"dnorm": self._dnorm_last} if self.adv_norm is not None else {}),
        }
        print(f"{PANEL_PREFIX} {json.dumps(panel, sort_keys=True)}", flush=True)
        self.inject_stats = Counter()
        return panel


def _rollout_delta_mean(sample, teacher_row, mask):
    """Mean over response-mask tokens of (applied teacher logp - the rollout engine's student logp), or None."""
    rlp = getattr(sample, "rollout_log_probs", None)
    if rlp is None:
        return None
    row = torch.as_tensor(rlp, dtype=torch.float32)
    if tuple(row.shape) != tuple(teacher_row.shape) or not bool(torch.isfinite(row).all()):
        return None
    return float((teacher_row.to(torch.float32) - row)[mask].mean().item())


def _delta_by_domain(rows):
    """{domain: {n, mean_delta}} from the per-sample rollout-side diagnostic."""
    by = {}
    for r in rows:
        d = r.get("rollout_delta_mean")
        if d is None:
            continue
        by.setdefault(r["domain"] or "unknown", []).append(d)
    return {k: {"n": len(v), "mean_delta": sum(v) / len(v)} for k, v in sorted(by.items())}


def _state():
    global _STATE
    if _STATE is None:
        path = os.environ.get(CONFIG_ENV)
        if not path:
            raise RuntimeError(f"{CONFIG_ENV} is not set: refusing to train without a hook config")
        _STATE = MopdHook(path)
        print(f"MOPD_HOOK_STATE mode={_STATE.mode} identity={IDENTITY[_STATE.mode]!r} teachers={_STATE.names} "
              f"adv_norm={_STATE.adv_norm} inject={'on' if _STATE.inject is not None else 'off'}", flush=True)
    return _STATE


async def get_reward(args, sample, **kwargs):
    """All teachers score the sampled tokens of this response, with extra repair rounds for a teacher whose five
    attempts all failed. The legacy fields still come from the label-routed teacher."""
    from Uni_OPD_utils.OPD_reward import get_reward as base

    state = _state()
    start = time.time()
    rm_manager = base.RMSystemManager(args)
    session_manager = base.RewardSessionManager()
    response_correct, rule_based_metadata = await base.get_rule_based_reward(args, sample)
    reward = await base._get_reward_fanout(
        args=args, sample=sample, rm_manager=rm_manager, session_manager=session_manager, start_time=start,
        response_correct=response_correct, rule_based_metadata=rule_based_metadata)
    for extra in range(state.extra_rounds):
        failed = [name for name, info in reward["per_teacher"].items() if info["failed"]]
        if not failed:
            break
        logger.warning(f"[mopd_hook] repair round {extra + 1}/{state.extra_rounds} for teachers {failed}")
        await asyncio.sleep(5.0 * (extra + 1))
        for name in failed:
            payload = rm_manager.build_payload_for_teacher(sample, name)
            result = await base._score_one_teacher(
                session_manager, name, functools.partial(rm_manager.get_next_url_for_teacher, name), payload)
            itl = result["res"].get("meta_info", {}).get("input_token_logprobs") if not result["failed"] else None
            reward["per_teacher"][name] = {
                "input_token_logprobs": itl, "reward_time": result["reward_time"],
                "url": result["url"], "failed": bool(result["failed"] or itl is None)}
            if name == reward.get("routed_teacher") and itl is not None:
                reward.pop(base.REWARD_FAILED_KEY, None)
                reward["meta_info"] = result["res"]["meta_info"]
                reward["teacher_url"] = result["url"]
    reward["reward_time"] = time.time() - start
    return reward


def post_process_rewards(args, samples, **kwargs):
    from Uni_OPD_utils.OPD_reward import post_process_rewards as base

    base.post_process_rewards(args, samples, **kwargs)  # legacy routed fields + sample.per_teacher_log_probs
    state = _state()
    new_lists, rows = state.apply(args, samples)
    state.finish(rows)
    return new_lists, new_lists
