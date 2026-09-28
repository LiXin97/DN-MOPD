# SPDX-License-Identifier: Apache-2.0
"""ParamMerge-Avg and ParamMerge-TA rows (paper App. A.2) from the three teachers' HF exports. CPU only.

    python recipes/qwen3.5/merge.py --kind avg|ta --size 2b [--base DIR] [--teacher math=DIR ...] [--out DIR]

  avg : theta = mean_k theta_k             float32 accumulation of w * theta_k (w = 1/3), saved in the first expert's
                                           dtype per tensor
  ta  : theta = theta_0 + lambda * sum_k (theta_k - theta_0),  lambda = 1 (configs/merge.yaml); float32 accumulation,
        saved in the experts' common dtype; lambda = 1/K would be avg and is refused
Key handling: the three teachers must have identical key sets and shapes; the base may carry extra keys only under
`mtp.` (the multi-token-prediction head, absent from the trainer's exports), which are excluded from both rows; any
other difference is refused. config / tokenizer / chat template / processor files are copied from the base after
checking that they equal the teachers'. Defaults: --base = the size's base model ($MODEL_ROOT/Qwen3.5-<S>, BASE_MODEL),
--teacher = $CKPT_ROOT/qwen3.5-<S>_teacher_<domain>_hf (TEACHER_MATH / TEACHER_CODE / TEACHER_IF),
--out = $CKPT_ROOT/qwen3.5-<S>_merge_<kind>_hf. A MERGE_PROVENANCE.json with input and output hashes is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
DOMAINS = ("math", "code", "ifeval")
TEACHER_ENV = {"math": "TEACHER_MATH", "code": "TEACHER_CODE", "ifeval": "TEACHER_IF"}


def load_config() -> dict:
    return yaml.safe_load((HERE / "configs" / "merge.yaml").read_text())


def average(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Equal-weight mean, float32 accumulation of w * t (w = 1/K), saved in the first tensor's dtype."""
    w = 1.0 / len(tensors)
    acc, dt = None, None
    for t in tensors:
        if dt is None:
            dt = t.dtype
        acc = t.to(torch.float32) * w if acc is None else acc + t.to(torch.float32) * w
    return acc.to(dt)


def task_arithmetic(base: torch.Tensor, sources: list[torch.Tensor], lam: float) -> torch.Tensor:
    """theta_0 + lam * sum_k (theta_k - theta_0); output dtype = the sources' common dtype."""
    if any(s.shape != base.shape for s in sources):
        raise ValueError("shape differs from the common base")
    if not base.is_floating_point():
        for s in sources:
            if s.dtype != base.dtype or not torch.equal(s, base):
                raise ValueError("non-floating buffer differs from the common base")
        return base.clone()
    dtypes = {s.dtype for s in sources}
    if len(dtypes) != 1 or not next(iter(dtypes)).is_floating_point:
        raise ValueError("experts must share one floating dtype")
    b = base.to(torch.float32)
    acc = b.clone()
    for s in sources:
        acc += lam * (s.to(torch.float32) - b)
    return acc.to(sources[0].dtype)


def check_lambda(lam: float, k: int) -> None:
    if abs(lam - 1.0 / k) < 1e-12:
        raise SystemExit("lambda = 1/k is ParamMerge-Avg; it is not an independent task-arithmetic row")


def index(d: str) -> dict:
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    out = {}
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        with safe_open(os.path.join(d, f), "pt") as h:
            for k in h.keys():
                out[k] = f
    if not out:
        raise SystemExit(f"no safetensors under {d}")
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def plan_keys(base_keys: set, teacher_keys: list[set], base_only_prefix: str) -> tuple[list[str], list[str]]:
    """(merged keys, excluded base-only keys); raises on any disallowed key difference."""
    keys = sorted(teacher_keys[0])
    for tk in teacher_keys[1:]:
        if sorted(tk) != keys:
            raise SystemExit("teacher key sets differ")
    missing_in_base = sorted(set(keys) - base_keys)
    if missing_in_base:
        raise SystemExit(f"{len(missing_in_base)} teacher keys absent from the base, e.g. {missing_in_base[:3]}")
    base_extra = sorted(base_keys - set(keys))
    bad = [k for k in base_extra if not k.startswith(base_only_prefix)]
    if bad:
        raise SystemExit(f"base-only keys outside {base_only_prefix!r}: {bad[:5]}")
    return keys, base_extra


def _default_base(size: str) -> str:
    if os.environ.get("BASE_MODEL"):
        return os.environ["BASE_MODEL"]
    out = subprocess.run(["bash", "-c", f'source "{HERE}/common.sh" && base_model {size}'], capture_output=True,
                         text=True)
    if out.returncode != 0:
        raise SystemExit(f"no base model for {size}: {out.stderr.strip()}")
    return out.stdout.strip().splitlines()[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=("avg", "ta"))
    ap.add_argument("--size", required=True, choices=("2b", "4b", "9b"))
    ap.add_argument("--base")
    ap.add_argument("--teacher", action="append", default=[], help="domain=DIR (math, code, ifeval)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config()
    ckpt_root = os.environ.get("CKPT_ROOT") or str(HERE.parents[1] / "outputs" / "checkpoints")
    teachers = {d: os.environ.get(TEACHER_ENV[d]) or f"{ckpt_root}/qwen3.5-{a.size}_teacher_{d}_hf" for d in DOMAINS}
    for t in a.teacher:
        d, _, p = t.partition("=")
        if d not in DOMAINS or not p:
            raise SystemExit(f"bad --teacher {t!r} (domain=DIR)")
        teachers[d] = p
    base = a.base or _default_base(a.size)
    out = a.out or f"{ckpt_root}/qwen3.5-{a.size}_merge_{a.kind}_hf"
    order = cfg["teacher_order"]
    srcs = [teachers[d] for d in order]
    lam = float(cfg["task_arithmetic_lambda"])
    if a.kind == "ta":
        check_lambda(lam, len(srcs))
    if os.path.exists(out):
        raise SystemExit(f"refuse to overwrite {out}")
    for f in cfg["must_match"]:
        ref = sha256(os.path.join(base, f))
        for s in srcs:
            if sha256(os.path.join(s, f)) != ref:
                raise SystemExit(f"{f} differs between the base and {s}")
    bmap, tmaps = index(base), [index(s) for s in srcs]
    keys, base_extra = plan_keys(set(bmap), [set(m) for m in tmaps], cfg["base_only_prefix"])
    out_dir = out + ".partial"
    os.makedirs(out_dir)
    handles = {}

    def tensor(d, m, k):
        p = os.path.join(d, m[k])
        if p not in handles:
            handles[p] = safe_open(p, "pt")
        return handles[p].get_tensor(k)

    shard, shard_bytes, shards, weight_map = {}, 0, [], {}
    stats = {"floating": 0, "non_floating": 0, "max_abs_delta_vs_base": 0.0}

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        name = f"tmp-{len(shards) + 1:05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shards.append(name)
        shard, shard_bytes = {}, 0

    t0 = time.time()
    for n, k in enumerate(keys):
        ts = [tensor(s, m, k) for s, m in zip(srcs, tmaps)]
        b = tensor(base, bmap, k)
        if any(t.shape != b.shape for t in ts):
            raise SystemExit(f"{k}: shape differs ({[tuple(t.shape) for t in ts]} vs base {tuple(b.shape)})")
        if a.kind == "avg":
            merged = average(ts)
        else:
            try:
                merged = task_arithmetic(b, ts, lam)
            except ValueError as exc:
                raise SystemExit(f"{k}: {exc}")
        if merged.is_floating_point():
            stats["floating"] += 1
            stats["max_abs_delta_vs_base"] = max(stats["max_abs_delta_vs_base"],
                                                 (merged.to(torch.float32) - b.to(torch.float32)).abs().max().item())
        else:
            stats["non_floating"] += 1
        shard[k] = merged.contiguous()
        shard_bytes += merged.numel() * merged.element_size()
        if shard_bytes >= int(cfg["shard_size_mb"]) * 1024 * 1024:
            flush()
        if n % 100 == 0:
            print(f"  {n}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    flush()
    total, final_map = len(shards), {}
    for i, name in enumerate(shards, 1):
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.rename(os.path.join(out_dir, name), os.path.join(out_dir, new))
        for k, v in weight_map.items():
            if v == name:
                final_map[k] = new
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.endswith(".safetensors"))
    json.dump({"metadata": {"total_size": size}, "weight_map": final_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    for f in cfg["aux_files"]:
        p = os.path.join(base, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))
    prov = {"kind": a.kind, "size": a.size,
            "formula": "mean_k theta_k" if a.kind == "avg" else "theta_0 + lambda * sum_k (theta_k - theta_0)",
            "lambda": None if a.kind == "avg" else lam, "k": len(srcs), "accumulation_dtype": "float32",
            "tensors": stats, "parameter_keys": len(keys), "excluded_base_only_keys": base_extra,
            "teachers": {d: {"config_sha256": sha256(os.path.join(teachers[d], "config.json"))} for d in order},
            "output_shards": {f: sha256(os.path.join(out_dir, f)) for f in sorted(set(final_map.values()))},
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(prov, open(os.path.join(out_dir, "MERGE_PROVENANCE.json"), "w"), indent=1)
    os.rename(out_dir, out)
    print(f"MERGE_OK kind={a.kind} size={a.size} out={out} shards={total} keys={len(keys)} "
          f"excluded={len(base_extra)} tensors={stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
