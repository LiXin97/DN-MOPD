# SPDX-License-Identifier: Apache-2.0
"""Teacher-trajectory bank of annealed injection (configs/injection_bank.yaml). Resumable phases, files under --parts:

  G  (vLLM, one GPU)       python injection_bank.py G --domain D --shard g --nshards N --teacher T --tokenizer S ...
         the domain's teacher samples n answers per prompt of shard g (prompts k with k % N == g), from the rollout's
         own prompt rendering (student chat template, enable_thinking=False); engine and request seed = seed + g
  A  (training env, CPU)   python injection_bank.py A --domain D --tokenizer S ...
         verdict of sample 0 with the trainer's rule-based verifier (strict: True or >= 1.0; code needs the judge),
         and a check that the rollout's own prompt ids (processor(text=...)) equal the generation's prompt ids
  B  (vLLM, one GPU)       python injection_bank.py B --domain D --shard g --nshards N --teacher T --tokenizer S ...
         the teacher's per-token log-probs of every kept trajectory (vLLM prompt_logprobs of prompt + response)
  C  (CPU)                 python injection_bank.py C --out BANK --teacher math=T --teacher code=T --teacher ifeval=T ...
         seals {meta, entries} with torch.save: one entry per prompt whose sample 0 finished, was verified correct and
         has matching prompt ids; entries are keyed by Uni_OPD_utils.mopd_hook.hook.prompt_key(rendered prompt) and
         hold {domain, teacher, prompt_ids, response_ids (trailing end tokens normalised to one <|im_end|>),
         teacher_logp}. Writes BANK.sha256 and BANK_META.json beside the bank.
Common options: --prompts (student prompts .jsonl), --parts DIR, --profile smoke.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DOMAINS = ("math", "code", "ifeval")
THINK_OFF_TAIL = "<think>\n\n</think>\n\n"


def params(profile: str | None) -> dict:
    cfg = yaml.safe_load((HERE.parent / "configs" / "injection_bank.yaml").read_text())
    out = {k: dict(v) for k, v in cfg.items() if k != "profiles"}
    if profile:
        for k, v in cfg["profiles"][profile].items():
            if k not in out["generation"]:
                raise SystemExit(f"profile {profile!r} overrides unknown key {k!r}")
            out["generation"][k] = v
    return out


def prompt_key(prompt) -> str:
    """== Uni_OPD_utils.mopd_hook.hook.prompt_key (the hook looks entries up by the rendered prompt text)."""
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(prompts: str, domain: str) -> list[dict]:
    rows = []
    for i, line in enumerate(open(prompts)):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("metadata") or {}).get("domain") == domain:
            rows.append(dict(r, idx=i))
    if not rows:
        raise SystemExit(f"{domain}: no prompts in {prompts}")
    return rows


def render(tok, messages) -> str:
    """The rollout's rendering (miles --apply-chat-template, kwargs {"enable_thinking": false}, tools=None)."""
    text = tok.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    if not text.endswith(THINK_OFF_TAIL):
        raise SystemExit("template drift: expected the non-thinking generation prompt (an empty think block)")
    return text


def end_ids(tok):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    extra = {tok.convert_tokens_to_ids("<|endoftext|>"), im_end}
    if tok.eos_token_id is not None:
        extra.add(tok.eos_token_id)
    return im_end, {i for i in extra if isinstance(i, int) and i >= 0}


def response_ids(gen_ids, im_end, ends) -> list[int]:
    ids = list(gen_ids)
    while ids and ids[-1] in ends:
        ids.pop()
    return ids + [im_end]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def g_rows(parts: Path, dom: str, nshards: int) -> list[dict]:
    rows = []
    for g in range(nshards):
        if not (parts / f"G_{dom}_s{g}.done").exists():
            raise SystemExit(f"G {dom} shard {g} not done")
        rows += _jsonl(parts / f"G_{dom}_s{g}.jsonl")
    return rows


# ------------------------------------------------------------------------------------------------ G
def phase_g(a, p) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gen = p["generation"]
    done = a.parts / f"G_{a.domain}_s{a.shard}.done"
    if done.exists():
        print(f"G {a.domain} shard {a.shard} already done", flush=True)
        return
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [r for k, r in enumerate(rows_of(a.prompts, a.domain)) if k % a.nshards == a.shard]
    out = a.parts / f"G_{a.domain}_s{a.shard}.jsonl"
    have = {r["key"] for r in _jsonl(out)}
    todo = []
    for r in rows:
        text = render(tok, r["prompt"])
        key = prompt_key(text)
        if key not in have:
            todo.append((key, r, tok(text, add_special_tokens=False)["input_ids"]))
    seed = int(gen["seed"]) + a.shard
    llm = LLM(model=str(a.teacher), tensor_parallel_size=1, seed=seed, max_num_seqs=gen["max_num_seqs"],
              max_model_len=gen["max_model_len"], gpu_memory_utilization=gen["gpu_memory_utilization"],
              enforce_eager=gen["enforce_eager"], enable_prefix_caching=gen["enable_prefix_caching"])
    sp = SamplingParams(n=gen["n"], temperature=gen["temperature"], top_p=gen["top_p"], max_tokens=gen["max_tokens"],
                        seed=seed)
    t0 = time.time()
    with out.open("a") as f:
        for i in range(0, len(todo), gen["chunk"]):
            chunk = todo[i:i + gen["chunk"]]
            outs = llm.generate([{"prompt_token_ids": list(pids)} for _, _, pids in chunk], sp, use_tqdm=False)
            for (key, r, pids), o in zip(chunk, outs):
                samples = [{"j": j, "gen_ids": list(c.token_ids), "text": c.text,
                            "finish_reason": c.finish_reason or "stop"} for j, c in enumerate(o.outputs)]
                f.write(json.dumps({"key": key, "idx": r["idx"], "domain": a.domain, "teacher": str(a.teacher),
                                    "shard": a.shard, "prompt_ids": list(pids), "samples": samples},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"G {a.domain} s{a.shard} {i + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


# ------------------------------------------------------------------------------------------------ A
def phase_a(a, p) -> None:
    from argparse import Namespace
    from types import SimpleNamespace

    from transformers import AutoProcessor, AutoTokenizer

    from Uni_OPD_utils.OPD_reward.rule_base_reward import get_rule_based_reward

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    proc = AutoProcessor.from_pretrained(a.tokenizer, trust_remote_code=True)
    src = {r["idx"]: r for r in rows_of(a.prompts, a.domain)}
    gen = g_rows(a.parts, a.domain, a.nshards)
    out = a.parts / f"A_{a.domain}.jsonl"
    done = {r["key"] for r in _jsonl(out)}
    todo = [(g, g["samples"][0]) for g in gen if g["key"] not in done and g["samples"]
            and g["samples"][0]["finish_reason"] == "stop"]
    print(f"A {a.domain}: {len(todo)} finished sample-0 answers to verify (done {len(done)})", flush=True)
    args = Namespace()

    def ids_match(g) -> tuple[bool, bool]:
        text = render(tok, src[g["idx"]]["prompt"])
        ids = proc(text=text)["input_ids"][0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return [int(x) for x in ids] == g["prompt_ids"], prompt_key(text) == g["key"]

    async def verdict(g, s):
        r = src[g["idx"]]
        ns = SimpleNamespace(response=s["text"], label=r["label"], metadata=r.get("metadata") or {},
                             teacher_model_name=r.get("teacher"), prompt=None)
        rc, _ = await get_rule_based_reward(args, ns)
        if rc is None or not isinstance(rc, (bool, int, float)):
            return None
        return (rc is True) or (isinstance(rc, (int, float)) and not isinstance(rc, bool) and float(rc) >= 1.0)

    async def run(batch):
        return await asyncio.gather(*(verdict(g, s) for g, s in batch))

    batch_size = int(p["verification"]["batch"])
    with out.open("a") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vs = asyncio.run(run(batch))
            for (g, s), v in zip(batch, vs):
                pm, km = ids_match(g)
                f.write(json.dumps({"key": g["key"], "j": s["j"], "domain": a.domain, "correct": v,
                                    "prompt_ids_match": pm, "key_match": km}) + "\n")
            f.flush()
            print(f"A {a.domain} {i + len(batch)}/{len(todo)}", flush=True)
    rows = _jsonl(out)
    (a.parts / f"A_{a.domain}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"A_DONE {a.domain} " + json.dumps({"n": len(rows), "correct": dict(collections.Counter(
        str(r["correct"]) for r in rows)), "prompt_ids_mismatch": sum(not r["prompt_ids_match"] for r in rows)}),
        flush=True)


def kept(parts: Path, dom: str, nshards: int, tok) -> tuple[dict, collections.Counter]:
    """{key: (gen_row, response_ids)} of the prompts whose sample 0 is finished, verified correct and id-matched."""
    im_end, ends = end_ids(tok)
    ver = {v["key"]: v for v in _jsonl(parts / f"A_{dom}.jsonl")}
    out, drop = {}, collections.Counter()
    for g in g_rows(parts, dom, nshards):
        s = g["samples"][0] if g["samples"] else None
        if s is None or s["finish_reason"] != "stop":
            drop["truncated"] += 1
            continue
        v = ver.get(g["key"])
        if v is None:
            raise SystemExit(f"{dom}: finished sample {g['key'][:12]}/0 has no phase-A row")
        if not (v["prompt_ids_match"] and v["key_match"]):
            drop["prompt_ids_mismatch"] += 1
            continue
        if v["correct"] is not True:
            drop["verifier_none" if v["correct"] is None else "wrong"] += 1
            continue
        out[g["key"]] = (g, response_ids(s["gen_ids"], im_end, ends))
    return out, drop


# ------------------------------------------------------------------------------------------------ B
def phase_b(a, p) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gen, sc = p["generation"], p["scoring"]
    done = a.parts / f"B_{a.domain}_s{a.shard}.done"
    if done.exists():
        print(f"B {a.domain} shard {a.shard} already done", flush=True)
        return
    if not (a.parts / f"A_{a.domain}.done").exists():
        raise SystemExit(f"B refused: A {a.domain} not done")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sel, _ = kept(a.parts, a.domain, a.nshards, tok)
    rows = [(key, g["prompt_ids"], rids) for key, (g, rids) in sorted(sel.items())]
    rows = [r for k, r in enumerate(rows) if k % a.nshards == a.shard]
    out = a.parts / f"B_{a.domain}_s{a.shard}.jsonl"
    have = {r["key"] for r in _jsonl(out)}
    todo = [r for r in rows if r[0] not in have]
    print(f"B {a.domain} shard {a.shard}: {len(todo)} trajectories to score (done {len(have)})", flush=True)
    if todo:
        llm = LLM(model=str(a.teacher), tensor_parallel_size=1, seed=int(sc["seed"]),
                  max_num_seqs=gen["max_num_seqs"], max_model_len=gen["max_model_len"],
                  gpu_memory_utilization=sc["gpu_memory_utilization"], enforce_eager=True, enable_prefix_caching=False)
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        with out.open("a") as f:
            for i in range(0, len(todo), int(sc["batch"])):
                batch = todo[i:i + int(sc["batch"])]
                seqs = [list(pids) + list(r) for _, pids, r in batch]
                res = llm.generate([{"prompt_token_ids": s} for s in seqs], sp, use_tqdm=False)
                for (key, pids, _r), s, o in zip(batch, seqs, res):
                    pl = o.prompt_logprobs
                    lp = [float(pl[pos][s[pos]].logprob) for pos in range(len(pids), len(s))]
                    f.write(json.dumps({"key": key, "teacher": str(a.teacher), "logp": lp}) + "\n")
                f.flush()
                print(f"B {a.domain} s{a.shard} {i + len(batch)}/{len(todo)}", flush=True)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


# ------------------------------------------------------------------------------------------------ C
def phase_c(a, p) -> None:
    import torch
    from transformers import AutoTokenizer

    gen = p["generation"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    teachers = dict(t.split("=", 1) for t in a.teacher)
    if sorted(teachers) != sorted(DOMAINS):
        raise SystemExit("C needs --teacher for all three domains")
    entries, per, drops = {}, {}, {}
    for dom in DOMAINS:
        sel, drop = kept(a.parts, dom, a.nshards, tok)
        drops[dom] = dict(drop)
        scores = {}
        for g in range(a.nshards):
            if not (a.parts / f"B_{dom}_s{g}.done").exists():
                raise SystemExit(f"C refused: B {dom} shard {g} not done")
            for b in _jsonl(a.parts / f"B_{dom}_s{g}.jsonl"):
                scores[b["key"]] = b
        n = 0
        for key, (grow, rids) in sel.items():
            b = scores.get(key)
            if b is None:
                raise SystemExit(f"{dom}: kept trajectory {key[:12]} has no phase-B score")
            lp = b["logp"]
            if len(lp) != len(rids) or not all(math.isfinite(x) and x <= 0.0 for x in lp):
                raise SystemExit(f"{dom}: unusable teacher log-probs for {key[:12]}")
            if len(rids) > int(gen["max_tokens"]):
                raise SystemExit(f"{dom}: response of {len(rids)} tokens exceeds max_tokens")
            if key in entries:
                raise SystemExit(f"duplicate prompt key {key[:12]}")
            entries[key] = {"domain": dom, "teacher": Path(teachers[dom]).name,
                            "prompt_ids": torch.tensor(grow["prompt_ids"], dtype=torch.int32),
                            "response_ids": torch.tensor(rids, dtype=torch.int32),
                            "teacher_logp": torch.tensor(lp, dtype=torch.float32)}
            n += 1
        per[dom] = n
    meta = {"rule": "one trajectory per prompt = sample 0 when finished, verified correct and id-matched",
            "generation": gen, "teachers": {d: Path(teachers[d]).name for d in DOMAINS}, "per_domain": per,
            "dropped_prompts": drops, "n_entries": len(entries),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    torch.save({"meta": meta, "entries": entries}, tmp)
    os.replace(tmp, out)
    digest = sha256(out)
    out.with_name(out.name + ".sha256").write_text(digest + "\n")
    out.with_name("BANK_META.json").write_text(json.dumps(dict(meta, sha256=digest), indent=1) + "\n")
    print("C_DONE " + json.dumps({"entries": len(entries), "per_domain": per, "sha256": digest, "dropped": drops}),
          flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["G", "A", "B", "C"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True, help="the student's base model (chat template and tokenizer)")
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int)
    ap.add_argument("--teacher", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--profile")
    a = ap.parse_args(argv)
    p = params(a.profile)
    a.nshards = a.nshards or int(p["generation"]["shards"])
    a.parts.mkdir(parents=True, exist_ok=True)
    if a.phase in ("G", "A", "B") and not a.domain:
        raise SystemExit("--domain is required")
    if a.phase in ("G", "B"):
        if len(a.teacher) != 1:
            raise SystemExit(f"{a.phase} needs exactly one --teacher DIR")
        a.teacher = a.teacher[0]
    if a.phase == "C" and not a.out:
        raise SystemExit("C needs --out")
    {"G": phase_g, "A": phase_a, "B": phase_b, "C": phase_c}[a.phase](a, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
