# SPDX-License-Identifier: Apache-2.0
"""SeqKD teacher answers: each domain's teacher answers its own student prompts once (label routing).

    python seqkd_generate.py worker --prompts F --domain D --model TEACHER --template-tokenizer STUDENT --out PART
                                    [--max-tokens 16384 --max-model-len 18432 --max-num-seqs 256 --temperature 1.0
                                     --top-p 1.0 --seed 42 --gpu-memory-utilization 0.90]
    python seqkd_generate.py merge  --out F --stats F --n N PART...

worker (needs vLLM; one GPU): renders the prompts with the student's chat template (enable_thinking=False, the rendering
of every OPD rollout and evaluation), samples one answer per prompt, and writes one row per prompt:
    {idx, prompt, label, domain, response, messages (user + assistant), metadata{finish_reason, teacher, teacher_model,
     n_prompt_tokens, n_response_tokens, ...}}
No answer is filtered for correctness. merge (CPU) concatenates the domain parts in prompt order and writes the
truncation statistics that ship with the corpus.
"""
import argparse
import json
import os
import sys


def _dom(r):
    d = r.get("domain")
    if d is None:
        d = (r.get("metadata") or {}).get("domain")
    return d


def worker(a) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    with open(a.prompts) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    sel = [(i, r) for i, r in enumerate(rows) if _dom(r) == a.domain]
    if not sel:
        raise SystemExit(f"no rows with domain={a.domain!r} in {a.prompts}")
    print(f"SEQKD_WORKER domain={a.domain} n={len(sel)} model={a.model}", flush=True)
    toker = AutoTokenizer.from_pretrained(a.template_tokenizer)
    prompts = [toker.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
               for _, r in sel]
    if not prompts[0].endswith("<think>\n\n</think>\n\n"):
        raise SystemExit("template drift: expected the non-thinking generation prompt (an empty think block)")
    llm = LLM(model=a.model, tensor_parallel_size=1, seed=a.seed, max_num_seqs=a.max_num_seqs,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization, enforce_eager=True)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, n=1, seed=a.seed)
    outs = llm.generate(prompts, sampling_params=sp)
    tmp = a.out + ".tmp"
    n_trunc = 0
    with open(tmp, "w") as f:
        for (idx, r), o in zip(sel, outs):
            c = o.outputs[0]
            fin = c.finish_reason or "stop"
            n_trunc += int(fin != "stop")
            meta = dict(r.get("metadata") or {})
            meta.update({"domain": a.domain, "finish_reason": fin, "teacher": a.domain, "teacher_model": a.model,
                         "n_prompt_tokens": len(o.prompt_token_ids), "n_response_tokens": len(c.token_ids)})
            f.write(json.dumps({"idx": idx, "prompt": r["prompt"], "label": r.get("label"), "domain": a.domain,
                                "response": c.text,
                                "messages": r["prompt"] + [{"role": "assistant", "content": c.text}],
                                "metadata": meta}, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    print(f"SEQKD_WORKER_DONE domain={a.domain} wrote={len(sel)} truncated={n_trunc} "
          f"({n_trunc / len(sel):.4f}) -> {a.out}", flush=True)
    return 0


def _block(rs):
    rl = sorted(r["metadata"]["n_response_tokens"] for r in rs)
    fin = {}
    for r in rs:
        fin[r["metadata"]["finish_reason"]] = fin.get(r["metadata"]["finish_reason"], 0) + 1
    return {"n": len(rs), "finish_reason_counts": fin, "truncation_rate": round(1 - fin.get("stop", 0) / len(rs), 4),
            "response_tokens": {"mean": round(sum(rl) / len(rl), 1), "p50": rl[len(rl) // 2],
                                "p90": rl[int(len(rl) * 0.9)], "max": rl[-1], "total": sum(rl)},
            "frac_with_think_tag": round(sum("<think>" in r["response"] for r in rs) / len(rs), 4)}


def merge(a) -> int:
    rows = []
    for p in a.parts:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["idx"])
    if len(rows) != a.n or [r["idx"] for r in rows] != list(range(a.n)):
        raise SystemExit(f"merge: {len(rows)} rows with idx gaps/duplicates, expected 0..{a.n - 1}")
    with open(a.out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(a.out + ".tmp", a.out)
    doms = sorted({r["domain"] for r in rows})
    stats = {"note": "teacher answers at the 16,384-token evaluation length, unfiltered; truncation reported",
             "overall": _block(rows), "per_domain": {d: _block([r for r in rows if r["domain"] == d]) for d in doms},
             "teachers": {d: sorted({r["metadata"]["teacher_model"] for r in rows if r["domain"] == d}) for d in doms}}
    with open(a.stats + ".tmp", "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(a.stats + ".tmp", a.stats)
    per = {d: stats["per_domain"][d]["truncation_rate"] for d in doms}
    print(f"SEQKD_MERGE_OK n={len(rows)} truncation overall={stats['overall']['truncation_rate']:.4f} per_domain={per}",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    for k in ("--prompts", "--domain", "--model", "--template-tokenizer", "--out"):
        w.add_argument(k, required=True)
    w.add_argument("--max-tokens", type=int, default=16384)
    w.add_argument("--max-model-len", type=int, default=18432)
    w.add_argument("--max-num-seqs", type=int, default=256)
    w.add_argument("--temperature", type=float, default=1.0)
    w.add_argument("--top-p", type=float, default=1.0)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--stats", required=True)
    m.add_argument("--n", type=int, required=True)
    m.add_argument("parts", nargs="+")
    a = ap.parse_args(argv)
    return worker(a) if a.cmd == "worker" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
