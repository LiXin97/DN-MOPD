# SPDX-License-Identifier: Apache-2.0
"""Sample answers for one model, suite and cap with vLLM, under the paper's frozen protocol.

    python -m eval.generate --model Qwen/Qwen3.5-2B --suite aime25 --max-tokens 8192
    python -m eval.generate --model /path/to/hf_export --name my_student --suite lcb_v6 --max-tokens 16384 \
        --out outputs/eval/my_student/lcb_v6/cap16384
    # one process per GPU over the suite's shards (optional):
    CUDA_VISIBLE_DEVICES=1 python -m eval.generate ... --shard 1

Protocol (eval/protocol.py; nothing here is configurable except the cap):
  * vLLM 0.18, tensor parallel 1, enforce_eager, engine seed 42; the six public suites use gpu_memory_utilization 0.90,
    max_model_len 32768 and max_num_seqs 64; MATH-500 uses its original settings (0.95, the model's own context,
    max_num_seqs 256, all prompts in one call, responses stripped);
  * temperature 1.0, top-p 1.0, top-k -1, presence penalty 0, sampling seed 42, max_tokens = the cap;
  * chat template with add_generation_prompt=True and enable_thinking=False; the prompts are the prepared catalogs
    (python -m eval.data.prepare), whose sha256 must equal the paper's evaluation catalogs;
  * n = 64 (AIME), 6 (LiveCodeBench), 16 (IFEval, IFBench, MATH-500) sampled answers per question;
  * questions go to the engine in chunks (4 AIME, 16 LiveCodeBench, 32 IF problems per call); every chunk is written
    atomically to <out>/raw/<first question>.json, so an interrupted run resumes where it stopped.

The default output directory is $OUT_ROOT/eval/<name>/<suite>/cap<cap> (OUT_ROOT defaults to <repo>/outputs); grade
it with `python -m eval.grade --gen <dir>`. Seeded sampling is not bitwise reproducible across GPU types, drivers or
vLLM versions, so regenerated answers are a fresh draw from the same protocol, not a copy of the paper's answers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from eval.common import REPO_ROOT, digest, prepared_dir, records, write_text_atomic
from eval.protocol import (CATALOG_SHA256, CHAT_TEMPLATE_SHA256, CONTEXT, ENABLE_THINKING, ENGINE, ENGINE_MATH500,
                           SAMPLING, SUITES, VLLM_VERSION, check_cap)

PROTOCOL_ID = 'dn-mopd-eval-v1'


def default_name(model: str) -> str:
    """A directory-safe model name: the checkpoint directory name, or the hub id with '/' replaced."""
    path = Path(model)
    return path.name if path.exists() else model.replace('/', '__')


def catalog(suite: str, data: Path, limit: Optional[int] = None) -> List[dict]:
    """The prepared catalog of a suite, checked against the paper's evaluation catalog hash."""
    path = data / 'catalog' / f'{suite}.jsonl'
    if not path.is_file():
        raise SystemExit(f'{path} is missing; run: python -m eval.data.prepare --suites {suite}')
    want = CATALOG_SHA256.get(suite) or json.loads((data / 'PREPARED.json').read_text())[suite]['catalog_sha256']
    if digest(path) != want:
        raise SystemExit(f'{path} is not the paper catalog (sha256 differs); re-run python -m eval.data.prepare')
    rows = records(path)
    if len(rows) != SUITES[suite].problems:
        raise SystemExit(f'{path}: {len(rows)} questions, expected {SUITES[suite].problems}')
    return rows[:limit] if limit else rows


def chunk_starts(count: int, chunk: int) -> List[int]:
    return list(range(0, count, chunk))


def shard_starts(count: int, chunk: int, shards: int, shard: int) -> List[int]:
    """Chunk starts owned by one shard: contiguous, disjoint, together covering the suite."""
    starts = chunk_starts(count, chunk)
    if shard not in range(shards):
        raise ValueError(f'shard must be in [0, {shards})')
    size = -(-len(starts) // shards)
    return starts[shard * size:(shard + 1) * size]


def validate_rows(rows: Sequence[dict], source: Sequence[dict], name: str, cap: int, n: int) -> None:
    """Generated rows must match their questions one to one and carry n samples with metadata."""
    if len(rows) != len(source):
        raise ValueError('incomplete rows')
    for r, q in zip(rows, source):
        if (r['id'], r['input_sha256']) != (q['id'], q['input_sha256']):
            raise ValueError(f'prompt identity differs: {q["id"]}')
        if (r['model'], r['cap'], r['n']) != (name, cap, n):
            raise ValueError(f'model/sampling identity differs: {q["id"]}')
        if len(r['responses']) != n or any(not isinstance(s, str) for s in r['responses']):
            raise ValueError(f'samples: {q["id"]}')
        if len(r['token_counts']) != n or len(r['finish_reasons']) != n:
            raise ValueError(f'generation metadata: {q["id"]}')


def packages() -> dict:
    out = {}
    for p in ('torch', 'transformers', 'vllm'):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_manifest(out: Path, params: dict) -> None:
    """The run protocol is written once; a resumed run must match it exactly."""
    path = out / 'MANIFEST.json'
    if path.exists():
        if json.loads(path.read_text()) != params:
            raise SystemExit(f'{path} records a different protocol, model or cap; refusing a mixed-protocol resume')
        return
    write_text_atomic(path, json.dumps(params, ensure_ascii=False, indent=2) + '\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='HF model directory or hub id')
    ap.add_argument('--suite', required=True, choices=list(SUITES))
    ap.add_argument('--max-tokens', type=int, required=True, choices=(8192, 16384), help='generation cap')
    ap.add_argument('--out', type=Path, default=None, help='output directory of this model, suite and cap')
    ap.add_argument('--name', default=None, help='model name used in records (default: directory name / hub id)')
    ap.add_argument('--shard', type=int, default=None,
                    help='generate only this shard of the suite (0-based; the suite has a fixed shard count)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N questions (smoke tests only)')
    a = ap.parse_args(argv)
    suite, cap = SUITES[a.suite], check_cap(a.max_tokens)
    name = a.name or default_name(a.model)
    out = a.out or Path(os.environ.get('OUT_ROOT', REPO_ROOT / 'outputs')) / 'eval' / name / suite.name / f'cap{cap}'
    rows = catalog(suite.name, prepared_dir(), a.limit)
    engine = ENGINE_MATH500 if suite.name == 'math500' else ENGINE
    starts = (chunk_starts(len(rows), suite.chunk) if a.shard is None
              else shard_starts(len(rows), suite.chunk, suite.shards, a.shard))

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    template = tok.chat_template if isinstance(tok.chat_template, str) else None
    template_sha = hashlib.sha256(template.encode()).hexdigest() if template else None
    if not template or 'enable_thinking' not in template:
        print('WARNING: the chat template does not implement enable_thinking; the protocol renders with '
              'enable_thinking=False', file=sys.stderr)
    if template_sha not in CHAT_TEMPLATE_SHA256.values():
        print(f'NOTE: chat template sha256 {template_sha} is not one of the paper\'s Qwen3.5 templates', file=sys.stderr)
    versions = packages()
    if versions['vllm'] != VLLM_VERSION:
        print(f'WARNING: vLLM {versions["vllm"]} differs from the paper\'s {VLLM_VERSION}', file=sys.stderr)

    params = {'protocol': PROTOCOL_ID, 'model': a.model, 'name': name, 'suite': suite.name, 'cap': cap, 'n': suite.n,
              'chunk': suite.chunk, 'questions': len(rows), 'subset': bool(a.limit),
              'catalog_sha256': digest(prepared_dir() / 'catalog' / f'{suite.name}.jsonl'),
              'sampling': {**SAMPLING, 'max_tokens': cap}, 'enable_thinking': ENABLE_THINKING, 'engine': engine,
              'chat_template_sha256': template_sha, 'packages': versions}
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, params)
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)

    def render(row: dict) -> str:
        return tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=ENABLE_THINKING)

    pending = []
    for start in starts:
        batch, path = rows[start:start + suite.chunk], raw / f'{start:04d}.json'
        if path.exists():
            validate_rows(json.loads(path.read_text()), batch, name, cap, suite.n)
        else:
            pending.append((path, batch))
    if not pending:
        print(f'GENERATE {name} {suite.name} cap{cap}: nothing left to generate in {out}')
        return 0

    prompts = {}
    for r in rows:            # validate the whole suite before spending GPU time; never trim a long question
        prompts[r['id']] = render(r)
        if suite.name != 'math500' and len(tok.encode(prompts[r['id']], add_special_tokens=False)) + cap > CONTEXT:
            raise SystemExit(f'question exceeds the fixed context: {r["id"]}')

    from vllm import LLM, SamplingParams
    if suite.name == 'math500':
        llm = LLM(model=a.model, tokenizer=a.model, **engine)
        params_sp = SamplingParams(temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'], max_tokens=cap,
                                   n=suite.n, seed=SAMPLING['seed'])
    else:
        llm = LLM(model=a.model, **engine)
        params_sp = SamplingParams(n=suite.n, max_tokens=cap, **SAMPLING)
    for path, batch in pending:
        outputs = llm.generate([prompts[r['id']] for r in batch], params_sp, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError('missing generated prompt')
        generated = []
        for r, o in zip(batch, outputs):
            if len(o.outputs) != suite.n:
                raise RuntimeError('incomplete samples')
            # No stripping for the six suites: strict IF grading and code extraction consume the text as generated.
            texts = [x.text.strip() if suite.name == 'math500' else x.text for x in o.outputs]
            generated.append({'id': r['id'], 'input_sha256': r['input_sha256'], 'model': name, 'cap': cap,
                              'n': suite.n, 'responses': texts, 'token_counts': [len(x.token_ids) for x in o.outputs],
                              'finish_reasons': [x.finish_reason for x in o.outputs]})
        validate_rows(generated, batch, name, cap, suite.n)
        write_text_atomic(path, json.dumps(generated, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        print(f'GENERATED {name} {suite.name} cap{cap} {path.name}', flush=True)
    print(f'GENERATE_DONE {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
