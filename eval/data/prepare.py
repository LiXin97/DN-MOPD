# SPDX-License-Identifier: Apache-2.0
"""Download the pinned public benchmark sources and rebuild the paper's evaluation catalogs, verifying every question.

    python -m eval.data.prepare                      # all seven suites into eval/data/prepared ($DN_MOPD_EVAL_DATA)
    python -m eval.data.prepare --suites aime25,ifeval
    python -m eval.data.prepare --source-dir DIR     # reuse already downloaded source files (checked by sha256)

For each suite this
  1. downloads the source file at its pinned revision (eval/data/sources.json) and checks its sha256;
  2. rebuilds each question exactly as the paper's evaluation did (the prompt as chat messages, the reference answer
     or the official tests);
  3. checks that every question's id and input hash equal the released catalog (reproduce/data/catalog; for MATH-500
     the released question list), in order, and that the rebuilt complete catalog has the sha256 of the catalog the
     paper's cells were generated from (eval/protocol.py);
  4. for LiveCodeBench, decodes each problem's official tests with the official lcb_runner and checks them against
     the catalog's test hash.
It prints PASS or FAIL per suite and exits non-zero on any failure.

The released catalogs of AIME 2025 and LiveCodeBench carry no benchmark text (AIME 2025: ids and input hashes only;
LiveCodeBench: ids, dates, difficulty and hashes). Their problems, reference answers and tests are rebuilt here from
the pinned public sources (about 700 MB of downloads for LiveCodeBench test5 and test6), and the complete catalog
must reproduce the evaluation catalog hash exactly.

Output: <out>/catalog/<suite>.jsonl, <out>/lcb_tests/<test_index>.json, <out>/PREPARED.json.
Requires pyarrow (AIME 2025 is a parquet file) and, for LiveCodeBench, the pinned lcb_runner
(python -m eval.external.fetch --only LiveCodeBench).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from eval.common import digest, jsonl_text, prepared_dir, records, released_dir, sha256_text, write_text_atomic
from eval.protocol import AIME_SUFFIX, CATALOG_SHA256, SUITES

SOURCES_FILE = Path(__file__).resolve().parent / 'sources.json'
ORDER = ('aime25', 'aime26', 'ifeval', 'ifbench', 'math500', 'lcb_v5', 'lcb_v6')
# Suites whose released catalog (reproduce/data/catalog) omits the benchmark text; prepare.py restores it.
TEXT_NOT_RELEASED = ('aime25', 'lcb_v5', 'lcb_v6')


def sources() -> Dict[str, dict]:
    return {k: v for k, v in json.loads(SOURCES_FILE.read_text()).items() if not k.startswith('_')}


def source_url(spec: dict) -> str:
    if spec['host'] == 'huggingface':
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
        return f'{endpoint}/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}'
    return f'https://raw.githubusercontent.com/{spec["repo"]}/{spec["revision"]}/{spec["file"]}'


def fetch_source(spec: dict, source_dir: Path) -> Path:
    """The pinned source file, downloaded once into source_dir and checked by sha256."""
    dest = source_dir / spec['repo'].replace('/', '__') / spec['revision'] / spec['file']
    if dest.is_file() and digest(dest) == spec['sha256']:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url, tmp = source_url(spec), dest.with_name(dest.name + '.part')
    for attempt in range(1, 5):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=300) as r, tmp.open('wb') as f:
                for block in iter(lambda: r.read(1 << 22), b''):
                    h.update(block)
                    f.write(block)
            break
        except OSError as exc:
            if attempt == 4:
                raise RuntimeError(f'download failed: {url}: {exc}') from exc
            time.sleep(5 * attempt)
    if h.hexdigest() != spec['sha256']:
        tmp.unlink()
        raise ValueError(f'{url}: sha256 {h.hexdigest()} differs from the pinned {spec["sha256"]}')
    tmp.replace(dest)
    return dest


# ---- catalog builders: the constructions of the paper's evaluation, unchanged ---------------------------------------

def _math_row(qid: str, input_sha256: str, problem: str, answer) -> dict:
    return {'id': qid, 'input_sha256': input_sha256, 'messages': [{'role': 'user', 'content': problem + AIME_SUFFIX}],
            'visible': problem, 'answer': answer}


def build_aime25(path: Path, spec: dict) -> List[dict]:
    """The input row is {problem, answer (str), id (int), suite}; its hash is sha256(json.dumps(row, sort_keys=True))."""
    import pyarrow.parquet as pq
    rows = []
    for r in pq.read_table(path).to_pylist():
        row = {'problem': r['problem'], 'answer': str(r['answer']), 'id': int(r['id']), 'suite': 'aime25'}
        rows.append(_math_row(str(row['id']), sha256_text(json.dumps(row, sort_keys=True)), row['problem'],
                              row['answer']))
    return rows


def build_aime26(path: Path, spec: dict) -> List[dict]:
    """Rows of aime2026.jsonl verbatim (exact decoded problem strings; answers stay integers)."""
    return [_math_row(str(r['id']), sha256_text(json.dumps(r, sort_keys=True)), r['problem'], r['answer'])
            for r in records(path)]


def _if_rows(path: Path, suite: str, provenance: str) -> List[dict]:
    """The public IF line (official fields, kwargs None-filtered as both official evaluation_libs do, plus a metadata
    envelope) is hashed as serialized; the prompt is one user turn, verbatim."""
    out = []
    for i, r in enumerate(records(path)):
        ids = r['instruction_id_list']
        if not ids:
            raise ValueError(f'{suite} row {i}: empty instruction_id_list')
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None}
                  for kw in (r.get('kwargs') or [{} for _ in ids])]
        if len(kwargs) != len(ids):
            raise ValueError(f'{suite} row {i}: kwargs/instruction length mismatch')
        line = json.dumps({'key': r['key'], 'prompt': r['prompt'], 'instruction_id_list': ids, 'kwargs': kwargs,
                           'suite': suite, 'domain': f'{suite}_public', 'data_source': provenance, 'src_index': i,
                           'metadata': {'suite': suite, 'n_instructions': len(ids), 'split': 'public_eval'}},
                          ensure_ascii=False) + '\n'
        out.append({'id': f'{suite}:{r["key"]}', 'input_sha256': sha256_text(line), 'visible': r['prompt'],
                    'messages': [{'role': 'user', 'content': str(r['prompt'])}], 'key': r['key'],
                    'prompt': r['prompt'], 'instruction_id_list': ids, 'kwargs': kwargs, 'suite': suite})
    return out


def build_ifeval(path: Path, spec: dict) -> List[dict]:
    return _if_rows(path, 'ifeval', spec['provenance'])


def build_ifbench(path: Path, spec: dict) -> List[dict]:
    return _if_rows(path, 'ifbench', spec['provenance'])


def build_math500(path: Path, spec: dict) -> List[dict]:
    """HuggingFaceH4/MATH-500 in its published order; the prompt is the problem plus the AIME answer instruction."""
    out = []
    for r in records(path):
        released = {'id': r['unique_id'], 'problem': r['problem'], 'answer': str(r['answer'])}
        out.append(_math_row(released['id'], sha256_text(json.dumps(released, sort_keys=True)), r['problem'],
                             released['answer']))
    return out


def build_lcb(path: Path, spec: dict, released: List[dict], tests_dir: Path) -> List[dict]:
    """Rebuild the complete LiveCodeBench rows (prompt rendered by the official template) and decode the official
    tests of every problem into tests_dir/<test_index>.json, checked against the released test hash."""
    from eval.graders.code_grader import catalog_row, evaluation_sample_json
    by_row = {r['source_row']: r for r in released}
    if len(by_row) != len(released) or any(r['source_file'] != spec['file'] for r in released):
        raise ValueError('released catalog does not index the pinned source file')
    built: Dict[str, dict] = {}
    count = 0
    with path.open(encoding='utf-8') as f:
        for row_index, line in enumerate(f):
            count += 1
            want = by_row.get(row_index)
            if want is None:
                continue
            if sha256_text(line) != want['input_sha256']:
                raise ValueError(f'input hash differs: {want["id"]}')
            base = catalog_row(line)
            tests = evaluation_sample_json(line)
            if sha256_text(tests) != want['test_sha256']:
                raise ValueError(f'decoded official tests differ from the released test hash: {want["id"]}')
            write_text_atomic(tests_dir / f'{want["test_index"]:04d}.json', tests)
            full = {'id': base['id'], 'input_sha256': sha256_text(line), 'messages': base['messages'],
                    'visible': base['visible'], 'test_index': want['test_index'], 'test_sha256': want['test_sha256'],
                    'tests': base['tests'], 'date': base['date'], 'difficulty': base['difficulty'],
                    'source_file': spec['file'], 'source_row': row_index}
            if {k: v for k, v in full.items() if k not in ('messages', 'visible')} != want:
                raise ValueError(f'rebuilt fields differ from the released catalog: {want["id"]}')
            built[want['id']] = full
    if count != spec['rows']:
        raise ValueError(f'{spec["file"]} has {count} rows, expected {spec["rows"]}')
    return [built[r['id']] for r in released]


BUILDERS: Dict[str, Callable] = {'aime25': build_aime25, 'aime26': build_aime26, 'ifeval': build_ifeval,
                                 'ifbench': build_ifbench, 'math500': build_math500}


def released_catalog(suite: str, released: Path) -> List[dict]:
    if suite == 'math500':
        return records(released / 'math500' / 'questions.jsonl')
    return records(released / 'catalog' / f'{suite}.jsonl')


def verify(suite: str, rows: List[dict], reference: List[dict]) -> Tuple[int, str]:
    """(questions checked, catalog sha256); raises on the first difference from the released catalog."""
    if len(rows) != SUITES[suite].problems or len(reference) != len(rows):
        raise ValueError(f'{suite}: {len(rows)} rebuilt questions, {len(reference)} released, '
                         f'{SUITES[suite].problems} expected')
    for i, (got, want) in enumerate(zip(rows, reference)):
        if suite == 'math500':
            same = (got['id'], got['visible'], got['answer']) == (want['id'], want['problem'], want['answer'])
        else:
            same = (got['id'], got['input_sha256']) == (want['id'], want['input_sha256'])
            same = same and all(got[k] == v for k, v in want.items())   # every released field, prompts included
        if not same:
            raise ValueError(f'{suite}: question {i} ({want["id"]}) differs from the released catalog')
    text = jsonl_text(rows)
    sha = sha256_text(text)
    if suite in CATALOG_SHA256 and sha != CATALOG_SHA256[suite]:
        raise ValueError(f'{suite}: rebuilt catalog sha256 {sha} differs from the evaluation catalog '
                         f'{CATALOG_SHA256[suite]}')
    return len(rows), sha


def prepare(suite: str, out: Path, source_dir: Path, released: Path) -> dict:
    spec = sources()[suite]
    path = fetch_source(spec, source_dir)
    reference = released_catalog(suite, released)
    if suite.startswith('lcb_'):
        rows = build_lcb(path, spec, reference, out / 'lcb_tests')
    else:
        rows = BUILDERS[suite](path, spec)
    count, sha = verify(suite, rows, reference)
    write_text_atomic(out / 'catalog' / f'{suite}.jsonl', jsonl_text(rows))
    return {'source': {k: spec[k] for k in ('host', 'repo', 'revision', 'file', 'sha256')}, 'questions': count,
            'catalog_sha256': sha, 'verified_against': 'released catalog (ids, input hashes, fields)'
            + ('' if suite == 'math500' else ' and evaluation catalog sha256')}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suites', default=','.join(ORDER), help='comma-separated subset of ' + ','.join(ORDER))
    ap.add_argument('--out', type=Path, default=None, help='output directory (default eval/data/prepared)')
    ap.add_argument('--source-dir', type=Path, default=None, help='download cache (default <out>/sources)')
    ap.add_argument('--released', type=Path, default=None, help='released records (default reproduce/data)')
    a = ap.parse_args()
    out = (a.out or prepared_dir()).resolve()
    source_dir = (a.source_dir or out / 'sources').resolve()
    released = (a.released or released_dir()).resolve()
    wanted = [s for s in a.suites.split(',') if s]
    unknown = sorted(set(wanted) - set(ORDER))
    if unknown:
        ap.error(f'unknown suites: {unknown}')
    state_path = out / 'PREPARED.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    failed = 0
    for suite in wanted:
        try:
            state[suite] = prepare(suite, out, source_dir, released)
            print(f'PASS {suite}: {state[suite]["questions"]} questions match the released catalog; '
                  f'catalog sha256 {state[suite]["catalog_sha256"][:16]}', flush=True)
        except Exception as exc:        # report every suite, then fail
            failed += 1
            state.pop(suite, None)
            print(f'FAIL {suite}: {exc}', flush=True)
        write_text_atomic(state_path, json.dumps(state, indent=1) + '\n')
    print('PREPARE ' + ('FAILED' if failed else 'OK') + f' ({len(wanted) - failed}/{len(wanted)} suites) -> {out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
