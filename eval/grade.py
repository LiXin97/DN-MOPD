# SPDX-License-Identifier: Apache-2.0
"""Grade a generation directory (python -m eval.generate) with the paper's graders, on CPU.

    python -m eval.grade --gen outputs/eval/<name>/<suite>/cap<cap>

  AIME 2025/2026   math-verify on the last \\boxed{} answer (eval/graders/math_grader.py)
  MATH-500         the same comparison as the paper's MATH-500 script (exceptions count as incorrect)
  LiveCodeBench    official lcb_runner extraction and tests, 6 s per test, 8 GiB bounded runtime
  IFEval, IFBench  official checkers, strict prompt accuracy, deterministic grading revision

Writes, beside the generations:
  graded.jsonl   every question with its responses, per-sample correctness and grader details
  scores.jsonl   the released per-question format (id, input_sha256, n, correct as a 0/1 string, mean_tokens,
                 cap_hits), read by eval.aggregate and eval.compare
  summary.json   mean_pass1 (avg@N), mean output tokens, cap-hit rate, grader pins
Grades are cached per chunk under grades/, so an interrupted grading run resumes.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

from eval.common import EVAL_DIR, digest, prepared_dir, records, sha256_text, write_json_atomic, write_text_atomic
from eval.generate import chunk_starts, validate_rows
from eval.protocol import IF_GRADING_REVISION, LCB_GRADE_WORKERS, SUITES
from eval.external import pins

_TESTS_DIR: Path = Path()


def grader_revision(suite: str) -> dict:
    """What a grade depends on: this package's grader sources and the pinned third-party commits."""
    kind = SUITES[suite].grader
    files = {'math_grader.py'} if kind in ('aime', 'math500') else (
        {'code_grader.py', 'lcb_runtime.py'} if kind == 'lcb' else {'if_grader.py'})
    value = {'grader': kind, 'sources': {f: digest(EVAL_DIR / 'graders' / f) for f in sorted(files)}}
    if kind == 'lcb':
        value['LiveCodeBench'] = pins()['LiveCodeBench']['commit']
    if kind == 'if':
        value.update({k: pins()[k]['commit'] for k in ('google-research', 'IFBench', 'nltk_data')},
                     if_grading_revision=IF_GRADING_REVISION)
    value['sha256'] = sha256_text(json.dumps(value, sort_keys=True))
    return value


def _grade_row(args) -> dict:
    """Grade one question (runs in a worker process: math-verify and the LCB runner rely on signals)."""
    kind, question, row = args
    if kind == 'aime':
        from eval.graders.math_grader import grade_aime
        return {**row, 'correct': grade_aime(row['responses'], question['answer'])}
    if kind == 'math500':
        from eval.graders.math_grader import grade_math500
        return {**row, 'correct': grade_math500(row['responses'], question['answer'])}
    from eval.graders.code_grader import score_code
    tests = _TESTS_DIR / f'{question["test_index"]:04d}.json'
    if not tests.is_file() or digest(tests) != question['test_sha256']:
        raise ValueError(f'official tests missing or changed: {tests} (run python -m eval.data.prepare)')
    return {**row, **score_code(json.loads(tests.read_text()), row['responses'])}


def grade_chunk(kind: str, questions: List[dict], rows: List[dict], executor) -> List[dict]:
    if kind == 'if':
        from eval.graders.if_grader import assert_coverage, grade_item, load_graders
        graders = load_graders()
        assert_coverage(questions, graders)
        return [{**r, **grade_item(q, r['responses'], graders)} for q, r in zip(questions, rows)]
    return list(executor.map(_grade_row, [(kind, q, r) for q, r in zip(questions, rows)]))


def main(argv=None) -> int:
    global _TESTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', type=Path, required=True, help='generation directory (contains MANIFEST.json and raw/)')
    ap.add_argument('--workers', type=int, default=LCB_GRADE_WORKERS,
                    help='questions graded in parallel (the paper used 4; LiveCodeBench verdicts depend on timeouts)')
    a = ap.parse_args(argv)
    gen = a.gen.resolve()
    m = json.loads((gen / 'MANIFEST.json').read_text())
    suite = SUITES[m['suite']]
    data = prepared_dir()
    _TESTS_DIR = data / 'lcb_tests'
    path = data / 'catalog' / f'{suite.name}.jsonl'
    if digest(path) != m['catalog_sha256']:
        raise SystemExit(f'{path} differs from the catalog the answers were generated from')
    questions = records(path)[:m['questions']]
    revision = grader_revision(suite.name)
    final: List[dict] = []
    context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=context) as executor:
        for start in chunk_starts(len(questions), suite.chunk):
            source = questions[start:start + suite.chunk]
            rawpath = gen / 'raw' / f'{start:04d}.json'
            if not rawpath.is_file():
                raise SystemExit(f'missing generation chunk {rawpath}; finish python -m eval.generate first')
            raw = json.loads(rawpath.read_text())
            validate_rows(raw, source, m['name'], m['cap'], m['n'])
            cache = gen / 'grades' / rawpath.name
            if cache.is_file():
                saved = json.loads(cache.read_text())
                if saved['raw_sha256'] == digest(rawpath) and saved['grader_revision'] == revision['sha256']:
                    final.extend(saved['rows'])
                    continue
            rows = grade_chunk(suite.grader, source, raw, executor)
            if any(len(r['correct']) != m['n'] or any(type(x) is not bool for x in r['correct']) for r in rows):
                raise RuntimeError('missing or invalid grades')
            write_json_atomic(cache, {'raw_sha256': digest(rawpath), 'grader_revision': revision['sha256'],
                                      'rows': rows})
            final.extend(rows)
            print(f'GRADED {suite.name} {rawpath.name}', flush=True)
    write_text_atomic(gen / 'graded.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in final))
    compact = [{'id': r['id'], 'input_sha256': r['input_sha256'], 'n': r['n'],
                'correct': ''.join('1' if x else '0' for x in r['correct']),
                'mean_tokens': round(sum(r['token_counts']) / len(r['token_counts']), 2),
                'cap_hits': sum(1 for x in r['finish_reasons'] if x == 'length')} for r in final]
    write_text_atomic(gen / 'scores.jsonl', ''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in compact))
    total = sum(r['n'] for r in final)
    scores = [sum(r['correct']) / r['n'] for r in final]
    summary = {'model': m['name'], 'suite': suite.name, 'cap': m['cap'], 'problems': len(final),
               'samples_per_problem': m['n'], 'subset': m['subset'], 'mean_pass1': sum(scores) / len(scores),
               'mean_output_tokens': sum(sum(r['token_counts']) for r in final) / total,
               'cap_hit_rate': sum(f == 'length' for r in final for f in r['finish_reasons']) / total,
               'graded_sha256': digest(gen / 'graded.jsonl'), 'grader_revision': revision,
               'manifest_sha256': digest(gen / 'MANIFEST.json')}
    if suite.grader == 'if':
        from eval.graders.if_grader import fallback_letter_rows
        fallback = {f['row'] for f in fallback_letter_rows(questions)}
        keep = [s for i, s in enumerate(scores) if i not in fallback]
        summary.update(metric='strict prompt accuracy', official_letter_fallback_rows=len(scores) - len(keep),
                       mean_pass1_excluding_letter_fallback=sum(keep) / len(keep))
    write_json_atomic(gen / 'summary.json', summary)
    print(f'GRADE_DONE {m["name"]} {suite.name} cap{m["cap"]}: {100 * summary["mean_pass1"]:.2f} '
          f'(avg@{m["n"]}, {len(final)} questions{", SUBSET" if m["subset"] else ""}); mean tokens '
          f'{summary["mean_output_tokens"]:.1f}; cap-hit rate {summary["cap_hit_rate"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
