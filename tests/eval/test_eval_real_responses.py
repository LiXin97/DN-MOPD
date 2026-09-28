# SPDX-License-Identifier: Apache-2.0
"""Re-grade real sampled answers of the paper's models and compare with the recorded correctness.

fixtures/real_responses.jsonl holds 68 answers (AIME, MATH-500, IFEval, IFBench and LiveCodeBench; half graded correct
in the paper, half incorrect) of the initial 2B student (cap 8,192) and the 9B DN-MOPD student (cap 16,384), chosen
among the shortest answers of mixed questions. Each recorded grade is first checked against the released per-question
records (reproduce/data), then re-derived with this package's graders. AIME 2025 reference answers and the
LiveCodeBench tests are not released; those re-grades need python -m eval.data.prepare and skip without it.

Set DN_MOPD_EVAL_REFERENCE_CELLS to a directory of complete generation cells laid out as
<root>/<model>/<suite>/cap<cap>/raw/<chunk>.json to also re-grade whole cells (slow; LiveCodeBench needs the
prepared official tests).
"""
import json
import os
from pathlib import Path

import pytest
from evaltest_helpers import (full_catalog, needs_if, needs_lcb, needs_math_verify, prepared_ready, real_responses,
                              released_catalog)

from eval.common import records, released_dir


def released_correct(row):
    root = released_dir()
    path = (root / 'math500' / f'cap{row["cap"]}' / f'{row["model"]}.jsonl' if row['suite'] == 'math500'
            else root / 'scores' / row['suite'] / f'cap{row["cap"]}' / f'{row["model"]}.jsonl')
    rec = {r['id']: r['correct'] for r in records(path)}
    return rec[row['id']][row['sample']] == '1'


@pytest.mark.parametrize('suite', ['aime25', 'aime26', 'math500', 'ifeval', 'ifbench', 'lcb_v5', 'lcb_v6'])
def test_fixture_matches_released_records(suite):
    rows = real_responses(suite)
    assert rows and {r['correct'] for r in rows} == {True, False}
    for r in rows:
        assert r['correct'] == released_correct(r), (r['model'], r['id'], r['sample'])


@needs_math_verify
@pytest.mark.parametrize('suite', ['aime25', 'aime26', 'math500'])
def test_math_regrade(suite):
    from eval.graders.math_grader import grade_aime, grade_math500
    grade = grade_math500 if suite == 'math500' else grade_aime
    catalog = full_catalog(suite)          # AIME 2025 answers are not released: they come from the prepared catalog
    for r in real_responses(suite):
        assert grade([r['response']], catalog[r['id']]['answer']) == [r['correct']], (r['model'], r['id'], r['sample'])


@needs_if
@pytest.mark.parametrize('suite', ['ifeval', 'ifbench'])
def test_if_regrade(suite):
    from eval.graders.if_grader import grade_item, load_graders
    catalog = released_catalog(suite)
    for r in real_responses(suite):
        got = grade_item(catalog[r['id']], [r['response']], load_graders())['correct']
        assert got == [r['correct']], (r['model'], r['id'], r['sample'])


@needs_lcb
@pytest.mark.skipif(not prepared_ready('lcb_v6'), reason='run python -m eval.data.prepare --suites lcb_v5,lcb_v6')
@pytest.mark.parametrize('suite', ['lcb_v5', 'lcb_v6'])
def test_lcb_regrade(suite):
    from eval.common import digest, prepared_dir
    from eval.graders.code_grader import score_code
    catalog = released_catalog(suite)
    for r in real_responses(suite):
        q = catalog[r['id']]
        tests = prepared_dir() / 'lcb_tests' / f'{q["test_index"]:04d}.json'
        assert digest(tests) == q['test_sha256']
        got = score_code(json.loads(tests.read_text()), [r['response']])['correct']
        assert got == [r['correct']], (r['model'], r['id'], r['sample'])


REFERENCE = os.environ.get('DN_MOPD_EVAL_REFERENCE_CELLS')


@pytest.mark.skipif(not REFERENCE, reason='set DN_MOPD_EVAL_REFERENCE_CELLS to re-grade whole reference cells')
def test_regrade_reference_cells(tmp_path):
    """Grade each reference cell with python -m eval.grade and compare every per-question correctness string with the
    released records. Cells whose model is not in the released records are skipped."""
    import subprocess
    import sys
    from eval.common import digest, prepared_dir
    from eval.protocol import ENGINE, SAMPLING, SUITES
    checked = 0
    for raw in sorted(Path(REFERENCE).glob('*/*/cap*/raw')):
        model, suite, cap = raw.parts[-4], raw.parts[-3], int(raw.parts[-2][3:])
        released = released_dir() / 'scores' / suite / f'cap{cap}' / f'{model}.jsonl'
        if suite not in SUITES or not released.is_file() or not prepared_ready(suite):
            continue
        gen = tmp_path / model / suite / f'cap{cap}'
        (gen / 'raw').mkdir(parents=True)
        for f in raw.glob('*.json'):
            (gen / 'raw' / f.name).symlink_to(f)
        s = SUITES[suite]
        manifest = {'protocol': 'dn-mopd-eval-v1', 'model': model, 'name': model, 'suite': suite, 'cap': cap,
                    'n': s.n, 'chunk': s.chunk, 'questions': s.problems, 'subset': False,
                    'catalog_sha256': digest(prepared_dir() / 'catalog' / f'{suite}.jsonl'),
                    'sampling': {**SAMPLING, 'max_tokens': cap}, 'enable_thinking': False, 'engine': ENGINE,
                    'chat_template_sha256': None, 'packages': {}}
        (gen / 'MANIFEST.json').write_text(json.dumps(manifest))
        subprocess.run([sys.executable, '-m', 'eval.grade', '--gen', str(gen)], check=True,
                       cwd=Path(__file__).resolve().parents[2])
        got = {r['id']: r['correct'] for r in records(gen / 'scores.jsonl')}
        want = {r['id']: r['correct'] for r in records(released)}
        assert got == want, f'{model} {suite} cap{cap}'
        checked += 1
    assert checked, 'no reference cell matched a released record'
