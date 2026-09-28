# SPDX-License-Identifier: Apache-2.0
"""Catalog preparation: the pinned sources, the evaluation catalog hashes and the rebuild checks.

Offline tests use the released catalogs (reproduce/data/catalog); for AIME 2026, IFEval and IFBench these are the
complete evaluation catalogs, for AIME 2025 and LiveCodeBench they carry ids and hashes only. Tests on a prepared
directory run when eval/data/prepare.py has been run ($DN_MOPD_EVAL_DATA); DN_MOPD_NETWORK_TESTS=1 additionally
downloads three small sources (AIME 2025, AIME 2026, IFBench) and rebuilds them.
"""
import json
import os

import pytest
from evaltest_helpers import prepared_ready

from eval.common import digest, jsonl_text, prepared_dir, records, released_dir, sha256_text
from eval.data import prepare
from eval.protocol import CATALOG_SHA256, SIX, SUITES


def test_every_suite_has_a_pinned_source():
    src = prepare.sources()
    assert set(src) == set(SUITES)
    for spec in src.values():
        assert len(spec['revision']) == 40 and len(spec['sha256']) == 64
        assert prepare.source_url(spec).startswith('https://')


def test_catalog_hashes_are_the_released_cells_hashes():
    cells = records(released_dir() / 'cells.jsonl')
    assert len(cells) == 960
    for suite in SIX:
        assert {c['catalog_sha256'] for c in cells if c['suite'] == suite} == {CATALOG_SHA256[suite]}


@pytest.mark.parametrize('suite', ['aime26', 'ifeval', 'ifbench'])
def test_released_complete_catalogs_pass_verification(suite):
    rows = records(released_dir() / 'catalog' / f'{suite}.jsonl')
    assert digest(released_dir() / 'catalog' / f'{suite}.jsonl') == CATALOG_SHA256[suite]
    count, sha = prepare.verify(suite, rows, rows)
    assert count == SUITES[suite].problems and sha == CATALOG_SHA256[suite]
    changed = [dict(r) for r in rows]
    changed[3]['messages'] = [{'role': 'user', 'content': changed[3]['messages'][0]['content'] + ' '}]
    with pytest.raises(ValueError):
        prepare.verify(suite, changed, rows)


def test_released_aime25_catalog_carries_no_text_or_answers():
    rows = records(released_dir() / 'catalog' / 'aime25.jsonl')
    assert [r['id'] for r in rows] == [str(i) for i in range(30)]
    assert all(set(r) == {'id', 'input_sha256'} and len(r['input_sha256']) == 64 for r in rows)
    assert set(prepare.TEXT_NOT_RELEASED) == {'aime25', 'lcb_v5', 'lcb_v6'}


def test_released_lcb_catalogs_omit_statements():
    for suite, count, offset in (('lcb_v5', 167, 713), ('lcb_v6', 175, 880)):
        rows = records(released_dir() / 'catalog' / f'{suite}.jsonl')
        assert len(rows) == count
        assert all('messages' not in r and 'visible' not in r for r in rows)
        assert sorted(r['test_index'] - r['source_row'] for r in rows) == [offset] * count


def test_no_released_text_for_the_restricted_suites():
    """Only the released catalogs of AIME 2026, IFEval and IFBench (permissive licenses) and the MATH-500 question list
    (MIT) carry benchmark text."""
    for suite in prepare.TEXT_NOT_RELEASED:
        for r in records(released_dir() / 'catalog' / f'{suite}.jsonl'):
            assert not {'messages', 'visible', 'answer', 'problem', 'prompt'} & set(r), (suite, r['id'])


def test_builders_on_synthetic_sources(tmp_path):
    aime = tmp_path / 'aime.jsonl'
    row = {'id': 7, 'problem': 'Find $x$.', 'answer': 12}
    aime.write_text(json.dumps(row) + '\n')
    built = prepare.build_aime26(aime, {})
    assert built == [{'id': '7', 'input_sha256': sha256_text(json.dumps(row, sort_keys=True)),
                      'messages': [{'role': 'user', 'content': 'Find $x$.' + prepare.AIME_SUFFIX}],
                      'visible': 'Find $x$.', 'answer': 12}]
    ifsrc = tmp_path / 'if.jsonl'
    ifsrc.write_text(json.dumps({'key': 5, 'prompt': 'Say hi.', 'instruction_id_list': ['a:b'],
                                 'kwargs': [{'n': 2, 'unused': None}]}) + '\n')
    (r,) = prepare._if_rows(ifsrc, 'ifbench', 'provenance text')
    assert r['kwargs'] == [{'n': 2}] and r['id'] == 'ifbench:5'
    assert r['messages'] == [{'role': 'user', 'content': 'Say hi.'}]


def test_math500_rebuild_matches_released_questions(tmp_path):
    released = records(released_dir() / 'math500' / 'questions.jsonl')
    hf = tmp_path / 'test.jsonl'
    hf.write_text(''.join(json.dumps({'problem': q['problem'], 'answer': q['answer'], 'unique_id': q['id']}) + '\n'
                          for q in released))
    rows = prepare.build_math500(hf, {})
    assert prepare.verify('math500', rows, released)[0] == 500


@pytest.mark.skipif(not all(prepared_ready(s) for s in SUITES), reason='run python -m eval.data.prepare')
def test_prepared_directory():
    state = json.loads((prepared_dir() / 'PREPARED.json').read_text())
    assert set(state) == set(SUITES)
    for suite in SIX:
        assert digest(prepared_dir() / 'catalog' / f'{suite}.jsonl') == CATALOG_SHA256[suite]
    for suite in ('lcb_v5', 'lcb_v6'):
        rows = records(prepared_dir() / 'catalog' / f'{suite}.jsonl')
        for r in rows[:5] + rows[-5:]:
            assert digest(prepared_dir() / 'lcb_tests' / f'{r["test_index"]:04d}.json') == r['test_sha256']
        assert len(rows) == SUITES[suite].problems


@pytest.mark.skipif(os.environ.get('DN_MOPD_NETWORK_TESTS') != '1', reason='set DN_MOPD_NETWORK_TESTS=1 (downloads)')
def test_prepare_downloads_and_verifies(tmp_path):
    for suite in ('aime25', 'aime26', 'ifbench'):
        info = prepare.prepare(suite, tmp_path, tmp_path / 'sources', released_dir())
        assert info['catalog_sha256'] == CATALOG_SHA256[suite]
        assert sha256_text(jsonl_text(records(tmp_path / 'catalog' / f'{suite}.jsonl'))) == CATALOG_SHA256[suite]
