# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the evaluation tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def third_party_ready(name: str) -> bool:
    from eval.external import require
    try:
        require(name)
        return True
    except (RuntimeError, KeyError):
        return False


def prepared_ready(suite: str) -> bool:
    from eval.common import prepared_dir
    return (prepared_dir() / 'catalog' / f'{suite}.jsonl').is_file()


def released_catalog(suite: str) -> Dict[str, dict]:
    from eval.common import records, released_dir
    if suite == 'math500':
        return {r['id']: r for r in records(released_dir() / 'math500' / 'questions.jsonl')}
    return {r['id']: r for r in records(released_dir() / 'catalog' / f'{suite}.jsonl')}


def full_catalog(suite: str) -> Dict[str, dict]:
    """Questions with text and answers: the released catalog, or, for suites whose released catalog omits the text
    (AIME 2025, LiveCodeBench), the catalog rebuilt by python -m eval.data.prepare (the test skips without it)."""
    from eval.common import prepared_dir, records
    from eval.data.prepare import TEXT_NOT_RELEASED
    if suite not in TEXT_NOT_RELEASED:
        return released_catalog(suite)
    if not prepared_ready(suite):
        pytest.skip(f'the released {suite} catalog carries no text; run python -m eval.data.prepare --suites {suite}')
    return {r['id']: r for r in records(prepared_dir() / 'catalog' / f'{suite}.jsonl')}


def real_responses(suite: str) -> List[dict]:
    rows = [json.loads(line) for line in (FIXTURES / 'real_responses.jsonl').open()]
    return [r for r in rows if r['suite'] == suite]


needs_math_verify = pytest.mark.skipif(not have_module('math_verify'), reason='math-verify is not installed')
needs_lcb = pytest.mark.skipif(not third_party_ready('LiveCodeBench'),
                               reason='run python -m eval.external.fetch --only LiveCodeBench')
needs_if = pytest.mark.skipif(not (third_party_ready('google-research') and third_party_ready('IFBench')
                                   and third_party_ready('nltk_data') and have_module('nltk')
                                   and have_module('langdetect')),
                              reason='run python -m eval.external.fetch and install the IF checker requirements')
