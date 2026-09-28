# SPDX-License-Identifier: Apache-2.0
"""Hand-made cases for the LiveCodeBench grader: official extraction, stdin and call-based tests, timeouts, the
8 GiB bounded runtime, and infrastructure failures that must raise instead of scoring."""
import json
from unittest.mock import patch

import pytest
from evaltest_helpers import needs_lcb

pytestmark = needs_lcb

STDIN = {'inputs': ['2\n', '5\n'], 'outputs': ['4\n', '10\n'], 'fn_name': None}
CALL = {'inputs': ['[1,2,3]', '[4,5]'], 'outputs': ['6', '9'], 'fn_name': 'total'}


def fenced(code: str) -> str:
    return '```python\n' + code + '\n```'


def sample(tests: dict) -> dict:
    return {'input_output': json.dumps(tests)}


def test_stdin_programs():
    from eval.graders.code_grader import score_code
    got = score_code(sample(STDIN), [fenced(c) for c in ('print(int(input())*2)', 'print(7)', 'while True: pass',
                                                         'def broken(')], timeout=1)
    assert got['correct'] == [True, False, False, False]


def test_call_based_programs():
    from eval.graders.code_grader import score_code
    got = score_code(sample(CALL), [fenced('class Solution:\n def total(self, x): return sum(x)'),
                                    fenced('class Solution:\n def total(self, x): return 0')], timeout=1)
    assert got['correct'] == [True, False]


def test_official_extraction():
    from eval.graders.code_grader import score_code
    assert score_code(sample(STDIN), ['print(int(input())*2)'])['correct'] == [False]   # unfenced output fails
    two_blocks = '```python\nprint(7)\n```\nRevised:\n```python\nprint(int(input())*2)\n```'
    assert score_code(sample(STDIN), [two_blocks])['correct'] == [True]                 # the LAST block is graded


def test_memory_bomb_fails_without_killing_the_checker():
    from eval.graders.code_grader import score_code
    from eval.graders.lcb_runtime import MEMORY_LIMIT_BYTES
    tests = {'inputs': [''], 'outputs': ['1'], 'fn_name': None}
    got = score_code(sample(tests), [fenced('x = [True] * 10**12\nprint(1)')], timeout=1)
    assert got['correct'] == [False]
    assert got['checker_metadata'][0].get('error_code') != -5
    assert 'MemoryError' in str(got['checker_metadata'])
    assert MEMORY_LIMIT_BYTES == 8 * 1024 ** 3


def test_swallowed_timeout_hits_the_global_limit():
    from eval.graders.code_grader import score_code
    tests = {'inputs': ['1\n'], 'outputs': ['2\n'], 'fn_name': None}
    codes = ['print(int(input())+1)', 'print(0)', 'while True: pass',
             'while True:\n try:\n  while True: pass\n except BaseException:\n  pass']
    got = score_code(sample(tests), [fenced(c) for c in codes], timeout=1)
    assert got['correct'] == [True, False, False, False]
    assert got['checker_metadata'][-1]['error_message'] == 'Global Time Limit Exceeded'
    assert got['checker_metadata'][-1]['global_timeout_seconds'] == 7
    assert got['test_results'][-1] == [-1]


def test_unexpected_checker_exit_is_not_a_model_failure():
    from eval.graders.code_grader import lcb_imports
    from eval.graders.lcb_runtime import check_correctness
    lcb_imports()

    def crash(*args):
        raise RuntimeError('synthetic checker infrastructure failure')

    with patch('lcb_runner.evaluation.compute_code_generation_metrics._temp_run', crash):
        with pytest.raises(RuntimeError, match='incomplete result'):
            check_correctness(sample({'inputs': ['1'], 'outputs': ['2']}), 'print(2)', timeout=1, debug=False)
