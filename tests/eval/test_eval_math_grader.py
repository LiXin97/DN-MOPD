# SPDX-License-Identifier: Apache-2.0
"""Hand-made cases for the AIME and MATH-500 graders (math-verify on the last boxed answer)."""
from evaltest_helpers import needs_math_verify

from eval.graders.math_grader import grade_aime, grade_math500, last_boxed_only_string, remove_boxed


def test_last_boxed_extraction():
    assert last_boxed_only_string('a \\boxed{1} b \\boxed{2}') == '\\boxed{2}'
    assert last_boxed_only_string('\\boxed{\\frac{1}{2}} end') == '\\boxed{\\frac{1}{2}}'
    assert last_boxed_only_string('no box here') is None
    assert last_boxed_only_string('unclosed \\boxed{12') is None
    assert last_boxed_only_string('\\fbox{7}') == '\\fbox{7}'
    assert remove_boxed('\\boxed{7}') == '7'
    assert remove_boxed('\\fbox{7}') is None          # only \boxed{...} yields an answer
    assert remove_boxed(None) is None


@needs_math_verify
def test_aime_grader_cases():
    # the form of the checker preflight of the paper's instrument
    assert grade_aime(['The answer is \\boxed{42}.', '\\boxed{43}', 'no box'], '42') == [True, False, False]
    got = grade_aime(['\\boxed{1} then \\boxed{345}', '\\boxed{345} then \\boxed{1}', '\\boxed{345.0}',
                      '\\boxed{0345}', '$\\boxed{345}$', '\\boxed{345'], 345)
    assert got == [True, False, True, True, True, False]


@needs_math_verify
def test_aime_grader_rejects_unparseable_gold():
    import pytest
    with pytest.raises(ValueError):
        grade_aime(['\\boxed{1}'], '')


@needs_math_verify
def test_math500_grader_cases():
    gold = '\\left( 3, \\frac{\\pi}{2} \\right)'
    got = grade_math500(['So $(r,\\theta) = \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}$.',
                         '\\boxed{(3, \\pi/2)}', '\\boxed{(3, \\pi)}', '', 'no box'], gold)
    assert got == [True, True, False, False, False]
    assert grade_math500(['\\boxed{p - q}', '\\boxed{q - p}'], 'p - q') == [True, False]
    assert grade_math500(['\\boxed{\\frac{14}{3}}', '\\boxed{4.6667}'], '\\frac{14}{3}') == [True, False]
