# SPDX-License-Identifier: Apache-2.0
"""Hand-made cases for the official IFEval and IFBench checkers under the paper's deterministic grading revision."""
import random

import pytest
from evaltest_helpers import needs_if, released_catalog

pytestmark = needs_if


def item(suite, ids, kwargs, prompt='Answer the request.'):
    return {'suite': suite, 'key': 0, 'prompt': prompt, 'instruction_id_list': ids, 'kwargs': kwargs}


def strict(it, responses):
    from eval.graders.if_grader import grade_item, load_graders
    return grade_item(it, responses, load_graders())['correct']


def test_ifeval_single_instructions():
    assert strict(item('ifeval', ['punctuation:no_comma'], [{}]), ['no commas here', 'one, two']) == [True, False]
    assert strict(item('ifeval', ['change_case:english_lowercase'], [{}]),
                  ['all lower case text', 'Not All Lower']) == [True, False]
    assert strict(item('ifeval', ['detectable_format:number_highlighted_sections'], [{'num_highlights': 2}]),
                  ['*first* and *second*', 'only *one*']) == [True, False]
    assert strict(item('ifeval', ['keywords:existence'], [{'keywords': ['apple', 'pear']}]),
                  ['an apple and a pear', 'an apple only']) == [True, False]
    assert strict(item('ifeval', ['length_constraints:number_words'], [{'relation': 'less than', 'num_words': 5}]),
                  ['three short words', 'this answer has far too many words in it']) == [True, False]


def test_ifeval_prompt_level_strict_needs_every_instruction():
    it = item('ifeval', ['punctuation:no_comma', 'change_case:english_lowercase'], [{}, {}])
    from eval.graders.if_grader import grade_item, load_graders
    got = grade_item(it, ['fine text', 'Fine text', 'fine, text'], load_graders())
    assert got['correct'] == [True, False, False]
    assert got['follow_instruction_list_strict'] == [[True, True], [True, False], [False, True]]


def test_ifbench_single_instructions():
    assert strict(item('ifbench', ['format:no_whitespace'], [{}]), ['nowhitespace', 'has whitespace']) == [True, False]
    assert strict(item('ifbench', ['count:word_count_range'], [{'min_words': 3.0, 'max_words': 5.0}]),
                  ['one two three four', 'one']) == [True, False]
    kw = {'keyword1': 'kaleidoscope', 'keyword2': 'nebula', 'keyword3': 'whisper', 'keyword4': 'labyrinth',
          'keyword5': 'paradox'}
    good = ' '.join(['kaleidoscope'] + ['nebula'] * 2 + ['whisper'] * 3 + ['labyrinth'] * 5 + ['paradox'] * 7)
    assert strict(item('ifbench', ['count:keywords_multiple'], [kw]), [good, 'kaleidoscope nebula']) == [True, False]


def test_grading_is_deterministic_and_keeps_the_callers_random_state():
    it = item('ifeval', ['punctuation:no_comma', 'keywords:existence'], [{}, {'keywords': ['apple']}])
    random.seed(123)
    before = random.getstate()
    first = strict(it, ['an apple', 'a pear, an apple', 'nothing'])
    assert random.getstate() == before
    assert strict(it, ['an apple', 'a pear, an apple', 'nothing']) == first == [True, False, False]


def test_unimplemented_instruction_aborts():
    from eval.graders.if_grader import assert_coverage, load_graders
    with pytest.raises(SystemExit, match='COVERAGE ABORT'):
        assert_coverage([item('ifeval', ['not:an_instruction'], [{}])], load_graders())


def test_released_catalogs_are_fully_covered_and_have_two_letter_fallback_rows():
    from eval.graders.if_grader import assert_coverage, fallback_letter_rows, load_graders
    for suite in ('ifeval', 'ifbench'):
        assert_coverage(list(released_catalog(suite).values()), load_graders())
    rows = fallback_letter_rows(list(released_catalog('ifeval').values()))
    assert sorted(r['letter'] for r in rows) == ['!', '#']
