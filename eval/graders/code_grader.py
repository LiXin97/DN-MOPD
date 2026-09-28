# SPDX-License-Identifier: Apache-2.0
"""LiveCodeBench grader: the official lcb_runner code extractor and test runner (LiveCodeBench @ 28fef95).

A response is correct when the code in its LAST fenced block (official OpenAIChat extraction) passes every official
test of the problem (public and private, none pruned), with a 6-second per-test timeout and the 8 GiB bounded runtime
of lcb_runtime.py. A response without a fenced block is incorrect. A checker infrastructure failure (lcb error
code -5, or an abnormal checker exit) raises instead of scoring.

Test verdicts depend on wall-clock timeouts, so a heavily loaded grading host can turn slow passing programs into
failures; the paper graded at most four problems of a cell in parallel.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from eval.protocol import LCB_TIMEOUT_SECONDS
from eval.external import require


def lcb_imports():
    """Import the official lcb_runner. Upstream loads its bundled few-shot resources relative to the repository root
    at import time, so the import runs from there; the working directory is restored before any program runs."""
    root = require('LiveCodeBench')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    previous = Path.cwd()
    try:
        os.chdir(root)
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.prompts.code_generation import get_generic_question_template_answer, PromptConstants
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
    finally:
        os.chdir(previous)
    return (CodeGenerationProblem, get_generic_question_template_answer, PromptConstants, extract_code, LMStyle,
            evaluate_generations_by_problem)


def score_code(sample: dict, responses: Sequence[str], timeout: int = LCB_TIMEOUT_SECONDS) -> dict:
    """Grade the sampled responses of one problem against its official tests (the decoded evaluation sample)."""
    _, _, _, extract, style, evaluate = lcb_imports()
    from eval.graders.lcb_runtime import install
    install()
    codes = [extract(r, style.OpenAIChat) for r in responses]
    results, metadata = evaluate((codes, sample, False, timeout))
    if len(results) != len(responses) or len(metadata) != len(responses):
        raise ValueError('Missing checker result')
    if any(m.get('error_code') == -5 for m in metadata):
        raise RuntimeError(f'Official test runner infrastructure failure: {metadata}')
    # Official convention: every test result must be > 0; negative error codes fail.
    correct = [bool(r) and all(x > 0 for x in r) for r in results]
    return {'correct': correct, 'test_results': results, 'checker_metadata': metadata,
            'extracted_code_sha256': [hashlib.sha256(c.encode()).hexdigest() for c in codes]}


def evaluation_sample_json(line: str) -> str:
    """Decode one line of the official code_generation_lite release into its official evaluation sample, serialized
    exactly as the paper's test cache stored it (json, indent 2, trailing newline); its sha256 is the catalog's
    test_sha256."""
    problem_class = lcb_imports()[0]
    problem = problem_class(**json.loads(line))
    return json.dumps(problem.get_evaluation_sample(), ensure_ascii=False, allow_nan=False, indent=2) + '\n'


def catalog_row(line: str) -> dict:
    """The catalog fields the paper derived from one official problem line (prompt rendering included)."""
    problem_class, prompt, constants, *_ = lcb_imports()
    p = problem_class(**json.loads(line))
    return {'id': p.platform.value + ':' + str(p.question_id),
            'messages': [{'role': 'system', 'content': constants.SYSTEM_MESSAGE_GENERIC},
                         {'role': 'user', 'content': prompt(p)}],
            'visible': p.question_content, 'date': p.contest_date.isoformat(), 'difficulty': p.difficulty.value,
            'tests': len(json.loads(p.get_evaluation_sample()['input_output'])['inputs'])}
