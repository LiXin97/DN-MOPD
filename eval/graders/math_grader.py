# SPDX-License-Identifier: Apache-2.0
"""Mathematics graders: math-verify (0.9.0) on the last \\boxed{...} of the answer.

Two variants are used in the paper and both are kept exactly:
  grade_aime     AIME 2025 and AIME 2026. The gold answer must parse, and a grader exception propagates (an
                 exception is an instrument failure, never a wrong answer).
  grade_math500  MATH-500, graded by the G-OPD evaluation script: responses are stripped at generation time, and a
                 math-verify exception counts as incorrect.
In both, an answer without a complete last \\boxed{...} is incorrect.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def last_boxed_only_string(string: str) -> Optional[str]:
    """The last \\boxed{...} (or \\fbox{...}) of a string, braces balanced; None if absent or unclosed.
    Verbatim from the G-OPD evaluation (math_eval/eval_math.py)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def remove_boxed(s: Optional[str]) -> Optional[str]:
    """Content of a \\boxed{...} string; None for anything else (including \\fbox and None).
    Verbatim from the G-OPD evaluation."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


def grade_aime(responses: Sequence[str], answer) -> List[bool]:
    """AIME grader: math_verify.verify(parse(boxed gold), parse(boxed last-boxed answer))."""
    from math_verify import parse, verify
    gold = parse('\\boxed{' + str(answer) + '}')
    if not gold:
        raise ValueError('Invalid gold math answer')
    result = []
    for text in responses:
        boxed = last_boxed_only_string(text)
        pred = remove_boxed(boxed) if boxed else None
        # Extraction failure is a model failure. Grader exceptions propagate.
        result.append(False if pred is None else bool(verify(gold, parse('\\boxed{' + pred + '}'))))
    return result


def grade_math500(responses: Sequence[str], answer) -> List[bool]:
    """MATH-500 grader (G-OPD acc_list_mathverify): the same comparison, exceptions count as incorrect."""
    from math_verify import parse, verify
    gold = str(answer)
    result = []
    for r in responses:
        boxed = remove_boxed(last_boxed_only_string(r)) if r else None
        if boxed is None:
            result.append(False)
            continue
        try:
            result.append(bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + boxed + "}"))))
        except Exception:
            result.append(False)
    return result
