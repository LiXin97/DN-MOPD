# SPDX-License-Identifier: Apache-2.0
"""Instruction-following (IF) verifier: a closed taxonomy of constraint checkers with no third-party imports.

This is the reward of the IF GRPO teacher and the correctness label of IF prompts on the student side. Every checker
is a pure function of the response string, so a reward is reproducible without the model or any network access.

A sample's label is a JSON list of constraint specs:
    [{"type": "bullet_count", "n": 3}, {"type": "must_exclude", "values": ["however"]}]
`score(response, constraints)` returns (all_satisfied, per_constraint_detail). The reward is strict: every
constraint must hold (binary, like the math and code verifiers).

Self-test: `python ifeval_checkers.py` (positive and negative cases per constraint type).
"""
from __future__ import annotations

import json
import re
import string

# ----------------------------------------------------------------- text primitives


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p for p in parts if p.strip()]


def _bullets(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if re.match(r"\s*[-*•]\s+\S", ln)]


def _numbered(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if re.match(r"\s*\d+[.)]\s+\S", ln)]


def _highlights(text: str) -> list[str]:
    return re.findall(r"\*\*(.+?)\*\*", text, flags=re.S)


def _placeholders(text: str) -> list[str]:
    return re.findall(r"\[[^\[\]\n]{1,60}\]", text)


# ----------------------------------------------------------------- checkers
# Each takes (response, spec) -> bool. Keep them total: never raise on odd input.

def _c_word_count_min(r, s):   return len(_words(r)) >= int(s["n"])
def _c_word_count_max(r, s):   return len(_words(r)) <= int(s["n"])
def _c_sentence_count(r, s):   return len(_sentences(r)) == int(s["n"])
def _c_bullet_count(r, s):     return len(_bullets(r)) == int(s["n"])
def _c_numbered_count(r, s):   return len(_numbered(r)) == int(s["n"])
def _c_paragraph_count(r, s):  return len(_paragraphs(r)) == int(s["n"])
def _c_highlight_count(r, s):  return len(_highlights(r)) >= int(s["n"])
def _c_placeholder_count(r, s):return len(_placeholders(r)) >= int(s["n"])


def _c_must_include(r, s):
    low = r.lower()
    return all(str(v).lower() in low for v in s["values"])


def _c_must_exclude(r, s):
    low = r.lower()
    return all(str(v).lower() not in low for v in s["values"])


def _c_forbidden_chars(r, s):
    low = r.lower()
    return all(str(ch).lower() not in low for ch in s["values"])


def _c_all_caps(r, s):
    letters = [ch for ch in r if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _c_all_lowercase(r, s):
    letters = [ch for ch in r if ch.isalpha()]
    return bool(letters) and all(ch.islower() for ch in letters)


def _c_starts_with(r, s):      return r.strip().startswith(str(s["value"]))
def _c_ends_with(r, s):        return r.strip().endswith(str(s["value"]))
def _c_no_commas(r, s):        return "," not in r


def _c_valid_json(r, s):
    t = r.strip()
    if t.startswith("```"):                       # tolerate a fenced block
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def _c_wrapped_in(r, s):
    t = r.strip()
    o, c = str(s["open"]), str(s["close"])
    return t.startswith(o) and t.endswith(c) and len(t) >= len(o) + len(c)


def _c_repeat_first(r, s):
    """Response must open by repeating the given text verbatim (case-insensitive)."""
    return r.strip().lower().startswith(str(s["value"]).strip().lower())


CHECKERS = {
    "word_count_min": _c_word_count_min,
    "word_count_max": _c_word_count_max,
    "sentence_count": _c_sentence_count,
    "bullet_count": _c_bullet_count,
    "numbered_count": _c_numbered_count,
    "paragraph_count": _c_paragraph_count,
    "highlight_count": _c_highlight_count,
    "placeholder_count": _c_placeholder_count,
    "must_include": _c_must_include,
    "must_exclude": _c_must_exclude,
    "forbidden_chars": _c_forbidden_chars,
    "all_caps": _c_all_caps,
    "all_lowercase": _c_all_lowercase,
    "starts_with": _c_starts_with,
    "ends_with": _c_ends_with,
    "no_commas": _c_no_commas,
    "valid_json": _c_valid_json,
    "wrapped_in": _c_wrapped_in,
    "repeat_first": _c_repeat_first,
}


def score(response: str, constraints) -> tuple[bool, dict]:
    """(all_satisfied, detail). Unknown constraint types are a HARD ERROR, never a silent pass —
    a typo in the data must not manufacture reward."""
    if isinstance(constraints, str):
        constraints = json.loads(constraints)
    if not constraints:
        raise ValueError("empty constraint list: an IF sample with no constraint is unscoreable")
    detail, ok_all = {}, True
    for i, spec in enumerate(constraints):
        t = spec["type"]
        if t not in CHECKERS:
            raise KeyError(f"unknown IF constraint type {t!r}; taxonomy is closed by design")
        try:
            ok = bool(CHECKERS[t](response or "", spec))
        except Exception:
            ok = False                     # a malformed response fails, it does not crash training
        detail[f"{i}:{t}"] = ok
        ok_all = ok_all and ok
    return ok_all, {"satisfied": sum(detail.values()), "total": len(detail), "per_constraint": detail}


def instruction_text(spec: dict) -> str:
    """Human-readable rendering of one constraint, for building the prompt."""
    t = spec["type"]
    return {
        "word_count_min": lambda: f"Use at least {spec['n']} words.",
        "word_count_max": lambda: f"Use at most {spec['n']} words.",
        "sentence_count": lambda: f"Answer in exactly {spec['n']} sentences.",
        "bullet_count": lambda: f"Answer as exactly {spec['n']} bullet points, each starting with '- '.",
        "numbered_count": lambda: f"Answer as exactly {spec['n']} numbered items, like '1. '.",
        "paragraph_count": lambda: f"Answer in exactly {spec['n']} paragraphs separated by a blank line.",
        "highlight_count": lambda: f"Highlight at least {spec['n']} sections using **double asterisks**.",
        "placeholder_count": lambda: f"Include at least {spec['n']} placeholders in square brackets, like [name].",
        "must_include": lambda: "Your answer must contain " + ", ".join(f"the word \"{v}\"" for v in spec["values"]) + ".",
        "must_exclude": lambda: "Your answer must not contain " + ", ".join(f"the word \"{v}\"" for v in spec["values"]) + ".",
        "forbidden_chars": lambda: "Do not use the letter " + " or ".join(f"'{c}'" for c in spec["values"]) + " anywhere.",
        "all_caps": lambda: "Write your entire answer in capital letters.",
        "all_lowercase": lambda: "Write your entire answer in lowercase letters only.",
        "starts_with": lambda: f"Begin your answer with \"{spec['value']}\".",
        "ends_with": lambda: f"End your answer with \"{spec['value']}\".",
        "no_commas": lambda: "Do not use any commas.",
        "valid_json": lambda: "Reply with valid JSON and nothing else.",
        "wrapped_in": lambda: f"Wrap your entire answer in {spec['open']} and {spec['close']}.",
        "repeat_first": lambda: f"Begin by repeating this exactly: \"{spec['value']}\"",
    }[t]()


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    cases = [
        ({"type": "word_count_min", "n": 3}, "one two three", "one two"),
        ({"type": "word_count_max", "n": 3}, "one two", "one two three four"),
        ({"type": "sentence_count", "n": 2}, "A cat sat. A dog ran.", "A cat sat."),
        ({"type": "bullet_count", "n": 2}, "- a\n- b", "- a"),
        ({"type": "numbered_count", "n": 2}, "1. a\n2. b", "1. a"),
        ({"type": "paragraph_count", "n": 2}, "para one\n\npara two", "para one"),
        ({"type": "highlight_count", "n": 1}, "see **this**", "see this"),
        ({"type": "placeholder_count", "n": 1}, "dear [name]", "dear friend"),
        ({"type": "must_include", "values": ["banana"]}, "a Banana here", "an apple here"),
        ({"type": "must_exclude", "values": ["however"]}, "so then", "However, no"),
        ({"type": "forbidden_chars", "values": ["e"]}, "a tall hill", "a tree"),
        ({"type": "all_caps"}, "HELLO THERE", "Hello There"),
        ({"type": "all_lowercase"}, "hello there", "Hello there"),
        ({"type": "starts_with", "value": "Answer:"}, "Answer: yes", "yes"),
        ({"type": "ends_with", "value": "END"}, "all done END", "all done"),
        ({"type": "no_commas"}, "a b c", "a, b"),
        ({"type": "valid_json"}, '{"a": 1}', "{a: 1}"),
        ({"type": "wrapped_in", "open": "<<", "close": ">>"}, "<<hi>>", "hi"),
        ({"type": "repeat_first", "value": "Explain X"}, "explain x, then...", "then..."),
    ]
    assert len(cases) == len(CHECKERS), f"self-test covers {len(cases)} of {len(CHECKERS)} checkers"
    for spec, good, bad in cases:
        ok_g, _ = score(good, [spec])
        ok_b, _ = score(bad, [spec])
        assert ok_g, f"{spec['type']}: positive case failed on {good!r}"
        assert not ok_b, f"{spec['type']}: negative case passed on {bad!r}"
        instruction_text(spec)                      # must render for every type
    ok, d = score("- a\n- b", [{"type": "bullet_count", "n": 2}, {"type": "no_commas"}])
    assert ok and d["satisfied"] == 2
    ok, d = score("- a, x\n- b", [{"type": "bullet_count", "n": 2}, {"type": "no_commas"}])
    assert not ok and d["satisfied"] == 1, "strict conjunction must fail if any constraint fails"
    try:
        score("x", [{"type": "no_such_check"}]); raise SystemExit("unknown type must raise")
    except KeyError:
        pass
    print(f"IFEVAL_CHECKERS_SELFTEST: PASS ({len(CHECKERS)} checkers, positive+negative each)")
