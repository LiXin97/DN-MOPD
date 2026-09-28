# SPDX-License-Identifier: Apache-2.0
"""Instruction-following graders: the official IFEval (google-research @ b24f2136) and IFBench (allenai @ 1091c4c3)
checkers, called through their own evaluation_lib; nothing in the checking path is reimplemented.

The paper's metric is STRICT prompt-level accuracy: a sampled answer is correct when it follows every instruction of
its prompt under test_instruction_following_strict. Loose results are recorded beside it but never used.

Grading revision "deterministic-official-v1-20260918": some official build_description paths draw missing arguments
from Python's `random`, and langdetect seeds a fresh detector per call. Both are fixed (random.seed(0) around every
check, restoring the caller's state; langdetect DetectorFactory.seed = 0) so that a grade never depends on sample order
or process history. Any instruction id that the official registry does not implement, and any exception raised by an
official checker, aborts grading: it can never be scored 0 or 1.

Requirements: nltk 3.10.2, langdetect 1.0.9, immutabledict 4.3.1, absl-py 2.5.0, emoji 2.15.0, syllapy 0.7.2 and the
compiled `regex` package, plus the pinned nltk data (python -m eval.external.fetch).
"""
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List

from eval.protocol import IF_GRADING_RANDOM_SEED
from eval.external import require

_GRADERS: Dict[str, dict] = {}


def load_graders() -> Dict[str, dict]:
    """Import both official evaluation_libs against the pinned sources and nltk data. Abort loudly on any gap."""
    if _GRADERS:
        return _GRADERS
    google = require('google-research')
    ifbench = require('IFBench')
    nltk_data = require('nltk_data')
    # nltk resources both checkers load; a missing one would make IFBench attempt a network download at import.
    missing = [r for r in ('tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords',
                           'taggers/averaged_perceptron_tagger_eng') if not (nltk_data / r).is_dir()]
    if missing:
        raise SystemExit(f'nltk data missing under {nltk_data}: {missing}; run python -m eval.external.fetch')
    try:
        import regex  # noqa: F401  (nltk's compiled dependency; a missing one surfaces here with a clear message)
    except ImportError as exc:
        raise SystemExit(f'the `regex` package is required by nltk: pip install regex ({exc})') from exc
    os.environ['NLTK_DATA'] = str(nltk_data)          # before any nltk import (IFBench uses setdefault)
    for path in (str(ifbench), str(google)):
        if path not in sys.path:
            sys.path.insert(0, path)                   # IFBench: flat modules; google: namespace package
    import nltk
    nltk.data.path.insert(0, str(nltk_data))
    # Language detection otherwise initializes a new entropy-seeded detector on each call.
    from langdetect import DetectorFactory
    DetectorFactory.seed = IF_GRADING_RANDOM_SEED

    from instruction_following_eval import evaluation_lib as g_lib            # IFEval
    from instruction_following_eval import instructions_registry as g_reg
    import evaluation_lib as b_lib                                            # IFBench (flat modules)
    import instructions_registry as b_reg
    if os.path.dirname(os.path.abspath(b_lib.__file__)) != os.path.abspath(ifbench):
        raise SystemExit(f'module shadowing: `evaluation_lib` resolved to {b_lib.__file__}, expected {ifbench}')
    _GRADERS.update({
        'ifeval': {'lib': g_lib, 'registry': g_reg.INSTRUCTION_DICT,
                   'provenance': 'google-research/instruction_following_eval @ b24f2136'},
        'ifbench': {'lib': b_lib, 'registry': b_reg.INSTRUCTION_DICT, 'provenance': 'allenai/IFBench @ 1091c4c3'},
    })
    return _GRADERS


def assert_coverage(items: List[dict], graders: Dict[str, dict]) -> None:
    """Every instruction id must be implemented by its suite's official registry, else abort."""
    problems = []
    for i, d in enumerate(items):
        if d.get('suite') not in graders:
            problems.append(f'row {i}: unknown suite {d.get("suite")!r}')
            continue
        registry = graders[d['suite']]['registry']
        problems += [f'row {i} (key={d.get("key")}): instruction id {iid!r} is not implemented'
                     for iid in d['instruction_id_list'] if iid not in registry]
    if problems:
        raise SystemExit('COVERAGE ABORT: ' + '; '.join(problems[:20]))


def _official_check(check, inp, mapping):
    # The same random draw for every answer, independent of model, sample order and strict/loose call order.
    state = random.getstate()
    try:
        random.seed(IF_GRADING_RANDOM_SEED)
        return check(inp, mapping)
    finally:
        random.setstate(state)


def fallback_letter_rows(items: List[dict]) -> List[dict]:
    """IFEval rows whose letter argument is not a single letter, so the official checker substitutes a random one
    (canonical IFEval has two, '#' and '!'). They stay in the suite; the summary reports a score without them."""
    found = []
    for i, row in enumerate(items):
        if row.get('suite') != 'ifeval':
            continue
        for iid, kw in zip(row['instruction_id_list'], row['kwargs']):
            letter = kw.get('letter')
            if iid == 'keywords:letter_frequency' and (not letter or len(letter) != 1
                                                        or not ('a' <= letter.lower() <= 'z')):
                found.append({'row': i, 'key': row.get('key'), 'letter': letter})
    return found


def grade_item(item: dict, responses: List[str], graders: Dict[str, dict]) -> dict:
    """Strict and loose official results of every sampled response of one prompt."""
    lib = graders[item['suite']]['lib']
    strict, loose, per_strict, per_loose = [], [], [], []
    for resp in responses:
        # A fresh InputExample per call: IFBench's strict check mutates inp.kwargs in place.
        inp = lib.InputExample(key=item.get('key', 0), instruction_id_list=list(item['instruction_id_list']),
                               prompt=item['prompt'], kwargs=[dict(kw) for kw in item['kwargs']])
        mapping = {item['prompt']: resp}
        out_s = _official_check(lib.test_instruction_following_strict, inp, mapping)
        out_l = _official_check(lib.test_instruction_following_loose, inp, mapping)
        strict.append(bool(out_s.follow_all_instructions))
        loose.append(bool(out_l.follow_all_instructions))
        per_strict.append([bool(x) for x in out_s.follow_instruction_list])
        per_loose.append([bool(x) for x in out_l.follow_instruction_list])
    return {'correct': strict, 'loose_correct': loose, 'follow_instruction_list_strict': per_strict,
            'follow_instruction_list_loose': per_loose}
