# SPDX-License-Identifier: Apache-2.0
"""Rule-based scalar reward for GRPO teacher training (no teacher servers involved).

`OPD_reward.get_reward.get_reward` constructs `RMSystemManager(args)`, which expects live teacher servers. Teacher
training is plain outcome RL from the same base model the student starts from, so it needs a reward that touches
nothing but the verifier. miles' contract for `--custom-rm-path` (`miles/rollout/rm_hub/__init__.py`) is a scalar;
this returns 1.0 / 0.0.

Domain dispatch is `sample.metadata["domain"]` (falling back to the teacher label):
    math (and anything unrecognised) -> grade_answer_verl, in a forked child with a time limit
    code / code-a3b                   -> the PRIME code-judge service (rule_base_reward.get_rule_based_reward)
    ifeval                            -> the vendored closed-taxonomy constraint checkers (ifeval_checkers.py)

Use: --custom-rm-path Uni_OPD_utils.OPD_reward.grpo_rule_reward.grpo_rule_reward
Environment: GRPO_MATH_TIMEOUT_S (seconds per math answer, default 15), GRPO_MATH_WORKERS (default 4).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
from argparse import Namespace

from Uni_OPD_utils.OPD_reward.rule_base_reward import get_rule_based_reward
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_ifeval_score = None


def _get_ifeval_score():
    """Load ifeval_checkers.py (beside this file) by path."""
    global _ifeval_score
    if _ifeval_score is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifeval_checkers.py")
        spec = importlib.util.spec_from_file_location("grpo_ifeval_checkers", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load ifeval checkers from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ifeval_score = mod.score
    return _ifeval_score


def _domain(sample: Sample) -> str:
    meta = getattr(sample, "metadata", None) or {}
    dom = meta.get("domain")
    if isinstance(dom, str) and dom:
        return dom.lower()
    return (getattr(sample, "teacher_model_name", None) or "").lower()


# ---------------------------------------------------------------------------------------------
# Math verifier time limit. grade_answer_verl has no timeout (sympy can hang on adversarial LaTeX). SIGALRM cannot be
# used in the reward's worker thread, so the grade runs in a forked child whose MAIN thread arms the alarm. A timeout
# scores 0.0 (unverifiable = not correct) and is counted; the pool is rebuilt if a child dies.
_MATH_TIMEOUT_S = float(os.environ.get("GRPO_MATH_TIMEOUT_S", "15"))
_MATH_WORKERS = int(os.environ.get("GRPO_MATH_WORKERS", "4"))
_math_pool = None
_math_timeouts = 0
_math_mech_failures = 0
_grade_fn = None


class _MathTimeout(Exception):
    pass


def _load_grade_answer_verl():
    """Load grade_answer_verl by file path: importing miles.rollout.rm_hub executes its __init__, which pulls in ray;
    math_utils itself needs only re, sympy and pylatexenc."""
    global _grade_fn
    if _grade_fn is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "miles", "rollout", "rm_hub", "math_utils.py")
        spec = importlib.util.spec_from_file_location("_grpo_math_utils", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load math_utils from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _grade_fn = mod.grade_answer_verl
    return _grade_fn


def _grade_math_in_child(response, label):
    """Runs in a forked child's main thread, so SIGALRM works. Returns True/False for a verdict and None for a timeout.
    Anything that prevents the verifier from running is raised, never converted into False."""
    import signal as _signal

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise _MathTimeout

    grade = _load_grade_answer_verl()          # raises on a broken environment -> parent falls back
    old = _signal.signal(_signal.SIGALRM, _on_alarm)
    _signal.setitimer(_signal.ITIMER_REAL, _MATH_TIMEOUT_S)
    try:
        return bool(grade(response, label))
    except _MathTimeout:
        return None
    except Exception:                          # the verifier itself rejecting odd input is a wrong answer
        return False
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old)


def _grade_math_guarded(response: str, label) -> float:
    """0.0/1.0. A verifier timeout is a legitimate 0.0 and is counted. A mechanism failure (broken pool, fork refused)
    must never become a silent 0.0 for every sample, so it falls back to grading in-process, loudly."""
    global _math_pool, _math_timeouts, _math_mech_failures
    import concurrent.futures as _cf
    import multiprocessing as _mp
    for attempt in (1, 2):
        try:
            if _math_pool is None:
                _math_pool = _cf.ProcessPoolExecutor(max_workers=_MATH_WORKERS,
                                                     mp_context=_mp.get_context("fork"))
            fut = _math_pool.submit(_grade_math_in_child, response or "", label)
            ok = fut.result(timeout=_MATH_TIMEOUT_S + 20)
            if ok is None:
                _math_timeouts += 1
                if _math_timeouts <= 5 or _math_timeouts % 50 == 0:
                    logger.warning("[grpo_rm] math verifier timed out after %.0fs (count=%d); scored 0.0",
                                   _MATH_TIMEOUT_S, _math_timeouts)
                return 0.0
            return 1.0 if ok else 0.0
        except Exception as exc:                      # broken pool / child killed: rebuild once, then fall back
            _math_mech_failures += 1
            logger.error("[grpo_rm] math grading subprocess failed (%s: %s); attempt %d/2, mech_failures=%d",
                         type(exc).__name__, str(exc)[:160], attempt, _math_mech_failures)
            try:
                if _math_pool is not None:
                    _math_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _math_pool = None
    logger.error("[grpo_rm] math grading subprocess unusable twice — falling back to IN-PROCESS grading "
                 "(unbounded: a sympy hang here stalls the rollout). mech_failures=%d", _math_mech_failures)
    try:
        return 1.0 if _load_grade_answer_verl()(response or "", label) else 0.0
    except Exception as exc:
        logger.error("[grpo_rm] in-process math grading also failed (%s); scoring 0.0", type(exc).__name__)
        return 0.0


async def grpo_rule_reward(args: Namespace, sample: Sample, **kwargs) -> float:
    dom = _domain(sample)
    if dom == "ifeval":
        # A malformed label is a data defect, not a wrong answer: raise rather than pay 0 reward for every sample.
        constraints = sample.label
        if isinstance(constraints, str):
            constraints = json.loads(constraints)
        try:
            ok, _ = _get_ifeval_score()(sample.response or "", constraints)
        except (KeyError, ValueError):
            raise
        except Exception as exc:  # a checker crashing on odd output is a wrong answer
            logger.warning("[grpo_rm] ifeval checker raised %s: %s", type(exc).__name__, exc)
            ok = False
        return 1.0 if ok else 0.0

    if dom not in ("code", "code-a3b"):
        # math and anything unrecognised: the guarded math verifier (identical verifier, bounded time)
        return await asyncio.to_thread(_grade_math_guarded, sample.response or "", sample.label)

    correct, _meta = await get_rule_based_reward(args, sample)
    # the code judge can return a float score; binarise it (>= 1.0 is correct). None (judge error) scores 0.0.
    if correct is None:
        return 0.0
    if isinstance(correct, bool):
        return 1.0 if correct else 0.0
    return 1.0 if float(correct) >= 1.0 else 0.0
