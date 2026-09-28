# SPDX-License-Identifier: Apache-2.0
"""HTTP code judge for the PRIME code verifier (the server half of PRIME_code_server; server.py holds the client).

Routes (must match server.py JUDGE_ROUTER / HEALTH_ROUTER):
    POST /judge   {"completion": str, "test_cases": dict | str, "continuous": bool}
                  -> {"success": bool | float, "metadata": ..., "error": str | None}
    GET  /health  -> {"status": "ok"}

Launch (see recipes/qwen3.5/start_code_judge.sh):
    PYTHONPATH=<repo>/miles python -m uvicorn Uni_OPD_utils.outcome_reward.PRIME_code_server.judge_app:app \
        --host 127.0.0.1 --port 17580 --workers 6

Execution model. Each /judge request runs PRIME's compute_score in a ProcessPoolExecutor built on the 'spawn' context
(clean single-threaded workers), and the worker forces the 'fork' start method before judging. Newer uvicorn starts
its --workers with 'spawn', a spawn-started process inherits start_method='spawn', and PRIME's check_correctness
(written for fork) would then spawn its helper processes, each re-importing PRIME_code against a ~6 s join timeout:
correct solutions were graded false at random. Forcing 'fork' in the judging process removes that nondeterminism.
check_correctness bounds each call at about 6-7 s for the whole test suite of a problem (inherited PRIME semantics).
A wedged worker is killed after CODE_JUDGE_TIMEOUT seconds (default 30) and the pool is rebuilt after
CODE_JUDGE_REBUILD_AFTER consecutive timeouts (default 3); a timed-out judgment returns success 0.0 with an error.
Pool size: CODE_JUDGE_POOL (default 16) per uvicorn worker process. Cap BLAS threads (OPENBLAS_NUM_THREADS=1, ...)
in the launcher.

SECURITY: compute_score exec()s model-generated code with only reliability_guard() (os.system/os.remove/... nulled
out in the child process). That is NOT a sandbox. Run this service as a low-privilege user with ulimits (the launcher
sets them), bound to localhost, and never point it at untrusted third-party completions.
"""

import json
import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("code_judge_forksafe")

app = FastAPI(title="PRIME code judge (fork-safe)", docs_url=None, redoc_url=None)

_POOL_SIZE = int(os.environ.get("CODE_JUDGE_POOL", "16"))
# Per-judgment wall clock. check_correctness bounds itself at ~6-7 s; anything past this is a WEDGED worker, and a
# wedged worker must be killed.
_JUDGE_TIMEOUT = float(os.environ.get("CODE_JUDGE_TIMEOUT", "30"))
_pool_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None


def _judge_in_child(completion: str, test_cases, continuous: bool):
    # Runs in a spawn worker: single-threaded, so check_correctness may fork safely. A spawn-started process inherits
    # start_method='spawn' (multiprocessing.spawn.prepare -> set_start_method(force=True)), so PRIME's
    # check_correctness -- written for fork -- would spawn its Manager server and its _temp_run child, each
    # re-importing PRIME_code against a 6 s join timeout; force 'fork' here.
    import multiprocessing as _mp

    if _mp.get_start_method(allow_none=True) != "fork":
        _mp.set_start_method("fork", force=True)
    from Uni_OPD_utils.outcome_reward.PRIME_code import compute_score

    return compute_score(completion, test_cases, continuous)


_REBUILD_AFTER = int(os.environ.get("CODE_JUDGE_REBUILD_AFTER", "3"))
_timeout_lock = threading.Lock()
_consecutive_timeouts = 0


def _kill_processes(pool) -> int:
    """Kill a pool's worker processes. `shutdown(wait=False)` does NOT: it drops the executor and
    leaves its workers running, so rebuilding on every timeout would leak workers."""
    n = 0
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            if proc.is_alive():
                proc.kill()
                n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


def _kill_pool_children() -> int:
    """Kill every worker of the current pool.

    `fut.result(timeout=...)` gives up on the FUTURE but leaves the worker running: a submission that
    wedges a worker (an unterminated program that check_correctness fails to bound) removes that worker
    from the pool permanently. Capacity then decays to zero, and because the handler returns
    {"success": 0.0} on timeout, a starved judge is indistinguishable from a model that cannot code.
    Killing the workers (and letting the pool be rebuilt) bounds the damage to the in-flight batch.
    """
    global _pool
    killed = 0
    with _pool_lock:
        if _pool is None:
            return 0
        for proc in list(getattr(_pool, "_processes", {}).values()):
            try:
                if proc.is_alive():
                    proc.kill()
                    killed += 1
            except Exception:  # noqa: BLE001
                pass
    return killed


def _get_pool(rebuild: bool = False) -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if rebuild and _pool is not None:
            old, _pool = _pool, None
            killed = _kill_processes(old)          # kill BEFORE shutdown, or the workers linger
            old.shutdown(wait=False, cancel_futures=True)
            logger.error("judge pool rebuilt; killed %d worker(s)", killed)
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=_POOL_SIZE, mp_context=multiprocessing.get_context("spawn")
            )
        return _pool


class JudgeRequest(BaseModel):
    completion: str
    test_cases: dict | str
    continuous: bool = False


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:  # noqa: BLE001
            return str(obj)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/judge")
def judge(req: JudgeRequest) -> dict:
    global _consecutive_timeouts
    for attempt in (0, 1):
        try:
            fut = _get_pool(rebuild=attempt > 0).submit(
                _judge_in_child, req.completion, req.test_cases, req.continuous
            )
            # check_correctness bounds itself at ~6-7 s; the wall clock only guards a wedged worker.
            success, metadata = fut.result(timeout=_JUDGE_TIMEOUT)
            with _timeout_lock:
                _consecutive_timeouts = 0          # a success clears the streak
            return {"success": success, "metadata": _jsonable(metadata), "error": None}
        except FutureTimeoutError:
            # A wedged worker never returns, so capacity decays and every later judgment is silently
            # graded WRONG. But rebuilding on EVERY timeout is worse than the disease: it discards
            # healthy in-flight judgments and (before _kill_processes) leaked workers. Rebuild only
            # after _REBUILD_AFTER consecutive timeouts; isolated ones just fail.
            with _timeout_lock:
                _consecutive_timeouts += 1
                n_consec = _consecutive_timeouts
                due = n_consec >= _REBUILD_AFTER
                if due:
                    _consecutive_timeouts = 0
            if due:
                _get_pool(rebuild=True)
            logger.error("judge timed out after %.0fs (consecutive=%d, rebuilt=%s)",
                         _JUDGE_TIMEOUT, n_consec, due)
            return {"success": 0.0, "metadata": None,
                    "error": f"TimeoutError: judgment exceeded {_JUDGE_TIMEOUT:.0f}s"}
        except BrokenProcessPool:
            logger.exception("judge pool broke (attempt %d)", attempt)
            continue
        except Exception as e:  # noqa: BLE001
            logger.exception("judge call failed")
            return {"success": 0.0, "metadata": None, "error": f"{type(e).__name__}: {e}"}
    return {"success": 0.0, "metadata": None, "error": "BrokenProcessPool: pool broke twice"}
