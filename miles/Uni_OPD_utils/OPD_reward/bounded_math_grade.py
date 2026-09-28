# SPDX-License-Identifier: Apache-2.0
"""Bounded math grading for the rule-based correctness label.

`grade_answer_verl` can end in `sympy.simplify` with no time limit, and it used to run synchronously inside the
rollout event loop: one pathological boxed answer (for example `\\boxed{99999999!}` against a non-integer reference)
freezes every pending generate/score coroutine. The grader itself is unchanged; it runs in killable child processes
(math_grader_worker.py) with a per-request time limit. A timed-out answer is scored False (`grader_timeout` in the
metadata) exactly as a grader exception is. On the on-policy-distillation path the correctness label is logged (and
read by the injection trigger); it never enters the distillation loss.

Plain `subprocess` workers are driven from a small thread pool: the rollout manager's event loop cannot spawn asyncio
subprocesses inside a Ray actor, while threads that wait on a pipe release the GIL, so the loop keeps running even when
a worker hangs in C code.

Environment: MATH_GRADER_TIMEOUT (seconds per answer, default 20), MATH_GRADER_WORKERS (default 8).
"""
import asyncio
import json
import logging
import os
import queue
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
TIMEOUT = float(os.environ.get("MATH_GRADER_TIMEOUT", "20"))
WORKERS = int(os.environ.get("MATH_GRADER_WORKERS", "8"))
START_TIMEOUT = 180.0
READY = b"MATH_GRADER_READY"
_WORKER_FILE = Path(__file__).with_name("math_grader_worker.py")
_POOL = {"workers": None, "executor": None, "timeouts": 0, "graded": 0}


class GraderTimeout(Exception):
    pass


class _Worker:
    def __init__(self):
        self.proc = None

    def _read_line(self, limit):
        deadline, buf = time.monotonic() + limit, b""
        while not buf.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise GraderTimeout()
            chunk = os.read(self.proc.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError("math grader worker exited")
            buf += chunk
        return buf

    def _start(self):
        self.proc = subprocess.Popen([sys.executable, "-u", str(_WORKER_FILE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if self._read_line(START_TIMEOUT).strip() != READY:
            raise RuntimeError("math grader worker did not start")

    def kill(self):
        if self.proc is not None:
            try:
                self.proc.kill(); self.proc.wait(timeout=10)
            except Exception:
                pass
        self.proc = None

    def grade(self, response, label):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self.proc.stdin.write((json.dumps([response, label]) + "\n").encode())
        return bool(json.loads(self._read_line(TIMEOUT)))


def _pool():
    if _POOL["workers"] is None:
        q = queue.Queue()
        for _ in range(WORKERS):
            q.put(_Worker())
        _POOL.update(workers=q, executor=ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="math_grader"))
    return _POOL["workers"], _POOL["executor"]


def _grade_sync(response, label):
    workers, _ = _pool()
    worker = workers.get()
    try:
        verdict = worker.grade(response, label)
        _POOL["graded"] += 1
        return verdict, {}
    except GraderTimeout:
        worker.kill()
        _POOL["timeouts"] += 1
        logger.warning(f"[math_grader] no verdict within {TIMEOUT:.0f}s; scored False, worker replaced "
                       f"(timeouts so far {_POOL['timeouts']}; answer tail {response[-80:]!r})")
        return False, {"grader_timeout": True}
    except Exception as e:                               # broken pipe, worker crash, start failure
        worker.kill()
        logger.warning(f"[math_grader] worker failure {type(e).__name__}: {e}; scored False")
        return False, {"grader_error": type(e).__name__}
    finally:
        workers.put(worker)


async def grade_math_bounded(response, label):
    """(correct, metadata). Never blocks the event loop; never raises for a slow or crashed grader."""
    _, executor = _pool()
    return await asyncio.get_running_loop().run_in_executor(executor, _grade_sync, response, label)
