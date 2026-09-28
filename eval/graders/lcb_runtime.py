# SPDX-License-Identifier: Apache-2.0
"""Official LiveCodeBench test runner with bounded child memory and complete timeout metadata.

Replaces lcb_runner's check_correctness (the official _temp_run / run_test are called unchanged) in two respects:
  * the child process runs under an 8 GiB RLIMIT_AS (or a tighter inherited limit), so a memory bomb in a model's
    program fails its tests instead of taking the grading host down;
  * the pinned runner indexes an empty metadata list when its child exceeds the global deadline; this wrapper returns
    the official all-tests-failed result (-1 per test) with explicit metadata instead. Any other unexpected child exit
    raises: it is an infrastructure failure, never a wrong answer.
"""
from __future__ import annotations

import json
import multiprocessing
import resource

from eval.protocol import LCB_MEMORY_LIMIT_BYTES

MEMORY_LIMIT_BYTES = LCB_MEMORY_LIMIT_BYTES
REAP_TIMEOUT_SECONDS = 60


class PostTestTransport:
    """Restore the caller's address-space limit only after run_test returns."""

    def __init__(self, sink, limits):
        self.sink = sink
        self.limits = limits

    def append(self, value):
        resource.setrlimit(resource.RLIMIT_AS, self.limits)
        self.sink.append(value)


def run_bounded(sample, generation, debug, result, metadata, timeout):
    from lcb_runner.evaluation.compute_code_generation_metrics import _temp_run
    previous = resource.getrlimit(resource.RLIMIT_AS)
    soft, hard = previous
    finite = [MEMORY_LIMIT_BYTES] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY]
    limit = min(finite)
    # Keep the existing hard ceiling so the original soft limit can be restored after the official run_test has
    # completed. The program itself is constrained by the same 8 GiB (or tighter inherited) soft limit.
    result_transport = PostTestTransport(result, previous)
    metadata_transport = PostTestTransport(metadata, previous)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    try:
        _temp_run(sample, generation, debug, result_transport, metadata_transport, timeout)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def check_correctness(sample, generation, timeout, debug=True):
    tests = len(json.loads(sample['input_output'])['inputs'])
    deadline = (timeout + 1) * tests + 5
    context = multiprocessing.get_context('fork')     # the start method of the paper's runs (Linux default <= 3.13)
    with context.Manager() as manager:
        result = manager.list()
        metadata = manager.list()
        proc = context.Process(target=run_bounded, args=(sample, generation, debug, result, metadata, timeout))
        try:
            proc.start()
            proc.join(timeout=deadline)
            timed_out = proc.is_alive()
            if timed_out:
                proc.kill()
                proc.join(timeout=REAP_TIMEOUT_SECONDS)
                if proc.is_alive():
                    raise RuntimeError('Unable to reap timed-out official checker')
            # Preserve successful upstream results, including ordinary code failures.
            if len(result) == 1 and len(metadata) == 1:
                if not timed_out and proc.exitcode != 0:
                    raise RuntimeError(f'Official checker exited abnormally: {proc.exitcode}')
                return result[0], metadata[0]
            if timed_out and not result and not metadata:
                return [-1] * tests, {
                    'error_code': -3,
                    'error_message': 'Global Time Limit Exceeded',
                    'global_timeout_seconds': deadline,
                    'child_exitcode': proc.exitcode,
                    'wrapper_revision': 'lcb-global-timeout-metadata-v1',
                }
            raise RuntimeError('Official checker returned an incomplete result: '
                               f'exitcode={proc.exitcode}, timed_out={timed_out}, '
                               f'results={len(result)}, metadata={len(metadata)}')
        finally:
            if proc.pid is not None and proc.is_alive():
                proc.kill()
                proc.join(timeout=REAP_TIMEOUT_SECONDS)


def install():
    """Route the official evaluate_generations_by_problem through the bounded check_correctness."""
    from lcb_runner.evaluation import compute_code_generation_metrics as official
    official.check_correctness = check_correctness
