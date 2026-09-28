import multiprocessing
import os
import sys
import traceback

from .testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(in_outs: dict | None, generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list
