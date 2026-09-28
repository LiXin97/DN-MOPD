# SPDX-License-Identifier: Apache-2.0
"""Child process of the bounded math grader: one JSON request per line on stdin ([response, label]), one JSON result
per line on stdout. It runs the unchanged grader (miles.rollout.rm_hub.math_utils.grade_answer_verl); the parent kills
it when a request exceeds the time limit (sympy.simplify has no limit of its own)."""
import importlib.util
import json
import sys
from pathlib import Path


def load_grader():
    """The grader file of this tree, loaded by path: importing the `miles.rollout` package would pull torch and sglang
    into every worker; math_utils.py itself needs only re, sympy and pylatexenc."""
    path = Path(__file__).resolve().parents[2] / "miles/rollout/rm_hub/math_utils.py"
    spec = importlib.util.spec_from_file_location("bounded_math_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade_answer_verl


def main():
    grade_answer_verl = load_grader()
    out = sys.stdout
    sys.stdout = sys.stderr                 # nothing but results may reach the pipe
    print("MATH_GRADER_READY", file=out, flush=True)
    for line in sys.stdin:
        response, label = json.loads(line)
        try:
            result = bool(grade_answer_verl(response, label))
        except Exception:                   # the in-process grader's exceptions are scored False too
            result = False
        print(json.dumps(result), file=out, flush=True)


if __name__ == "__main__":
    main()
