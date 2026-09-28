# SPDX-License-Identifier: Apache-2.0
"""Concurrency check of the PRIME code judge.

`/health` answers from a judge whose worker pool is exhausted, and the judge grades a timed-out judgment as
`success: 0.0` (wrong), so a starved judge is indistinguishable from a model that cannot code. This check sends N
trivial correct submissions and a few non-terminating ones at once and requires every verdict to be right within the
deadline. Exit 0 = usable, 3 = degraded.

    python judge_check.py [--ep 127.0.0.1:17580] [--n 20] [--bad 4] [--deadline 90]
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OK = "```python\ndef solve():\n    print(2)\nsolve()\n```"
BAD = "```python\nwhile True:\n    pass\n```"
CASES = {"inputs": [""], "outputs": ["2"]}


def call(ep: str, completion: str, timeout: float):
    body = json.dumps({"completion": completion, "test_cases": CASES}).encode()
    t = time.time()
    try:
        req = urllib.request.Request(f"http://{ep}/judge", data=body, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy
        return time.time() - t, json.loads(opener.open(req, timeout=timeout).read()).get("success")
    except Exception as e:  # noqa: BLE001
        return time.time() - t, f"FAIL {type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="127.0.0.1:17580")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--bad", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=90.0)
    a = ap.parse_args()
    jobs = [(OK, True)] * a.n + [(BAD, False)] * a.bad
    t0 = time.time()
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: call(a.ep, j[0], a.deadline), jobs))
    wall = time.time() - t0
    wrong = sum(1 for (_, got), (_, want) in zip(res, jobs) if got is not want)
    lat = sorted(r[0] for r in res)
    print(f"JUDGE_CHECK ep={a.ep} n={len(jobs)} wall={wall:.1f}s p50={lat[len(lat) // 2]:.2f}s "
          f"max={lat[-1]:.2f}s wrong={wrong}/{len(jobs)}")
    if wrong or wall > a.deadline:
        print("JUDGE_CHECK DEGRADED: restart the judge before a code run")
        return 3
    print("JUDGE_CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
