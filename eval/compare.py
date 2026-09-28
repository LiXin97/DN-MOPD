# SPDX-License-Identifier: Apache-2.0
"""Paired bootstrap difference A - B, exactly as the paper computes its intervals (reproduce/reproduce.py).

    python -m eval.compare q35_9b_s_mdnorm q35_9b_s_route                       # released records, both caps
    python -m eval.compare mymodel q35_9b_s_route --records-a outputs/eval --cap 16384
    python -m eval.compare A B --domain total,math,code,if,aime25 --cap 8192
    python -m eval.compare q35_4b_s_mdnorm q35_4b_base --domain math500 --cap 8192

Each contrast (a domain, the Total, one suite, or MATH-500) resamples the questions of each of its suites with the same
indices for both models, B = 10,000 replicates, from a fresh numpy.random.default_rng(20260923) per contrast, and
reports the point difference in percentage points with the 2.5/97.5 percentiles. Suites are resampled in the order
AIME25, AIME26, LCB v5, LCB v6, IFEval, IFBench. Intervals are conditional on one training run per model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.common import released_dir
from eval.protocol import CAPS, DOMAINS, SUITES
from eval.stats import Records, contrast

CHOICES = (*DOMAINS, *SUITES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', help='model key of A')
    ap.add_argument('b', help='model key of B')
    ap.add_argument('--cap', type=int, choices=CAPS, action='append', help='evaluation cap(s) (default: both)')
    ap.add_argument('--domain', default='total,math,code,if',
                    help='comma-separated contrasts: ' + ', '.join(CHOICES))
    ap.add_argument('--records', type=Path, default=None, help='records of both models (default reproduce/data)')
    ap.add_argument('--records-a', type=Path, default=None, help='records of A, if they live elsewhere')
    ap.add_argument('--records-b', type=Path, default=None, help='records of B, if they live elsewhere')
    ap.add_argument('--no-bootstrap', action='store_true', help='point differences only')
    ap.add_argument('--json', action='store_true', help='print JSON instead of a table')
    a = ap.parse_args(argv)
    base = a.records or released_dir()
    rec_a, rec_b = Records.detect(a.records_a or base), Records.detect(a.records_b or base)
    wanted = [d for d in a.domain.split(',') if d]
    bad = sorted(set(wanted) - set(CHOICES))
    if bad:
        ap.error(f'unknown contrast(s) {bad}; choose from {", ".join(CHOICES)}')
    out = []
    for cap in a.cap or CAPS:
        for name in wanted:
            suites = DOMAINS.get(name, (name,))
            delta, ci, _ = contrast(rec_a, a.a, rec_b, a.b, cap, suites, boot=not a.no_bootstrap)
            out.append({'a': a.a, 'b': a.b, 'cap': cap, 'contrast': name, 'delta_pp': delta, 'ci95': ci})
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f'{a.a} - {a.b} (percentage points; paired bootstrap B = 10,000, seed 20260923)')
    for r in out:
        ci = '' if r['ci95'] is None else f'  95% CI [{r["ci95"][0]:+.2f}, {r["ci95"][1]:+.2f}]'
        print(f'  cap {r["cap"]:5d}  {r["contrast"]:8s} {r["delta_pp"]:+7.2f}{ci}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
