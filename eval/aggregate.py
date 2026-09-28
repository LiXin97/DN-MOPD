# SPDX-License-Identifier: Apache-2.0
"""Suite, domain and Total scores, mean output tokens and cap-hit rate from per-question records.

    python -m eval.aggregate                                  # the paper's 80 models (reproduce/data), both caps
    python -m eval.aggregate --records outputs/eval           # your graded runs: <root>/<model>/<suite>/cap<cap>/
    python -m eval.aggregate --models q35_9b_s_mdnorm,q35_9b_s_route --cap 8192
    python -m eval.aggregate --csv levels.csv                 # also write every row as CSV
    python -m eval.aggregate --math500                        # MATH-500 avg@16 instead of the six suites

Scores are percentages (avg@N x 100). Domain scores average their two suites; Total averages all six. The token
columns are the mean output tokens of each domain (mean of its two suites), and cap_hit_rate is the mean over the six
suites of the fraction of sampled answers that stopped at the cap. A domain or Total with a missing suite is shown
as '-'.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from eval.common import released_dir
from eval.protocol import CAPS, SIX
from eval.stats import Records, level_row

COLUMNS = (*SIX, 'math', 'code', 'if', 'total', 'math_tokens', 'code_tokens', 'if_tokens', 'cap_hit_rate')


def fmt(value, digits: int = 2) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records', type=Path, default=None,
                    help='released records (default reproduce/data) or a directory of graded runs')
    ap.add_argument('--models', default='', help='comma-separated model keys (default: all)')
    ap.add_argument('--cap', type=int, choices=CAPS, action='append', help='evaluation cap(s) (default: both)')
    ap.add_argument('--csv', type=Path, default=None, help='write the rows to this CSV file')
    ap.add_argument('--math500', action='store_true', help='report MATH-500 avg@16 (released layout: math500/)')
    a = ap.parse_args(argv)
    root = a.records or released_dir()
    if not root.is_dir():
        print(f'no such records directory: {root}', file=sys.stderr)
        return 2
    rec = Records.detect(root)
    models = [m for m in a.models.split(',') if m] or (
        sorted({p.stem for p in (rec.root / 'math500').glob('cap*/*.jsonl')}) if a.math500 and rec.layout == 'released'
        else rec.models())
    caps = a.cap or list(CAPS)
    if a.math500:
        print(f'{"model":34s} {"cap":>6s} {"math500":>8s}')
        for model in models:
            for cap in caps:
                if rec.has(model, 'math500', cap):
                    print(f'{model:34s} {cap:6d} {100 * np.mean(list(rec.scores(model, "math500", cap).values())):8.2f}')
        return 0
    rows = [level_row(rec, m, c) for m in models for c in caps if any(rec.has(m, x, c) for x in SIX)]
    if not rows:
        print('no records found under', rec.root, file=sys.stderr)
        return 1
    head = f'{"model":34s} {"cap":>6s} ' + ' '.join(f'{c:>8s}' for c in COLUMNS)
    print(head)
    for r in rows:
        cells = [fmt(r[c], 4 if c == 'cap_hit_rate' else 1 if c.endswith('_tokens') else 2) for c in COLUMNS]
        print(f'{r["model"]:34s} {r["cap"]:6d} ' + ' '.join(f'{v:>8s}' for v in cells))
    if a.csv:
        with a.csv.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['model', 'cap', *COLUMNS])
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
