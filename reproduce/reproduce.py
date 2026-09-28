#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recompute the paper's Qwen3.5 scores and paired intervals from the per-question records, and check them against
the numbers the paper's tables and figures were rendered from (data/paper_tables/).

  python reproduce.py                 # full check (paired bootstrap, B = 10,000; a few minutes on one CPU)
  python reproduce.py --no-bootstrap  # point estimates only (seconds)

Definitions (paper Section 3 and App. A.3):
  question score  = fraction of the sampled answers graded correct
  suite score     = mean question score (avg@N; x100 for percentages)
  domain score    = mean of its two suites (math = AIME25/26, code = LCB v5/v6, IF = IFEval/IFBench)
  Total           = mean of the six suite scores
  paired interval = 2.5/97.5 percentiles of B = 10,000 bootstrap replicates, seed 20260923, resampling questions
                    within each suite with the same indices for both models; a fresh generator per contrast.
  seed read       = the seed-42/43/44 Total draws of DN-MOPD - Label share one generator (in seed order); the
                    three-seed mean interval resamples a seed and then one of its draws, three times per replicate.
Requires numpy only. Writes results/levels.csv and results/checks.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent            # reproduce/: data/ and results/ live beside this script
DATA = ROOT / 'data'
SUITES = ('aime25', 'aime26', 'lcb_v5', 'lcb_v6', 'ifeval', 'ifbench')
DOMAINS = {'math': ('aime25', 'aime26'), 'code': ('lcb_v5', 'lcb_v6'), 'if': ('ifeval', 'ifbench'), 'total': SUITES}
B, SEED = 10000, 20260923
CELLS = {(r['model'], r['suite'], r['cap']): r for r in map(json.loads, open(DATA / 'cells.jsonl'))}
_cache = {}


def per_question(model, suite, cap):
    k = (model, suite, cap)
    if k not in _cache:
        rows = [json.loads(line) for line in open(DATA / f'scores/{suite}/cap{cap}/{model}.jsonl')]
        out = {r['id']: r['correct'].count('1') / len(r['correct']) for r in rows}
        want = CELLS[k]['mean_pass1']
        if abs(sum(out.values()) / len(out) - want) > 1e-9 or len(out) != CELLS[k]['problems']:
            raise SystemExit(f'score file disagrees with its cell summary: {k}')
        _cache[k] = out
    return _cache[k]


def m500(model, cap):
    rows = [json.loads(line) for line in open(DATA / f'math500/cap{cap}/{model}.jsonl')]
    return {r['id']: r['correct'].count('1') / len(r['correct']) for r in rows}


def paired(xa, xb, rng, boot):
    if set(xa) != set(xb):
        raise SystemExit('question ids differ')
    ids = sorted(xa)
    va, vb = np.array([xa[i] for i in ids]), np.array([xb[i] for i in ids])
    d = 100 * (va.mean() - vb.mean())
    if not boot:
        return d, None
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    return d, 100 * (va[idx].mean(axis=1) - vb[idx].mean(axis=1))


def contrast(a, b, cap, domain='total', boot=True, rng=None):
    rng = rng if rng is not None else np.random.default_rng(SEED)
    suites = DOMAINS[domain]
    pt, draws = 0.0, np.zeros(B)
    for s in suites:
        d, bt = paired(per_question(a, s, cap), per_question(b, s, cap), rng, boot)
        pt += d / len(suites)
        if boot:
            draws += bt / len(suites)
    ci = [float(v) for v in np.percentile(draws, [2.5, 97.5])] if boot else None
    return float(pt), ci, draws


def seed_read(size, cap, boot):
    label = {42: 's_route', 43: 'ctl_observe_s43', 44: 'ctl_observe_s44'}
    dn = {42: 's_mdnorm', 43: 'ctl_dnorm_s43', 44: 'ctl_dnorm_s44'}
    rng = np.random.default_rng(SEED)
    per, draws = {}, {}
    for sd in (42, 43, 44):
        pt, ci, bt = contrast(f'q35_{size}_{dn[sd]}', f'q35_{size}_{label[sd]}', cap, boot=boot, rng=rng)
        per[str(sd)] = {'delta': pt, 'ci95': ci}
        draws[sd] = bt
    out = {'per_seed': per, 'mean': float(np.mean([v['delta'] for v in per.values()]))}
    if boot:
        keys, two = sorted(draws), np.zeros(B)
        for _ in range(3):
            pick, col = rng.integers(0, 3, size=B), rng.integers(0, B, size=B)
            two += np.array([draws[keys[p]][c] for p, c in zip(pick, col)]) / 3
        out['two_level_ci95'] = [float(v) for v in np.percentile(two, [2.5, 97.5])]
    return out


class Checker:
    def __init__(self):
        self.n, self.worst, self.fail = 0, 0.0, []

    def eq(self, what, got, want, tol):
        if got is None:
            return
        dev = abs(float(got) - float(want))
        self.n += 1
        self.worst = max(self.worst, dev)
        if dev > tol:
            self.fail.append(f'{what}: reproduced {got:.6f}, paper data {float(want):.6f}')


def levels():
    rows = []
    for model in sorted({m for m, _, _ in CELLS}):
        for cap in (8192, 16384):
            s = {x: 100 * np.mean(list(per_question(model, x, cap).values())) for x in SUITES}
            dom = {d: float(np.mean([s[x] for x in v])) for d, v in DOMAINS.items()}
            toks = {x: CELLS[(model, x, cap)]['mean_output_tokens'] for x in SUITES}
            hits = {x: CELLS[(model, x, cap)]['cap_hit_rate'] for x in SUITES}
            rows.append({'model': model, 'cap': cap, **{x: s[x] for x in SUITES}, **dom,
                         'math_tokens': np.mean([toks[x] for x in DOMAINS['math']]),
                         'code_tokens': np.mean([toks[x] for x in DOMAINS['code']]),
                         'if_tokens': np.mean([toks[x] for x in DOMAINS['if']]),
                         'cap_hit_rate': float(np.mean(list(hits.values())))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-bootstrap', action='store_true')
    boot = not ap.parse_args().no_bootstrap
    P = DATA / 'paper_tables'
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    ck = Checker()

    lv = levels()                                                     # every cell is checked inside per_question
    with open(out / 'levels.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(lv[0]))
        w.writeheader()
        w.writerows(lv)
    by = {(r['model'], r['cap']): r for r in lv}
    for r in csv.DictReader(open(P / 'q35_extended_results.csv')):   # App. C levels
        got = by[(f"q35_{r['size']}_{r['method']}", int(r['evaluation_cap']))]
        for x in (*SUITES, 'math', 'code', 'if', 'total'):
            ck.eq(f"level {r['size']} {r['method']} {x}", got[x], r[x], 1e-6)
    print(f'levels: {len(lv)} model-cap rows from {len(CELLS)} cells', flush=True)

    def check(tag, a, b, cap, dom, delta, ci, tol):
        d, c, _ = contrast(a, b, cap, dom, boot)
        ck.eq(f'{tag} {a}-{b} {cap} {dom} delta', d, delta, tol)
        if c is not None and ci is not None:
            ck.eq(f'{tag} {a}-{b} {cap} {dom} lo', c[0], ci[0], tol)
            ck.eq(f'{tag} {a}-{b} {cap} {dom} hi', c[1], ci[1], tol)

    for r in csv.DictReader(open(P / 'q35_total_contrasts.csv')):    # Tables 1-2, Figure 1
        check('total', r['a'], r['b'], int(r['cap']), 'total', r['delta_total'], [r['ci95_low'], r['ci95_high']], 1e-6)
    print('total contrasts done', flush=True)
    for r in csv.DictReader(open(P / 'q35_story_contrasts.csv')):    # domain contrasts (6 decimals)
        k = lambda m: f"q35_{r['size']}_{m}"
        check('domain', k(r['a']), k(r['b']), int(r['cap']), r['domain'], r['delta'], [r['ci_low'], r['ci_high']],
              2e-6)
    print('domain contrasts done', flush=True)
    for r in csv.DictReader(open(P / 'q35_extended_contrasts.csv')):   # 160-update and annealed comparisons
        k = lambda m: f"q35_{r['size']}_{m}"
        check('extended', k(r['method_a']), k(r['method_b']), int(r['evaluation_cap']), 'total', r['delta_total_pp'],
              [r['paired_ci95_low'], r['paired_ci95_high']], 1e-6)
    rv = json.loads((P / 'q35_controls.json').read_text())    # App. C.3, C.5-C.7 controls
    for c in rv['contrasts']:
        k = lambda m: f"q35_{c['size']}_{m}"
        check(c['section'], k(c['x']), k(c['y']), c['cap'], c['domain'], c['delta'], c['ci95'], 1e-6)
    print('control contrasts done', flush=True)
    for s in rv['seeds']:
        got = seed_read(s['size'], s['cap'], boot)
        for sd, v in s['per_seed'].items():
            ck.eq(f"seed {s['size']} {s['cap']} s{sd} delta", got['per_seed'][sd]['delta'], v['delta'], 1e-6)
            if boot:
                for j, e in enumerate(('lo', 'hi')):
                    ck.eq(f"seed {s['size']} {s['cap']} s{sd} {e}", got['per_seed'][sd]['ci95'][j], v['ci95'][j], 1e-6)
        ck.eq(f"seed {s['size']} {s['cap']} mean", got['mean'], s['mean'], 1e-6)
        if boot:
            for j, e in enumerate(('lo', 'hi')):
                ck.eq(f"seed {s['size']} {s['cap']} two-level {e}", got['two_level_ci95'][j], s['two_level_ci95'][j],
                      1e-6)
    names = {'base': 'base', 'route': 's_route', 'mdnorm': 's_mdnorm', 'singlemath': 's_singlemath'}
    for c in rv['math500']['contrasts']:                             # App. C.7 MATH-500
        xa, xb = m500(f"q35_{c['size']}_{names[c['a']]}", c['cap']), m500(f"q35_{c['size']}_{names[c['b']]}", c['cap'])
        d, bt = paired(xa, xb, np.random.default_rng(SEED), boot)
        ck.eq(f"math500 {c['size']} {c['cap']} {c['a']}-{c['b']} delta", d, c['delta'], 1e-6)
        if boot:
            lo, hi = np.percentile(bt, [2.5, 97.5])
            ck.eq(f"math500 {c['size']} {c['cap']} {c['a']}-{c['b']} lo", lo, c['ci95'][0], 1e-6)
            ck.eq(f"math500 {c['size']} {c['cap']} {c['a']}-{c['b']} hi", hi, c['ci95'][1], 1e-6)

    res = {'bootstrap': boot, 'values_checked': ck.n, 'max_abs_deviation': ck.worst, 'failures': ck.fail}
    (out / 'checks.json').write_text(json.dumps(res, indent=1) + '\n')
    print(f"checked {ck.n} values against the paper data; max |deviation| {ck.worst:.2e}; "
          f"{'ALL MATCH' if not ck.fail else f'{len(ck.fail)} MISMATCHES'}")
    for f in ck.fail[:20]:
        print('  ', f)
    raise SystemExit(1 if ck.fail else 0)


if __name__ == '__main__':
    main()
