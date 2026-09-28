# SPDX-License-Identifier: Apache-2.0
"""eval.stats, eval.aggregate and eval.compare on the released per-question records reproduce reproduce.py and the
paper's tables: every level, every Total, extended, control and MATH-500 contrast with its paired interval
(set DN_MOPD_SLOW_TESTS=1 to also check the 168 domain contrasts)."""
import csv
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest
from evaltest_helpers import REPO

from eval.protocol import BOOTSTRAP_SEED, DOMAINS, SIX
from eval.stats import Records, contrast, level_row, paired

TABLES = REPO / 'reproduce' / 'data' / 'paper_tables'


@pytest.fixture(scope='module')
def rec():
    return Records(REPO / 'reproduce' / 'data', 'released')


@pytest.fixture(scope='module')
def reproduce():
    spec = importlib.util.spec_from_file_location('reproduce_script', REPO / 'reproduce' / 'reproduce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(got, want, tol=1e-6):
    return abs(float(got) - float(want)) <= tol


def test_levels_equal_reproduce(rec, reproduce):
    ref = reproduce.levels()
    assert len(ref) == 160 and len(rec.models()) == 80
    for want in ref:
        got = level_row(rec, want['model'], want['cap'])
        for k, v in want.items():
            if k not in ('model', 'cap'):
                assert got[k] == pytest.approx(float(v), abs=1e-12), (want['model'], want['cap'], k)


def test_levels_match_paper_tables(rec):
    for r in csv.DictReader(open(TABLES / 'q35_extended_results.csv')):
        got = level_row(rec, f'q35_{r["size"]}_{r["method"]}', int(r['evaluation_cap']))
        for k in (*SIX, 'math', 'code', 'if', 'total'):
            assert close(got[k], r[k]), (r['size'], r['method'], k)


def test_total_contrasts_match_paper(rec):
    rows = list(csv.DictReader(open(TABLES / 'q35_total_contrasts.csv')))
    assert len(rows) == 48
    for r in rows:
        d, ci, _ = contrast(rec, r['a'], rec, r['b'], int(r['cap']), DOMAINS['total'])
        assert close(d, r['delta_total']) and close(ci[0], r['ci95_low']) and close(ci[1], r['ci95_high']), r


def test_extended_and_control_contrasts_match_paper(rec):
    for r in csv.DictReader(open(TABLES / 'q35_extended_contrasts.csv')):
        k = lambda m: f'q35_{r["size"]}_{m}'
        d, ci, _ = contrast(rec, k(r['method_a']), rec, k(r['method_b']), int(r['evaluation_cap']), DOMAINS['total'])
        assert close(d, r['delta_total_pp']) and close(ci[0], r['paired_ci95_low']) and close(ci[1], r['paired_ci95_high'])
    for c in json.loads((TABLES / 'q35_controls.json').read_text())['contrasts']:
        k = lambda m: f'q35_{c["size"]}_{m}'
        d, ci, _ = contrast(rec, k(c['x']), rec, k(c['y']), c['cap'], DOMAINS[c['domain']])
        assert close(d, c['delta']) and close(ci[0], c['ci95'][0]) and close(ci[1], c['ci95'][1]), c


def test_math500_contrasts_match_paper(rec):
    names = {'base': 'base', 'route': 's_route', 'mdnorm': 's_mdnorm', 'singlemath': 's_singlemath'}
    for c in json.loads((TABLES / 'q35_controls.json').read_text())['math500']['contrasts']:
        a, b = f'q35_{c["size"]}_{names[c["a"]]}', f'q35_{c["size"]}_{names[c["b"]]}'
        d, ci, _ = contrast(rec, a, rec, b, c['cap'], ('math500',))
        assert close(d, c['delta']) and close(ci[0], c['ci95'][0]) and close(ci[1], c['ci95'][1]), c


@pytest.mark.skipif(os.environ.get('DN_MOPD_SLOW_TESTS') != '1', reason='set DN_MOPD_SLOW_TESTS=1 (about a minute)')
def test_domain_contrasts_match_paper(rec):
    for r in csv.DictReader(open(TABLES / 'q35_story_contrasts.csv')):
        k = lambda m: f'q35_{r["size"]}_{m}'
        d, ci, _ = contrast(rec, k(r['a']), rec, k(r['b']), int(r['cap']), DOMAINS[r['domain']])
        assert close(d, r['delta'], 2e-6) and close(ci[0], r['ci_low'], 2e-6) and close(ci[1], r['ci_high'], 2e-6)


def test_paired_uses_shared_indices():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    xa = {'q1': 1.0, 'q2': 0.0, 'q3': 0.5}
    d, draws = paired(xa, xa, rng)
    assert d == 0.0 and np.all(draws == 0.0)       # identical models: every replicate is exactly zero
    with pytest.raises(ValueError):
        paired(xa, {'q1': 1.0}, rng)


def test_runs_layout_equals_released_layout(rec, tmp_path):
    """Graded runs (<root>/<model>/<suite>/cap<cap>/scores.jsonl + summary.json) give the same numbers."""
    models, cap = ('q35_4b_s_mdnorm', 'q35_4b_s_route'), 8192
    for m in models:
        for s in SIX:
            dst = tmp_path / m / s / f'cap{cap}'
            dst.mkdir(parents=True)
            shutil.copy(rec.path(m, s, cap), dst / 'scores.jsonl')
            (dst / 'summary.json').write_text(json.dumps(rec.summary(m, s, cap)))
    runs = Records.detect(tmp_path)
    assert runs.layout == 'runs' and runs.models() == sorted(models)
    for m in models:
        assert level_row(runs, m, cap) == level_row(rec, m, cap)
    assert contrast(runs, models[0], rec, models[1], cap, SIX)[:2] == contrast(rec, models[0], rec, models[1], cap,
                                                                               SIX)[:2]


def test_cli_aggregate_and_compare(tmp_path, capsys, reproduce):
    from eval import aggregate, compare
    out = tmp_path / 'levels.csv'
    assert aggregate.main(['--csv', str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    ref = {(r['model'], r['cap']): r for r in reproduce.levels()}
    assert len(rows) == 160
    for r in rows:
        assert close(r['total'], ref[(r['model'], int(r['cap']))]['total'], 1e-9)
    capsys.readouterr()
    assert compare.main(['q35_9b_s_mdnorm', 'q35_9b_s_route', '--cap', '8192', '--domain', 'total', '--json']) == 0
    got = json.loads(capsys.readouterr().out)[0]
    want = [r for r in csv.DictReader(open(TABLES / 'q35_total_contrasts.csv'))
            if (r['a'], r['b'], r['cap']) == ('q35_9b_s_mdnorm', 'q35_9b_s_route', '8192')][0]
    assert close(got['delta_pp'], want['delta_total']) and close(got['ci95'][1], want['ci95_high'])
