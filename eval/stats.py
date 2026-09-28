# SPDX-License-Identifier: Apache-2.0
"""Scores and paired bootstrap of the paper (the definitions of reproduce/reproduce.py, computed the same way).

  question score  fraction of a question's sampled answers graded correct
  suite score     100 x mean question score (avg@N)
  domain score    mean of its two suite scores (math = AIME25/26, code = LCB v5/v6, IF = IFEval/IFBench)
  Total           mean of the six suite scores
  paired interval 2.5/97.5 percentiles of B = 10,000 bootstrap replicates, seed 20260923, resampling questions within
                  each suite with the same indices for both models; a fresh generator per contrast

Per-question records use the released format: one JSON line per question with `id`, `input_sha256`, `n`, `correct`
(one '0'/'1' character per sampled answer), `mean_tokens` and `cap_hits`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval.protocol import BOOTSTRAP_B, BOOTSTRAP_SEED, DOMAINS, SIX

Scores = Dict[str, float]


def question_scores(rows: Sequence[dict]) -> Scores:
    """{question id: fraction of correct sampled answers}."""
    out = {r['id']: r['correct'].count('1') / len(r['correct']) for r in rows}
    if len(out) != len(rows):
        raise ValueError('duplicate question id')
    return out


class Records:
    """Per-question records of evaluated models, from either layout:

      released  reproduce/data: scores/<suite>/cap<cap>/<model>.jsonl, math500/cap<cap>/<model>.jsonl, and the cell
                summaries cells.jsonl (exact mean output tokens and cap-hit rates);
      runs      a directory of graded generations: <root>/<model>/<suite>/cap<cap>/{scores.jsonl,summary.json}.
    """

    def __init__(self, root: Path, layout: str):
        if layout not in ('released', 'runs'):
            raise ValueError(layout)
        self.root, self.layout = Path(root), layout
        self._rows: Dict[Tuple[str, str, int], List[dict]] = {}
        self._cells: Optional[Dict[Tuple[str, str, int], dict]] = None

    @classmethod
    def detect(cls, root: Path) -> 'Records':
        root = Path(root)
        if (root / 'scores').is_dir() or (root / 'cells.jsonl').is_file():
            return cls(root, 'released')
        return cls(root, 'runs')

    def path(self, model: str, suite: str, cap: int) -> Path:
        if self.layout == 'released':
            top = 'math500' if suite == 'math500' else f'scores/{suite}'
            return self.root / top / f'cap{cap}' / f'{model}.jsonl'
        return self.root / model / suite / f'cap{cap}' / 'scores.jsonl'

    def has(self, model: str, suite: str, cap: int) -> bool:
        return self.path(model, suite, cap).is_file()

    def rows(self, model: str, suite: str, cap: int) -> List[dict]:
        key = (model, suite, cap)
        if key not in self._rows:
            path = self.path(model, suite, cap)
            if not path.is_file():
                raise FileNotFoundError(f'no records for {model} {suite} cap {cap}: {path}')
            with path.open() as f:
                self._rows[key] = [json.loads(line) for line in f if line.strip()]
        return self._rows[key]

    def models(self) -> List[str]:
        if self.layout == 'released':
            return sorted({p.stem for p in (self.root / 'scores').glob('*/cap*/*.jsonl')})
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob('*/cap*/scores.jsonl')))

    def recorded_summary(self, model: str, suite: str, cap: int) -> Optional[dict]:
        """The cell summary written at grading time (cells.jsonl or summary.json), if any."""
        if self.layout == 'released':
            if self._cells is None:
                path = self.root / 'cells.jsonl'
                self._cells = {}
                if path.is_file():
                    for line in path.open():
                        r = json.loads(line)
                        self._cells[(r['model'], r['suite'], r['cap'])] = r
            return self._cells.get((model, suite, cap))
        path = self.root / model / suite / f'cap{cap}' / 'summary.json'
        return json.loads(path.read_text()) if path.is_file() else None

    def summary(self, model: str, suite: str, cap: int) -> dict:
        """Cell summary: exact mean output tokens and cap-hit rate when recorded, else derived from the records."""
        recorded = self.recorded_summary(model, suite, cap)
        if recorded is not None:
            return recorded
        rows = self.rows(model, suite, cap)
        return {'mean_output_tokens': float(np.mean([r['mean_tokens'] for r in rows])),
                'cap_hit_rate': sum(r['cap_hits'] for r in rows) / sum(r['n'] for r in rows)}

    def scores(self, model: str, suite: str, cap: int) -> Scores:
        """Question scores, checked against the cell summary's mean_pass1 and question count when one is recorded."""
        out = question_scores(self.rows(model, suite, cap))
        want = self.recorded_summary(model, suite, cap) or {}
        if 'mean_pass1' in want and (abs(sum(out.values()) / len(out) - want['mean_pass1']) > 1e-9
                                     or len(out) != want['problems']):
            raise ValueError(f'records disagree with their cell summary: {model} {suite} cap {cap}')
        return out


def level_row(rec: Records, model: str, cap: int) -> dict:
    """Suite, domain and Total scores, mean output tokens per domain and the mean cap-hit rate of one model at one
    cap. A domain (or Total) whose suites are not all present is None."""
    present = [x for x in SIX if rec.has(model, x, cap)]
    s = {x: (100 * np.mean(list(rec.scores(model, x, cap).values())) if x in present else None) for x in SIX}
    dom = {d: (float(np.mean([s[x] for x in v])) if all(s[x] is not None for x in v) else None)
           for d, v in DOMAINS.items()}
    summ = {x: rec.summary(model, x, cap) for x in present}
    tok = {d: (float(np.mean([summ[x]['mean_output_tokens'] for x in DOMAINS[d]]))
               if all(x in summ for x in DOMAINS[d]) else None) for d in ('math', 'code', 'if')}
    hits = float(np.mean([summ[x]['cap_hit_rate'] for x in SIX])) if len(summ) == len(SIX) else None
    return {'model': model, 'cap': cap, **s, **dom, 'math_tokens': tok['math'], 'code_tokens': tok['code'],
            'if_tokens': tok['if'], 'cap_hit_rate': hits}


def paired(xa: Scores, xb: Scores, rng: np.random.Generator, boot: bool = True, replicates: int = BOOTSTRAP_B):
    """Point difference (pp) of two models on one suite, and its bootstrap replicates (questions resampled with the
    same indices for both models)."""
    if set(xa) != set(xb):
        raise ValueError('question ids differ')
    ids = sorted(xa)
    va, vb = np.array([xa[i] for i in ids]), np.array([xb[i] for i in ids])
    d = 100 * (va.mean() - vb.mean())
    if not boot:
        return d, None
    idx = rng.integers(0, len(ids), size=(replicates, len(ids)))
    return d, 100 * (va[idx].mean(axis=1) - vb[idx].mean(axis=1))


def contrast(rec_a: Records, a: str, rec_b: Records, b: str, cap: int, suites: Sequence[str], boot: bool = True,
             rng: Optional[np.random.Generator] = None, replicates: int = BOOTSTRAP_B):
    """(delta, [lo, hi] or None, replicates) of a - b averaged over `suites`; each suite is resampled in turn from one
    generator, a fresh one seeded 20260923 unless `rng` is given."""
    rng = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    pt, draws = 0.0, np.zeros(replicates)
    for s in suites:
        d, bt = paired(rec_a.scores(a, s, cap), rec_b.scores(b, s, cap), rng, boot, replicates)
        pt += d / len(suites)
        if boot:
            draws += bt / len(suites)
    ci = [float(v) for v in np.percentile(draws, [2.5, 97.5])] if boot else None
    return float(pt), ci, draws
