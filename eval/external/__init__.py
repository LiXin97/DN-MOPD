# SPDX-License-Identifier: Apache-2.0
"""Pinned third-party graders (fetched by `python -m eval.external.fetch`, never vendored).

PINS.json lists, for each source, the repository, the commit and the sha256 of every file used. The .py hashes are the
hashes of the grader sources behind the paper's numbers. `require(name)` returns the local directory of a source after
checking every pinned file; a missing or modified file is an error, never a silent fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from eval.common import digest, third_party_dir

PINS_FILE = Path(__file__).resolve().parent / 'PINS.json'
SOURCES = ('LiveCodeBench', 'google-research', 'IFBench', 'nltk_data')
_VERIFIED: Dict[str, Path] = {}


def pins() -> dict:
    return {k: v for k, v in json.loads(PINS_FILE.read_text()).items() if not k.startswith('_')}


def require(name: str) -> Path:
    """Local directory of a pinned source, verified file by file against PINS.json (cached per process)."""
    if name in _VERIFIED:
        return _VERIFIED[name]
    spec = pins()[name]
    root = third_party_dir() / name
    missing, changed = [], []
    for rel, want in spec['files'].items():
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif digest(path) != want:
            changed.append(rel)
    if missing or changed:
        raise RuntimeError(f'{name} at {root} is not the pinned copy (missing {missing[:3]}, changed {changed[:3]}); '
                           f'run: python -m eval.external.fetch --only {name}')
    _VERIFIED[name] = root
    return root
