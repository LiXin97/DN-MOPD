# SPDX-License-Identifier: Apache-2.0
"""Paths, hashing and small file helpers shared by the evaluation tools.

Locations (each can be overridden by an environment variable):
  DN_MOPD_ROOT          repository root (default: the parent of this package)
  DN_MOPD_THIRD_PARTY   fetched official graders and nltk data (default: eval/third_party)
  DN_MOPD_EVAL_DATA     prepared question catalogs and LiveCodeBench tests (default: eval/data/prepared)
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, List

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('DN_MOPD_ROOT', EVAL_DIR.parent)).resolve()


def third_party_dir() -> Path:
    """Directory that eval/external/fetch.py fills with the pinned official graders."""
    return Path(os.environ.get('DN_MOPD_THIRD_PARTY', EVAL_DIR / 'third_party')).resolve()


def prepared_dir() -> Path:
    """Directory that eval/data/prepare.py fills with the complete catalogs and the LiveCodeBench tests."""
    return Path(os.environ.get('DN_MOPD_EVAL_DATA', EVAL_DIR / 'data' / 'prepared')).resolve()


def released_dir() -> Path:
    """The released per-question records and question catalogs (reproduce/data)."""
    return REPO_ROOT / 'reproduce' / 'data'


def digest(path: os.PathLike) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def records(path: os.PathLike) -> List[dict]:
    """Rows of a JSON-lines file (blank lines ignored)."""
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_text(rows: Iterable[dict]) -> str:
    """The catalog serialization: one json.dumps(row, ensure_ascii=False) per line."""
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)


def write_text_atomic(path: os.PathLike, text: str) -> None:
    """Write a file through a unique temporary name and fsync, so a crash never leaves a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path: os.PathLike, value: Any) -> None:
    """JSON with indent=2 and a trailing newline (the byte format of the paper's cell files)."""
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
