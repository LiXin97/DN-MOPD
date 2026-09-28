# SPDX-License-Identifier: Apache-2.0
"""Download the pinned official graders and nltk data used for the paper's evaluation, and verify every file.

    python -m eval.external.fetch                 # all sources into eval/third_party/ (or $DN_MOPD_THIRD_PARTY)
    python -m eval.external.fetch --only IFBench  # one source
    python -m eval.external.fetch --check         # verify what is present, download nothing

Sources (commit and per-file sha256 in PINS.json):
  LiveCodeBench     lcb_runner: official code extraction, test decoding and test runner (MIT)
  google-research   instruction_following_eval: the official IFEval checkers (Apache-2.0)
  IFBench           the official IFBench checkers (Apache-2.0)
  nltk_data         punkt, punkt_tab, stopwords and the perceptron taggers used by both IF checkers

Files are fetched from raw.githubusercontent.com at the pinned commit. Nothing is installed; the graders add these
directories to sys.path at load time.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from eval.common import third_party_dir
from eval.external import SOURCES, pins, require


def raw_url(repo_url: str, commit: str, rel: str) -> str:
    owner_repo = repo_url.rstrip('/').split('github.com/')[1]
    return f'https://raw.githubusercontent.com/{owner_repo}/{commit}/{rel}'


def download(url: str, dest: Path, want: str, attempts: int = 4) -> None:
    """Download url to dest and check its sha256; retry transient failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except OSError as exc:
            if attempt == attempts:
                raise RuntimeError(f'download failed: {url}: {exc}') from exc
            time.sleep(2 * attempt)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(f'sha256 mismatch for {url}: got {got}, pinned {want}')
    tmp = dest.with_name(dest.name + '.part')
    tmp.write_bytes(data)
    tmp.replace(dest)


def fetch(name: str, root: Path) -> None:
    spec = pins()[name]
    dest = root / name
    for rel, want in spec['files'].items():
        path = dest / rel
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == want:
            continue
        download(raw_url(spec['url'], spec['commit'], rel), path, want)
    if name == 'nltk_data':          # nltk reads the unpacked directories (e.g. nltk_data/tokenizers/punkt/)
        for rel in spec['files']:
            with zipfile.ZipFile(dest / rel) as z:
                z.extractall(dest / Path(rel).relative_to('packages').parent)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=SOURCES, action='append', help='fetch only these sources')
    ap.add_argument('--check', action='store_true', help='verify the local copies without downloading')
    a = ap.parse_args()
    root = third_party_dir()
    failed = 0
    for name in a.only or SOURCES:
        try:
            if not a.check:
                fetch(name, root)
            path = require(name)
            print(f'PASS {name} {pins()[name]["commit"][:12]} -> {path}')
        except Exception as exc:      # report every source, then fail
            failed += 1
            print(f'FAIL {name}: {exc}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
