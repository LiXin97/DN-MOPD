# SPDX-License-Identifier: Apache-2.0
"""Evaluation tests (CPU only; no test generates with a GPU). Puts the repository root on sys.path.

Tests that need optional pieces skip with a reason when they are absent:
  math-verify                       math grader tests
  python -m eval.external.fetch  LiveCodeBench and IF checker tests
  python -m eval.data.prepare       tests on the rebuilt catalogs and the decoded LiveCodeBench tests
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for path in (str(REPO), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)
