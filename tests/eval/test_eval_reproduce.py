# SPDX-License-Identifier: Apache-2.0
"""reproduce/reproduce.py recomputes the paper's values from the released records and prints ALL MATCH.

The point estimates run by default (seconds); DN_MOPD_SLOW_TESTS=1 also runs the full paired bootstrap (a few
minutes). The script runs on a copy so that its results/ directory is written outside the repository.
"""
import os
import shutil
import subprocess
import sys

import pytest
from evaltest_helpers import REPO


def run(tmp_path, *args):
    shutil.copy(REPO / 'reproduce' / 'reproduce.py', tmp_path / 'reproduce.py')
    (tmp_path / 'data').symlink_to(REPO / 'reproduce' / 'data')
    return subprocess.run([sys.executable, str(tmp_path / 'reproduce.py'), *args], capture_output=True, text=True)


def test_point_estimates_all_match(tmp_path):
    r = run(tmp_path, '--no-bootstrap')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'ALL MATCH' in r.stdout
    assert (tmp_path / 'results' / 'levels.csv').is_file()


@pytest.mark.skipif(os.environ.get('DN_MOPD_SLOW_TESTS') != '1', reason='set DN_MOPD_SLOW_TESTS=1 (a few minutes)')
def test_full_bootstrap_all_match(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'checked 1440 values' in r.stdout and 'ALL MATCH' in r.stdout
