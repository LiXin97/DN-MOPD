# SPDX-License-Identifier: Apache-2.0
"""Recipe shell scripts: bash syntax, shellcheck (when installed), headers, and argument errors (no GPU needed)."""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import RECIPES

SCRIPTS = sorted(RECIPES.glob("*.sh"))
ENTRY = [p for p in SCRIPTS if p.name != "common.sh"]


def test_every_recipe_script_is_present():
    names = {p.name for p in SCRIPTS}
    assert {"train_teacher_grpo.sh", "train_student.sh", "gen_seqkd.sh", "train_seqkd_sft.sh", "export_hf.sh",
            "start_teachers.sh", "start_code_judge.sh", "build_injection_bank.sh", "smoke_test.sh",
            "common.sh"} <= names
    assert (RECIPES / "merge.py").is_file()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    out = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_headers_and_strict_mode(script):
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert lines[1] == "# SPDX-License-Identifier: Apache-2.0"
    if script.name != "common.sh":
        assert "set -euo pipefail" in text or "set -uo pipefail" in text
        assert 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"' in text


def _shellcheck():
    exe = shutil.which("shellcheck") or os.path.join(os.path.dirname(sys.executable), "shellcheck")
    return exe if os.path.exists(exe) else None


def test_shellcheck_clean():
    exe = _shellcheck()
    if exe is None:
        pytest.skip("shellcheck is not installed")
    out = subprocess.run([exe, "-x", "-S", "style", *[p.name for p in SCRIPTS]], cwd=RECIPES, capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def _run(script, *args, **env):
    e = dict(os.environ, PYTHON=sys.executable, **env)
    return subprocess.run(["bash", str(RECIPES / script), *args], capture_output=True, text=True, env=e, timeout=300)


@pytest.mark.parametrize("script,args,msg", [
    ("train_student.sh", (), "usage"),
    ("train_student.sh", ("nope", "2b"), "unknown METHOD"),
    ("train_student.sh", ("dn_mopd", "7b"), "SIZE must be"),
    ("train_student.sh", ("dn_mopd", "2b", "x"), "SEED must be"),
    ("train_teacher_grpo.sh", ("physics", "2b"), "DOMAIN must be"),
    ("train_teacher_grpo.sh", ("math",), "usage"),
    ("export_hf.sh", (), "usage"),
    ("gen_seqkd.sh", ("1b",), "SIZE must be"),
    ("train_seqkd_sft.sh", (), "usage"),
    ("build_injection_bank.sh", ("3b",), "SIZE must be"),
    ("start_teachers.sh", (), "usage"),
])
def test_argument_errors(script, args, msg):
    out = _run(script, *args)
    assert out.returncode != 0 and msg in out.stderr, (out.returncode, out.stderr[-400:])


def test_student_reports_missing_inputs(tmp_path):
    out = _run("train_student.sh", "dn_mopd", "2b", BASE_MODEL=str(tmp_path), DATA_ROOT=str(tmp_path),
               OUT_ROOT=str(tmp_path / "o"), CKPT_ROOT=str(tmp_path / "c"))
    assert out.returncode != 0 and "missing student data" in out.stderr
    (tmp_path / "student").mkdir()
    (tmp_path / "student" / "train_3domain.jsonl").write_text("{}\n")
    out = _run("train_student.sh", "dn_mopd", "2b", BASE_MODEL=str(tmp_path), DATA_ROOT=str(tmp_path),
               OUT_ROOT=str(tmp_path / "o"), CKPT_ROOT=str(tmp_path / "c"))
    assert out.returncode != 0 and "teacher" in out.stderr and "TEACHER_MATH" in out.stderr


def test_common_helpers(tmp_path):
    script = f"""
set -euo pipefail
source "{RECIPES}/common.sh"
check_size 4b
echo "ray=$(ray_address 2001:db8::1 6379) $(ray_address 198.51.100.1 6379)"
echo "teacher=$(TEACHER_IF=/x teacher_export 2b ifeval) $(CKPT_ROOT=/c teacher_export 9b math)"
mkdir -p {tmp_path}/run/iter_0000010 {tmp_path}/run/iter_0000020
touch {tmp_path}/run/iter_0000010/meta.json
echo 10 > {tmp_path}/run/latest_checkpointed_iteration.txt
prune_partial_saves {tmp_path}/run
iter_done {tmp_path}/run 10 && echo done10
iter_done {tmp_path}/run 20 || echo notdone20
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=dict(os.environ, PYTHON=sys.executable))
    assert out.returncode == 0, out.stderr
    assert "ray=[2001:db8::1]:6379 198.51.100.1:6379" in out.stdout
    assert "teacher=/x /c/qwen3.5-9b_teacher_math_hf" in out.stdout
    assert "done10" in out.stdout and "notdone20" in out.stdout
    assert not (tmp_path / "run" / "iter_0000020").exists() and (tmp_path / "run" / "iter_0000010").exists()


def test_gpus_free_requires_every_listed_gpu_idle(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/sh\nprintf '0, 3\\n1, 900\\n2, 40000\\n3, 0\\n'\n")
    (fake / "nvidia-smi").chmod(0o755)
    env = dict(os.environ, PYTHON=sys.executable, PATH=f"{fake}:{os.environ['PATH']}")
    run = lambda gpus: subprocess.run(["bash", "-c", f'source "{RECIPES}/common.sh"; gpus_free {gpus}'],
                                      capture_output=True, text=True, env=env)
    assert run("0,1,3").returncode == 0
    assert run("0,2").returncode != 0                      # GPU 2 is busy
    assert run("3,7").returncode != 0                      # GPU 7 does not exist
    (fake / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n")      # a driver that cannot be queried
    out = run("0")
    assert out.returncode != 0 and "nvidia-smi failed" in out.stderr
