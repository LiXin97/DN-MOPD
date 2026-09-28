#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a DN-MOPD environment against the versions used for the paper.

Prints the Python, package, CUDA, driver and GPU versions, and warns when something differs from the pins in
``env/requirements-{train,eval}.txt`` (the single source of truth; this script only reads them).

Usage::

    python env/check_env.py                 # detect the environment from the installed packages
    python env/check_env.py --env train     # or --env eval
    python env/check_env.py --strict        # exit with status 1 if there is any warning
    python env/check_env.py --no-torch      # skip importing torch (faster; no CUDA/cuDNN details)

Only the standard library is required; torch is imported only for the CUDA details.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

EXPECTED_PYTHON = {"train": (3, 12), "eval": (3, 10)}
# Minimum NVIDIA driver branch for the CUDA runtime bundled in the wheels (CUDA 13.0 -> R580, CUDA 12.8 -> R570).
MIN_DRIVER = {"train": 580, "eval": 570}
MIN_GLIBC = {"train": (2, 34), "eval": (2, 28)}
# Packages whose mismatch most likely changes numerics or breaks the run; listed first in the report.
KEY_PACKAGES = {
    "train": ["torch", "sglang", "sglang-router", "transformers", "ray", "flashinfer-python", "flash-attn-4",
              "flash-linear-attention", "numpy", "math-verify"],
    "eval": ["vllm", "torch", "transformers", "flashinfer-python", "numpy", "math-verify", "nltk"],
}


def norm(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def base_version(version: str) -> str:
    """Drop a local version label ("2.11.0+cu130" -> "2.11.0")."""
    return version.split("+", 1)[0]


def read_pins(path: Path) -> Dict[str, str]:
    """{normalized name: version} for every ``name[extras]==version`` line of a requirements file."""
    pins: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;]+)", line)
        if m:
            pins[norm(m.group(1))] = m.group(3)
    return pins


def installed_version(name: str) -> Optional[str]:
    """Installed version, or None. An editable SGLang install reports 0.0.0; read its generated version file."""
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    if norm(name) == "sglang" and base_version(version) in ("0.0.0", "0.0.0.dev0"):
        spec = importlib.util.find_spec("sglang")
        if spec and spec.origin:
            vfile = Path(spec.origin).parent / "_version.py"
            if vfile.exists():
                m = re.search(r"__version__\s*=\s*(?:version\s*=\s*)?['\"]([^'\"]+)['\"]", vfile.read_text())
                if m:
                    return m.group(1) + " (editable)"
    return version


def detect_env() -> str:
    if installed_version("vllm"):
        return "eval"
    if installed_version("sglang"):
        return "train"
    return "train" if sys.version_info[:2] == EXPECTED_PYTHON["train"] else "eval"


def run(cmd: List[str]) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def gpu_rows() -> List[Dict[str, str]]:
    out = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader,nounits"])
    rows = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            rows.append(dict(zip(["index", "name", "memory_mib", "driver", "compute_cap"], parts)))
    return rows


def torch_report(warn) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    try:
        import torch  # noqa: PLC0415 (optional, heavy)
    except Exception as exc:  # pragma: no cover - depends on the machine
        warn(f"torch cannot be imported: {exc!r}")
        return lines
    lines.append(("torch", f"{torch.__version__} (CUDA build {torch.version.cuda})"))
    try:
        lines.append(("cuDNN", str(torch.backends.cudnn.version())))
    except Exception as exc:  # a mismatched cuDNN raises here
        warn(f"cuDNN cannot be initialised: {str(exc).splitlines()[0]}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        caps = sorted({f"sm_{a}{b}" for a, b in (torch.cuda.get_device_capability(i) for i in range(n))})
        lines.append(("CUDA devices (torch)", f"{n} ({', '.join(caps)})"))
    else:
        warn("torch.cuda.is_available() is False: no usable GPU or driver")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", choices=["train", "eval", "auto"], default="auto")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 on any warning")
    ap.add_argument("--no-torch", action="store_true", help="do not import torch")
    args = ap.parse_args()

    env = detect_env() if args.env == "auto" else args.env
    warnings: List[str] = []
    warn = warnings.append
    report: List[Tuple[str, str]] = []

    # Python, OS, glibc
    py = sys.version_info[:3]
    report.append(("environment", env))
    report.append(("python", f"{py[0]}.{py[1]}.{py[2]} ({sys.executable})"))
    if py[:2] != EXPECTED_PYTHON[env]:
        warn(f"python {py[0]}.{py[1]} differs from the paper's {EXPECTED_PYTHON[env][0]}.{EXPECTED_PYTHON[env][1]}")
    report.append(("platform", f"{platform.system()} {platform.machine()}"))
    libc, libc_ver = platform.libc_ver()
    if libc == "glibc" and libc_ver:
        report.append(("glibc", libc_ver))
        if tuple(int(x) for x in libc_ver.split(".")[:2]) < MIN_GLIBC[env]:
            warn(f"glibc {libc_ver} < {'.'.join(map(str, MIN_GLIBC[env]))} (wheels of this environment need it)")

    # Packages
    req = HERE / f"requirements-{env}.txt"
    pins = read_pins(req) if req.exists() else {}
    if not pins:
        warn(f"no pins found in {req.name}")
    order = [norm(k) for k in KEY_PACKAGES[env]] + sorted(k for k in pins if k not in map(norm, KEY_PACKAGES[env]))
    pkg_lines = []
    for name in order:
        want = pins.get(name)
        have = installed_version(name)
        if have is None:
            status = "MISSING"
            if want:
                warn(f"{name} is not installed (pinned: {want})")
        elif want and base_version(have.split(" ")[0]) != base_version(want):
            status = f"MISMATCH (pinned: {want})"
            warn(f"{name} {have} != {want}")
        else:
            status = "ok"
        pkg_lines.append((name, have or "-", status))

    # This repository's packages
    repo_pkgs = [("dn_mopd", "dn_mopd")] + ([("miles", "miles")] if env == "train" else [])
    for label, module in repo_pkgs:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            warn(f"{label} is not importable; install it with pip install --no-deps -e "
                 f"{'./miles' if module == 'miles' else '.'}")
            report.append((label, "not importable"))
            continue
        origin = Path(spec.origin).resolve()
        inside = REPO in origin.parents
        report.append((label, f"{'this repository' if inside else 'OUTSIDE this repository'}: {origin.parent}"))
        if not inside:
            warn(f"{label} is imported from {origin.parent}, not from this repository "
                 f"(the PyPI project named 'miles' is unrelated)" if module == "miles" else
                 f"{label} is imported from {origin.parent}, not from this repository")

    # GPUs and driver
    gpus = gpu_rows()
    if gpus:
        for g in gpus:
            report.append((f"gpu {g['index']}", f"{g['name']}, {int(float(g['memory_mib'])) // 1024} GiB, "
                                                f"sm_{g['compute_cap'].replace('.', '')}, driver {g['driver']}"))
        major = int(gpus[0]["driver"].split(".")[0])
        if major < MIN_DRIVER[env]:
            warn(f"driver {gpus[0]['driver']} is older than R{MIN_DRIVER[env]}, which the bundled CUDA runtime needs")
        if len(gpus) < 8:
            warn(f"{len(gpus)} GPU(s) visible; the paper's recipes assume one node with 8 GPUs")
        small = [g for g in gpus if float(g["memory_mib"]) < 79 * 1024]
        if small:
            warn(f"{len(small)} GPU(s) have < 80 GB; see env/install.md for memory caveats")
    else:
        warn("nvidia-smi not found or no GPU visible")

    if not args.no_torch:
        report.extend(torch_report(warn))

    # CUDA 13 libraries bundled with torch must be findable by processes started outside Ray (train only).
    if env == "train":
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            cu13 = Path(spec.origin).parent.parent / "nvidia" / "cu13" / "lib"
            ld = os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if cu13.is_dir() and str(cu13) not in ld:
                warn(f"{cu13} is not on LD_LIBRARY_PATH; teacher servers started outside Ray may fail "
                     f"to load libnvrtc.so.13 (see env/install.md)")

    # Output
    width = max(len(k) for k, _ in report) + 2
    print("DN-MOPD environment check")
    print("=" * 25)
    for k, v in report:
        print(f"{k:<{width}}{v}")
    print()
    nw = max(len(n) for n, _, _ in pkg_lines) + 2 if pkg_lines else 10
    vw = max(len(v) for _, v, _ in pkg_lines) + 2 if pkg_lines else 10
    print(f"{'package':<{nw}}{'installed':<{vw}}status")
    for n, v, s in pkg_lines:
        print(f"{n:<{nw}}{v:<{vw}}{s}")
    print()
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("No warnings: the environment matches the pinned versions.")
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
