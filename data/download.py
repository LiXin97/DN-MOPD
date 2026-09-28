#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download the released training prompt sets from the Hugging Face Hub and verify them against data/MANIFEST.json.

The four files live in the public dataset XINLI1997/DN-MOPD-Data under the same paths as in data/:
  student/train_3domain.jsonl   2,700 prompts (every student run)          46 MB
  teacher/math_train.jsonl      8,000 prompts (GRPO math teachers)        3.8 MB
  teacher/code_train.jsonl      8,000 prompts (GRPO code teachers)        917 MB
  teacher/if_train.jsonl        8,000 prompts (GRPO IF teachers)          3.7 MB

The dataset revision is pinned in data/MANIFEST.json ("hub"), and every file is checked against the sha256 listed
there. A file that is already present with the right hash is not downloaded again. data/build/build_data.py rebuilds
the same bytes from the public upstream datasets instead.

usage:
  python data/download.py                                   # all four files
  python data/download.py --only student/train_3domain.jsonl --only teacher/math_train.jsonl
  python data/download.py --check                           # verify the local files only (no network)
  HF_ENDPOINT=https://hf-mirror.example python data/download.py   # a Hub mirror

Backends: huggingface_hub when it is installed (the training environment has it), else plain HTTPS from
$HF_ENDPOINT/datasets/<repo>/resolve/<revision>/<path>; `--backend` forces one. The exit code is 0 when every
requested file matches the manifest, 1 when a file is missing, fails to download or has a different sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent
STAGING = ".download"          # per-output-directory staging area; removed after a successful run
CHUNK = 1 << 22


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"{endpoint()}/datasets/{repo_id}/resolve/{revision}/{path}"


def download_http(repo_id: str, revision: str, path: str, dest: Path) -> None:
    """Stream one file over plain HTTP(S) into `dest`, resuming a partial `dest.part` when the server allows it."""
    url = resolve_url(repo_id, revision, path)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "dn-mopd-data-download/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:          # the partial file is already complete (or stale): start over
            part.unlink()
            return download_http(repo_id, revision, path, dest)
        raise
    with resp:
        resumed = have and resp.status == 206
        total = resp.headers.get("Content-Length")
        total = int(total) + (have if resumed else 0) if total is not None else None
        done = have if resumed else 0
        print(f"  GET {url}" + (f" ({total / 1e6:.1f} MB)" if total else ""), flush=True)
        next_report = 0.1
        with open(part, "ab" if resumed else "wb") as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and total > 50e6 and done / total >= next_report:
                    print(f"  {path}: {100 * done / total:.0f}%", flush=True)
                    next_report += 0.1
    part.replace(dest)


def download_hub(repo_id: str, revision: str, path: str, dest: Path) -> None:
    """One file through huggingface_hub (honours HF_ENDPOINT, HF_TOKEN and resumes partial downloads)."""
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_root = dest.parents[len(Path(path).parts) - 1]     # dest = <staging_root>/<path>
    got = Path(hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset", revision=revision,
                               local_dir=str(staging_root), endpoint=endpoint()))
    if got.resolve() != dest.resolve():
        shutil.move(str(got), str(dest))


def pick_backend(name: str) -> str:
    if name != "auto":
        return name
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return "http"
    return "hub"


def normalise(rel: str) -> str:
    rel = rel.strip().lstrip("./")
    return rel[len("data/"):] if rel.startswith("data/") else rel


def check_remote_manifest(fetch, staging: Path, wanted: List[str], files: Dict[str, dict]) -> Optional[str]:
    """Compare the dataset revision's own MANIFEST.json with the local one for the requested files. Returns an error
    message, or None. A revision without a MANIFEST.json is accepted: every file is still checked by hash."""
    dest = staging / "MANIFEST.remote.json"
    try:
        fetch("MANIFEST.json", dest)
    except Exception as e:  # noqa: BLE001 - reported, not fatal: the per-file hashes are checked anyway
        print(f"note: could not read the revision's MANIFEST.json ({type(e).__name__}); checking files by hash only")
        return None
    try:
        remote = json.loads(dest.read_text(encoding="utf-8")).get("files", {})
    except (ValueError, AttributeError):
        return "the dataset revision's MANIFEST.json is not valid JSON"
    for rel in wanted:
        theirs = (remote.get(rel) or {}).get("sha256")
        if theirs is not None and theirs != files[rel]["sha256"]:
            return (f"this dataset revision lists sha256 {theirs} for {rel}, but data/MANIFEST.json expects "
                    f"{files[rel]['sha256']}; use the pinned revision (the default)")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None, metavar="PATH",
                    help="a file to fetch, relative to data/ (repeatable); default: every file in the manifest")
    ap.add_argument("--revision", default=None, help="dataset revision (default: the one pinned in the manifest)")
    ap.add_argument("--repo-id", default=None, help="dataset repository (default: the one in the manifest)")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR, help="where the files go (default: data/)")
    ap.add_argument("--manifest", type=Path, default=DATA_DIR / "MANIFEST.json", help=argparse.SUPPRESS)
    ap.add_argument("--backend", choices=("auto", "hub", "http"), default="auto",
                    help="huggingface_hub or plain HTTP (default: huggingface_hub when installed)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the local files only; download nothing")
    mode.add_argument("--force", action="store_true", help="download again even when a local file matches")
    a = ap.parse_args(argv)

    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    files: Dict[str, dict] = manifest["files"]
    hub = manifest.get("hub", {})
    repo_id = a.repo_id or hub.get("repo_id")
    revision = a.revision or hub.get("revision")
    if not a.check and not (repo_id and revision):
        ap.error("the manifest pins no dataset repository/revision; pass --repo-id and --revision")
    wanted = [normalise(p) for p in a.only] if a.only else list(files)
    unknown = sorted(set(wanted) - set(files))
    if unknown:
        ap.error(f"not in the manifest: {', '.join(unknown)}; choose from {', '.join(files)}")

    out_dir: Path = a.out_dir
    status: Dict[str, str] = {}
    todo: List[str] = []
    for rel in wanted:
        path = out_dir / rel
        if path.exists() and not a.force:
            if path.stat().st_size == files[rel]["bytes"] and sha256_file(path) == files[rel]["sha256"]:
                status[rel] = "OK"
                continue
            if a.check:
                status[rel] = "MISMATCH"
                continue
        elif a.check:
            status[rel] = "MISSING"
            continue
        todo.append(rel)

    if todo:
        backend = pick_backend(a.backend)
        fetch_one = download_hub if backend == "hub" else download_http
        staging = out_dir / STAGING
        staging.mkdir(parents=True, exist_ok=True)
        print(f"dataset {repo_id} @ {revision} via {backend} from {endpoint()}", flush=True)
        if revision != hub.get("revision"):
            print(f"note: revision {revision} is not the pinned {hub.get('revision')}; files are still checked "
                  "against data/MANIFEST.json", flush=True)

        def fetch(rel: str, dest: Path) -> None:
            fetch_one(repo_id, revision, rel, dest)

        err = check_remote_manifest(fetch, staging, todo, files)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        for rel in todo:
            staged = staging / rel
            try:
                fetch(rel, staged)
            except Exception as e:  # noqa: BLE001 - any transport error is a failed file, reported below
                print(f"  failed: {rel}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                status[rel] = "FAILED"
                continue
            got = sha256_file(staged)
            if staged.stat().st_size != files[rel]["bytes"] or got != files[rel]["sha256"]:
                print(f"  {rel}: sha256 {got} != {files[rel]['sha256']}", file=sys.stderr, flush=True)
                staged.unlink()
                status[rel] = "MISMATCH"
                continue
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, out_dir / rel)
            status[rel] = "FETCHED"
        if all(status[r] in ("OK", "FETCHED") for r in todo):
            shutil.rmtree(staging, ignore_errors=True)

    for rel in wanted:
        print(f"{status[rel]:9s} {rel}  sha256={files[rel]['sha256'][:16]}...", flush=True)
    bad = [r for r in wanted if status[r] not in ("OK", "FETCHED")]
    if bad:
        hint = "run: python data/download.py" if a.check else "rerun to resume, or rebuild: data/build/build_data.py"
        print(f"{len(bad)} of {len(wanted)} file(s) not verified ({hint})", file=sys.stderr)
        return 1
    print(f"all {len(wanted)} file(s) match data/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
