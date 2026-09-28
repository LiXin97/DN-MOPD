#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upload one exported DN-MOPD model and its model card to the Hugging Face Hub.

The default is a DRY RUN: the script checks the export directory and the card, and prints exactly what would be
uploaded. Nothing is sent unless ``--yes`` is given, and ``--dry-run`` always wins over ``--yes``.

Usage::

    python hf/make_cards.py                                   # writes hf/cards/<name>.md
    python hf/upload.py --export-dir /path/to/hf_export --repo-id <user>/DN-MOPD-Qwen3.5-9B \\
        --card hf/cards/DN-MOPD-Qwen3.5-9B.md                 # dry run: checks and plan only
    python hf/upload.py ... --yes                             # really upload (private repo by default)
    python hf/upload.py ... --yes --public                    # create the repo as public

What is uploaded: every weight/config/tokenizer file of the export directory, plus the card as ``README.md``.
The export's own ``README.md`` (the base model's card, if present) and ``.gitattributes`` are never uploaded.
The export directory is never modified. The files are linked into a staging directory, and a resumable
``upload_large_folder`` sends them from there.

Authentication: ``HF_TOKEN`` or a prior ``huggingface-cli login``. The token is never printed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
ALLOWED_SUFFIXES = {".json", ".safetensors", ".jinja", ".txt", ".model", ".tiktoken"}
ALLOWED_NAMES = {"LICENSE", "NOTICE", "merges.txt", "vocab.json"}
NEVER_UPLOAD = {"README.md", ".gitattributes"}
FORBIDDEN_PATTERNS = [  # training state or local artefacts that must never end up on the Hub
    r"optim", r"\.pt$", r"\.pth$", r"\.bin$", r"rng_state", r"scheduler", r"trainer_state", r"\.log$", r"events\.out",
]
TEXT_SCAN = [  # local paths or secrets in the small text files would leak through the upload
    (re.compile(r"(^|[\"'\s=:])/(mnt|home|root|tmp|data|scratch|opt)/"), "absolute local path"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "Hugging Face token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}"), "API key"),
]
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def safetensors_keys(path: Path) -> List[str]:
    """Tensor names from a safetensors header (no tensor data is read)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def check_export(export: Path, card: Path, repo_id: str) -> Tuple[List[Path], List[str], List[str]]:
    """(files to upload, errors, notes). Errors block the upload; notes are informational."""
    errors: List[str] = []
    notes: List[str] = []
    if not REPO_ID.match(repo_id):
        errors.append(f"invalid repo id {repo_id!r} (expected <user-or-org>/<name>)")
    if not export.is_dir():
        return [], [f"export directory not found: {export}"], notes
    for name in REQUIRED:
        if not (export / name).is_file():
            errors.append(f"missing {name}")
    if not card.is_file():
        errors.append(f"model card not found: {card} (run hf/make_cards.py first)")
    else:
        text = card.read_text()
        if not text.startswith("---\n") or "license:" not in text.split("---", 2)[1]:
            errors.append("model card has no YAML front matter with a license")
        if "{{" in text:
            errors.append("model card still contains template placeholders")
        if repo_id.split("/")[-1] not in text:
            notes.append("the card does not mention the repo name; check that card and repo id belong together")

    files: List[Path] = []
    for p in sorted(export.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(export).as_posix()
        if rel in NEVER_UPLOAD:
            notes.append(f"skipped {rel} (replaced by the model card)" if rel == "README.md" else f"skipped {rel}")
            continue
        if any(re.search(pat, rel) for pat in FORBIDDEN_PATTERNS):
            errors.append(f"refusing to upload training/local artefact: {rel}")
            continue
        if "/" in rel or not (p.suffix in ALLOWED_SUFFIXES or p.name in ALLOWED_NAMES):
            errors.append(f"unexpected file (not a weight, config or tokenizer file): {rel}")
            continue
        files.append(p)

    # Weight index consistency: every referenced shard is present, and no shard is left out of the index.
    index_path = export / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        missing = [s for s in shards if not (export / s).is_file()]
        if missing:
            errors.append(f"index references missing shard(s): {missing}")
        present = sorted(p.name for p in export.glob("*.safetensors"))
        extra = [s for s in present if s not in shards]
        if extra:
            errors.append(f"safetensors file(s) not in the index: {extra}")
        if not missing:
            found = set()
            for s in shards:
                found.update(safetensors_keys(export / s))
            if found != set(weight_map):
                errors.append(f"index and shard headers disagree on {len(found ^ set(weight_map))} tensor name(s)")
        n_mtp = sum(k.startswith("mtp.") for k in weight_map)
        notes.append(f"{len(weight_map)} tensors in {len(shards)} shard(s); mtp.* tensors: {n_mtp} "
                     "(the released exports omit the base model's 15 mtp.* tensors)")
    cfg_path = export / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        notes.append(f"architectures={cfg.get('architectures')} model_type={cfg.get('model_type')}")

    # Small text files must not leak local paths or secrets.
    for p in files:
        if p.suffix == ".safetensors" or p.stat().st_size > 50_000_000:
            continue
        text = p.read_text(errors="replace")
        for pat, why in TEXT_SCAN:
            if pat.search(text):
                errors.append(f"{p.name}: contains {why}")
    return files, errors, notes


def stage(files: List[Path], card: Path, staging: Path) -> None:
    """Symlink the export files and the card (as README.md) into an empty staging directory."""
    staging.mkdir(parents=True, exist_ok=True)
    for p in files:
        (staging / p.name).symlink_to(p.resolve())
    shutil.copyfile(card, staging / "README.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--export-dir", type=Path, required=True, help="HF-format export directory of one model")
    ap.add_argument("--repo-id", required=True, help="target model repo, e.g. <user>/DN-MOPD-Qwen3.5-9B")
    ap.add_argument("--card", type=Path, required=True, help="model card generated by hf/make_cards.py")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    ap.add_argument("--revision", default=None, help="branch to upload to (default: main)")
    ap.add_argument("--staging-dir", type=Path, default=None, help="where to build the staging directory")
    ap.add_argument("--dry-run", action="store_true", help="only check and print the plan (the default)")
    ap.add_argument("--yes", action="store_true", help="really upload; without it the script is a dry run")
    args = ap.parse_args()

    files, errors, notes = check_export(args.export_dir, args.card, args.repo_id)
    total = sum(p.stat().st_size for p in files)
    live = args.yes and not args.dry_run

    print(f"repo        : {args.repo_id} ({'public' if args.public else 'private'}, model)")
    print(f"export dir  : {args.export_dir}")
    print(f"model card  : {args.card} -> README.md")
    print(f"files       : {len(files)} + README.md, {human(total)}")
    for p in files:
        print(f"  {p.name:<45} {human(p.stat().st_size):>12}")
    for n in notes:
        print(f"note        : {n}")
    for e in errors:
        print(f"ERROR       : {e}")
    if errors:
        print("Refusing to upload: fix the errors above.")
        return 1
    if not live:
        print("DRY RUN: nothing was uploaded. Re-run with --yes to upload.")
        return 0

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 (only needed for a real upload)
    except ImportError:
        print("ERROR       : huggingface_hub is not installed (pip install 'huggingface_hub>=0.25')")
        return 1
    api = HfApi()
    staging = args.staging_dir or Path(tempfile.mkdtemp(prefix="dn_mopd_upload_"))
    if staging.exists() and any(p for p in staging.iterdir() if p.name != ".cache"):
        print(f"ERROR       : staging directory {staging} is not empty")
        return 1
    stage(files, args.card, staging)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, folder_path=str(staging), repo_type="model",
                            revision=args.revision, private=not args.public)
    print(f"uploaded {len(files) + 1} files to https://huggingface.co/{args.repo_id}")
    print(f"staging directory (symlinks and upload cache; safe to delete): {staging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
