#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the released training prompt sets from public sources and check them against data/MANIFEST.json.

This is the from-scratch alternative to `python data/download.py`, which fetches the same bytes from the pinned
Hugging Face dataset revision.

Outputs (paths relative to --out-dir, default: the repository's data/ directory):
  student/train_3domain.jsonl   2,700 prompts: 900 math + 900 code + 900 instruction following (IF)
  teacher/math_train.jsonl      8,000 math prompts   (GRPO math teacher)
  teacher/code_train.jsonl      8,000 code prompts   (GRPO code teacher; about 0.9 GB)
  teacher/if_train.jsonl        8,000 IF prompts     (GRPO IF teacher)

Sources (downloaded from the Hugging Face Hub at pinned revisions, sha256-checked):
  --source upstream (default)  zwhe99/DeepMath-103K (MIT) and PRIME-RL/Eurus-2-RL-Data (MIT)
  --source gopd                Keven16/G-OPD-Training-Data, the intermediate copy the paper's pools were read from;
                               it holds exactly the rows derived below, so both sources give identical outputs
IF prompts are generated locally by gen_ifeval_data.py (no download).

How the pools are derived (verified row by row against the intermediate copy):
  math pool  DeepMath-103K questions in first-occurrence order, deduplicated by question text, kept if any occurrence
             has difficulty >= 6 (57,046 questions); prompt = question + MATH_SUFFIX, label = the final answer of the
             first occurrence
  code pool  the 25,276 Eurus-2-RL-Data train rows with ability == "code", in order; the system message is dropped and
             CODE_SUFFIX appended to the user message; label = reward_model.ground_truth (JSON test cases)
  src_index  the position of a row in its pool

How the released sets are assembled from the pools:
  student    math/code: the rows listed in ids/student_math_code_ids.json (a pass-rate selection under a model outside
             the paper, so it is published as ids), interleaved math, code, math, ...; IF: the 8,000 generated prompts
             shuffled with random.Random(42), first 900; the 2,700 rows are then shuffled with the same generator
  teachers   math/code: the 900 student rows of the domain first, then the pool in order, skipping prompts already
             taken (hash of the message text) and rows without a label, until 8,000; IF: all 8,000 generated prompts

usage:
  pip install pyarrow
  python data/build/build_data.py                                   # everything, check against MANIFEST.json
  python data/build/build_data.py --only teacher/code_train.jsonl   # one file
  python data/build/build_data.py --source gopd --cache-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent
sys.path.insert(0, str(HERE))

MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE_SUFFIX = "\nYou need to think first then write the Python code."
N_TEACHER = 8000
N_STUDENT_PER_DOMAIN = 900
STUDENT_SEED = 42
IF_SEED = 20260803
IF_SOURCE = "synthetic-ifeval-v1"

HF_SOURCES: Dict[str, Dict] = {
    "deepmath": {
        "repo": "zwhe99/DeepMath-103K", "revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "files": {
            "data/train-00000-of-00010.parquet": "d6412432f30425e848a224dc641e681eb1ed51b970d52536eda7daefc01d8c8b",
            "data/train-00001-of-00010.parquet": "eef9d3012456239eb0f4cd462ac7bebb7d6c4f675c41329c680ef8a506ded512",
            "data/train-00002-of-00010.parquet": "20e4dc6527d94c2057a9b727f8b58994395de602e284b3294f7d2451f75d681a",
            "data/train-00003-of-00010.parquet": "c5b20135f93e7890da191973ad77d88e67c95144ccc169ec592489f044e7c38a",
            "data/train-00004-of-00010.parquet": "4153e531d78ca278d2e12e6d08099b66126b26b99ca6a92b3d6f95c156818749",
            "data/train-00005-of-00010.parquet": "110586bdc6f35b0434bccbd582f1e97e8328da0752ccd36590a50178885f0360",
            "data/train-00006-of-00010.parquet": "e39c00ed42a6a1af74ddc042840884a59adcb9139a23f535e399ddf0c292ef76",
            "data/train-00007-of-00010.parquet": "fdeb213b5c2d0bb50f1081ae48b7ce4fa38147efe623de5d6d836a32f4044dad",
            "data/train-00008-of-00010.parquet": "d8e5b1417f0364312896d259efbea0600c1fac22eacb2f71187c5b0e9704f388",
            "data/train-00009-of-00010.parquet": "67639c6620cabce348e91c2cc331c4877307ed1a5d8fbd1c0d7dcca561cea8df",
        },
    },
    "eurus": {
        "repo": "PRIME-RL/Eurus-2-RL-Data", "revision": "9776b13264b5aaa0b16495fcf086a0a8d86fd655",
        "files": {"train.parquet": "16feb28654b54924307ea0b5e866f65aca4980e3b3b891a8ab68ce52b5f460dd"},
    },
    "gopd": {
        "repo": "Keven16/G-OPD-Training-Data", "revision": "c9bc6783733858dd4892f26f95e4aea0942e91f1",
        "files": {
            "DeepMath-103K/train_filtered_level6.parquet":
                "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597",
            "Eurus/code_train.parquet": "ea31611a8f32e2fcf3f24a484b7433868bc47facca1cf950867460b0300f7b7e",
        },
    },
}
OUTPUTS = ("student/train_3domain.jsonl", "teacher/math_train.jsonl", "teacher/code_train.jsonl",
           "teacher/if_train.jsonl")

Row = Dict  # a pool row: {"data_source": str, "prompt": list[dict] | str, "label": str | None}


# ------------------------------------------------------------------------------------------------ downloads
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: str, path: str, cache_dir: Path) -> Path:
    """The pinned file `path` of HF dataset `source`, from the cache or downloaded, sha256-verified."""
    spec = HF_SOURCES[source]
    expected = spec["files"][path]
    local = cache_dir / spec["repo"].replace("/", "__") / spec["revision"] / path
    if local.exists() and sha256_file(local) == expected:
        return local
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{spec['repo']}/resolve/{spec['revision']}/{path}"
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 22)
    got = sha256_file(tmp)
    if got != expected:
        raise SystemExit(f"sha256 mismatch for {spec['repo']}/{path}: {got} != {expected}")
    tmp.replace(local)
    return local


# ------------------------------------------------------------------------------------------------ pools
def _parquet():
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise SystemExit("build_data.py needs pyarrow for the math/code pools: pip install pyarrow") from e
    return pq, pc


def math_pool_upstream(cache_dir: Path) -> List[Row]:
    """DeepMath-103K -> deduplicated questions with max difficulty >= 6, first-occurrence order and answer."""
    pq, _ = _parquet()
    answer: Dict[str, str] = {}
    max_difficulty: Dict[str, float] = {}
    for path in sorted(HF_SOURCES["deepmath"]["files"]):
        table = pq.read_table(fetch("deepmath", path, cache_dir), columns=["question", "final_answer", "difficulty"])
        for r in table.to_pylist():
            q = r["question"]
            if q not in answer:
                answer[q] = r["final_answer"]
                max_difficulty[q] = r["difficulty"]
            else:
                max_difficulty[q] = max(max_difficulty[q], r["difficulty"])
    return [{"data_source": "DeepMath-103K", "prompt": [{"content": q + MATH_SUFFIX, "role": "user"}], "label": a}
            for q, a in answer.items() if max_difficulty[q] >= 6]


def code_pool_upstream(cache_dir: Path) -> List[Row]:
    """Eurus-2-RL-Data train -> the code rows, system message dropped, CODE_SUFFIX appended."""
    pq, pc = _parquet()
    pf = pq.ParquetFile(fetch("eurus", "train.parquet", cache_dir))
    rows: List[Row] = []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["data_source", "prompt", "ability", "reward_model"])
        for r in table.filter(pc.equal(table["ability"], "code")).to_pylist():
            users = [m["content"] for m in r["prompt"] if m["role"] == "user"]
            if len(users) != 1:
                raise SystemExit(f"unexpected Eurus code prompt with {len(users)} user messages")
            rows.append({"data_source": r["data_source"],
                         "prompt": [{"content": users[0] + CODE_SUFFIX, "role": "user"}],
                         "label": (r["reward_model"] or {}).get("ground_truth")})
    return rows


def _gopd_pool(path: str, cache_dir: Path) -> List[Row]:
    pq, _ = _parquet()
    pf = pq.ParquetFile(fetch("gopd", path, cache_dir))
    rows: List[Row] = []
    for rg in range(pf.num_row_groups):   # row groups keep memory bounded on the 1.6 GB code file
        for r in pf.read_row_group(rg, columns=["data_source", "prompt", "reward_model"]).to_pylist():
            rows.append({"data_source": r["data_source"], "prompt": r["prompt"],
                         "label": (r["reward_model"] or {}).get("ground_truth")})
    return rows


POOLS: Dict[str, Dict[str, Callable[[Path], List[Row]]]] = {
    "upstream": {"math": math_pool_upstream, "code": code_pool_upstream},
    "gopd": {"math": lambda c: _gopd_pool("DeepMath-103K/train_filtered_level6.parquet", c),
             "code": lambda c: _gopd_pool("Eurus/code_train.parquet", c)},
}


# ------------------------------------------------------------------------------------------------ records
def record(domain: str, src_index: int, row: Row) -> Dict:
    """A released row. `teacher` routes the prompt to its domain teacher; metadata.domain selects the verifier."""
    return {"prompt": row["prompt"], "label": row["label"], "domain": domain, "teacher": domain,
            "metadata": {"domain": domain, "data_source": row["data_source"], "src_index": src_index}}


def prompt_hash(prompt) -> str:
    """Duplicate key of the teacher pools: sha1 of the concatenated message texts."""
    if isinstance(prompt, list):
        prompt = "".join(str(m.get("content", "")) for m in prompt)
    return hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()


def if_records() -> List[Dict]:
    from gen_ifeval_data import generate
    return [record("ifeval", r["src_index"], {"data_source": IF_SOURCE, "prompt": r["prompt"], "label": r["label"]})
            for r in generate(N_TEACHER, IF_SEED)]


def student_seed_rows(domain: str, pool: List[Row]) -> List[Dict]:
    ids = json.loads((HERE / "ids" / "student_math_code_ids.json").read_text())[domain]
    return [record(domain, i, pool[i]) for i in ids]


def teacher_rows(domain: str, seeds: List[Dict], pool: List[Row], n: int = N_TEACHER) -> List[Dict]:
    rows = list(seeds)
    seen = {prompt_hash(r["prompt"]) for r in rows}
    for idx, row in enumerate(pool):
        if len(rows) >= n:
            break
        key = prompt_hash(row["prompt"])
        if key in seen or row["label"] is None:
            continue
        seen.add(key)
        rows.append(record(domain, idx, row))
    if len(rows) < n:
        raise SystemExit(f"{domain}: only {len(rows)} teacher rows available")
    return rows


def student_rows(math_seeds: List[Dict], code_seeds: List[Dict], if_rows: List[Dict]) -> List[Dict]:
    rng = random.Random(STUDENT_SEED)
    shuffled_if = list(if_rows)
    rng.shuffle(shuffled_if)
    chosen_if = [dict(r, prompt=[{"role": "user", "content": r["prompt"]}])
                 for r in shuffled_if[:N_STUDENT_PER_DOMAIN]]
    out = math_seeds + code_seeds + chosen_if
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------------------ main
def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(POOLS), default="upstream")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    ap.add_argument("--only", nargs="+", choices=OUTPUTS, default=list(OUTPUTS))
    ap.add_argument("--manifest", type=Path, default=DATA_DIR / "MANIFEST.json")
    a = ap.parse_args(argv)

    wanted = set(a.only)
    ifr = if_records()
    pools: Dict[str, List[Row]] = {}
    seeds: Dict[str, List[Dict]] = {}
    needs = {"math": {"student/train_3domain.jsonl", "teacher/math_train.jsonl"},
             "code": {"student/train_3domain.jsonl", "teacher/code_train.jsonl"}}
    for domain in ("math", "code"):
        if wanted & needs[domain]:
            pools[domain] = POOLS[a.source][domain](a.cache_dir)
            seeds[domain] = student_seed_rows(domain, pools[domain])
            print(f"{domain} pool: {len(pools[domain])} rows ({a.source})", flush=True)

    outputs: Dict[str, Callable[[], List[Dict]]] = {
        "student/train_3domain.jsonl": lambda: student_rows(seeds["math"], seeds["code"], ifr),
        "teacher/math_train.jsonl": lambda: teacher_rows("math", seeds["math"], pools["math"]),
        "teacher/code_train.jsonl": lambda: teacher_rows("code", seeds["code"], pools["code"]),
        "teacher/if_train.jsonl": lambda: ifr,
    }
    manifest = json.loads(a.manifest.read_text())["files"] if a.manifest.exists() else {}
    status = 0
    for rel in OUTPUTS:
        if rel not in wanted:
            continue
        path = a.out_dir / rel
        write_jsonl(path, outputs[rel]())
        got = sha256_file(path)
        expected = manifest.get(rel, {}).get("sha256")
        verdict = "MATCH" if got == expected else ("NO MANIFEST ENTRY" if expected is None else "MISMATCH")
        status |= verdict != "MATCH"
        print(f"{verdict:9s} {rel}  sha256={got}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
