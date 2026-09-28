# SPDX-License-Identifier: Apache-2.0
"""The released prompt sets: hashes, schema, internal consistency, and the offline parts of the rebuild.

The four jsonl files are not stored in git (`python data/download.py` fetches them). Tests that read a file skip
cleanly when it is absent; the manifest and IF-generator tests need no downloaded file.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(DATA / "build"))

import build_data  # noqa: E402
import gen_ifeval_data  # noqa: E402

CHECKERS, score = gen_ifeval_data.CHECKERS, gen_ifeval_data.score
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text())["files"]
# Row-level checks read these three files; the 0.9 GB code file is covered by the hash test only.
ROW_CHECKED = ["student/train_3domain.jsonl", "teacher/math_train.jsonl", "teacher/if_train.jsonl"]


def _require(*rels):
    missing = [rel for rel in rels if not (DATA / rel).exists()]
    if missing:
        pytest.skip(f"data/{missing[0]} is not downloaded (the prompt sets are not in git; "
                    f"fetch them with: python data/download.py)")


def _rows(rel):
    _require(rel)
    with open(DATA / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_lists_the_four_training_sets():
    assert sorted(MANIFEST) == sorted(build_data.OUTPUTS)


def test_manifest_pins_the_hub_dataset():
    hub = json.loads((DATA / "MANIFEST.json").read_text())["hub"]
    assert hub["repo_id"] == "XINLI1997/DN-MOPD-Data" and hub["repo_type"] == "dataset"
    assert re.fullmatch(r"[0-9a-f]{40}", hub["revision"])


@pytest.mark.parametrize("rel", sorted(MANIFEST))
def test_file_hash_and_size(rel):
    _require(rel)
    path = DATA / rel
    assert path.stat().st_size == MANIFEST[rel]["bytes"]
    assert _sha256(path) == MANIFEST[rel]["sha256"]


@pytest.mark.parametrize("rel", ROW_CHECKED)
def test_schema_and_counts(rel):
    rows = _rows(rel)
    assert len(rows) == MANIFEST[rel]["rows"]
    assert dict(collections.Counter(r["domain"] for r in rows)) == MANIFEST[rel]["domains"]
    for r in rows:
        assert list(r) == ["prompt", "label", "domain", "teacher", "metadata"]
        assert list(r["metadata"]) == ["domain", "data_source", "src_index"]
        assert r["domain"] == r["teacher"] == r["metadata"]["domain"] in ("math", "code", "ifeval")
        assert isinstance(r["label"], str) and r["label"]
        if isinstance(r["prompt"], list):
            assert len(r["prompt"]) == 1 and r["prompt"][0]["role"] == "user" and r["prompt"][0]["content"]
        else:
            assert rel == "teacher/if_train.jsonl" and r["prompt"]


@pytest.mark.parametrize("rel", ROW_CHECKED)
def test_labels_are_verifiable(rel):
    for r in _rows(rel):
        if r["domain"] == "ifeval":
            cons = json.loads(r["label"])
            assert cons and all(c["type"] in CHECKERS for c in cons)
            assert not score("", cons)[0]                    # an empty answer never earns the reward
        elif r["domain"] == "code":
            tests = json.loads(r["label"])
            assert tests["inputs"] and len(tests["inputs"]) == len(tests["outputs"])


def test_if_generator_reproduces_the_teacher_if_set():
    rows = build_data.if_records()
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == MANIFEST["teacher/if_train.jsonl"]["sha256"]


def test_student_set_is_consistent_with_ids_and_teacher_pools():
    _require("student/train_3domain.jsonl", "teacher/math_train.jsonl", "teacher/if_train.jsonl")
    student = _rows("student/train_3domain.jsonl")
    ids = json.loads((DATA / "build" / "ids" / "student_math_code_ids.json").read_text())
    by = collections.defaultdict(list)
    for r in student:
        by[r["domain"]].append(r)
    for d in ("math", "code"):
        assert sorted(r["metadata"]["src_index"] for r in by[d]) == ids[d]
    # teacher pools start with the student's rows of that domain, in id order
    math_teacher = _rows("teacher/math_train.jsonl")
    student_math = {r["metadata"]["src_index"]: r for r in by["math"]}
    for i, t in zip(ids["math"], math_teacher[:900]):
        assert t["metadata"]["src_index"] == i and t["prompt"] == student_math[i]["prompt"]
    # the student IF rows are teacher IF rows with the prompt wrapped as one user message
    teacher_if = {r["metadata"]["src_index"]: r for r in _rows("teacher/if_train.jsonl")}
    for r in by["ifeval"]:
        t = teacher_if[r["metadata"]["src_index"]]
        assert r["prompt"] == [{"role": "user", "content": t["prompt"]}] and r["label"] == t["label"]


def test_student_ifeval_selection_matches_the_seeded_shuffle():
    student = _rows("student/train_3domain.jsonl")
    ifr = build_data.if_records()
    by_domain = {d: sorted((r for r in student if r["domain"] == d), key=lambda r: r["metadata"]["src_index"])
                 for d in ("math", "code")}                       # pre-shuffle order = ascending ids
    rebuilt = build_data.student_rows(by_domain["math"], by_domain["code"], ifr)
    assert rebuilt == student


@pytest.mark.parametrize("rel", ROW_CHECKED)
def test_no_internal_paths_in_rows(rel):
    _require(rel)
    pattern = re.compile(r"/mnt/|/home/|/tmp/[A-Za-z]|\\\\mnt\\\\")
    text = (DATA / rel).read_text(encoding="utf-8")
    assert not pattern.search(text), rel


def test_if_generator_uses_the_trainers_checkers():
    """One copy of the IF constraint checkers: the data build loads the trainer's file (the IF teachers' reward)."""
    trainer = DATA.parent / "miles" / "Uni_OPD_utils" / "OPD_reward" / "ifeval_checkers.py"
    assert trainer.is_file()
    assert Path(gen_ifeval_data.checkers.__file__).resolve() == trainer.resolve()
    assert not (DATA / "build" / "ifeval_checkers.py").exists()
