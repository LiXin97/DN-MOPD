# SPDX-License-Identifier: Apache-2.0
"""Generate the synthetic instruction-following (IF) training prompts, offline and deterministically.

Every prompt is a generic writing task plus 1-3 verifiable constraints drawn from a closed taxonomy. The label is the
JSON constraint list, graded by `ifeval_checkers.score` (strict: every constraint must hold). The prompts are ours
(no third-party text); see data/DATA_LICENSES.md.

The paper's IF pool is `--n 8000 --seed 20260803`. This reproduces it byte for byte (before the field pruning done by
build_data.py); the IF teacher trains on all 8,000 rows and the student set takes 900 of them.

usage:
  python gen_ifeval_data.py --n 8000 --seed 20260803 --out if_train_raw.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# The constraint checkers are the trainer's own (the reward of the IF teachers and the IF correctness label), kept in
# one place. The trainer loads this file by path as well, so it is loaded by path here instead of being imported from
# a package: data/ needs no installed trainer, only the file in this repository.
CHECKERS_PATH = Path(__file__).resolve().parents[2] / "miles" / "Uni_OPD_utils" / "OPD_reward" / "ifeval_checkers.py"


def _load_checkers():
    if not CHECKERS_PATH.is_file():
        raise SystemExit(f"missing {CHECKERS_PATH}: the IF generator uses the trainer's constraint checkers")
    spec = importlib.util.spec_from_file_location("dn_mopd_data_ifeval_checkers", CHECKERS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checkers = _load_checkers()
CHECKERS, instruction_text, score = checkers.CHECKERS, checkers.instruction_text, checkers.score

# ---------------------------------------------------------------- base tasks
# Deliberately generic and answerable without external knowledge: the domain under test is
# CONSTRAINT COMPLIANCE, not factual recall, so a wrong fact must never cost reward.
_TOPICS_TRAIN = [
    "a rainy afternoon", "learning to cook", "a long train journey", "an old bookshop",
    "how tides work", "why bread rises", "a city at night", "the first day at a new job",
    "keeping a garden alive", "the sound of a violin", "moving to another town",
    "an unfinished letter", "the smell of coffee", "a bicycle repair", "winter mornings",
    "a lighthouse keeper", "how bridges carry load", "a lost umbrella", "the last bus home",
    "why leaves change colour", "an empty stadium", "a handwritten recipe",
    "a crowded ferry", "the noise of a printing press", "a shortcut through an alley",
    "how compasses point north", "an orchard in spring", "a misplaced key",
    "the first snow of the year", "learning a second language", "a village market day",
    "why kettles whistle", "an abandoned railway", "a shared umbrella", "night fishing",
    "the weight of a backpack", "how locks work", "a postcard that arrived late",
    "an argument about directions", "the taste of burnt toast", "a paper aeroplane",
    "why mirrors fog", "a queue at the bakery", "the smell of rain on dust",
    "an old family photograph", "learning to swim", "a broken escalator",
    "how yeast works", "a walk before dawn", "the sound of distant thunder",
    "a chipped mug", "why boats float", "an unread book", "the hum of power lines",
    "a lost glove in winter", "how sundials tell time", "a rooftop garden",
    "an interrupted phone call", "the first bicycle", "a jar of loose change",
    "why ice cracks", "a delayed train", "the smell of fresh paint", "an empty theatre",
    "learning to whistle", "a garden fence", "how wells draw water", "a candle burning down",
    "an old wristwatch", "the sound of typing", "a fallen branch", "why kites fly",
    "a hotel corridor", "the taste of cold water", "an unfamiliar street",
    "how thermometers work", "a stack of unopened mail", "the last day of summer",
    "a squeaking door", "why soap cleans", "an early morning market",
    "the sound of a metronome", "a forgotten password", "how bells resonate",
    "a bench by the river", "an overgrown path", "the smell of a library",
    "a torn map", "why honey does not spoil", "an empty swimming pool",
    "learning to sew", "a foghorn at night", "the weight of wet clothes",
    "how magnets attract", "a birthday without cake", "an unplayed piano",
    "the taste of lemon", "a windy balcony", "why shadows lengthen",
    "an old toolbox", "the sound of gravel underfoot", "a shared kitchen",
    "how sponges absorb", "a missed connection", "the first cup of tea",
    "an unfinished puzzle", "why glass is transparent", "a laundry line in summer",
    "the smell of cut grass", "a narrow staircase", "how pulleys lift",
    "an empty picture frame", "the sound of a distant train", "a cracked pavement",
    "why milk curdles", "an old coat pocket", "learning to skate",
    "the light in a stairwell", "a bus with one passenger", "how sand forms",
]
_FORMS = [
    "Write a short description of {t}.",
    "Explain {t} to someone who has never encountered it.",
    "Describe {t} without using technical jargon.",
    "Give practical advice about {t}.",
    "Summarise what matters most about {t}.",
    "Write a brief reflection on {t}.",
    "Explain why someone might find {t} memorable.",
    "Describe {t} as it would appear to a first-time visitor.",
    "Write a short note to a friend about {t}.",
    "List what someone should know about {t}.",
]

# ---------------------------------------------------------------- constraint sampling
_WORDS_POOL = ["harbour", "lantern", "quietly", "gravity", "orange", "signal", "wooden", "market"]
_STARTERS = ["Answer:", "Here is my reply:", "In short:"]
_ENDERS = ["THE END", "-- done --", "That is all."]


def _sample_constraint(rng: random.Random, used: set) -> dict:
    """One constraint, avoiding types already used in this sample and self-contradictions."""
    while True:
        t = rng.choice(sorted(CHECKERS))
        if t in used:
            continue
        # mutually exclusive families
        if t == "all_caps" and "all_lowercase" in used:      continue
        if t == "all_lowercase" and "all_caps" in used:      continue
        if t == "valid_json" and used & {"bullet_count", "numbered_count", "paragraph_count",
                                         "sentence_count", "highlight_count", "wrapped_in",
                                         "starts_with", "ends_with", "repeat_first"}: continue
        if t in {"bullet_count", "numbered_count", "paragraph_count", "sentence_count",
                 "highlight_count", "wrapped_in", "starts_with", "ends_with",
                 "repeat_first"} and "valid_json" in used: continue
        if t == "forbidden_chars" and used & {"must_include", "repeat_first", "starts_with",
                                              "ends_with"}: continue
        if t in {"must_include", "repeat_first", "starts_with", "ends_with"} and \
           "forbidden_chars" in used: continue
        break
    if t in ("word_count_min",):        return {"type": t, "n": rng.choice([30, 40, 60, 80])}
    if t in ("word_count_max",):        return {"type": t, "n": rng.choice([60, 90, 120, 160])}
    if t == "sentence_count":           return {"type": t, "n": rng.choice([2, 3, 4, 5])}
    if t in ("bullet_count", "numbered_count", "paragraph_count"):
        return {"type": t, "n": rng.choice([2, 3, 4])}
    if t in ("highlight_count", "placeholder_count"):
        return {"type": t, "n": rng.choice([1, 2, 3])}
    if t in ("must_include", "must_exclude"):
        return {"type": t, "values": rng.sample(_WORDS_POOL, rng.choice([1, 2]))}
    if t == "forbidden_chars":          return {"type": t, "values": [rng.choice(["e", "a", "z"])]}
    if t == "starts_with":              return {"type": t, "value": rng.choice(_STARTERS)}
    if t == "ends_with":                return {"type": t, "value": rng.choice(_ENDERS)}
    if t == "wrapped_in":               return {"type": t, "open": "<<", "close": ">>"}
    if t == "repeat_first":             return {"type": t, "value": "REQUEST"}   # filled in later
    return {"type": t}


# Constraints an EMPTY response satisfies vacuously; a sample made only of these is not
# discriminative (caught by the assertion in _make_row on the first generation run).
_VACUOUS_ON_EMPTY = {"must_exclude", "forbidden_chars", "no_commas", "word_count_max"}


def _make_row(rng, topics, idx, split):
    base = rng.choice(_FORMS).format(t=rng.choice(topics))
    k = rng.choice([1, 2, 2, 3])                       # 1-3 constraints, mode 2
    used, cons = set(), []
    for _ in range(k):
        c = _sample_constraint(rng, used)
        if c["type"] == "repeat_first":
            c["value"] = base
        used.add(c["type"]); cons.append(c)
    # every sample needs at least one constraint an empty answer FAILS, else the reward is
    # satisfiable by saying nothing
    while all(c["type"] in _VACUOUS_ON_EMPTY for c in cons):
        c = _sample_constraint(rng, used)
        if c["type"] in _VACUOUS_ON_EMPTY:
            continue
        if c["type"] == "repeat_first":
            c["value"] = base
        used.add(c["type"]); cons.append(c)
    lines = [instruction_text(c) for c in cons]
    prompt = base + "\n\nFollow ALL of these requirements exactly:\n" + \
             "\n".join(f"- {l}" for l in lines)
    # sanity: an empty response must fail, otherwise the sample is not discriminative
    assert not score("", cons)[0], f"degenerate constraint set: {cons}"
    return {
        "prompt": prompt,
        "label": json.dumps(cons, ensure_ascii=False),
        "domain": "ifeval",
        "true_domain": "ifeval",
        "teacher": "ifeval",
        "data_source": "synthetic-ifeval-v1",
        "src_index": idx,
        "metadata": {"domain": "ifeval", "true_domain": "ifeval", "src_index": idx,
                     "data_source": "synthetic-ifeval-v1", "n_constraints": len(cons),
                     "split": split},
    }


def generate(n: int = 8000, seed: int = 20260803) -> List[Dict]:
    """The first `n` unique prompts of the generator seeded with `seed` (the paper: n=8000, seed=20260803)."""
    rng = random.Random(seed)
    # Exact duplicates are rejected: a duplicated prompt is a wasted sample and inflates its effective epoch count.
    rows, seen_prompts, tries = [], set(), 0
    while len(rows) < n:
        tries += 1
        if tries > 50 * n:
            raise SystemExit(f"generator exhausted: only {len(rows)} unique prompts of {n}")
        r = _make_row(rng, _TOPICS_TRAIN, len(rows), "train")
        if r["prompt"] in seen_prompts:
            continue
        seen_prompts.add(r["prompt"])
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = generate(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: n={len(rows)}")


if __name__ == "__main__":
    main()
