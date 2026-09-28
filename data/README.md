# Training prompt sets

These are the exact prompt sets of the paper's Qwen3.5 experiments, in the same rows and the same order. All fields
the trainer reads are byte-identical.

The files are not stored in git. They are hosted in the Hugging Face dataset
[XINLI1997/DN-MOPD-Data](https://huggingface.co/datasets/XINLI1997/DN-MOPD-Data) under the same paths, and
`data/download.py` fetches them from the revision pinned in `MANIFEST.json`:

```bash
python data/download.py                                    # all four files (about 0.97 GB), sha256-checked
python data/download.py --only student/train_3domain.jsonl # one file (repeat --only for more)
python data/download.py --check                            # verify the local files; no download
```

| File | Rows | Size | Used by |
|---|---|---|---|
| `student/train_3domain.jsonl` | 2,700 (900 math, 900 code, 900 IF) | 46 MB | every student run |
| `teacher/math_train.jsonl` | 8,000 | 3.8 MB | GRPO math teachers |
| `teacher/code_train.jsonl` | 8,000 | 917 MB | GRPO code teachers |
| `teacher/if_train.jsonl` | 8,000 | 3.7 MB | GRPO IF teachers |

- **Student runs** use the student set: Label, DN-MOPD, single-teacher OPD, uniform pool, dynamic router, annealed
  injection, the controls, and the SeqKD-SFT generation.
- **Hashes, sizes and counts** are in `MANIFEST.json`, together with the pinned dataset revision (`hub`).
- **Downloads.** `download.py` uses `huggingface_hub` when it is installed (the training environment has it) and plain
  HTTPS otherwise (`--backend hub|http` forces one). `HF_ENDPOINT` selects a Hub mirror, and `--revision` another
  dataset revision. A file already present with the right hash is not downloaded again. Every file is checked against
  `MANIFEST.json`, and the exit code is non-zero if any file is missing or differs.
- **Check the files** with `python data/download.py --check` or `python -m pytest tests/core -k data`. The data
  tests skip when the files have not been downloaded.
- **Rebuild them from public sources** with `data/build/build_data.py` instead of downloading them (see below).

## Row schema

Every row has the same five keys.

```json
{"prompt": [{"content": "...", "role": "user"}],
 "label": "...",
 "domain": "math",
 "teacher": "math",
 "metadata": {"domain": "math", "data_source": "DeepMath-103K", "src_index": 13934}}
```

| Key | Read by the trainer as | Content |
|---|---|---|
| `prompt` | `--input-key prompt` | One user message. The chat template is applied with `enable_thinking=false`. In `teacher/if_train.jsonl` the prompt is a plain string, which the trainer wraps as one user message, the same form as in the student set. |
| `label` | `--label-key label` | Depends on the domain: see the list below. |
| `teacher` | teacher routing | The name of the teacher that scores the prompt in OPD (label routing). It always equals `domain`. |
| `metadata.domain` | `--metadata-key metadata` | Selects the verifier. It is also the domain DN-MOPD normalizes over. |
| `metadata.data_source`, `metadata.src_index` | provenance only | The source dataset, and the row's position in its source pool (see below). |
| `domain` | convenience copy | Read by the SeqKD-SFT generation script. |

The `label` holds:
- **math**: the reference answer, graded by the rule-based math verifier;
- **code**: a JSON string `{"inputs": [...], "outputs": [...]}` of stdin/stdout test cases;
- **IF**: a JSON list of constraints, graded strictly by the trainer's constraint checkers
  (`miles/Uni_OPD_utils/OPD_reward/ifeval_checkers.py`).

The trainer drops prompts longer than 2,048 tokens after templating (`--rollout-max-prompt-len 2048`). No released
prompt is that long: the longest is 1,228 tokens with the Qwen3.5 tokenizer, so every row is used.

**Fields removed relative to the training files.** The files the paper trained on also carried the following fields:
- `true_domain` (top level and in `metadata`), always equal to `domain`;
- duplicate top-level `data_source` and `src_index`;
- `metadata.n_constraints` and `metadata.split` on IF rows;
- `pass_rate` (top level and in `metadata`) on the 1,800 student math/code rows, and on the same rows in the teacher
  pools. It is an internal difficulty estimate from a model outside the paper.

No training code reads any of them. Row order, prompts, labels, teacher names and domains are unchanged, so training
on these files is identical.

## How each file was built

**Source pools.** Both are rebuilt by `build/build_data.py`. They were verified row by row against the intermediate
copy [Keven16/G-OPD-Training-Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data), which the paper read
them from.
- **Math pool** (57,046 prompts) comes from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K).
  - The questions are deduplicated by text, in first-occurrence order.
  - A question is kept if any of its occurrences has `difficulty >= 6`.
  - The prompt is the question followed by `"\nPlease reason step by step, and put your final answer within
    \boxed{}."`.
  - The label is the first occurrence's `final_answer`.
- **Code pool** (25,276 prompts) is every `ability == "code"` row of
  [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data) `train`, in order.
  - The system message is dropped.
  - The user message is followed by `"\nYou need to think first then write the Python code."`.
  - The label is `reward_model.ground_truth`.
- **`src_index`** is the position of a row in its pool.

**IF prompts.** `build/gen_ifeval_data.py --n 8000 --seed 20260803` generates the IF prompts. Each is a generic
writing task with 1 to 3 verifiable constraints from a closed set of 19 types. Every constraint set is checked so that
an empty answer fails it. All 8,000 are ours; no third-party text is used.

The generator draws constraint types from, and checks every constraint set with, the trainer's IF checkers
(`miles/Uni_OPD_utils/OPD_reward/ifeval_checkers.py`, the reward of the IF teachers). There is one copy of that file:
the trainer loads it by path, and `gen_ifeval_data.py` does the same, so the data build needs the repository checkout
but no installed trainer.

**Student set.**
- **Math and code rows.** It takes 900 math and 900 code prompts, listed in `build/ids/student_math_code_ids.json`.
  - They were chosen, before the Qwen3.5 experiments, as prompts whose pass rate under Qwen3-4B lay in
    [0.125, 0.875]. The pass rate came from 8 samples at temperature 1.0 and top-p 0.95, with the non-thinking
    template.
  - 900 were then drawn at random per domain (seed 42).
  - The rows are interleaved math, code, math, code, and so on.
  - Code sub-sources: 370 codecontests, 282 taco, 156 apps, 92 codeforces.
- **IF rows.** The 8,000 IF prompts are shuffled with `random.Random(42)` and the first 900 are kept. Their prompts are
  wrapped as one user message.
- **Final shuffle.** The 2,700 rows are shuffled with the same generator.

**Teacher pools.**
- **Math and code.** Each pool starts with the student's 900 rows of that domain. It continues through the pool in
  order and skips prompts already taken (by a hash of the message text) and rows without a label, until it has 8,000
  rows.
- **IF.** The teacher uses all 8,000 generated prompts.

## What is reproducible from public data

Every file is reproducible, and `build_data.py` rebuilds all four byte for byte (sha256 equal to `MANIFEST.json`):
- **From the upstream sources.** It uses DeepMath-103K at rev. `5cf055d` and Eurus-2-RL-Data at rev. `9776b13`
  (default, about 3.9 GB of downloads).
- **From the intermediate copy.** `--source gopd` uses G-OPD-Training-Data at rev. `c9bc678` (about 1.7 GB).

One step is not reproducible from public data alone: *choosing* the 1,800 student math/code prompts. That choice used
pass rates of a model outside the paper, so it is published as ids. Given the ids, everything else is deterministic:
- the pools and the IF generator;
- the seed-42 shuffles;
- the teacher pools, which are seeded by the student rows.

```bash
pip install pyarrow
python data/build/build_data.py                                   # all four files, checked against MANIFEST.json
python data/build/build_data.py --only teacher/code_train.jsonl   # one file
python data/build/build_data.py --source gopd                     # smaller download, same bytes
```

The script behaves as follows:
- **Checks.** Downloads are pinned by revision and sha256.
- **Mirrors.** `HF_ENDPOINT` selects a Hub mirror.
- **Cache.** Files are cached in `data/build/cache/`, which git ignores.
- **Memory.** Building the code files holds the 25,276 code rows in memory, a few GB.
- **Exit code.** It is non-zero if any output differs from `MANIFEST.json`.

## Content notes

- Some code test cases contain strings that look like private IP addresses or access keys. They come from
  programming problems about parsing such strings. They are upstream data, and they are kept unchanged because
  the reward uses them.
- Licenses: see [DATA_LICENSES.md](DATA_LICENSES.md).
