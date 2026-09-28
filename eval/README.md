# Evaluation

This package runs the paper's evaluation protocol on any model that vLLM can serve, on any machine with a GPU: it
samples answers with vLLM, grades them with the official public checkers, and computes the paper's scores and paired
bootstrap intervals. The graders, the question catalogs and the statistics are the ones behind the paper's numbers:

- the question catalogs are rebuilt from pinned public sources and must reproduce, question by question, the catalogs
  the paper's answers were generated from (`reproduce/data/catalog`);
- the graders re-grade the paper's own sampled answers to exactly the recorded correctness (below);
- `eval.aggregate` and `eval.compare` reproduce every score and interval in the paper from the released records
  (`reproduce/`).

## Install

```bash
pip install -r eval/requirements.txt       # vLLM 0.18.0, math-verify 0.9.0, the IF checker dependencies, ...
python -m eval.external.fetch           # official graders at pinned commits (LiveCodeBench, IFEval, IFBench, nltk data)
python -m eval.data.prepare                # download the benchmarks at pinned revisions and verify every question
```

`eval.external.fetch` downloads, and checks by sha256, the files of: LiveCodeBench `lcb_runner` at
[`28fef95`](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24) (MIT),
google-research `instruction_following_eval` at
[`b24f2136`](https://github.com/google-research/google-research/tree/b24f2136e8ef405b900b5619760126304f190941/instruction_following_eval)
(Apache-2.0), IFBench at [`1091c4c3`](https://github.com/allenai/IFBench/tree/1091c4c3de6c1f6ed12c012ed68f11ea450b0117)
(Apache-2.0), and the nltk data packages (punkt, punkt_tab, stopwords, the perceptron taggers) at nltk_data
`550b6625`. `PINS.json` lists every file and hash; the `.py` hashes are those of the grader sources used for the paper.
Nothing is vendored. Files go to `eval/third_party/` (or `$DN_MOPD_THIRD_PARTY`).

`eval.data.prepare` writes `eval/data/prepared/` (or `$DN_MOPD_EVAL_DATA`) and prints PASS or FAIL per suite. For each
suite it downloads the pinned source, checks its sha256, rebuilds every question exactly as the paper's evaluation
did, and checks each question's id and input hash (and every released field) against `reproduce/data/catalog` and the
complete catalog's sha256 against the catalog the paper's cells were generated from. For LiveCodeBench it also decodes
each problem's official tests with the official `lcb_runner` and checks their hash. The AIME 2025 problems and
answers and the LiveCodeBench problem statements are not redistributed in this repository (their released catalogs
hold ids and hashes only; see `reproduce/THIRD_PARTY.md`), so this step is required before evaluating those suites.
For LiveCodeBench the two release files (`test5.jsonl`, `test6.jsonl`, about 700 MB) are downloaded and the decoded
tests take about 1.4 GB. A machine without internet access can use directories fetched and
prepared elsewhere through `DN_MOPD_THIRD_PARTY` and `DN_MOPD_EVAL_DATA`.

| Suite | Questions | Samples per question | Pinned source | Text in `reproduce/data` |
|---|---|---|---|---|
| `aime25` | 30 | 64 | HF `yentinglin/aime_2025` @ `6f71d77b` | no (ids and hashes; the card declares no license) |
| `aime26` | 30 | 64 | HF `math-ai/aime26` @ `79037aeb` | yes (Apache-2.0) |
| `lcb_v5` | 167 | 6 | HF `livecodebench/code_generation_lite` @ `0fe84c39`, `test5.jsonl` | no (ids, dates, hashes) |
| `lcb_v6` | 175 | 6 | the same release, `test6.jsonl` | no (ids, dates, hashes) |
| `ifeval` | 541 | 16 | google-research `instruction_following_eval/data/input_data.jsonl` @ `b24f2136` | yes (Apache-2.0) |
| `ifbench` | 300 | 16 | allenai/IFBench `data/IFBench_test.jsonl` @ `1091c4c3` | yes (ODC-BY-1.0) |
| `math500` | 500 | 16 | HF `HuggingFaceH4/MATH-500` @ `6e4ed1a2` | yes (MIT) |

LiveCodeBench v5 and v6 are the problems *added* in those releases (the files `test5.jsonl` and `test6.jsonl`,
contest dates 2024-09-22 to 2025-01-04 and 2025-01-04 to 2025-04-06), not the cumulative `release_v5`/`release_v6` sets. IFEval
uses the copy in the google-research repository, which differs from the Hugging Face copy `google/IFEval` in the
wording of one prompt (key 2785, where the repository's wording matches its arguments). AIME 2025 is the
`yentinglin/aime_2025` parquet; the question input hash is taken over `{problem, answer, id, suite}`.

## Evaluate a model

```bash
M=/path/to/hf_model            # or a hub id such as Qwen/Qwen3.5-4B
for s in aime25 aime26 lcb_v5 lcb_v6 ifeval ifbench; do
  python -m eval.generate --model $M --name mymodel --suite $s --max-tokens 8192   # GPU
  python -m eval.grade --gen outputs/eval/mymodel/$s/cap8192                        # CPU
done
python -m eval.aggregate --records outputs/eval --cap 8192
python -m eval.compare mymodel q35_4b_s_mdnorm --records-a outputs/eval --cap 8192     # vs a released model
```

- `eval.generate` writes `$OUT_ROOT/eval/<name>/<suite>/cap<cap>/` (`OUT_ROOT` defaults to `outputs/`, `--out`
  overrides). Each suite has a fixed number of shards (AIME 4, LiveCodeBench 3, IF 2): run
  `CUDA_VISIBLE_DEVICES=k python -m eval.generate ... --shard k` once per shard to spread a suite over GPUs; every
  chunk of questions is written atomically, so an interrupted run resumes where it stopped, and a resume with another
  model, cap or protocol is refused.
- `eval.grade` writes `graded.jsonl` (responses with every grader detail), `scores.jsonl` (the released per-question
  format: `id`, `input_sha256`, `n`, `correct` as a 0/1 string, `mean_tokens`, `cap_hits`) and `summary.json`
  (`mean_pass1` = avg@N, mean output tokens, cap-hit rate, grader pins). Grades are cached per chunk.
- `eval.aggregate` prints suite, domain and Total scores (x100), the mean output tokens of each domain and the mean
  cap-hit rate; with no `--records` it prints the paper's 80 models from `reproduce/data`.
- `eval.compare A B` prints A - B for Total and each domain (or any suite, or `math500`) with the paired 95% interval,
  for both caps unless `--cap` is given. `--records-a` / `--records-b` let one side come from your runs and the other
  from the released records.

## The protocol

Everything below is fixed in `eval/protocol.py`; only the cap (8,192 or 16,384 tokens) is a choice.

- **Engine.** vLLM 0.18.0, tensor parallel 1, `enforce_eager=True`, engine seed 42. The six public suites use
  `gpu_memory_utilization=0.90`, `max_model_len=32768` (prompt + cap must fit; no question is ever trimmed) and
  `max_num_seqs=64`, and go to the engine in chunks of 4 (AIME), 16 (LiveCodeBench) or 32 (IF) questions. MATH-500
  was run by a separate script and keeps its settings: `gpu_memory_utilization=0.95`, the model's own context length,
  `max_num_seqs=256`, all 500 prompts in one call.
- **Sampling.** Temperature 1.0, top-p 1.0, top-k -1, presence penalty 0, seed 42, `max_tokens` = the cap; n = 64
  (AIME), 6 (LiveCodeBench), 16 (IFEval, IFBench, MATH-500).
- **Prompts.** The catalog's chat messages rendered with `apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)`. AIME and MATH-500: one user turn, the problem followed by
  `"\nPlease reason step by step, and put your final answer within \boxed{}."`. LiveCodeBench: the official system
  message and question template of `lcb_runner`. IF: the prompt verbatim as one user turn. Answers of the six suites are
  graded exactly as generated; MATH-500 answers are stripped of surrounding whitespace, as in the paper.
- **Graders.**
  - AIME: math-verify 0.9.0 on the last `\boxed{...}`, `verify(parse(gold), parse(answer))`; an answer without a
    complete box is wrong; a grader exception aborts. MATH-500: the same comparison, with exceptions counted as wrong.
  - LiveCodeBench: the official extraction (the last fenced code block) and the official test runner over all public
    and private tests, 6 s per test, run under an 8 GiB address-space limit; a program without a fenced block is
    wrong; a runner infrastructure failure aborts instead of scoring.
  - IFEval, IFBench: the official checkers; a sampled answer is correct when it follows every instruction under the
    strict check (strict prompt accuracy). Grading revision `deterministic-official-v1-20260918`: `random` is seeded
    with 0 around each check and langdetect with 0, so a grade never depends on sample order; an unimplemented
    instruction or a checker exception aborts. Two IFEval prompts ask for a non-letter character count and make the
    official checker substitute a random letter; they stay in the suite and `summary.json` also reports the score
    without them.
- **Scores.** A question's score is the fraction of its answers graded correct; a suite score is the mean over
  questions (avg@N); a domain averages its two suites; Total averages the six.
- **Intervals.** Paired bootstrap over questions within each suite, same indices for both models, B = 10,000, a fresh
  `numpy.random.default_rng(20260923)` per contrast, 2.5/97.5 percentiles.

**What can differ when you rerun.** Seeded sampling is not bitwise reproducible across GPU types, drivers, CUDA or vLLM
versions, so regenerated answers are a new draw from the same protocol, not the paper's answers; scores move within
sampling noise. `eval.generate` records the package versions and the chat template hash in `MANIFEST.json` and warns
when vLLM is not 0.18.0 or the template is not one of the paper's Qwen3.5 templates. LiveCodeBench verdicts depend on
wall-clock timeouts: grade on an unloaded machine (the paper graded at most four problems of a cell in parallel).
The paper graded math and code under Python 3.12 and IF under Python 3.10, with the pinned versions above.

## How the package was checked

- **Catalogs.** `eval.data.prepare` rebuilt all seven suites from the public sources; every question matched the
  released catalog (ids and input hashes, plus every released field), the six complete catalogs, including AIME 2025
  and LiveCodeBench whose text is not released, matched the sha256 of the paper's evaluation catalogs, and all 342
  decoded LiveCodeBench test sets matched their recorded hashes.
- **Graders.** The paper's own raw answers were re-graded with this package for whole cells: the 2B initial student
  at 8,192 and the 9B DN-MOPD student at 16,384 on all six suites and MATH-500, and a control run (4B, 8,192) on
  IFEval and LiveCodeBench v5; 64,354 sampled answers in all. Every AIME, MATH-500, IFEval and IFBench grade (59,248)
  and 5,101 of 5,106 LiveCodeBench grades equal the released records. The five exceptions are correct but slow
  programs of the 9B student (each needed 10 to 48 s over all its tests in the paper's run) that hit the 6 s per-test limit
  on our slower, shared grading machine; two of them pass again when re-graded alone. No grade flipped the other way.
  `tests/eval/test_eval_real_responses.py` repeats the re-grade on a shipped sample of 68 answers (the AIME 2025 and
  LiveCodeBench parts need the prepared data and skip without it).
- **Statistics.** `tests/eval/test_eval_stats.py` checks every level against `reproduce.py` and every Total, extended,
  control and MATH-500 contrast against the paper's tables (1e-6).

## Tests and smoke test

```bash
python -m pytest tests/eval                      # CPU only; tests needing the fetched graders or prepared data skip
DN_MOPD_SLOW_TESTS=1 python -m pytest tests/eval # also the 168 domain contrasts and the full reproduce.py bootstrap
CUDA_VISIBLE_DEVICES=0 bash eval/smoke_test.sh   # GPU: a few questions of every suite, graded; prints SMOKE PASS/FAIL
```

`smoke_test.sh` uses `Qwen/Qwen3.5-2B` by default (`SMOKE_MODEL` to change it) and prints, beside each score, the
paper's score of the 2B initial student on the same questions.

## Files

| File | Role |
|---|---|
| `protocol.py` | the frozen protocol constants and evaluation catalog hashes |
| `generate.py`, `grade.py`, `aggregate.py`, `compare.py` | the four commands |
| `stats.py` | per-question records (released and run layouts), levels, paired bootstrap |
| `graders/` | `math_grader.py`, `code_grader.py` + `lcb_runtime.py`, `if_grader.py` |
| `data/prepare.py`, `data/sources.json` | pinned sources and the verified catalog rebuild |
| `external/fetch.py`, `external/PINS.json` | pinned official graders (fetched into `eval/third_party/`, not tracked) |
| `smoke_test.sh` | GPU end-to-end check |
