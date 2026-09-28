# Reproduce the paper's numbers from the per-question records

This directory holds the per-question evaluation records behind every Qwen3.5 table and figure of
*Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation* (DN-MOPD), and `reproduce.py`,
which recomputes the paper's scores and paired intervals from those records and checks them against the values the
paper's tables and figures were rendered from. If you use the records, please cite the paper (`CITATION.cff` at the
repository root).

## Quick start

Requires Python 3 and numpy only.

```bash
python reproduce/reproduce.py --no-bootstrap   # scores and point estimates: 640 values, about 7 s
python reproduce/reproduce.py                  # plus every paired interval (B = 10,000): 1,440 values, about 2.5 min
```

Expected last line:

```
checked 1440 values against the paper data; max |deviation| 5.00e-07; ALL MATCH
```

Measured on one machine with Python 3.12 and numpy 2.3.5: 7.3 s without and 147 s with the bootstrap. The script
writes `reproduce/results/levels.csv` (every score of every model at both caps) and `reproduce/results/checks.json`.

`reproduce.py` recomputes, from the per-question records:

- every suite score, domain score, Total, mean response length and cap-hit rate, for 80 models x 2 evaluation caps;
- every paired contrast and 95% interval reported for Qwen3.5 (Totals, domains, annealed injection and the 160-update
  continuations, the control runs);
- the three-seed two-level interval;
- the MATH-500 contrasts.

Every per-question file is first checked against its cell summary (`data/cells.jsonl`). The script then compares
1,440 reported values against `data/paper_tables/` and prints `ALL MATCH` when every value agrees to within 1e-6 (2e-6
for the domain contrasts, which that table stores with 6 decimals). The largest deviation, 5e-7, comes from that
rounding.

The same definitions are implemented by the evaluation package (`eval/stats.py`): `python -m eval.aggregate` prints
these scores and `python -m eval.compare A B` any paired contrast, from these records or from your own graded runs.
`tests/eval/test_eval_stats.py` checks that both give identical numbers.

## Definitions

A question's score is the fraction of its sampled answers that are graded correct. We sample 64 answers per question
for AIME 2025 and 2026, 6 for LiveCodeBench v5 and v6, 16 for IFEval and IFBench (strict prompt accuracy), and 16 for
MATH-500. Each suite score is the mean question score (avg@N, an estimate of single-answer accuracy), times 100. Each
domain averages its two suites (math: AIME 2025/2026; code: LiveCodeBench v5/v6; IF: IFEval/IFBench), and Total
averages all six suites. MATH-500 is reported separately.

Paired intervals resample questions within each suite with the same indices for both models: B = 10,000 replicates,
a fresh `numpy.random.default_rng(20260923)` per contrast, suites resampled in the order AIME25, AIME26, LCB v5,
LCB v6, IFEval, IFBench, and the 2.5/97.5 percentiles of the averaged replicates. They are conditional on one student
training run and one expert pool per size. The three-seed read shares one generator across the seed-42/43/44 contrasts
(in seed order) and resamples a seed and then one of its replicates, three times per replicate.

Evaluation used the non-thinking chat rendering, temperature and top-p 1.0, generation seed 42, and caps of 8,192 and
16,384 tokens; `eval/README.md` gives the complete protocol and the tools to run it on new models.

## Contents

| Path | Contents | Paper |
|---|---|---|
| `reproduce.py` | Recomputation and check of all Qwen3.5 numbers. | Tables 1-5, Fig. 1, App. B, C, E.1 |
| `data/models.csv` | The 80 evaluated models: key, size, paper name, family, method configuration, training seed, updates, model-identity hash and chat-template hash. | Tables 1-2, App. A-C |
| `data/cells.jsonl` | One row per (model, suite, cap), 960 rows: score, problems, samples per problem, mean output tokens, cap-hit rate, and provenance hashes (graded output, grader revision, evaluation manifest, model identity, question catalog) plus the package versions (torch 2.10.0, transformers 4.57.6, vLLM 0.18.0). `catalog_sha256` is the hash of the complete evaluation catalog; for AIME 2025 and LiveCodeBench, whose released catalogs omit the text, it is the hash that `python -m eval.data.prepare` reproduces. | App. A.3, A.5 |
| `data/scores/<suite>/cap<cap>/<model>.jsonl` | One row per question: `id`, `input_sha256`, `n`, `correct` (one '0'/'1' character per sampled answer, in sample order), `mean_tokens`, `cap_hits`. | All Qwen3.5 scores |
| `data/catalog/<suite>.jsonl` | The fixed question lists, in evaluation order, with ids and input hashes. AIME 2026, IFEval and IFBench include the prompts as rendered to the model (chat messages) and the reference answers; these three files are byte-identical to the evaluation catalogs. AIME 2025 and LiveCodeBench are not redistributed: AIME 2025 rows hold the question id and input hash only (no text, no answer), and LiveCodeBench rows the public problem id, release date, difficulty, input hash and official test index and hash. `python -m eval.data.prepare` rebuilds their complete catalogs from the pinned public sources and verifies every hash (see `THIRD_PARTY.md`). | App. A.3 |
| `data/math500/` | MATH-500 per-question correctness (16 answers, math-verify) for the initial student, Label, DN-MOPD and the math single-teacher student at every size and both caps, with the question list and generation settings. | Sec. 3.1, App. C.7 |
| `data/traces/dn_multiplier_trajectory.json` | Per-batch DN statistics of the three DN-MOPD runs, including their continuation beyond 80 updates: pooled std, per-domain std, raw and clipped multiplier, token counts. | Fig. 4c, Sec. 2, App. D.2-D.3 |
| `data/traces/dn_feedback_mechanism.json` | Per-domain log-ratio dispersion and multiplier summaries over the recorded batches. | Fig. 4a, Table D.1, App. D.2 |
| `data/traces/variance_share.json` | Token share and each domain's share of the pooled log-ratio variance. | App. D.5 |
| `data/traces/gradient_diagnostic_4b.json` | Per-domain gradient norms, cosines and contribution shares under equal and DN weights (4B). | App. D.5 |
| `data/traces/label_unapplied_multipliers.json` | The DN multipliers the seed 42-44 Label runs would have received (recorded, not applied). | App. D.5 |
| `data/paper_tables/q35_total_contrasts.csv` | Total contrasts (DN-MOPD vs Label, the single teachers, SeqKD-SFT and task arithmetic). | Fig. 1c, App. C.3 |
| `data/paper_tables/q35_story_contrasts.csv` | Domain-level contrasts against the initial student and between methods. | Sec. 3.1, App. C.3, E.1 |
| `data/paper_tables/q35_extended_results.csv`, `q35_extended_contrasts.csv` | Annealed injection and the 160-update continuations. | Table 5, App. C.1-C.3 |
| `data/paper_tables/q35_controls.json` | Seeds 43/44, frozen update-0 multipliers, fixed weights, the 160-update 9B comparison, MATH-500, variance shares and gradient summaries. | Table 3, App. C.3, C.5-C.7, D.5 |
| `data/paper_tables/appendix_support.json` | The earlier Qwen3-4B comparison and the annealed-injection coverage counts. | App. C.1, E.2 |

Total size of `data/`: 38 MB (1,005 files).

## Model keys

The records keep the evaluation's model keys, because the provenance hashes in `data/cells.jsonl` refer to them. Keys
have the form `q35_<size>_<name>` with `<size>` in `2b`, `4b`, `9b`; `data/models.csv` gives the paper name of every
key. The `<name>` values are:

| Key name | Paper name |
|---|---|
| `base` | Initial student (Qwen3.5 at that size) |
| `t_math`, `t_code`, `t_ifeval` | RL experts (GRPO teachers) for math, code and instruction following |
| `s_singlemath`, `s_singlecode`, `s_singleifeval` | Single-teacher OPD with the math, code or IF expert |
| `s_pool` | Uniform pool |
| `s_dynamic` | Dynamic router |
| `s_route` | Label, i.e. multi-teacher OPD with label routing |
| `s_mdnorm` | DN-MOPD |
| `s_manneal` | Annealed injection |
| `*160` (e.g. `s_mdnorm160`, `s_route160`, `s_singlecode160`) | The same run continued to 160 updates |
| `sft_seqkd_s42` | SeqKD-SFT |
| `merge_avg` | ParamMerge-Avg (uniform average of the three experts) |
| `merge_ta1` | ParamMerge-TA (task arithmetic, lambda = 1) |
| `ctl_dnorm_s43`, `ctl_dnorm_s44` | DN-MOPD with student seeds 43 and 44 |
| `ctl_observe_s42`, `ctl_observe_s43`, `ctl_observe_s44` | Label with seeds 42, 43 and 44; these runs record the DN multipliers without applying them (`ctl_observe_s42` is a same-seed rerun of `s_route`) |
| `ctl_fixcal_s42` | The frozen update-0 DN multipliers |
| `ctl_fix21q_s42` | Fixed weights (2, 1, 0.25) |
| `ctl_fixmath_s42` | Fixed weights (2, 1, 1) |
| `ctl_fixif_s42` | Fixed weights (1, 1, 0.25) |

Fixed weights are listed in the order math, code, IF. Every student not marked otherwise uses seed 42 and 80 updates.

## Not included

- **Model weights and full model responses.** Each score row carries the hash of its graded output, and each cell the
  hash of the model identity, so the records can be tied to the checkpoints. A small sample of real responses with
  their grades is used by the tests (`tests/eval/fixtures/`).
- **AIME 2025 problems and answers, and LiveCodeBench problem statements and tests** (see `THIRD_PARTY.md`); they
  are fetched from their pinned public sources and checked against the catalogs by `eval/data/prepare.py`.
  `reproduce.py` does not need them.

## License

The records are released under CC BY 4.0 (`LICENSE`). Benchmark content inside the catalogs (AIME 2026, IFEval,
IFBench, MATH-500) keeps its original license (`THIRD_PARTY.md`). `reproduce.py` is code, under the repository's Apache License 2.0.
