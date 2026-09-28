# Third-party content in `reproduce/data`

The per-question scores, cell summaries, model list, traces and paper tables are our own records (CC BY 4.0, see
`LICENSE`). The question catalogs identify the benchmark questions the scores refer to. Benchmark text is included
only where its source declares a license that permits redistribution (AIME 2026, IFEval, IFBench, MATH-500); that
content is not ours, keeps the license of its source (below), and is included so that each score can be tied to its
exact question. If you redistribute it, follow the source's terms. For AIME 2025 and LiveCodeBench the catalogs hold
identifiers and hashes only.

| File | Content included | Source (pinned revision) | License of the source |
|---|---|---|---|
| `data/catalog/aime25.jsonl` | **no problem text and no answers**: only the question id (the source's `id` column, 0-29) and the sha256 of the source row (`{problem, answer, id, suite: "aime25"}`, JSON with sorted keys) | Hugging Face dataset [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) @ `6f71d77b0b89b9dabe07ab466c51df33f514df7f`, file `data/train-00000-of-00001-243207c6c994e1bd.parquet` | the dataset card declares no license, so nothing from it is redistributed here; the AIME problems are published by the Mathematical Association of America (MAA) |
| `data/catalog/aime26.jsonl` | the 30 AIME 2026 problems (as chat messages with the answer instruction appended) and their answers | [`math-ai/aime26`](https://huggingface.co/datasets/math-ai/aime26) @ `79037aebdb6580008fb960d17cb21fd3099083e3` | Apache-2.0, as declared by the dataset card (`license: apache-2.0`; full text in the repository's root `LICENSE`); please also cite the dataset as its card asks. The AIME problems are published by the MAA |
| `data/catalog/ifeval.jsonl` | the 541 IFEval prompts with their instruction ids and arguments | `instruction_following_eval/data/input_data.jsonl` of [google-research/google-research](https://github.com/google-research/google-research) @ `b24f2136e8ef405b900b5619760126304f190941` (the copy published as [`google/IFEval`](https://huggingface.co/datasets/google/IFEval)) | Apache-2.0 |
| `data/catalog/ifbench.jsonl` | the 300 IFBench test prompts with their instruction ids and arguments | `data/IFBench_test.jsonl` of [allenai/IFBench](https://github.com/allenai/IFBench) @ `1091c4c3de6c1f6ed12c012ed68f11ea450b0117` (published as [`allenai/IFBench_test`](https://huggingface.co/datasets/allenai/IFBench_test)) | data: ODC-BY-1.0 (code: Apache-2.0) |
| `data/math500/questions.jsonl` | the 500 MATH-500 problems and reference answers | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, the subset of MATH (Hendrycks et al., 2021) selected in [openai/prm800k](https://github.com/openai/prm800k) | MIT (MATH and prm800k) |
| `data/catalog/lcb_v5.jsonl`, `data/catalog/lcb_v6.jsonl` | **no problem statements and no tests**: only the public LiveCodeBench problem id, contest date, difficulty, the sha256 of the official problem record and of its decoded tests, and the record's position in the official release | [`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) @ `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, files `test5.jsonl` (167 problems) and `test6.jsonl` (175 problems) | the problems originate from LeetCode, AtCoder and Codeforces; they are not redistributed here |

The AIME 2025 problems and answers and the LiveCodeBench statements and tests are rebuilt locally from their pinned
public sources by `eval/data/prepare.py` (`python -m eval.data.prepare`), which re-downloads every source above at its
pinned revision, checks each question's id and input hash against these catalogs, and checks that each rebuilt
complete catalog has exactly the sha256 of the catalog the paper's answers were generated from.

The per-question records under `data/scores/` and `data/math500/cap*/` contain no benchmark text: each row holds a
question id, the input hash, and the correctness of each sampled answer. The grading test fixture
`tests/eval/fixtures/real_responses.jsonl` holds 68 of our models' answers (8 of them to AIME 2025 questions); it has
no problem-statement or reference-answer fields (the answers are model outputs, which may restate short parts of a
problem), and its AIME 2025 re-grade reads the reference answers from the prepared catalog.
