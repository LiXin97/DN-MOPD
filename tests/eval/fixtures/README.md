# Test fixtures

`real_responses.jsonl`: 68 sampled answers from the paper's evaluation (the initial Qwen3.5-2B student at a cap of
8,192 tokens and the 9B DN-MOPD student at 16,384), eight to twelve per suite for AIME 2025, AIME 2026, MATH-500,
IFEval, IFBench and LiveCodeBench v5/v6, half graded correct in the paper and half incorrect, chosen among the shortest
answers. Each row: suite, model key, cap, question id, sample index, the response text exactly as generated, and its
recorded grade; no problem statements and no reference answers. `tests/eval/test_eval_real_responses.py` checks every
grade against the released records in `reproduce/data` and re-grades every response (the AIME 2025 reference answers
and the LiveCodeBench tests come from `python -m eval.data.prepare`, since they are not redistributed). Like the records, the responses are released under CC BY 4.0
(`reproduce/LICENSE`).
