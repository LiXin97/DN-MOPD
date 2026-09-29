---
license: apache-2.0
license_link: LICENSE
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
tags:
- dn-mopd
- on-policy-distillation
- multi-teacher-distillation
- qwen3.5
- student
- dn-mopd-student
---

# DN-MOPD-Qwen3.5-2B

A Qwen3.5-2B student trained with **DN-MOPD** (Domain-Normalized Multi-Teacher On-Policy Distillation). Three same-size RL experts (math, code, instruction following) teach one student on its own responses; each prompt is scored by the expert of its domain, and DN-MOPD rescales each domain's token-level feedback by its measured spread, w_d = clip(σ_all / σ_d, 0.25, 4), so that no domain dominates the shared update.

**Paper:** *Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation*
([arXiv:2609.35347](https://arxiv.org/abs/2609.35347), [project page](https://lixin.ai/DN-MOPD)) · **Code:** [github.com/LiXin97/DN-MOPD](https://github.com/LiXin97/DN-MOPD)

## Model details

|  |  |
|---|---|
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Method | DN-MOPD: label-routed on-policy distillation with per-domain advantage scaling w_d = clip(σ_all / σ_d, 0.25, 4) |
| Teachers (same size) | [math](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-math), [code](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-code), [IF](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-if) |
| Training | 80 updates from the base model, student seed 42 |
| Precision | bfloat16 |
| Chat format | non-thinking (`enable_thinking=False`) |
| License | Apache-2.0 (same as the base model) |

## Training recipe

- **Prompts:** 2,700 training prompts, 900 each for mathematics, code and instruction following; each prompt carries its domain label and is scored by that domain's expert (label routing).
- **Advantage:** per sampled token, teacher log-probability minus the actor-recomputed student log-probability, used in the clipped policy-gradient OPD loss (ratio clip 0.2/0.2); no KL or entropy term.
- **DN-MOPD scaling:** on every batch, σ_d is the population standard deviation of the teacher–rollout log-ratios over the valid response tokens of domain d, and σ_all pools all domains; each domain's advantages are multiplied by w_d = clip(σ_all / σ_d, 0.25, 4) (w_d = 1 if a statistic is degenerate). Signs are preserved.
- **Batching:** 64 prompts × 8 responses = 512 responses per update, one optimizer step per rollout batch.
- **Lengths:** prompt ≤ 2,048 tokens, response ≤ 8,192 tokens, temperature 1.0.
- **Optimizer:** Adam, learning rate 1e-6 (constant after 5 warm-up updates), betas (0.9, 0.98), weight decay 0.1, gradient clipping 1.0.
- **Length of training:** 80 updates, student seed 42.

The full recipe, with the launch scripts for every row of the paper's tables, is in
[`recipes/qwen3.5/`](https://github.com/LiXin97/DN-MOPD/tree/main/recipes/qwen3.5) and [`docs/recipe.md`](https://github.com/LiXin97/DN-MOPD/blob/main/docs/recipe.md).

## Usage

This model was trained and evaluated with the **non-thinking** chat format. Pass `enable_thinking=False` to the chat
template. The evaluation settings in the paper were temperature 1.0 and top-p 1.0, with up to
16,384 new tokens (8,192 in the appendix).

**vLLM** (the paper used vLLM 0.18.0):

```python
from vllm import LLM, SamplingParams

llm = LLM(model="XINLI1997/DN-MOPD-Qwen3.5-2B", max_model_len=32768)
params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=16384, seed=42)
messages = [{"role": "user", "content": "Find the sum of all positive divisors of 36. Put the final answer in \\boxed{}."}]
outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
print(outputs[0].outputs[0].text)
```

**Transformers** (Qwen3.5 needs `transformers>=5`; the paper's training environment used 5.12.1):

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("XINLI1997/DN-MOPD-Qwen3.5-2B")
model = AutoModelForImageTextToText.from_pretrained("XINLI1997/DN-MOPD-Qwen3.5-2B", dtype=torch.bfloat16, device_map="auto")
messages = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=1.0, top_p=1.0)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Evaluation

Paper Table 2 (Qwen3.5-2B; each domain averages its two tasks: AIME25/AIME26, LiveCodeBench v5/v6, IFEval/IFBench):

| Model | Math | Code | IF | Total |
|---|:---:|:---:|:---:|:---:|
| **DN-MOPD-Qwen3.5-2B** (DN-MOPD) | 20.7 | 16.3 | 49.9 | 29.0 |
| Label (label-routed MOPD) | 17.0 | 13.4 | 49.5 | 26.6 |
| Initial student (Qwen3.5-2B) | 17.6 | 11.3 | 43.3 | 24.0 |

Scores (%) from the paper; training seed 42; 16,384-token evaluation cap; non-thinking chat template; temperature 1.0, top-p 1.0, generation seed 42. AIME25/AIME26: avg@64. LiveCodeBench v5/v6 (167/175 disjoint problems): avg@6. IFEval/IFBench: strict prompt accuracy, avg@16. Total: mean of the six task scores.

Six-task Total, DN-MOPD minus Label (pp), this checkpoint vs. the seed-42 Label checkpoint, with paired 95% bootstrap intervals (questions resampled within each task, B = 10,000):

- 16K cap: +2.36 [+1.58, +3.17] (DN-MOPD 28.96 vs. Label 26.60)
- 8K cap: +2.92 [+2.02, +3.85] (DN-MOPD 27.87 vs. Label 24.96)

Across student seeds 42/43/44 (16K), the mean gain over Label is +2.34 [+1.86, +2.84]; only the seed-42 model is released.

MATH-500 (16 answers per question), DN-MOPD minus Label: 16K +5.95 [+4.95, +6.96]; 8K +6.85 [+5.73, +7.99].

## Files

- Weights in Hugging Face format (`Qwen3_5ForConditionalGeneration`, bfloat16), exported from the FSDP training
  checkpoint.
- **The export omits the 15 multi-token-prediction tensors (`mtp.*`) of the base model.** All other tensors have the
  base model's names and shapes. MTP-based speculative decoding is therefore not available with this checkpoint.
  Ordinary decoding is unaffected: the paper's evaluations used exactly these files.
- `config.json`, the tokenizer files and `chat_template.jinja` are the base model's, unchanged.
- The vision encoder is carried over from the base model. Training and evaluation used text only.
- `LICENSE` is the base model's Apache-2.0 license.

## Limitations

- DN-MOPD is not the best integration recipe overall: in the paper, SeqKD-SFT and task-arithmetic merging (ParamMerge-TA) reach higher Totals under their own recipes. DN-MOPD's gains are over label-routed multi-teacher OPD and, by a smaller margin whose intervals sometimes include zero, over the strongest single-teacher student.
- Gains are largest in mathematics; code and IF gains are smaller and less consistent (at 9B, code does not improve at the 16K cap). The IF multiplier usually sits at the 0.25 lower bound, and the clipping bounds were not tuned per size.
- Trained on 2,700 prompts in three domains with a single seed (42) and one expert pool per size, within one model family; results for other teacher–student configurations may differ.
- Trained with responses of at most 8,192 tokens and evaluated only in non-thinking mode; thinking mode, multimodal inputs, other languages and safety behaviour were not evaluated beyond the base model.

## Citation

```bibtex
@article{li2026dnmopd,
  title   = {Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation},
  author  = {Li, Xin and Jiang, Hao and Gao, Xin and Wang, Annan and Xie, Yuchen and Guo, Jinghao and Qu, Xingwei and Zhang, Yichi and Yuen, Chau},
  journal = {arXiv preprint arXiv:2609.35347},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.35347}
}
```

This model is a fine-tuned derivative of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by the Qwen team,
released under the Apache License 2.0.
