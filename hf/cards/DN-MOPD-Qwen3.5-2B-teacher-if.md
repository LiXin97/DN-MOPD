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
- teacher
- grpo
---

# DN-MOPD-Qwen3.5-2B-teacher-if

The Qwen3.5-2B **instruction-following (IF) expert** used as a frozen teacher in the DN-MOPD paper: Qwen3.5-2B trained with GRPO on instruction-following (IF) prompts. It is one of three same-size experts (math, code, IF) that the Qwen3.5-2B students learn from.

**Paper:** *Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation*
([arXiv:2609.35347](https://arxiv.org/abs/2609.35347), [project page](https://lixin.ai/DN-MOPD)) · **Code:** [github.com/LiXin97/DN-MOPD](https://github.com/LiXin97/DN-MOPD)

## Model details

|  |  |
|---|---|
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Role | Instruction-following (IF) expert; frozen teacher of the 2B students |
| Training | GRPO from the base model, 400 updates, seed 42 |
| Precision | bfloat16 |
| Chat format | non-thinking (`enable_thinking=False`) |
| License | Apache-2.0 (same as the base model) |

## Training recipe

- **Algorithm:** GRPO on instruction-following (IF) prompts with a verifiable reward; no KL or entropy term.
- **Batching:** 128 prompts per rollout, 8 responses per prompt, 256 responses per optimizer step. Dynamic sampling drops prompt groups without reward variation (at most 8 generation batches per rollout).
- **Lengths:** prompt ≤ 2,048 tokens, response ≤ 8,192 tokens, temperature 1.0.
- **Optimizer:** Adam, learning rate 1e-6 (constant after 10 warm-up updates), betas (0.9, 0.98), weight decay 0.1, gradient clipping 1.0.
- **Length of training:** 400 updates (runs were capped at 400), seed 42.

The full recipe, with the launch scripts for every row of the paper's tables, is in
[`recipes/qwen3.5/`](https://github.com/LiXin97/DN-MOPD/tree/main/recipes/qwen3.5) and [`docs/recipe.md`](https://github.com/LiXin97/DN-MOPD/blob/main/docs/recipe.md).

## Usage

This model was trained and evaluated with the **non-thinking** chat format. Pass `enable_thinking=False` to the chat
template. The evaluation settings in the paper were temperature 1.0 and top-p 1.0, with up to
16,384 new tokens (8,192 in the appendix).

**vLLM** (the paper used vLLM 0.18.0):

```python
from vllm import LLM, SamplingParams

llm = LLM(model="XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-if", max_model_len=32768)
params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=16384, seed=42)
messages = [{"role": "user", "content": "Find the sum of all positive divisors of 36. Put the final answer in \\boxed{}."}]
outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
print(outputs[0].outputs[0].text)
```

**Transformers** (Qwen3.5 needs `transformers>=5`; the paper's training environment used 5.12.1):

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-if")
model = AutoModelForImageTextToText.from_pretrained("XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-if", dtype=torch.bfloat16, device_map="auto")
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
| **DN-MOPD-Qwen3.5-2B-teacher-if** (IF expert) | 15.6 | 12.5 | 52.8 | 27.0 |
| Initial student (Qwen3.5-2B) | 17.6 | 11.3 | 43.3 | 24.0 |

Scores (%) from the paper; training seed 42; 16,384-token evaluation cap; non-thinking chat template; temperature 1.0, top-p 1.0, generation seed 42. AIME25/AIME26: avg@64. LiveCodeBench v5/v6 (167/175 disjoint problems): avg@6. IFEval/IFBench: strict prompt accuracy, avg@16. Total: mean of the six task scores.

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

- A specialist: it was trained for one domain only. In the table above it scores below the initial model on Math.
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
