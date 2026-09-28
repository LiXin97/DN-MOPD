---
license: apache-2.0
license_link: LICENSE
base_model: {{base_model}}
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
{{extra_tags}}
---

# {{name}}

{{summary}}

**Paper:** *Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation*
([project page]({{project_url}}), arXiv: coming soon) · **Code:** [{{code_url_short}}]({{code_url}})

## Model details

{{details_table}}

## Training recipe

{{recipe}}

The full recipe, with the launch scripts for every row of the paper's tables, is in
[`recipes/qwen3.5/`]({{code_url}}/tree/main/recipes/qwen3.5) and [`docs/recipe.md`]({{code_url}}/blob/main/docs/recipe.md).

## Usage

This model was trained and evaluated with the **non-thinking** chat format. Pass `enable_thinking=False` to the chat
template.{{thinking_note}} The evaluation settings in the paper were temperature 1.0 and top-p 1.0, with up to
16,384 new tokens (8,192 in the appendix).

**vLLM** (the paper used vLLM 0.18.0):

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{{repo_id}}", max_model_len=32768)
params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=16384, seed=42)
messages = [{"role": "user", "content": "Find the sum of all positive divisors of 36. Put the final answer in \\boxed{}."}]
outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
print(outputs[0].outputs[0].text)
```

**Transformers** (Qwen3.5 needs `transformers>=5`; the paper's training environment used 5.12.1):

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{repo_id}}")
model = AutoModelForImageTextToText.from_pretrained("{{repo_id}}", dtype=torch.bfloat16, device_map="auto")
messages = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=1.0, top_p=1.0)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Evaluation

{{evaluation}}

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

{{limitations}}

## Citation

```bibtex
@misc{li2026dnmopd,
  title  = {Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation},
  author = {Li, Xin and Jiang, Hao and Gao, Xin and Wang, Annan and Xie, Yuchen and Guo, Jinghao and Qu, Xingwei and Zhang, Yichi and Yuen, Chau},
  year   = {2026},
  url    = {https://lixin.ai/DN-MOPD}
}
```

This model is a fine-tuned derivative of [{{base_model}}](https://huggingface.co/{{base_model}}) by the Qwen team,
released under the Apache License 2.0.
