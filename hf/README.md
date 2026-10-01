# Hugging Face release tooling

Model cards and upload tooling for the DN-MOPD checkpoints. **All 21 models are released** (2026-10-01) as public
repositories under the Hugging Face account `XINLI1997`, grouped in the [DN-MOPD collection](https://huggingface.co/collections/XINLI1997/dn-mopd-6aba5f0df7bd8732d7205ed4).
The README of each repository is the card in `cards/`, byte for byte.

| File | Purpose |
|---|---|
| `models.json` | The released models: name, size, role, update count and paper row. |
| `paper_results.json` | Verbatim copy of the paper's tables. It is the only source of the numbers in the cards. |
| `model_card_template.md` | The card template. `{{...}}` fields are filled by `make_cards.py`. |
| `make_cards.py` | Renders `cards/<name>.md` for every model; `--check` fails if a card is out of date. |
| `cards/` | The generated cards, one per model. |
| `upload.py` | Uploads one export directory and its card. A dry run unless `--yes` is given. |
| `MANIFEST.md` | The released models, with sizes and the model-identity hashes of the evaluation records. |

## Released repositories

| Model | Repositories |
|---|---|
| DN-MOPD student, 80 updates (the paper's main model) | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B) |
| Label baseline student (MOPD with label routing), 80 updates | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-baseline-label`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-baseline-label) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-baseline-label) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-baseline-label) |
| GRPO experts (frozen teachers), math | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-teacher-math`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-teacher-math) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-teacher-math) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-math) |
| GRPO experts (frozen teachers), code | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-teacher-code`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-teacher-code) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-teacher-code) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-code) |
| GRPO experts (frozen teachers), IF | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-teacher-if`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-teacher-if) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-teacher-if) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-teacher-if) |
| DN-MOPD continued to 160 updates | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-160updates`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-160updates) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-160updates) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-160updates) |
| Label continued to 160 updates | `XINLI1997/DN-MOPD-Qwen3.5-{9B,4B,2B}-baseline-label-160updates`: [9B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-9B-baseline-label-160updates) · [4B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-4B-baseline-label-160updates) · [2B](https://huggingface.co/XINLI1997/DN-MOPD-Qwen3.5-2B-baseline-label-160updates) |

That is 15 core repositories (about 162 GB) and 6 160-update continuations (about 65 GB). Every uploaded file was
checked against the sha256 of the export it came from. [MANIFEST.md](MANIFEST.md) lists each model with its
evaluation-record identity hash.

To render the cards for another namespace, run `python hf/make_cards.py --namespace <name>` and pass matching
`--repo-id` values.

## Workflow

```bash
# 1) Render the cards (numbers come only from paper_results.json)
python hf/make_cards.py
python hf/make_cards.py --check          # CI: cards are up to date

# 2) Dry run: validates the export directory and prints what would be uploaded
python hf/upload.py --export-dir /path/to/export/DN-MOPD-Qwen3.5-9B \
    --repo-id XINLI1997/DN-MOPD-Qwen3.5-9B --card hf/cards/DN-MOPD-Qwen3.5-9B.md

# 3) Real upload (private by default); needs HF_TOKEN or `huggingface-cli login`
#    (only needed to re-upload; the 21 released repositories already hold these files)
python hf/upload.py ... --yes
```

The dry run refuses an export that contains optimizer or other training state, unexpected files, local absolute
paths or tokens in text files, or an index that does not match the shard headers. It also refuses a card that
still contains template fields.

Each export directory from training contains a copy of the base model's `README.md` and `LICENSE`. `upload.py`
replaces that README with the generated card. It keeps the base model's Apache-2.0 `LICENSE`, and it never modifies
the export directory itself.

## What every card states

- The base model (Qwen/Qwen3.5-*, Apache-2.0) and the training method, with the recipe summary and links to the code.
- The paper's scores for that model: Table 1 at 9B, Table 2 at 4B/2B, and Table 5 for the 160-update models. They
  are shown next to the initial model, and each student also appears next to its Label or DN-MOPD counterpart.
  DN-MOPD cards add the paired DN-MOPD − Label intervals and the MATH-500 contrast.
- That the export omits the base model's 15 `mtp.*` tensors, so MTP speculative decoding is unavailable.
- The chat format: non-thinking, `enable_thinking=False`. Qwen3.5-4B and 9B think by default.
- The limitations and scope, from the paper: DN-MOPD is not the best integration recipe overall, since SeqKD-SFT and
  task arithmetic score higher. There is one expert pool per size, one seed per released model, and the models are
  text-only.
- The citation.
