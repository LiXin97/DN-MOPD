# Reproduce DN-MOPD

Evaluate a released checkpoint and retrain the paper's models. To check the paper's numbers on a CPU in seconds, see
the [Quick start](../README.md#quick-start) of the README.

## Evaluate a released checkpoint (one GPU)

From the repository root, in the paper's evaluation environment (Python 3.10, vLLM 0.18.0, PyTorch 2.10, pinned
versions):

```bash
# 1. Evaluation environment and the official LiveCodeBench / IFEval / IFBench checkers
conda create -y -n dnmopd-eval python=3.10 && conda activate dnmopd-eval
pip install -r env/requirements-eval.txt -c env/constraints-eval.txt
pip install --no-deps -e .    # the dn_mopd package (it needs only PyTorch)
python -m eval.external.fetch
python env/check_env.py --env eval

# 2. The 2B DN-MOPD student, about 4.4 GB, into the Hugging Face cache
hf download XINLI1997/DN-MOPD-Qwen3.5-2B

# 3. The six benchmarks: download and verify them (once)
python -m eval.data.prepare

# 4. Generate answers (GPU) and grade them (CPU)
for s in aime25 aime26 lcb_v5 lcb_v6 ifeval ifbench; do
  python -m eval.generate --model XINLI1997/DN-MOPD-Qwen3.5-2B --name my_model \
      --suite $s --max-tokens 16384
  python -m eval.grade --gen outputs/eval/my_model/$s/cap16384
done

# 5. Scores and average, then a paired comparison with the paper's 2B MOPD model
python -m eval.aggregate --records outputs/eval --cap 16384
python -m eval.compare my_model q35_2b_s_route --records-a outputs/eval --cap 16384
```

The settings match the paper: non-thinking mode, temperature 1.0, top-p 1.0, and 64 / 6 / 16 samples per question
for math / code / IF. Answers are sampled, so scores vary slightly from run to run.

## Train it yourself

Every run in the paper used one node with 8 GPUs (NVIDIA B200, 180 GB each). A teacher takes 20–38 h; a student
takes about 2.7 h (2B) to 4.5 h (9B). Training uses its own environment (Python 3.12, PyTorch 2.11 with CUDA 13.0,
SGLang 0.5.15.post1, transformers 5.12.1):

```bash
# training environment
conda create -y -n dnmopd-train python=3.12 && conda activate dnmopd-train
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r env/requirements-train.txt -c env/constraints-train.txt
pip install --no-deps -e ./miles -e .
python env/check_env.py --env train

python data/download.py              # training prompts from Hugging Face, about 1 GB
export MODEL_ROOT=/path/to/models    # holds Qwen3.5-2B/ (downloaded if missing)

for d in math code ifeval; do        # 1) one GRPO-trained teacher per domain
  bash recipes/qwen3.5/train_teacher_grpo.sh $d 2b
done
bash recipes/qwen3.5/train_student.sh label   2b 42   # 2) MOPD baseline, seed 42
bash recipes/qwen3.5/train_student.sh dn_mopd 2b 42   # 3) DN-MOPD, seed 42
```

Rerun the same command to resume after an interruption. The code teacher and the students run model-written programs
in a local judge. It is not a sandbox, so use a machine you control.
