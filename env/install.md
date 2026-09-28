# Installation

DN-MOPD uses two Python environments, because the trainer and the evaluator need different `transformers` majors:

| Environment | Python | Used for | Key versions |
|---|---|---|---|
| **train** | 3.12 | GRPO teachers, OPD students (Label, DN-MOPD and every other row), SeqKD-SFT, merges, HF export, training-time graders | torch 2.11.0 (CUDA 13.0), SGLang 0.5.15.post1, transformers 5.12.1, Ray 2.56.1, math-verify 0.9.0 |
| **eval** | 3.10 | vLLM generation for the benchmark tables, graders, aggregation, paired bootstrap | vLLM 0.18.0, torch 2.10.0 (CUDA 12.8), transformers 4.57.6, math-verify 0.9.0 |

Both requirement files pin the exact versions used for the paper. The `constraints-*.txt` files are the complete
frozen package lists of the paper's environments. Pass them with `-c` to keep every transitive dependency at the
version the paper used. Every pin resolves from public PyPI, apart from the three PyTorch wheels of the training
environment, which come from the PyTorch CUDA 13.0 index.

If you only want to check the paper's numbers from the released evaluation records, you do not need either environment
or a GPU. See [Reproduce the tables without a GPU](#reproduce-the-tables-without-a-gpu).

> If your `pip.conf` points to a private mirror, add `--index-url https://pypi.org/simple` to the `pip install -r ...`
> commands below, or check that your mirror carries the same versions.

## Hardware and drivers

**What the paper used.** Every run used one node with 8 NVIDIA B200 GPUs (180 GB each):

- **Students** (all OPD rows): an FSDP actor with a colocated SGLang rollout engine on GPUs 0–3, and the three frozen
  teachers served by SGLang (tensor parallel 1) on GPUs 4, 5 and 6. GPU 7 is idle.
- **GRPO teachers**: all 8 GPUs.
- **Evaluation**: one vLLM engine (tensor parallel 1) per GPU.

**System requirements:**

| Requirement | Training env | Evaluation env |
|---|---|---|
| OS / CPU | Linux x86_64 | Linux x86_64 |
| glibc | ≥ 2.34 (the SGLang 0.5.15.post1 wheels are `manylinux_2_34`; e.g. Ubuntu 22.04+, RHEL 9+) | ≥ 2.28 |
| CUDA runtime | 13.0, shipped inside the PyTorch wheels (no system CUDA toolkit is needed) | 12.8, shipped inside the wheels |
| NVIDIA driver | one that supports CUDA 13.0 (R580 or newer) | one that supports CUDA 12.8 (R570 or newer) |

**Other GPUs.** We have only run the code on B200. Nothing in DN-MOPD itself depends on the GPU type: the method
adds one standard deviation per domain per batch. Other GPUs need more care:

- **H100 / H200.** Untested. They should work with the same wheels; SGLang and FlashInfer pick an attention backend
  for each architecture. With 80 GB per GPU, the 9B student will probably need memory-only changes: gradient
  checkpointing, a smaller `--sglang-mem-fraction-static` (0.60 in the paper) or more GPUs for the actor
  ([docs/hardware.md](../docs/hardware.md#smaller-gpus)). Shorter responses would change the recipe.
- **A100 80 GB.** Untested. FlashAttention-4 targets Hopper and Blackwell, so SGLang must fall back to its
  FlashInfer or Triton backends. Expect the same memory limits as on H100, and lower throughput.
- **Evaluation** runs one model per GPU. The 9B model in bf16 needs about 19 GB for its weights, and the rest of the
  GPU memory serves as KV cache. The paper used `gpu_memory_utilization=0.90`, `max_model_len=32768` and
  `max_num_seqs=64`. Smaller GPUs work with a lower `max_num_seqs`.

## 1. Training environment (`train`, Python 3.12)

### With conda

```bash
conda create -y -n dnmopd-train python=3.12
conda activate dnmopd-train

# 1) PyTorch 2.11.0, CUDA 13.0 build. Install it first, from the PyTorch index.
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 2) SGLang, FlashInfer, FlashAttention-4, Ray, transformers, graders, ... (public PyPI)
pip install -r env/requirements-train.txt -c env/constraints-train.txt

# 3) This repository: the miles trainer fork and the dn_mopd package, both editable
pip install --no-deps -e ./miles
pip install --no-deps -e .

# 4) Check
python env/check_env.py --env train
```

Notes:

- Install PyTorch first, as in step 1. It makes the CUDA build explicit and matches the paper environment's `+cu130`
  wheels. PyPI's Linux x86_64 wheels of `torch==2.11.0` and `torchvision==0.26.0` are CUDA 13.0 builds as well, so
  a PyPI-only install also works. Do not mix CUDA builds: a cu128 `torchvision` next to a cu130 `torch` makes
  `import sglang` fail in torchvision's CUDA-version check.
- `pip install -e ./miles --no-deps` installs **this repository's** fork of miles. The PyPI package named `miles` is
  an unrelated project, so never `pip install miles`. The fork's own dependencies are pinned in
  `requirements-train.txt`.
- `pip install -e .` installs the framework-agnostic `dn_mopd` package from the `pyproject.toml` at the repository
  root. `dn_mopd` itself needs only torch.
- The paper's environment was built without Megatron-LM, TransformerEngine or apex; the students and teachers use
  miles' FSDP backend. You do not need them either.

### With uv

```bash
uv venv --python 3.12 .venv-train
source .venv-train/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
uv pip install -r env/requirements-train.txt -c env/constraints-train.txt
uv pip install --no-deps -e ./miles -e .
python env/check_env.py --env train
```

Do not put both indexes into one `uv pip install` with `--index-strategy unsafe-best-match`. That also pulls the
`+cu130` builds of `torchao` and `torchcodec`, whereas the paper used their PyPI builds.

### CUDA libraries at run time

PyTorch 2.11 (cu130) ships its CUDA libraries inside `site-packages/nvidia/` (for example
`site-packages/nvidia/cu13/lib`). The dynamic loader does not search there. A teacher SGLang server started outside Ray
therefore fails at import with `libnvrtc.so.13: cannot open shared object file`. The launchers in `recipes/` prepend
these directories to `LD_LIBRARY_PATH` before they start any server (`recipes/qwen3.5/common.sh`). If you start
servers yourself, do the same:

```bash
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
for d in "$SP"/nvidia/*/lib "$SP"/nvidia/*/lib64 "$SP"/nvidia/*/*/lib; do
  [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH
```

Run `ray stop --force` before relaunching a job on the same machine.

### Exact-provenance SGLang (optional)

The paper installed SGLang in editable mode from git tag `v0.5.15.post1`
(commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`), with no local patches. The PyPI wheel
`sglang==0.5.15.post1` is built from the same release and declares the same core dependencies. To build from the
commit instead, which needs a Rust toolchain (`cargo`):

```bash
git clone https://github.com/sgl-project/sglang.git third_party/sglang
git -C third_party/sglang checkout 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
pip install --no-build-isolation -e "third_party/sglang/python"
```

`--no-build-isolation` is required: the build imports the already installed torch.

## 2. Evaluation environment (`eval`, Python 3.10)

### With conda

```bash
conda create -y -n dnmopd-eval python=3.10
conda activate dnmopd-eval
pip install -r env/requirements-eval.txt -c env/constraints-eval.txt
pip install --no-deps -e .          # dn_mopd (optional for evaluation; lets check_env.py find it)
python -m eval.external.fetch       # official checkers + nltk data at pinned commits, sha256-verified
python env/check_env.py --env eval
```

### With uv

```bash
uv venv --python 3.10 .venv-eval
source .venv-eval/bin/activate
uv pip install -r env/requirements-eval.txt -c env/constraints-eval.txt
uv pip install --no-deps -e .
python -m eval.external.fetch
python env/check_env.py --env eval
```

### Official checkers (not on PyPI)

The benchmark graders call the official code of each benchmark, pinned by commit:

| Checker | Source | Commit |
|---|---|---|
| LiveCodeBench (`lcb_runner`: extraction and test execution) | https://github.com/LiveCodeBench/LiveCodeBench | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |
| IFEval (`instruction_following_eval/`) | https://github.com/google-research/google-research | `b24f2136e8ef405b900b5619760126304f190941` |
| IFBench | https://github.com/allenai/IFBench | `1091c4c3de6c1f6ed12c012ed68f11ea450b0117` |

`python -m eval.external.fetch` downloads exactly these files, plus the nltk data both IF checkers need, into
`eval/third_party/` (git ignores it; `DN_MOPD_THIRD_PARTY` sets another directory), and checks each file's sha256
against `eval/external/PINS.json`. Nothing is vendored and nothing is pip-installed: the graders put the
fetched directories on `sys.path` when they load. Their pip dependencies (`absl-py`, `langdetect`, `immutabledict`,
`nltk`, `emoji`, `syllapy`, `regex`) are pinned in `requirements-eval.txt`. The benchmark questions themselves are
rebuilt from their pinned public sources by `python -m eval.data.prepare`, which checks every question against
the released catalog.

In the paper, only generation ran in this environment. The math and LiveCodeBench answers were graded in the
training environment, with the same `math-verify==0.9.0`. The IFEval/IFBench checkers ran with the pure-Python
packages pinned in `requirements-eval.txt`. The eval environment above holds all graders, so a single environment can
both generate and grade. To grade math exactly as the paper did, it uses the grading environment's ANTLR runtime
(`antlr4-python3-runtime==4.9.3`; see "Known differences" below).

## 3. Tests (CPU)

The test suites run on CPU. `env/requirements-dev.txt` pins the test tools: pytest (the version already in both
requirement files) and, optionally, `shellcheck-py`, with which `tests/trainer` also lints the recipe scripts. Install
it into each environment that runs tests, then run `make check` from the repository root:

```bash
pip install -r env/requirements-dev.txt            # in the train and in the eval environment
make check TRAIN_PYTHON=/path/to/dnmopd-train/bin/python EVAL_PYTHON=/path/to/dnmopd-eval/bin/python
```

| Tests | Environment | Needs |
|---|---|---|
| `tests/core` (`dn_mopd`, the data manifest and download) | train (any env with torch works) | the data tests skip until `python data/download.py` has fetched the prompt sets |
| `tests/trainer` (hook, loss, rewards, recipes, scripts) | train | the miles fork and SGLang imports; no GPU |
| `tests/eval` (graders, pipeline, statistics, `reproduce.py`) | eval | grader tests skip until `python -m eval.external.fetch` and `python -m eval.data.prepare` have run |

`make check` runs the three suites one after the other (`make check-core`, `check-trainer`, `check-eval` run one
each). Without `TRAIN_PYTHON` / `EVAL_PYTHON`, every suite uses `python`, the active environment.

## Reproduce the tables without a GPU

`reproduce/` recomputes every Qwen3.5 number in the paper from the released per-question records. It needs only
Python 3 and numpy.

```bash
python -m venv .venv-repro && source .venv-repro/bin/activate
pip install "numpy>=2"
python reproduce/reproduce.py --no-bootstrap   # point estimates (seconds)
python reproduce/reproduce.py                  # plus the paired bootstrap intervals (a few minutes); prints ALL MATCH
```

## Docker (optional)

`env/Dockerfile.train` and `env/Dockerfile.eval` build the two environments on public NVIDIA CUDA base images
(`nvidia/cuda:13.0.2-devel-ubuntu24.04` and `nvidia/cuda:12.8.1-devel-ubuntu22.04`). They are provided as a starting
point and have not been built as part of the release checks. The build context is `env/`. You mount the repository
at run time, and the entrypoint installs `./miles` and `dn_mopd` in editable mode.

```bash
docker build -f env/Dockerfile.train -t dn-mopd:train env/
docker build -f env/Dockerfile.eval  -t dn-mopd:eval  env/
docker run --gpus all --ipc=host --shm-size=64g -it -v "$PWD":/workspace/DN-MOPD dn-mopd:train \
    python env/check_env.py --env train
```

## Known differences from the paper's environments

These are the only intentional differences between the files here and the environments that produced the paper:

1. **SGLang** comes from the PyPI wheel of the same release (`0.5.15.post1`), not from an editable git install.
   In the paper's environment pip reported the editable install as `sglang 0.0.0`; at runtime,
   `sglang.__version__` was `0.5.15.post1`.
2. **cuDNN.** The paper's training environment also had `nvidia-cudnn-cu12==9.16.0.29`, a leftover compatibility pin
   from an earlier torch build. That wheel writes the same `nvidia/cudnn/lib/libcudnn.so.9` as PyTorch's own
   `nvidia-cudnn-cu13==9.19.0.56` and overwrote it. PyTorch then raises a cuDNN version-mismatch error as soon as
   cuDNN is initialised. The pin is omitted here. <!-- TODO verify: GPU smoke test with the omitted pin -->
3. **SGLang extras.** The paper's environment was built with `sglang[all]`, which adds diffusion, tracing and HTTP/2
   extras that the training code does not use. They are not in `requirements-train.txt`; their versions remain in
   the constraints file.
4. **Graders.** The IFEval/IFBench checker packages (`absl-py`, `langdetect`, `immutabledict`, `nltk`, `emoji`,
   `syllapy`) were loaded from a side directory in the paper. Here they are regular pinned requirements of the eval
   environment, at the same versions.
5. **ANTLR runtime of the eval environment.** The paper's generation environment had
   `antlr4-python3-runtime==4.13.2`; its math grading ran in the training environment, which has `4.9.3`. math-verify
   parses LaTeX answers through this runtime, and the two versions grade some answers differently (one of the 68
   paper answers shipped in `tests/eval/fixtures/` flips under 4.13.2). Because the eval environment here also grades,
   it pins `4.9.3`. Nothing else in the environment depends on the runtime, and the resolved package set is otherwise
   unchanged.

`env/check_env.py` prints the installed versions and warns about any deviation from these pins.
