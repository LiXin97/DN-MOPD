# Hardware

Every result in the paper was trained on **one node with 8 NVIDIA B200 GPUs (180 GB each)**; no job spans nodes. The
recipes assume one 8-GPU Linux node with a CUDA 13-capable driver (the PyTorch 2.11 + cu130 environment of
`env/install.md`) and do not hard-code a GPU model.

## Per job

| Job | GPUs | Notes |
|---|---|---|
| GRPO teacher | 8 | FSDP actor + 8 colocated SGLang rollout engines; peak 150-170 GB per GPU in the paper runs (rollout pool 60 % of the GPU, released while the actor trains; gradient checkpointing on) |
| OPD student | 7 (0-3 student, 4-6 teachers) | 4 GPUs for the FSDP actor + colocated rollout, one SGLang server per teacher; GPU 7 is not used |
| SeqKD teacher answers | 3 | one vLLM process per domain (TP 1) |
| SeqKD-SFT | 8 | FSDP only, gradient checkpointing on |
| Injection bank | 8 | one vLLM process per shard |
| Evaluation | 1 per process | see `eval/` |
| Code judge, merges, HF export | CPU | |

GPU indices, teacher ports and memory fractions are in `recipes/qwen3.5/configs/student_opd.yaml`
(`student_gpus`, `teacher_servers`); `STUDENT_GPUS` and `GPUS` override them per run. Changing the number of student
GPUs changes only data parallelism: the global batch (512) and micro-batch (1) stay fixed.

## Smaller GPUs

The paper settings leave little headroom below ~140 GB per GPU for the 9B student (8,192-token responses, micro-batch
1, no gradient checkpointing). Options that change memory but not the update:

- `EXTRA_TRAINER_ARGS="--gradient-checkpointing --log-probs-chunk-size 2048"` for students (the teachers and SFT
  already use them);
- lower `sglang_mem_fraction_static` (student rollout pool) or the teacher servers' `mem_fraction_static` in
  `configs/student_opd.yaml`: they size KV caches, not scores;
- a different number of student GPUs (`STUDENT_GPUS`), as long as it divides the global batch of 512.

Tensor parallelism is not supported on this FSDP path (the Megatron-free log-prob fallback needs an unsharded
vocabulary). Lowering the batch sizes, response lengths or learning-rate schedule changes the recipe and is not a
reproduction.

## CPU, memory and disk

- **CPU**: the code judge runs model-written programs in a process pool (6 uvicorn workers x 16 processes by default,
  `CODE_JUDGE_WORKERS` / `CODE_JUDGE_POOL`); math grading uses a few worker processes. The paper runs had 22-31 CPU
  cores per node, which was enough; more cores lower code-reward latency.
- **Host memory**: Ray, the rollout manager and the judge; a few hundred GB is comfortable for the 9B runs (the colocated
  engines offload weights to host memory while the actor trains).
- **Disk**: an FSDP save (model + optimizer) is ~13 GB (2B), ~26 GB (4B), ~51 GB (9B). A student run writes 8 saves, a
  400-update teacher 40, SFT 2. A Hugging Face export is ~4.2 / 8.5 / 18 GB. Resuming needs only the newest complete
  save (`latest_checkpointed_iteration.txt`), so older `iter_*` directories can be removed.
- **Network**: none during training once the models and data are local (set `HF_HUB_OFFLINE=1`). The teacher servers,
  the code judge and Ray bind to the local node; the code judge listens on 127.0.0.1 only.

## Wall-clock

See [recipe.md](recipe.md#wall-clock-memory-and-disk): an 80-update student takes about 2.7 h (2B) to 4.5 h (9B), a
teacher 20-38 h, SeqKD-SFT about 1 h, on the paper's hardware.

## Safety of the code judge

The PRIME judge executes model-generated programs with only PRIME's `reliability_guard` (dangerous `os` / `shutil`
functions disabled in the child); it is not a sandbox. `start_code_judge.sh` runs it with ulimits (no core files,
process count, 8 GiB address space per process, 1 GiB file size), niced and bound to localhost. Run the recipes on a
machine or container you control, as an unprivileged user.
