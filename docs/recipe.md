# Training recipe (Qwen3.5 2B / 4B / 9B)

This page lists every setting of every trained row in the paper, how to run it with `recipes/qwen3.5/`, what it costs,
and the known quirks of the runs behind the paper's numbers. The values below are also the values in
`recipes/qwen3.5/configs/`; the scripts read them from there.

- [Pipeline](#pipeline)
- [Paths and environment](#paths-and-environment)
- [GPU layout](#gpu-layout)
- [Rows, methods and config keys](#rows-methods-and-config-keys)
- [GRPO teachers](#grpo-teachers)
- [OPD students](#opd-students)
- [DN-MOPD and the control runs](#dn-mopd-and-the-control-runs)
- [Annealed injection](#annealed-injection)
- [SeqKD-SFT](#seqkd-sft)
- [Weight merges](#weight-merges)
- [Hugging Face export](#hugging-face-export)
- [Wall-clock, memory and disk](#wall-clock-memory-and-disk)
- [Known quirks of the paper runs](#known-quirks-of-the-paper-runs)

## Pipeline

All commands run from the repository root on one node with 8 GPUs (see [hardware.md](hardware.md)). Every training
script is resumable: rerun the same command after an interruption.

```bash
export MODEL_ROOT=/path/to/models          # holds Qwen3.5-2B/, Qwen3.5-4B/, Qwen3.5-9B/ (downloaded from Hugging Face)
export PYTHON=python RAY=ray               # the training environment (env/install.md)
export VLLM_PYTHON=/path/to/eval-env/bin/python   # vLLM, for SeqKD answers and the injection bank

# 0. the four prompt sets (not in git): download from Hugging Face and check them against data/MANIFEST.json
python data/download.py

# 1. GRPO teachers (8 GPUs each)
for d in math code ifeval; do bash recipes/qwen3.5/train_teacher_grpo.sh $d 2b; done

# 2. OPD students (GPUs 0-3 student, 4-6 teachers); METHOD as in the table below
bash recipes/qwen3.5/train_student.sh label 2b
bash recipes/qwen3.5/train_student.sh dn_mopd 2b            # seed 42, 80 updates
bash recipes/qwen3.5/train_student.sh dn_mopd 2b 43         # another seed
bash recipes/qwen3.5/train_student.sh dn_mopd 2b 42 160     # continue the same run to 160 updates

# 3. annealed injection needs the size's trajectory bank first
bash recipes/qwen3.5/build_injection_bank.sh 2b
bash recipes/qwen3.5/train_student.sh annealed_injection 2b

# 4. SeqKD-SFT and the weight merges
bash recipes/qwen3.5/gen_seqkd.sh 2b && bash recipes/qwen3.5/train_seqkd_sft.sh 2b
python recipes/qwen3.5/merge.py --kind avg --size 2b
python recipes/qwen3.5/merge.py --kind ta  --size 2b
```

Each run ends with a Hugging Face export (`EXPORT_HF=0` skips it) that the evaluation pipeline (`eval/`) reads.
`bash recipes/qwen3.5/smoke_test.sh` is a short end-to-end check of the whole path (about an hour on 8 GPUs).

## Paths and environment

| Variable | Default | Meaning |
|---|---|---|
| `DN_MOPD_ROOT` | the repository | code root |
| `MODEL_ROOT` | `$DN_MOPD_ROOT/models` | base models as `$MODEL_ROOT/Qwen3.5-<S>`; `BASE_MODEL=<dir>` overrides; a missing model is downloaded to the Hugging Face cache unless `HF_HUB_OFFLINE=1` |
| `DATA_ROOT` | `$DN_MOPD_ROOT/data` | `student/train_3domain.jsonl`, `teacher/{math,code,if}_train.jsonl` (`python data/download.py` fetches them into `data/`) |
| `CKPT_ROOT` | `$DN_MOPD_ROOT/outputs/checkpoints` | FSDP checkpoints and HF exports |
| `OUT_ROOT` | `$DN_MOPD_ROOT/outputs` | logs, per-run configs, hook audit files, SeqKD corpora, injection banks, service logs |
| `TEACHER_MATH`, `TEACHER_CODE`, `TEACHER_IF` | `$CKPT_ROOT/qwen3.5-<S>_teacher_<domain>_hf` | the teachers a student distills from |
| `PYTHON`, `RAY`, `VLLM_PYTHON` | `python`, `ray`, `$PYTHON` | interpreters (training env; vLLM env) |
| `CODE_JUDGE_PORT` | `17580` | the local code judge |
| `DN_MOPD_SANITIZE_NCCL` | `1` | unset inherited `NCCL_*` / `UCX_*` and set `NCCL_NET_PLUGIN=none` (single node); `0` keeps your settings |
| `RAY_TMPDIR`, `RAY_PORT`, `LOCAL_IP` | `/tmp/ray_dn_mopd`, `6379`, first `hostname -I` address | Ray head node; every training script runs `ray stop --force` first, so use a dedicated node |
| `EXTRA_TRAINER_ARGS` | empty | extra miles options, e.g. memory-only `--gradient-checkpointing` on smaller GPUs |

Run names: teachers `qwen3.5-<S>_teacher_<domain>`, students `qwen3.5-<S>_<METHOD>_s<SEED>`, SFT
`qwen3.5-<S>_seqkd_sft`, merges `qwen3.5-<S>_merge_{avg,ta}_hf`.

## GPU layout

| Job | GPUs | What runs where |
|---|---|---|
| GRPO teacher | 0-7 | FSDP actor on 8 GPUs with 8 colocated SGLang rollout engines (TP 1); the code judge on the CPU |
| OPD student | 0-3 student, 4-6 teachers | FSDP actor on 4 GPUs with 4 colocated SGLang engines; math teacher GPU 4 (port 13141), code teacher GPU 5 (13142), IF teacher GPU 6 (13140), SGLang TP 1, `--chunked-prefill-size 4096`, memory fractions 0.35 / 0.35 / 0.85; GPU 7 idle; the code judge on the CPU |
| SeqKD answers | 0, 1, 2 | one vLLM process per domain |
| SeqKD-SFT | 0-7 | FSDP only (`--debug-train-only`) |
| Injection bank | 0-7 | one vLLM process per shard (8 shards per domain); verification on the CPU |
| Merge, HF export | CPU | |

The colocated rollout engines release their memory while the actor trains (`--colocate` implies offloading).

## Rows, methods and config keys

`METHOD` is the first argument of `train_student.sh`; the hook fields are the keys of the run config that
`Uni_OPD_utils.mopd_hook` reads (`$OUT_ROOT/runs/<run>/hook_config.json`, written from `configs/methods.yaml`). The last
column is the model key of the per-question records in `reproduce/` (`q35_<size>_<key>`).

| Paper row | Recipe | Route / hook fields | Record key |
|---|---|---|---|
| RL experts (GRPO teachers) | `train_teacher_grpo.sh {math,code,ifeval} S` | `configs/teacher_grpo.yaml` | `t_math`, `t_code`, `t_ifeval` |
| Label (MOPD with label routing), seed 42 | `train_student.sh label S` | legacy route: the label map `{default: code, math, code, ifeval}` is the router | `s_route` |
| Label through the hook | `train_student.sh label_hook S` | `mode: label` | (same target as `label`; see quirks) |
| DN-MOPD | `train_student.sh dn_mopd S [SEED]` | `mode: label`, `adv_norm: per_domain_scale` | `s_mdnorm`, `ctl_dnorm_s43`, `ctl_dnorm_s44` |
| Single-teacher OPD (math / code / IF) | `train_student.sh single_{math,code,if} S` | legacy route, every map key -> that teacher | `s_singlemath`, `s_singlecode`, `s_singleifeval` |
| Uniform pool | `train_student.sh uniform_pool S` | `mode: pool` | `s_pool` |
| Dynamic router | `train_student.sh dynamic_router S` | `mode: dynamic` | `s_dynamic` |
| Annealed injection | `build_injection_bank.sh S`, then `train_student.sh annealed_injection S` | `mode: label`, `inject_bank_path`, `inject_adv_const: 1.0`, `inject_anneal_steps: 40` | `s_manneal` |
| 160-update continuations | `train_student.sh {label,dn_mopd,single_code,single_if} S 42 160` | as the 80-update row | `s_route160`, `s_mdnorm160`, `s_singlecode160`, `s_singleifeval160` (9B) |
| Control: Label with recorded, unapplied DN multipliers (seeds 42-44) | `train_student.sh observe S [SEED]` | `mode: label`, `adv_norm: observe_domain_scale` | `ctl_observe_s42/43/44` |
| Control: frozen update-0 multipliers | `train_student.sh frozen_update0 S` | `adv_norm: fixed_domain_scale`, `adv_norm_fixed` = the size's row of `controls.json` | `ctl_fixcal_s42` |
| Control: fixed weights (2, 1, 0.25) | `train_student.sh fixed_w S` | `adv_norm: fixed_domain_scale`, `adv_norm_fixed: {math: 2, code: 1, ifeval: 0.25}` | `ctl_fix21q_s42` |
| Control: fixed weights (2, 1, 1) | `train_student.sh fixed_math S` | `adv_norm_fixed: {math: 2, code: 1, ifeval: 1}` | `ctl_fixmath_s42` (2B, 4B) |
| Control: fixed weights (1, 1, 0.25) | `train_student.sh fixed_if S` | `adv_norm_fixed: {math: 1, code: 1, ifeval: 0.25}` | `ctl_fixif_s42` (2B, 4B) |
| SeqKD-SFT | `gen_seqkd.sh S`, `train_seqkd_sft.sh S` | `configs/seqkd.yaml` | `sft_seqkd_s42` |
| ParamMerge-Avg / ParamMerge-TA | `merge.py --kind avg / ta --size S` | `configs/merge.yaml` | `merge_avg`, `merge_ta1` |

The SFT-initialised student runs are not part of the paper and have no recipe.

## GRPO teachers

`configs/teacher_grpo.yaml`. One specialist per domain, trained from the student's own base model; rewards are binary.

| Setting | Value |
|---|---|
| Algorithm | GRPO (`--advantage-estimator grpo`, group-normalised rewards), no KL term (`--kl-loss-coef 0`), no entropy bonus |
| Rollout | 128 prompts x 8 responses = 1,024 responses per update; temperature 1.0, top-p 1.0; non-thinking chat template |
| Lengths | prompt <= 2,048 tokens (longer prompts are dropped; none in the released data), response <= 8,192 tokens |
| Optimisation | global batch 256 responses (4 optimizer steps per update), micro-batch 1, `--balance-data`; Adam, lr 1e-6 constant after 10 warm-up steps, betas 0.9 / 0.98, weight decay 0.1, gradient clip 1.0 |
| Dynamic sampling | groups whose rewards do not vary are dropped (`anchor_group_filter`), over-sampling batch 128, at most 8 generation batches per update |
| Memory options | `--gradient-checkpointing --log-probs-chunk-size 2048 --recompute-loss-function` (memory only) |
| Seeds | `--rollout-seed 42` (data order and SGLang sampling); `--seed` is the miles default 1234 |
| Rewards | math: `grade_answer_verl` on the answer, 15 s limit per answer (timeout = 0); code: the PRIME judge (a judge error = 0); IF: the vendored checkers, every constraint must hold |
| Saves | every 10 updates |
| Updates used (paper) | 9B: math 250, code 200, IF 400; 4B: math 320, code 300, IF 400; 2B: 400 each |

The runs were capped at 400 updates or a wall-clock budget; the paper uses the last complete checkpoint within the
budget. `train_teacher_grpo.sh` trains to exactly that count (the learning rate is constant after warm-up, so this is
the same trajectory as the longer run up to that point).

## OPD students

`configs/student_opd.yaml`, shared by every student method.

| Setting | Value |
|---|---|
| Objective | on-policy distillation with the sampled-token reverse-KL advantage `A_t = log p_T(y_t) - log pi_old(y_t)` (actor-recomputed pre-update log-prob) in the PPO-clipped loss, clip 0.2 / 0.2; token positions whose teacher score failed get advantage 0; no KL-to-reference, no entropy term |
| Loss reduction | mean over valid tokens within each response, then over responses (the miles default) |
| Rollout | 64 prompts x 8 responses = 512 per update, temperature 1.0, top-p 1.0, non-thinking chat template; prompt <= 2,048, response <= 8,192 tokens |
| Optimisation | global batch 512 (one optimizer step per update), micro-batch 1, `--balance-data`; Adam, lr 1e-6 constant after 5 warm-up steps, betas 0.9 / 0.98, weight decay 0.1, gradient clip 1.0 |
| Updates | 80 (160 for the continuations) |
| Seeds | the student seed (42, 43, 44) sets `--rollout-seed`; `--seed` stays 1234 |
| Saves | every 10 updates |
| Teachers | the size's three GRPO teachers (Qwen3.5 of the same size), served by SGLang; the rollout manager keeps up to 96 scoring requests in flight |
| Data | `data/student/train_3domain.jsonl`: 900 math, 900 code, 900 IF prompts; each prompt's `teacher` field (= its domain) is routed through the label map |

Correctness of every response is also graded (math: `grade_answer_verl` with a 20 s limit, code: the judge, IF: the
checkers) and logged as `response_correct`; it enters no student loss (annealed injection reads it to choose which
response to replace).

## DN-MOPD and the control runs

Once per rollout batch, `Uni_OPD_utils.mopd_hook` computes, from the label teacher's scores and the rollout engine's
cached log-probs of the same tokens,

    r_{i,t}   = log p_{T_d}(y_t) - log pi_rollout(y_t)          on the valid response tokens
    sigma_d   = population std of r over the tokens of domain d, sigma_all = population std over all tokens
    w_d       = clip(sigma_all / sigma_d, 0.25, 4)              (w_d = 1 if a std is 0 or has < 2 tokens)

and attaches `w_d` to each sample of domain `d` (`dn_adv_scale`). The rollout manager carries it into the training
batch and the loss multiplies every advantage of the response by it (`MOPD_DN_APPLIED` log line); the scaled advantage
enters the unchanged clipped loss. Setting every `w_d = 1` recovers Label. The statistic, the raw and clipped
multipliers and the token counts are printed every batch (`MOPD_HOOK_PANEL {... "dnorm": ...}`) and written per
sample to `$OUT_ROOT/runs/<run>/audit/`.

Controls (same hook, same statistic, a different applied multiplier; `configs/controls.json`):

| Control | `adv_norm` | Applied `w` (math, code, IF) |
|---|---|---|
| observe | `observe_domain_scale` | 1, 1, 1 (Label update; the DN multipliers are recorded) |
| frozen update-0 multipliers | `fixed_domain_scale` | 9B 1.768, 0.888, 0.25; 4B 2.009, 0.811, 0.341; 2B 2.102, 0.822, 0.431 (the size's seed-42 DN-MOPD multipliers at update 0, 3 decimals) |
| fixed weights | `fixed_domain_scale` | 2, 1, 0.25 |
| fixed math | `fixed_domain_scale` | 2, 1, 1 |
| fixed IF | `fixed_domain_scale` | 1, 1, 0.25 |

## Annealed injection

`configs/methods.yaml` (`annealed_injection`) and `configs/injection_bank.yaml`.

- **Bank** (`build_injection_bank.sh S`): for each student prompt, its domain's teacher samples 4 answers (vLLM,
  temperature 1.0, top-p 1.0, at most 8,192 tokens, seed 42 + shard over 8 shards). Sample 0 is kept when it finished,
  the trainer's verifier grades it correct (strict), and its prompt token ids equal the rollout's. The teacher then
  scores the kept trajectory (per-token log-probs), and trailing end tokens are normalised to one `<|im_end|>`.
  Coverage in the paper: 2B 1,733 prompts (math 599, code 363, IF 771), 4B 2,243 (794 / 686 / 763), 9B 2,328
  (792 / 724 / 812).
- **Training**: label-mode Label targets; in every prompt group whose prompt the bank holds, the lowest-index
  verifier-wrong response (else the lowest-index response) is replaced by the bank trajectory. Its advantage is the
  constant `c(t) = 1.0 * max(0, 1 - t / 40)` on every token at update `t` (weighted imitation through the same clipped
  loss); from update 40 on nothing is injected. A response without a verdict (for example a code-judge error) stops
  the batch, so a healthy judge is required.

## SeqKD-SFT

`configs/seqkd.yaml`.

| Stage | Setting |
|---|---|
| Teacher answers | each domain's teacher answers its 900 student prompts once (label routing); vLLM, temperature 1.0, top-p 1.0, seed 42, at most 16,384 tokens, max model length 18,432, 256 sequences, eager mode; no correctness filter. Truncation rate in the paper: 2B 2.3 %, 4B 0.4 %, 9B 0.6 % |
| SFT | 4 epochs over the 2,700 answers = 84 updates of 128 answers (global batch 128, micro-batch 1, 8 GPUs); loss on answer tokens only (`sft_loss`, per token), end token only for finished answers; prompt <= 8,192, answer <= 16,384 tokens; Adam, lr 1e-5 with cosine decay to 1e-6 after 5 warm-up updates, betas 0.9 / 0.98, weight decay 0.1, clip 1.0; `--gradient-checkpointing --log-probs-chunk-size 4096`; saves at 42 and 84 |

## Weight merges

`merge.py`, `configs/merge.yaml`: ParamMerge-Avg is the uniform mean of the three teachers (float32 accumulation, saved
in the first expert's dtype); ParamMerge-TA is `theta_0 + 1.0 * sum_k (theta_k - theta_0)` from the base model
(float32 accumulation, saved in the experts' dtype). The base model's 15 `mtp.*` tensors (absent from the teachers'
exports) are left out; any other key or shape difference is refused. Config and tokenizer files come from the base
after a check that they equal the teachers'.

## Hugging Face export

`export_hf.sh RUN_DIR ITER SIZE` runs `miles/tools/convert_fsdp_to_hf.py` in its passthrough mode: the FSDP tensors
are written under the base repository's own key names with their own dtype, and the base config and tokenizer files
are copied. The result is a `Qwen3_5ForConditionalGeneration` checkpoint (vision tower included) without the 15
`mtp.*` tensors; about 4.2 GB (2B), 8.5 GB (4B) and 18 GB (9B). Single-file exports get a generated
`model.safetensors.index.json`.

## Wall-clock, memory and disk

Measured on one node with 8 x B200 (180 GB); approximate, including start-up and teacher scoring.

| Job | 2B | 4B | 9B |
|---|---|---|---|
| GRPO teacher (hours for the updates used) | math 22.5 (400), code 38 (400), IF 21.5 (400) | math 33.5 (320), code 36.5 (300), IF 26.5 (400) | math 31 (250), code 32 (200), IF 26 (400) |
| OPD student, 80 updates | ~2.7 h | ~3.5 h | ~3.3-4.5 h |
| Annealed-injection bank (paper runs) | ~0.5 h | ~0.5 h | ~0.5 h |
| SeqKD-SFT, 84 updates | ~1 h | ~1 h | ~1 h |

- GPU memory: the GRPO teachers reach 150-170 GB per GPU (the colocated SGLang pool is 60 % of each GPU during
  rollout). The student runs use the same 0.60 pool on GPUs 0-3 and no gradient checkpointing; on GPUs smaller than
  ~140 GB set `EXTRA_TRAINER_ARGS="--gradient-checkpointing --log-probs-chunk-size 2048"` (memory only) and, if
  needed, lower the SGLang memory fractions in `configs/student_opd.yaml` (they size KV caches, not scores).
- Disk: one FSDP save (model + optimizer) is ~13 GB (2B), ~26 GB (4B), ~51 GB (9B); a student keeps 8 saves, a
  400-update teacher 40. Only the newest complete save is needed to resume, so older `iter_*` directories can be
  deleted.

## Known quirks of the paper runs

1. **Seeds.** The student seed (42, 43, 44) set only `--rollout-seed` (data order and SGLang sampling). `--seed`
   stayed at the miles default 1234 in every run, teachers included. The recipes do the same.
2. **Label ran on the legacy route, DN-MOPD on the hook.** The seed-42 Label row (`s_route`) and the single-teacher
   rows used the trainer's native per-sample dispatch (one teacher scores each response). DN-MOPD, the pool and
   dynamic rows, annealed injection and the controls used `mopd_hook`, which scores every response with all three
   teachers and, in label mode, asserts on every sample that its target equals the legacy routed scores. The
   `observe` control at seed 42 is a same-seed Label rerun through the hook. Both routes are available here
   (`label`, `label_hook` / `observe`); the paper's Label row is `label`.
3. **Student prompt pool.** The math and code student prompts were filtered to a pass rate in [0.125, 0.875] under an
   earlier base model, not Qwen3.5 (see `data/README.md`).
4. **The IF multiplier sits at the 0.25 floor.** During the first 80 updates the IF domain's DN-MOPD multiplier was at
   the lower clip in 100 % (9B), 56 % (4B) and 71 % (2B) of the batches; math was up-weighted (mean 1.6-2.2) and code
   stayed near or below 1 at 2B / 4B.
5. **Teacher update counts differ by size and domain** (the wall-clock budget; see the teacher table). Every student of
   a size distills from the same three checkpoints.
6. **Verifier labels.** The paper's Label and single-teacher runs graded student IF responses with the math grader
   (IF correctness logged as mostly wrong) and graded math without a time limit; the hook runs used the IF checkers and
   the time-limited math grader. These labels are logged only and enter no student loss. The recipes use the latter
   verifier for every run.
7. **An identity convert hook was dropped.** The paper's student runs passed
   `--custom-convert-samples-to-train-data-path` to a module that, with its feature flag off, called the default
   conversion unchanged; the recipes omit it.
8. **Teacher-server memory fractions** were 0.35 (math, code) and 0.85 (IF) because of the launcher's slot layout;
   they size the KV pool only.
9. **Annealing length.** Injection anneals over 40 updates (half of the 80-update run).
10. **160 updates** were obtained by continuing the 80-update run in place (same directory, same data order and
    constant learning rate), which is what `train_student.sh METHOD S 42 160` does.
11. **The DN statistic uses the rollout engine's cached log-probs**, while the advantage it scales uses the
    actor-recomputed pre-update log-probs (as in the paper's Algorithm 1).
12. **Resuming Qwen3.5 runs** loads the optimizer state partially (the vision tower never gets a gradient, so it has no
    Adam state); model weights are loaded strictly.
13. **GRPO code reward.** A code-judge error scores 0, the same as a wrong program; a degraded judge therefore looks
    like a weak model. `start_code_judge.sh` checks the judge under concurrency before every run that needs it.
