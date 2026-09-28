# Changes to Uni-OPD / miles

`miles/` in this repository is a fork of the `miles/` directory of
[Uni-OPD](https://github.com/WenjinHou/Uni-OPD) at commit `08fcec0` (Apache License 2.0; miles itself carries
"Copyright 2025 Zhipu AI", see `miles/LICENSE`; the Uni-OPD repository's license file is kept as
`miles/LICENSE-Uni-OPD`). Every upstream file is kept. This page lists every change, as required by section 4(b) of
the Apache License 2.0; each modified Python file also starts with a `# Modified by the DN-MOPD authors (2026): ...`
line, and each new file carries `# SPDX-License-Identifier: Apache-2.0`.

`dn-mopd-vs-uni-opd-08fcec0.patch` is the complete diff (21 modified files, 12 new files). To reproduce the fork from
an upstream checkout:

```bash
git clone https://github.com/WenjinHou/Uni-OPD && cd Uni-OPD && git checkout 08fcec0
git apply /path/to/DN-MOPD/patches/dn-mopd-vs-uni-opd-08fcec0.patch     # or: patch -p1 < ...
```

## Method: the multi-teacher OPD hook and DN-MOPD

| File | Change |
|---|---|
| `Uni_OPD_utils/mopd_hook/{__init__,core,hook}.py` (new) | The reward-side hook of every multi-teacher student except the legacy-route rows. Targets: label (the domain's teacher), uniform pool (log of the 1/3 mixture), dynamic router (per-response minimum-likelihood teacher). DN-MOPD (`adv_norm: per_domain_scale`): per rollout batch, `w_d = clip(sigma_all / sigma_d, 0.25, 4)` from the population std of (teacher - rollout log-prob) over valid response tokens, attached to every sample as `dn_adv_scale`; controls `fixed_domain_scale` and `observe_domain_scale`. Annealed teacher-trajectory injection (`inject_*`, attached as `inject_adv_const`). Per-batch audit files and a `MOPD_HOOK_PANEL` log line. |
| `miles/ray/rollout.py` | `dn_adv_scale` and `inject_adv_const` are carried from the samples into the training data and into every data-parallel shard. |
| `miles/backends/training_utils/loss.py` | In the `on_policy_distillation` branch, the per-sample advantages are multiplied by `dn_adv_scale` (`apply_dn_adv_scale`, logged as `MOPD_DN_APPLIED`) and replaced by the constant on injected samples (`apply_inject_adv_const`); both are no-ops when absent (Label and the other methods). One-token responses keep a 1-D log-prob tensor (`reshape(-1)` instead of `squeeze(-1)`). |

## All-teacher scoring (used by the hook)

| File | Change |
|---|---|
| `Uni_OPD_utils/OPD_reward/get_reward.py` | Imports fixed (`exps.OPD.utils.reward.*` -> `Uni_OPD_utils.OPD_reward.*`). New `_score_one_teacher` / `_get_reward_fanout`: every registered teacher scores the response concurrently; the legacy fields still come from the label-routed teacher, and only its failure marks the sample failed. |
| `Uni_OPD_utils/OPD_reward/post_process_rewards.py` | Import fixed. `reward["per_teacher"]` is parsed into `sample.per_teacher_log_probs` (a failed teacher, or one whose echoed token ids do not match the response, gets a sentinel row). The legacy route is unchanged. |
| `Uni_OPD_utils/OPD_reward/reward_manager.py` | Imports fixed. The teacher server list / map paths can be set with `OPD_TEACHER_SERVER_LIST` / `OPD_TEACHER_SERVER_MAP`. Per-teacher interface: `teacher_names`, `resolve_real_name`, `get_next_url_for_teacher`, `build_payload_for_teacher`. |
| `miles/utils/types.py` | `Sample.per_teacher_log_probs`. |
| `miles/backends/fsdp_utils/actor.py` | `teacher_log_probs` is passed to the loss micro-batches (the on-policy-distillation policy loss reads it; without it the FSDP backend cannot run OPD). |

## Rewards and verifiers

| File | Change |
|---|---|
| `Uni_OPD_utils/OPD_reward/rule_base_reward.py` | Imports fixed (the upstream file imported modules that are not in the repository). The verifier is chosen by `sample.metadata["domain"]` (falling back to the teacher label). IF responses are graded by the vendored checkers (strict). Math grading runs in killable worker processes with a time limit. A code-judge error or a verifier exception is returned as `None` (no verdict) instead of a wrong answer. The code-judge client is created lazily. |
| `Uni_OPD_utils/OPD_reward/bounded_math_grade.py`, `math_grader_worker.py` (new) | The time-limited math grading used by `rule_base_reward` (unchanged `grade_answer_verl` in child processes). |
| `Uni_OPD_utils/OPD_reward/ifeval_checkers.py` (new) | Closed-taxonomy instruction-following constraint checkers (no third-party imports): the reward of the IF teachers and the IF correctness label. |
| `Uni_OPD_utils/OPD_reward/grpo_rule_reward.py` (new) | Scalar reward of the GRPO teachers: math (time-limited `grade_answer_verl`), code (the PRIME judge), IF (the checkers). |
| `Uni_OPD_utils/anchor_group_filter.py` (new) | Dynamic sampling for the GRPO teachers: zero-variance groups are dropped, with at most 8 generation batches per rollout (`DYNAMIC_SAMPLING_MAX_GEN_BATCHES`). |
| `Uni_OPD_utils/outcome_reward/PRIME_code_server/judge_app.py` (new) | The HTTP server of the PRIME code judge (upstream ships only the client): judging in a spawn-started process pool that forces the fork start method, with wedged-worker recovery. |
| `Uni_OPD_utils/outcome_reward/PRIME_code_server/server.py` | The endpoint list is read from `CODE_JUDGE_ENDPOINTS` or the file next to the module (upstream: a path relative to the working directory). |
| `Uni_OPD_utils/outcome_reward/PRIME_code_server/available_endpoints.json` | `["127.0.0.1:17580"]` (the local judge started by the recipes) instead of a placeholder. |

## SeqKD-SFT

| File | Change |
|---|---|
| `Uni_OPD_utils/seqkd/{__init__,sft_rollout}.py` (new) | Data-only rollout for SFT on teacher answers: non-thinking chat template (as every rollout and evaluation), loss on response tokens only, end token only for finished answers. |

## Qwen3.5 and environment compatibility (Megatron-free FSDP, transformers 5, recent SGLang)

| File | Change |
|---|---|
| `miles/backends/fsdp_utils/checkpoint.py` | The optimizer state is loaded with `allow_partial_load=True`: the Qwen3.5 vision tower never receives a gradient in text-only training, so its Adam state is never saved and a strict resume fails. |
| `miles/backends/fsdp_utils/actor.py` | Optional `ring_flash_attn` import (context parallelism only); `_no_split_modules` may be a set under transformers 5. |
| `miles/backends/fsdp_utils/parallel.py` | Optional `ring_flash_attn` import. |
| `miles/backends/megatron_utils/lora_utils.py` | Optional `megatron` import (the rollout side imports LoRA constants from here). |
| `miles/utils/ppo_utils.py` | `compute_log_probs` falls back to a plain log-softmax gather without Megatron (unsharded vocabulary only). |
| `miles/backends/sglang_utils/arguments.py` | Reads either the old (`sglang_data_parallel_size`, ...) or the new (`sglang_dp_size`, ...) SGLang argument names. |
| `miles/rollout/sglang_rollout.py` | For text-only prompts of the multimodal Qwen3.5 checkpoint, `multimodal_train_inputs` keeps only tensors and is `None` without multimodal input. |
| `miles/backends/training_utils/data.py` | Only tensors in `multimodal_train_inputs` are moved to the GPU. |
| `miles/backends/training_utils/log_utils.py` | The rollout-data log reducer accepts 0-dim tensors and skips `None` per-sample values. |
| `miles/ray/rollout_logs.py` | Logs `reward/mean`, `reward/std`, `reward/frac_correct` per rollout; `response_correct` is logged only when a verdict exists. |
| `Uni_OPD_utils/margin_calibration/margin_shift.py` | Imports fixed (`miles.miles.*` -> `miles.*`; `loss.py` imports this module unconditionally). |
| `tools/convert_fsdp_to_hf.py` | A passthrough export that keeps the origin repository's key names and dtype (needed for the multimodal Qwen3.5 checkpoints; the 15 `mtp.*` tensors, which the trainer never holds, are reported and left out), a text-config retry and language-tower rename for the instantiate-then-load path, and a refusal to write a checkpoint that matches fewer than 90% of the target keys. |

## Other

| File | Change |
|---|---|
| `LICENSE-Uni-OPD` (new) | Copy of the Uni-OPD repository's license file (the root of the upstream repository is not part of this fork). |
