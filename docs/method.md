# DN-MOPD: the method and where it lives in the code

"Routing determines where feedback comes from, but not how much it counts."

## Setting

- One initial model. Three copies of it are RL-trained into specialists (experts) for mathematics, code and
  instruction following (IF). A fourth copy is the student π_u.
- Every training prompt x carries a domain label d ∈ {math, code, IF}, and T_d is that domain's specialist.
- **Label-routed multi-teacher on-policy distillation (MOPD, "Label").** The student samples a response to each
  prompt. The specialist of the prompt's domain then scores every token of that response, and this gives the
  student a dense learning signal.

## Token advantage of label-routed OPD

With h_t = (x, y_<t) the prefix before token t:

$$
A_t \;=\; \log p_{T_d}(y_t \mid h_t) \;-\; \log \pi_u(y_t \mid h_t).
$$

A positive A_t encourages the sampled token, and a negative one discourages it.

## Why a correction is needed

For Qwen3.5 expert pools at 9B, 4B and 2B:

- Label does not outperform the strongest single-teacher student at any size, and it transfers little of the math
  expert's gain.
- The feedback is unbalanced. In the first training batch, token-level log-ratios from the IF expert are 2.3–4.4× as
  dispersed as the pooled signal, while math feedback is about half as dispersed.
- All domains update the same parameters. For the initial 4B student under equal weights, the IF loss supplies 94% of
  the combined gradient. IF supplies about 1% of the response tokens but up to half of the pooled log-ratio variance.

## Domain normalization

For each response i and each valid response token t, let

$$
r_{i,t} \;=\; \ell^{T}_{i,t} \;-\; \ell^{\mathrm{roll}}_{i,t},
$$

where ℓ^T is the teacher log-probability and ℓ^roll is the log-probability cached by the rollout engine when the
response was sampled. On every batch:

- σ_d is the **population** standard deviation of r over the valid response tokens of domain d;
- σ_all is the population standard deviation over **all** valid tokens of the batch, pooled across domains. It is
  computed from the concatenated tokens and is not an average of the σ_d.

$$
w_d \;=\; \operatorname{clip}\!\left(\frac{\sigma_{\mathrm{all}}}{\sigma_d},\; 0.25,\; 4\right),
\qquad
\tilde A_t \;=\; w_d \, A_t .
$$

Properties:

- For a nondegenerate, unclipped domain, Std(w_d · r_d) = σ_all. The rule amplifies feedback with a small spread and
  attenuates feedback with a large spread.
- It is a positive rescaling with no mean subtraction, so **every advantage keeps its sign**.
- **Degenerate statistics:** w_d = 1 if either standard deviation is zero or has fewer than two observations.
- **Label is the special case** in which every w_d = 1.

## Student update

The scaled advantage is detached, and A is recomputed from the actor's own log-probabilities just before the update:

$$
\tilde A_{i,t} \;=\; \operatorname{stopgrad}\!\big(w_{d_i}\,(\ell^{T}_{i,t} - \ell^{\mathrm{actor}}_{i,t})\big).
$$

It enters the unchanged clipped OPD loss:

$$
L_{i,t}(\theta) \;=\; -\min\!\Big\{\rho_{i,t}(\theta)\,\tilde A_{i,t},\;
\operatorname{clip}\big(\rho_{i,t}(\theta),\,1-\eta_-,\,1+\eta_+\big)\,\tilde A_{i,t}\Big\},
\qquad
\rho_{i,t}(\theta) \;=\; \frac{\pi_\theta(y_{i,t} \mid h_{i,t})}{\exp(\ell^{\mathrm{actor}}_{i,t})}.
$$

The loss is averaged over the valid tokens of each response, then over responses. The paper uses η₋ = η₊ = 0.2 and
no KL or entropy term.

Two different log-probabilities are involved. The **scale** statistic (σ, and hence w_d) uses the rollout
log-probabilities ℓ^roll. The **advantage** that is scaled uses the actor-recomputed log-probabilities ℓ^actor.

## Algorithm 1 (one DN-MOPD training iteration)

1. Sample responses y_i ~ π_u(· | x_i).
2. Score each y_i with its domain's teacher T_{d_i}.
3. r_{i,t} ← ℓ^T_{i,t} − ℓ^roll_{i,t} on valid tokens.
4. σ_all ← Std({r_{i,t}}).
5. For each domain d: σ_d ← Std({r_{i,t} : d_i = d}); w_d ← clip(σ_all / σ_d, 0.25, 4). Use w_d = 1 if either
   statistic is degenerate.
6. Recompute A_{i,t} ← ℓ^T_{i,t} − ℓ^actor_{i,t}.
7. Ã_{i,t} ← stopgrad(w_{d_i} A_{i,t}).
8. Update π_u on Ã with the clipped OPD objective.

## Cost and scope

- No additional teacher call, teacher model or learned router. The operation reuses the rollout log-ratios, computes
  one pooled standard deviation and one per domain, and scales the existing advantages.
- It changes neither **which** teacher supervises a prompt nor **how many** prompts each domain receives.

## What the rule does in practice

- The multiplier amplifies math feedback by about 1.5–2.1 at every size.
- It keeps code near 1 at 4B and 2B.
- The IF multiplier sits at the 0.25 floor in most batches, so clipping bounds that domain's scale rather than
  equalizing it.

The per-update trajectories (σ_all, the per-domain σ, and the raw and clipped multipliers) are part of the released
records (`reproduce/data/traces/dn_multiplier_trajectory.json`).

## Control modes used in the paper

These controls isolate where the gain comes from. All of them keep label routing.

| Control | What changes | Multipliers (math, code, IF) |
|---|---|---|
| Label | nothing | (1, 1, 1) |
| Observe-only | Label update; the DN multipliers are computed and recorded every batch but never applied | (1, 1, 1) applied |
| Fixed global weights | constant weights instead of per-batch estimates | (2, 1, 0.25) |
| Math ×2 only | constant | (2, 1, 1) |
| IF ×0.25 only | constant | (1, 1, 0.25) |
| Frozen update-0 multipliers | DN-MOPD's first-batch multipliers, held constant | 9B (1.768, 0.888, 0.25); 4B (2.009, 0.811, 0.341); 2B (2.102, 0.822, 0.431) |

Fixed weights must lie in [0.25, 4]. The frozen update-0 values come from the paper's control configuration.

## Where it lives in the code

All paths are relative to the repository root. `miles/` is the trainer fork; its Python package is `miles/miles/`,
and the multi-teacher code is in `miles/Uni_OPD_utils/`. One DN-MOPD update passes through four places:

1. **Multipliers, once per rollout batch.** `miles/miles/ray/rollout.py`, `RolloutManager._post_process_rewards`,
   calls the reward post-processing hook that the recipes register with
   `--custom-reward-post-process-path Uni_OPD_utils.mopd_hook.hook.post_process_rewards`. In
   `miles/Uni_OPD_utils/mopd_hook/hook.py`, `post_process_rewards` runs `MopdHook.apply` on the **whole** rollout batch
   (64 prompts × 8 responses), before the batch is split across training ranks. For each response, `apply` takes the
   label teacher's log-probabilities minus the rollout engine's cached log-probabilities (`sample.rollout_log_probs`)
   on the response tokens and groups them by `sample.metadata["domain"]`. It then calls
   `core.dn_multipliers(pairs, lower=0.25, upper=4.0)` in `miles/Uni_OPD_utils/mopd_hook/core.py`, which returns w_d
   (population standard deviations; w_d = 1 for a degenerate statistic) and the statistics printed in the
   `MOPD_HOOK_PANEL` line. The run config's `adv_norm` picks the multiplier that is applied: `per_domain_scale`
   (DN-MOPD, the measured w_d), `fixed_domain_scale` (`adv_norm_fixed`, each weight in [0.25, 4]) or
   `observe_domain_scale` (1.0). All three require label routing (`mode: label`). `apply` stores the result on every
   sample as `sample.dn_adv_scale`.
2. **Transport into the training batch.** `miles/miles/ray/rollout.py`: `RolloutManager._convert_samples_to_train_data`
   copies `dn_adv_scale` into the training data next to the teacher log-probs, and
   `RolloutManager._split_train_data_by_dp` carries it into every data-parallel shard. Runs without the hook (Label on
   the original route, single-teacher, uniform pool, dynamic router) have no `dn_adv_scale` key.
3. **Scaling the advantage.** `miles/miles/backends/training_utils/loss.py`, `compute_advantages_and_returns`, branch
   `on_policy_distillation`: the advantage is `(teacher_logp - actor_logp) * teacher_valid_mask`, with the
   actor-recomputed log-probabilities (`rollout_data["log_probs"]`). `apply_dn_adv_scale(advantages,
   rollout_data.get("dn_adv_scale"))` multiplies each response's advantages by its multiplier and prints a
   `MOPD_DN_APPLIED` line; without the key it returns the advantages unchanged.
4. **Loss.** `policy_loss_function` in the same file passes the scaled advantages to the unchanged PPO-style clipped
   loss, `compute_policy_loss` in `miles/miles/utils/ppo_utils.py` (clip 0.2 / 0.2, no KL or entropy term).

| Piece | Location | Notes |
|---|---|---|
| Framework-agnostic implementation (torch only) | `dn_mopd/normalizer.py`, `dn_mopd/advantages.py`, `dn_mopd/loss.py`, `dn_mopd/distributed.py` | `domain_multipliers` and `DomainNormalizer` (`.multipliers`, `.multipliers_batched`; modes `dn`, `observe`, `fixed`, `frozen`; bounds default to (0.25, 4)), `reverse_kl_advantages`, `scale_advantages[_batched]`, `clipped_opd_loss[_batched]`, and `all_reduce_statistics` / `merge_statistics` for trainers whose ranks each hold a shard (`sync=True`). The API is documented in [dn_mopd/README.md](../dn_mopd/README.md). |
| Unit tests (CPU) | `tests/core/` | Spread identity, clipping, degenerate statistics, sign preservation, the Label equivalence, the loss reduction, the batched and distributed forms, the modes, and agreement with the supplement's reference implementation. |
| Trainer hook | `miles/Uni_OPD_utils/mopd_hook/core.py` (`dn_multipliers`), `miles/Uni_OPD_utils/mopd_hook/hook.py` (`MopdHook.apply`, `post_process_rewards`) | Step 1 above. The hook also implements the other multi-teacher rows (`mode: pool`, `mode: dynamic`) and annealed injection (`inject_adv_const`, applied by `apply_inject_adv_const` in `loss.py`). |
| Carrying the multiplier | `miles/miles/ray/rollout.py` (`RolloutManager._convert_samples_to_train_data`, `RolloutManager._split_train_data_by_dp`) | Step 2 above. |
| Applying the multiplier | `miles/miles/backends/training_utils/loss.py` (`compute_advantages_and_returns`, `apply_dn_adv_scale`) | Step 3 above; the loss itself (`compute_policy_loss`, `miles/miles/utils/ppo_utils.py`) is unchanged. |
| Trainer tests (CPU) | `tests/trainer/` | `test_hook_core.py` and `test_hook_modes.py` (hook), `test_transport_and_loss.py` (steps 2–4). |
| Choosing a method | `recipes/qwen3.5/train_student.sh METHOD SIZE [SEED] [UPDATES]`, `recipes/qwen3.5/configs/methods.yaml` | `dn_mopd`, `label`, `label_hook`, `observe`, `fixed_w`, `fixed_math`, `fixed_if`, `frozen_update0`, plus the other rows of the paper's tables. The recipe writes the hook's run config to `$OUT_ROOT/runs/<run>/hook_config.json`. |

The trainer hook keeps its own float32 computation (`core.dn_multipliers`), which is what the paper's runs used. The
`dn_mopd` package accumulates float64 statistics; the two agree to about 1e-6 relative (`tests/trainer/test_hook_core.py`
checks the agreement).

### Label in this codebase

In the paper, the seed-42 Label student ran on the trainer's original label-routing path, without the DN-MOPD hook.
The DN-MOPD runs used the hook in label-routing mode. The release keeps both ways of running Label:

1. `train_student.sh label ...`: the original label-routing path, as used for the paper's seed-42 Label row;
2. `train_student.sh label_hook ...` runs the hook with no multiplier. `train_student.sh observe ...` runs the hook in
   observe-only mode. The paper's observe-only runs (seeds 42, 43 and 44) are Label runs of this kind, and they
   record the multipliers DN-MOPD would have applied.

Through the hook, Label's teacher targets equal those of the original path, and this is asserted on every sample.

## Reproducibility notes

- **Seeds.** The student seed (42, 43 or 44) sets the rollout seed, which controls data shuffling and the SGLang
  sampling seed. The trainer's global `--seed` stayed at the miles default of 1234 in all runs.
  [docs/recipe.md](recipe.md) documents this and the other quirks.
- The multiplier statistics use population standard deviations (`unbiased=False`).
- The clip range [0.25, 4] was the same in every run and was not tuned per size.
