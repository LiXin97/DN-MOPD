# dn_mopd

A small, framework-agnostic implementation of **DN-MOPD** (domain-normalized multi-teacher on-policy distillation).
The only dependency is `torch`. You can drop it into any on-policy distillation (OPD) or RL trainer, such as TRL, verl,
OpenRLHF or miles.

```bash
pip install -e .            # from the repository root; or copy the dn_mopd/ directory into your project
python -m pytest tests/core # CPU tests
python -m dn_mopd.examples.toy_opd_loop   # runnable CPU toy, a few seconds
```

## What it computes

Label-routed multi-teacher OPD scores every prompt of domain *d* with that domain's teacher *T_d*. Each sampled token
then gets the reverse-KL advantage `A_t = log p_{T_d}(y_t | h_t) - log pi(y_t | h_t)`.

DN-MOPD adds one operation per rollout batch:

```
r      = teacher log-prob - rollout log-prob            (valid response tokens only)
sigma_d   = population std of r over the tokens of domain d
sigma_all = population std of r over all tokens of the batch (pooled, not averaged)
w_d    = clip(sigma_all / sigma_d, 0.25, 4)              (w_d = 1 if a std is zero or has < 2 tokens)
A~     = stopgrad(w_d * A)                               -> the unchanged clipped OPD loss
```

Setting every `w_d = 1` recovers label-routed MOPD ("Label").

## The three lines

The inputs are padded `(B, T)` tensors: `teacher_logp`, `rollout_logp` (cached by the inference engine at sampling
time), `old_logp` (the trainer's pre-update recomputation), `new_logp` (with gradients), the response `mask`, and
`domain_ids` of shape `(B,)`.

```python
from dn_mopd import (DomainNormalizer, reverse_kl_advantages, scale_advantages_batched,
                     clipped_opd_loss_batched)

normalizer = DomainNormalizer(["math", "code", "ifeval"])            # mode="dn", bounds=(0.25, 4)

# once per rollout batch
weights, diag = normalizer.multipliers_batched(teacher_logp - rollout_logp, mask, domain_ids)   # (1)
adv = scale_advantages_batched(reverse_kl_advantages(teacher_logp, old_logp, mask),
                               domain_ids, weights)                                              # (2)
# every optimizer step on this batch
loss = clipped_opd_loss_batched(new_logp, old_logp, adv, mask)                                   # (3)
logger.log(diag.to_metrics())                                        # dn/<domain>/{std,factor,applied,...}
```

If your trainer keeps responses as lists of 1-D tensors (packed sequences), use the list forms instead:
`normalizer.multipliers([(domain, r_valid_tokens), ...])`, `scale_advantages` and `clipped_opd_loss`.

**Which log-probs go where.** The paper estimated the scale `r` from the rollout engine's cached log-probs. The
advantage `A` uses the trainer's recomputed pre-update log-probs, the same ones the PPO ratio divides by. A trainer
without cached rollout log-probs can pass `old_logp` for both.

## Modes

| `mode` | applied multiplier | paper row |
|---|---|---|
| `"dn"` | `clip(sigma_all / sigma_d)`, re-measured every batch | DN-MOPD |
| `"observe"` | 1.0 (the statistic is still measured and reported) | Label |
| `"fixed"` | constants from `fixed={...}` | fixed-weight controls |
| `"frozen"` | the first measurable batch's value, held from then on | frozen multipliers |

In every mode the statistic is computed and reported in `diag`.

**Frozen mode.** For the frozen control, the paper applied each size's update-0 multipliers, rounded to three
decimals, through `fixed`. `frozen` measures those values on the fly instead. Its state (the frozen values and the step
counter) is in `normalizer.state_dict()`: save it with your checkpoints and restore it with `load_state_dict`.

**Bounds.** `bounds=(lower, upper)` is configurable. The default is the paper's `(0.25, 4)`; `(0, math.inf)`
disables clipping.

## Where to call it in a trainer

- **Single controller** (the driver holds the whole rollout batch, as in verl's driver-side advantage computation or
  miles' reward post-processing): call it once on the full batch. `sync=False` is the default.
- **Data-parallel ranks, each holding a shard** (SPMD trainers): construct the normalizer with `sync=True` and call it
  once per rollout batch on every rank. Call it on all of the rank's samples, before micro-batching.
  - The per-domain sufficient statistics (responses, tokens, sum, sum of squares) are all-reduced in float64 over
    `process_group` with `torch.distributed`, so `sigma` is computed on the global batch.
  - Without an initialized process group it falls back to the local batch.
  - For other collectives (for example Ray), pass `sync=callable`. The callable receives the local `(4, D)` statistics
    and returns the global ones; `dn_mopd.merge_statistics` sums gathered shards.
- **Loss normalization with micro-batches or shards.** Pass `num_responses=<responses in the optimizer batch>` to
  `clipped_opd_loss(_batched)`. The per-shard losses then add up to the batch mean: the token mean within each
  response, then the mean over responses.

## Numerics

- **Statistics.** They are accumulated in float64 as sufficient statistics, with a population variance
  `E[r^2] - E[r]^2`. A variance below `1e-12 * E[r^2]` is treated as zero, which is a cancellation guard for constant
  domains.
- **Agreement with the paper.** The paper's trainer called `torch.std(unbiased=False)` on float32 tensors in one
  process. The two agree to about 1e-6 relative.
- **The test oracle.** The supplementary reference implementation is kept verbatim in
  `tests/core/dn_reference_supplement.py`.
- **Invalid values.** Values at masked positions are ignored, including NaN. A non-finite value at a valid token
  raises an error.
