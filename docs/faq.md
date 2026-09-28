# FAQ

### Which GPUs do I need?

Every run in the paper used **one node with 8 NVIDIA B200 GPUs (180 GB each)**:

- **Students** (every OPD row): the FSDP actor and its colocated SGLang rollout engine run on GPUs 0–3. The three
  frozen teachers are served by SGLang (tensor parallel 1) on GPUs 4, 5 and 6. GPU 7 is idle.
- **GRPO teachers**: all 8 GPUs.
- **Evaluation**: one vLLM engine (tensor parallel 1) per GPU.

The training environment ships CUDA 13.0 in its wheels and needs a driver that supports it (R580 or newer). The
evaluation environment ships CUDA 12.8. We have not run the code on other GPUs: H100/H200 and A100 are untested, and
we have no measured settings for 80 GB GPUs. The paper's student settings leave little headroom below about 140 GB
per GPU for the 9B student; [hardware.md](hardware.md#smaller-gpus) lists the options that change memory use but not
the update (gradient checkpointing, smaller SGLang memory fractions, a different number of student GPUs). See also
[env/install.md](../env/install.md#hardware-and-drivers).

### How long does a run take?

On the paper's hardware (one node with 8 B200 GPUs), from the run logs, including start-up and teacher scoring:

| Job | 2B | 4B | 9B |
|---|---|---|---|
| GRPO teacher, hours (updates used) | math 22.5 (400), code 38 (400), IF 21.5 (400) | math 33.5 (320), code 36.5 (300), IF 26.5 (400) | math 31 (250), code 32 (200), IF 26 (400) |
| OPD student, 80 updates | ~2.7 h | ~3.5 h | ~3.3–4.5 h |
| SeqKD-SFT, 84 updates | ~1 h | ~1 h | ~1 h |

[recipe.md](recipe.md#wall-clock-memory-and-disk) also lists the annealed-injection bank, GPU memory and disk use.
We do not report evaluation times. The sizes of the runs are:

- a student run is 80 updates (160 for the continuations), each with 64 prompts × 8 responses of up to 8,192 tokens;
- a GRPO teacher run is capped at 400 updates of 128 prompts × 8 responses;
- the evaluation samples 64 answers per AIME question, 6 per LiveCodeBench problem and 16 per IFEval/IFBench prompt,
  with caps of 8,192 and 16,384 tokens.

### Can I use DN-MOPD with other models or another trainer?

Yes. The rule needs only two inputs per response: a domain label, and the token-level log-ratios between the
teacher and the rollout log-probabilities. The `dn_mopd` package implements the multipliers, the advantage scaling
and the clipped OPD loss with torch alone, so it can be called from any trainer. The README has a short example.
The `miles/` fork is the trainer used for the paper.

The evidence covers **one model family (Qwen3.5, at 9B, 4B and 2B), with one expert pool per size**. DN-MOPD was
selected on an earlier Qwen3 development setup. An earlier Qwen3-4B comparison, under a different setup, found no
clear gain, so the benefit depends on the teacher–student configuration. Check it on your own setup against Label
(every w_d = 1).

### Why clip the multiplier to [0.25, 4]?

In the paper's words, the bounds [0.25, 4] limit the adjustment when a scale ratio is extreme, and clipping can
prevent full equalization. No domain's feedback is scaled by more than a factor of 4 in either direction, and every
advantage keeps its sign.

The operation and its clipping bounds were selected on an earlier Qwen3 development setup (at 80 updates) and
transferred unchanged to all three Qwen3.5 sizes. They were not tuned per size. In the Qwen3.5 runs:

- the IF multiplier sits at the 0.25 floor in most batches, so the rule bounds rather than equalizes that domain's
  scale;
- the math multiplier stays inside the bounds (about 1.5–2.1), and code stays near 1 at 4B and 2B.

### Is σ computed per rank or globally?

**Globally, over the whole rollout batch.** The trainer computes the multipliers once per rollout batch (64 prompts ×
8 responses), in the reward post-processing step of the rollout manager, before the batch is split across training
ranks. Every rank therefore uses the same w_d. σ_all pools the valid response tokens of all domains; it is not an
average of the per-domain σ_d. Both are population standard deviations. If you port the rule to another trainer,
compute σ on the full batch, or all-reduce the sufficient statistics (token count, sum and sum of squares per domain)
instead of computing σ per rank. `dn_mopd.DomainNormalizer(..., sync=True)` does this all-reduce.

### Which log-probabilities go into σ, and which into the advantage?

σ (and hence w_d) uses the teacher log-probabilities minus the **rollout** log-probabilities cached at sampling time.
The advantage that is scaled uses the teacher log-probabilities minus the **actor-recomputed** log-probabilities, and
it is detached before it enters the clipped OPD loss. See [method.md](method.md#student-update).

### What if a domain is missing from a batch or has almost no tokens?

A domain that is absent gets no multiplier, because none of its responses need one. If a standard deviation is zero
or has fewer than two observations, that domain's w_d is 1, the Label value.

### Why is Label reproduced in two ways?

In the paper, the seed-42 Label student ran on the trainer's original label-routing path, without the DN-MOPD hook.
The DN-MOPD runs used the hook in label-routing mode. The release offers both:

1. `recipes/qwen3.5/train_student.sh label SIZE SEED` runs the original path. This is how the paper's seed-42 Label
   row was produced.
2. `train_student.sh label_hook ...` runs the hook with no multiplier. `train_student.sh observe ...` runs the hook
   in observe-only mode: it records the multipliers DN-MOPD would apply but never applies them. The paper's
   seed-43/44 Label runs and a same-seed rerun of Label (seed 42) are observe-only runs.

Having both lets you check that the hook reduces exactly to Label when every w_d = 1. Through the hook, Label's
teacher targets are asserted to equal those of the original path on every sample.

### What does the "seed" of a student run control?

The student seed (42, 43 or 44) sets the rollout seed. That seed controls the data shuffling and the SGLang sampling
seed. The trainer's global `--seed` stayed at the miles default of 1234 in every run. The recipes reproduce this
behaviour, and [recipe.md](recipe.md) documents it.

### Does DN-MOPD change which teacher supervises a prompt, or the data mix?

No. DN-MOPD changes neither which teacher supervises a prompt nor how many prompts each domain receives. It only
rescales each domain's advantages. Setting every w_d = 1 gives back Label.

### Where does the gain come from?

The fixed-weight controls answer this:

- Changing the teacher assignment (a uniform pool or a dynamic router) brings no consistent gain.
- Doubling only the math weight recovers about half of DN-MOPD's improvement at 4B and little at 2B.
- Lowering only the IF weight to 0.25 recovers most of the improvement, and it raises math by about three points.
- Weights fixed at DN-MOPD's first-batch multipliers, or at a global (2, 1, 0.25), show no detectable difference from
  DN-MOPD at 9B and 4B. At 2B, per-batch estimation outperforms first-batch weights by 1.10 points.

### Is DN-MOPD the best way to combine specialists?

Not overall. In the paper, SeqKD-SFT and task-arithmetic merging (ParamMerge-TA) keep higher Totals under their own
training recipes. DN-MOPD is the best method within the multi-teacher OPD block, and it improves on Label at every
size, both evaluation caps and all three student seeds. Its lead over the strongest single-teacher student is
smaller, and some of those intervals include zero.

### Thinking or non-thinking mode?

Non-thinking, throughout. GRPO teacher training, student training and evaluation all render prompts with the chat
template and `enable_thinking=False`.
Qwen3.5-4B and 9B think by default, so pass this flag when you use the released models.

### Why do the exported checkpoints lack the `mtp.*` tensors?

The Hugging Face export omits the base model's 15 multi-token-prediction tensors. Ordinary decoding is unaffected, and
all of the paper's evaluations used these exports. MTP-based speculative decoding is not available with them.
