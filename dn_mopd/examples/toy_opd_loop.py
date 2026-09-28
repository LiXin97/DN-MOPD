# SPDX-License-Identifier: Apache-2.0
"""A minimal multi-teacher on-policy distillation loop with DN-MOPD, in pure PyTorch on CPU (a few seconds).

The "language models" are tiny bigram tables with a per-domain bias, so the example needs no download and no GPU.
Three frozen teachers (one per domain) differ in sharpness, so the spread of the teacher-student log-ratio differs
by domain: exactly the imbalance DN-MOPD normalizes. The student samples responses, every response is scored by its
domain's teacher (label routing), and the three DN-MOPD lines sit between scoring and the loss.

    python -m dn_mopd.examples.toy_opd_loop                 # DN-MOPD
    python -m dn_mopd.examples.toy_opd_loop --mode observe  # Label (multipliers measured, not applied)
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch

from dn_mopd import DomainNormalizer, clipped_opd_loss_batched, reverse_kl_advantages, scale_advantages_batched

DOMAINS = ("math", "code", "ifeval")
TEACHER_SHARPNESS = (4.0, 1.5, 0.5)     # sharper teacher -> wider log-ratio spread in that domain
VOCAB, MAX_LEN, BOS = 16, 12, 0


class ToyLM(torch.nn.Module):
    """p(y_t | y_{t-1}, domain) = softmax(table[y_{t-1}] + bias[domain])."""

    def __init__(self, scale: float, generator: torch.Generator) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(scale * torch.randn(VOCAB, VOCAB, generator=generator))
        self.bias = torch.nn.Parameter(scale * torch.randn(len(DOMAINS), VOCAB, generator=generator))

    def dist(self, domain_ids: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Next-token log-probs ``(..., V)`` given previous tokens ``prev`` and the response's domain."""
        bias = self.bias[domain_ids].reshape(domain_ids.shape + (1,) * (prev.dim() - 1) + (VOCAB,))
        return torch.log_softmax(self.table[prev] + bias, dim=-1)

    def log_probs(self, domain_ids: torch.Tensor, prev: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` log-probs of ``tokens`` given ``prev``."""
        return self.dist(domain_ids, prev).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout(student: ToyLM, domain_ids: torch.Tensor, generator: torch.Generator
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one response per prompt: tokens, previous tokens, rollout log-probs and valid mask, all ``(B, T)``."""
    batch = domain_ids.shape[0]
    prev = torch.full((batch,), BOS, dtype=torch.long)
    tokens, prevs, logps = [], [], []
    for _ in range(MAX_LEN):
        logp = student.dist(domain_ids, prev)
        y = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        tokens.append(y)
        prevs.append(prev)
        logps.append(logp.gather(-1, y[:, None]).squeeze(-1))
        prev = y
    lengths = torch.randint(1, MAX_LEN + 1, (batch,), generator=generator)   # responses stop at different lengths
    mask = torch.arange(MAX_LEN)[None, :] < lengths[:, None]
    return torch.stack(tokens, 1), torch.stack(prevs, 1), torch.stack(logps, 1), mask


def run(steps: int = 30, mode: str = "dn", batch: int = 48, ppo_epochs: int = 2, lr: float = 0.05,
        seed: int = 0, verbose: bool = True) -> List[Dict[str, float]]:
    """Train the toy student for ``steps`` rollout batches; returns one metrics dict per step."""
    gen = torch.Generator().manual_seed(seed)
    teachers = [ToyLM(s, gen).requires_grad_(False) for s in TEACHER_SHARPNESS]
    student = ToyLM(0.1, gen)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    normalizer = DomainNormalizer(DOMAINS, mode=mode)       # bounds default to the paper's (0.25, 4)
    history = []
    for step in range(steps):
        domain_ids = torch.randint(0, len(DOMAINS), (batch,), generator=gen)
        tokens, prev, rollout_logp, mask = rollout(student, domain_ids, gen)
        with torch.no_grad():
            all_teachers = torch.stack([t.log_probs(domain_ids, prev, tokens) for t in teachers])   # (3, B, T)
            teacher_logp = all_teachers[domain_ids, torch.arange(batch)]      # label routing: own-domain teacher
            old_logp = student.log_probs(domain_ids, prev, tokens)            # trainer's pre-update recomputation

        # ---- DN-MOPD ------------------------------------------------------------------------------------------
        # (1) once per rollout batch: per-domain multipliers from the rollout-side log-ratio
        weights, diag = normalizer.multipliers_batched(teacher_logp - rollout_logp, mask, domain_ids)
        # (2) reverse-KL OPD advantage (teacher minus pre-update student log-prob), scaled per domain, detached
        adv = scale_advantages_batched(reverse_kl_advantages(teacher_logp, old_logp, mask), domain_ids, weights)
        for _ in range(ppo_epochs):
            # (3) the unchanged clipped OPD loss: token mean per response, then mean over responses
            loss = clipped_opd_loss_batched(student.log_probs(domain_ids, prev, tokens), old_logp, adv, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # ------------------------------------------------------------------------------------------------------

        rkl = (old_logp - teacher_logp) * mask
        row: Dict[str, float] = {"step": float(step), "loss": float(loss.detach())}
        for i, d in enumerate(DOMAINS):
            sel = domain_ids == i
            row[f"w/{d}"] = float(weights[i])
            row[f"rkl/{d}"] = float(rkl[sel].sum() / mask[sel].sum().clamp_min(1))
        history.append(row)
        if verbose:
            ws = " ".join(f"w[{d}]={row[f'w/{d}']:.2f}" for d in DOMAINS)
            ks = " ".join(f"rkl[{d}]={row[f'rkl/{d}']:.3f}" for d in DOMAINS)
            print(f"step {step:3d}  loss {row['loss']:+.4f}  {ws}  {ks}  sigma_all={diag.global_std:.3f}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--mode", default="dn", choices=["dn", "observe", "frozen"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)    # the tensors are tiny; many intra-op threads only add synchronization overhead
    run(steps=args.steps, mode=args.mode, seed=args.seed)


if __name__ == "__main__":
    main()
