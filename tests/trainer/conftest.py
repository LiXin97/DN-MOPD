# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the trainer tests (CPU only). The miles fork and the recipe library are put on sys.path; torch.compile
is disabled so that the compiled loss helpers run eagerly on CPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO = Path(__file__).resolve().parents[2]
MILES = REPO / "miles"
RECIPES = REPO / "recipes" / "qwen3.5"
for p in (str(MILES), str(RECIPES / "lib"), str(RECIPES)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
import torch  # noqa: E402

TEACHERS = ["code_teacher", "ifeval_teacher", "math_teacher"]
LABEL_MAP = {"default": "code_teacher", "math": "math_teacher", "code": "code_teacher", "ifeval": "ifeval_teacher"}
DOMAINS = ("math", "code", "ifeval")
N, RBS = 8, 6            # responses per prompt, prompts per rollout (6 groups: two per domain)


def make_batch(step=0, seed=0, rbs=RBS, n=N, scale=None, correct=None):
    """One synthetic rollout batch of miles Samples with all-teacher scores, legacy routed scores and rollout log-probs.
    Group gi has domain DOMAINS[gi % 3]; `scale` sets each domain's log-ratio spread."""
    from miles.utils.types import Sample

    g = torch.Generator().manual_seed(1000 * seed + step)
    scale = scale or {"math": 0.3, "code": 1.0, "ifeval": 3.0}
    samples = []
    for gi in range(rbs):
        dom = DOMAINS[gi % 3]
        prompt = f"<|im_start|>user\nprompt {gi} ({dom})<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompt_ids = [100 + gi] * (4 + gi % 3)
        for j in range(n):
            length = 3 + (gi + j) % 6
            s = Sample(prompt=prompt, index=step * rbs * n + gi * n + j, group_index=step * rbs + gi)
            s.tokens = prompt_ids + [1000 + (7 * j + gi) % 50 for _ in range(length)]
            s.response_length = length
            s.loss_mask = [1] * length
            s.metadata = {"domain": dom}
            s.teacher_model_name = dom
            s.status = Sample.Status.COMPLETED
            rows = {t: -torch.rand(length, generator=g) * scale[dom] * (1 + k) for k, t in enumerate(TEACHERS)}
            s.per_teacher_log_probs = rows
            s.teacher_log_probs = rows[LABEL_MAP[dom]].clone()
            s.rollout_log_probs = (-torch.rand(length, generator=g) * scale[dom]).tolist()
            s.response_correct = (j % 2 == 0) if correct is None else correct(gi, j)
            s.response = "answer"
            samples.append(s)
    return samples


class Args:
    """The trainer arguments the hook reads."""

    def __init__(self, n=N, rbs=RBS, estimator="on_policy_distillation"):
        self.advantage_estimator = estimator
        self.n_samples_per_prompt = n
        self.rollout_batch_size = rbs


@pytest.fixture
def hook_module():
    from Uni_OPD_utils.mopd_hook import hook

    return hook


@pytest.fixture
def write_config(tmp_path):
    """write_config(**fields) -> path of a hook run config (label mode, three teachers, an audit dir by default)."""

    def _write(**fields):
        cfg = {"mode": "label", "n_samples": N, "teachers": TEACHERS, "label_map": LABEL_MAP,
               "audit_dir": str(tmp_path / "audit")}
        cfg.update(fields)
        cfg = {k: v for k, v in cfg.items() if v is not None}
        path = tmp_path / f"cfg_{len(list(tmp_path.glob('cfg_*.json')))}.json"
        path.write_text(json.dumps(cfg))
        return path

    return _write


@pytest.fixture(autouse=True)
def _no_server_map(monkeypatch):
    """The hook prefers OPD_TEACHER_SERVER_MAP over the config's label_map; tests use the config's."""
    monkeypatch.delenv("OPD_TEACHER_SERVER_MAP", raising=False)
