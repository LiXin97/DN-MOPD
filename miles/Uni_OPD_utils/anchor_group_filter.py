# SPDX-License-Identifier: Apache-2.0
"""Dynamic-sampling filter for the GRPO teachers: drop zero-reward-variance groups, with a bounded number of
generation batches.

miles implements the drop itself (`miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`) and
`generate_rollout` keeps over-sampling until `rollout_batch_size` groups survive. That loop has no bound, so a model
whose groups are almost all zero-variance keeps generating indefinitely. This wrapper adds the bound: up to
`DYNAMIC_SAMPLING_MAX_GEN_BATCHES` (default 8) generation batches' worth of groups are filtered on reward variance
exactly as the miles filter does; past that budget the rollout accepts whatever arrives, so a step always completes.
Accepting a zero-variance group does not corrupt the update: with rewards_normalization and grpo_std_normalization
on (the defaults) its advantage is reward - group_mean = 0 for every member, so it contributes no gradient.

The filter is called as `fn(args, group)` without a rollout id, so a rollout boundary is inferred from `kept`
reaching `rollout_batch_size` (the loop's own exit condition); the counters reset there.

Use: --dynamic-sampling-filter-path Uni_OPD_utils.anchor_group_filter.check_reward_nonzero_std_bounded
"""

import os

import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput

__all__ = ["check_reward_nonzero_std_bounded", "reset_state", "get_state"]

_state = {"seen": 0, "kept": 0, "dropped": 0, "released": 0}


def reset_state():
    for k in _state:
        _state[k] = 0


def get_state():
    return dict(_state)


def _flatten(samples):
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_reward_nonzero_std_bounded(args, samples, **kwargs):
    target = int(getattr(args, "rollout_batch_size", 0) or 0)
    budget = target * int(os.environ.get("DYNAMIC_SAMPLING_MAX_GEN_BATCHES", "8"))

    if target and _state["kept"] >= target:      # the previous rollout finished: this group starts a new one
        reset_state()
    _state["seen"] += 1

    if budget and _state["seen"] > budget:       # at most N generation batches: stop filtering, finish the step
        _state["kept"] += 1
        _state["released"] += 1
        return DynamicFilterOutput(keep=True, reason=None)

    rewards = [s.get_reward_value(args) for s in _flatten(samples)]
    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-8)
    if keep:
        _state["kept"] += 1
    else:
        _state["dropped"] += 1
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"zero_std_{round(rewards[0], 1)}")
