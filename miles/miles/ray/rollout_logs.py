# Modified by the DN-MOPD authors (2026): log the mean scalar reward (reward/mean, reward/std, reward/frac_correct)
# of every rollout; response_correct is logged only when at least one verdict exists.
import logging
from argparse import Namespace
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from miles.utils.misc import load_function
from miles.utils.types import Sample

from ..utils.metric_utils import has_repetition

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def log_eval_rollout(
    rollout_id: int,
    args: Namespace,
    data,
    extra_metrics: dict[str, Any] | None = None,
):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        # logger.info(f"Eval {rollout_id} {key}: rewards={rewards}")
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def parse_url(raw_url: str) -> str:
    url = urlparse(raw_url)
    return url.netloc


#! tensorboard 的 rollout 指标在这里处理
def log_rollout(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics,
    rollout_time: float,
) -> None:
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples: list[Sample]):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}  # 字典合并更新运算符
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_OPD_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    # Mean reward of the rollout (a reward curve for GRPO runs). On the OPD path sample.reward is a dict of reward
    # components, so a numeric component is used when present. Logging must never stop training.
    try:
        _rw = []
        for s in samples:
            r = getattr(s, "reward", None)
            if r is None:
                continue
            if isinstance(r, dict):
                for key in ("reward", "score", "total", "acc"):
                    if isinstance(r.get(key), (int, float)):
                        r = r[key]
                        break
                else:
                    nums = [v for v in r.values() if isinstance(v, (int, float))]
                    if not nums:
                        continue
                    r = sum(nums) / len(nums)
            if isinstance(r, bool) or not isinstance(r, (int, float)):
                continue
            _rw.append(float(r))
        if _rw:
            log_dict["reward/mean"] = float(np.mean(_rw))
            log_dict["reward/std"] = float(np.std(_rw))
            log_dict["reward/frac_correct"] = float(np.mean([r >= 1.0 for r in _rw]))
    except Exception as e:  # never let a logging line take down a training run
        log_dict["reward/logging_error"] = repr(e)[:120]
    return log_dict


def compute_perf_metrics_from_samples(args, samples: list[Sample], rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    # 记录 rollout time 和 reward time 的分布
    rollout_times = [sample.rollout_time for sample in samples if sample.rollout_time > 0]
    if rollout_times:
        log_dict["sample_rollout_time/max"] = max(rollout_times)
        log_dict["sample_rollout_time/min"] = min(rollout_times)
        log_dict["sample_rollout_time/avg"] = sum(rollout_times) / len(rollout_times)

    reward_times: list[float] = []
    for sample in samples:
        if sample.reward_time > 1e-5:
            reward_times.append(sample.reward_time)
        elif isinstance(sample.reward, dict) and "reward_time" in sample.reward and sample.reward["reward_time"] > 1e-5:
            reward_times.append(sample.reward["reward_time"])

    if reward_times:
        log_dict["sample_reward_time/max"] = max(reward_times)
        log_dict["sample_reward_time/min"] = min(reward_times)
        log_dict["sample_reward_time/avg"] = sum(reward_times) / len(reward_times)

    teacher_urls: list[str] = []
    for sample in samples:
        if isinstance(sample.reward, dict) and "teacher_url" in sample.reward:
            teacher_urls.append(parse_url(sample.reward["teacher_url"]))

    if teacher_urls:
        counter = Counter(teacher_urls)
        total = len(teacher_urls)
        for host_port, count in sorted(counter.items()):
            log_dict[f"teacher_url/{host_port}"] = count / total

    use_llm_flags: list[bool] = []
    llm_times: list[float] = []
    for sample in samples:
        if not isinstance(sample.reward, dict):
            continue
        if "use_llm" not in sample.reward:
            continue
        use_llm_flags.append(bool(sample.reward["use_llm"]))
        if sample.reward.get("use_llm") and "llm_time" in sample.reward:
            llm_times.append(float(sample.reward["llm_time"]))

    if use_llm_flags:
        log_dict["llm_judge/use_llm_ratio"] = sum(use_llm_flags) / len(use_llm_flags)
        if llm_times:
            log_dict["llm_judge/llm_time/max"] = max(llm_times)
            log_dict["llm_judge/llm_time/min"] = min(llm_times)
            log_dict["llm_judge/llm_time/avg"] = sum(llm_times) / len(llm_times)

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator in ["ppo"]:
        return {}
    if args.n_samples_per_prompt <= 1:
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups: list[list[Sample]] = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_OPD_metrics(args, all_samples: list[Sample]):
    if args.advantage_estimator != "on_policy_distillation":
        return {}

    metrics = {}
    response_corrects = []
    for sample in all_samples:
        if sample.response_correct is not None:
            response_corrects.append(sample.response_correct)
        elif "response_correct" in sample.reward and sample.reward["response_correct"] is not None:
            response_corrects.append(sample.reward["response_correct"])

    # a batch whose verdicts are all None (verifier errors) has no correctness to report
    if response_corrects:
        metrics["response_correct"] = np.mean(response_corrects)
    return metrics
