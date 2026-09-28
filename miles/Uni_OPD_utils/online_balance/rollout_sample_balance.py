import logging
import math
import random

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _mask_excess_samples(
    correct_samples: list[Sample],
    incorrect_samples: list[Sample],
    ratio: float,
) -> tuple[int, int]:
    """根据目标比例 (correct:incorrect = ratio:1)，mask 掉多余的样本。

    特殊值：
        - ratio = 0:   只保留错误样本，mask 掉所有正确样本。
        - ratio = inf: 只保留正确样本，mask 掉所有错误样本。
        - ratio = 1:   正确:错误 = 1:1（默认行为）。

    返回 mask 后的 (correct_count, incorrect_count)。
    """
    n_correct = len(correct_samples)
    n_incorrect = len(incorrect_samples)

    # 特殊值：ratio = 0，只保留错误样本
    if ratio == 0:
        for s in correct_samples:
            s.remove_sample = True
        return 0, n_incorrect

    # 特殊值：ratio = inf，只保留正确样本
    if math.isinf(ratio):
        for s in incorrect_samples:
            s.remove_sample = True
        return n_correct, 0

    if n_correct == 0 or n_incorrect == 0:
        # 无法平衡，不做任何 mask
        return n_correct, n_incorrect

    # 目标：n_correct_keep / n_incorrect_keep = ratio
    # 以较少的一方为基准，mask 较多的一方
    desired_correct = int(n_incorrect * ratio)
    desired_incorrect = int(n_correct / ratio) if ratio > 0 else n_correct

    if n_correct > desired_correct and desired_correct >= 1:
        # 正确样本过多，需要 mask 一部分正确样本
        to_mask = n_correct - desired_correct
        masked_samples = random.sample(correct_samples, to_mask)
        for s in masked_samples:
            s.remove_sample = True
        return desired_correct, n_incorrect
    elif n_incorrect > desired_incorrect and desired_incorrect >= 1:
        # 错误样本过多，需要 mask 一部分错误样本
        to_mask = n_incorrect - desired_incorrect
        masked_samples = random.sample(incorrect_samples, to_mask)
        for s in masked_samples:
            s.remove_sample = True
        return n_correct, desired_incorrect
    else:
        # 已经满足比例或无法调整
        return n_correct, n_incorrect


def _filter_global(samples: list[list[Sample]], ratio: float) -> None:
    """全局模式：展平所有样本后按比例筛选。"""
    flat_samples = [s for group in samples for s in group]
    correct_samples: list[Sample] = [s for s in flat_samples if s.response_correct is True]
    incorrect_samples: list[Sample] = [s for s in flat_samples if s.response_correct is not True]

    n_correct_before = len(correct_samples)
    n_incorrect_before = len(incorrect_samples)
    total = len(flat_samples)

    n_correct_after, n_incorrect_after = _mask_excess_samples(correct_samples, incorrect_samples, ratio)

    logger.info(
        f"[Filter Global] "
        f"Before: correct={n_correct_before}, incorrect={n_incorrect_before}, "
        f"ratio={n_correct_before / n_incorrect_before:.4f} (total={total}). "
        f"After: correct={n_correct_after}, incorrect={n_incorrect_after}, "
        f"ratio={n_correct_after / n_incorrect_after:.4f} "
        f"(masked={n_correct_before - n_correct_after + n_incorrect_before - n_incorrect_after})."
        if n_incorrect_before > 0 and n_incorrect_after > 0
        else f"[Filter Global] Before: correct={n_correct_before}, incorrect={n_incorrect_before} (total={total}). "
        f"After: correct={n_correct_after}, incorrect={n_incorrect_after}. "
        f"Cannot compute ratio (zero incorrect samples)."
    )


def _filter_per_group(samples: list[list[Sample]], ratio: float) -> None:
    """Sample 模式：在每个 prompt group 内按比例筛选。"""
    total_correct_before = 0
    total_incorrect_before = 0
    total_correct_after = 0
    total_incorrect_after = 0
    total = 0

    for group_samples in samples:
        correct_samples = [s for s in group_samples if s.response_correct is True]
        incorrect_samples = [s for s in group_samples if s.response_correct is not True]

        total += len(group_samples)
        total_correct_before += len(correct_samples)
        total_incorrect_before += len(incorrect_samples)

        n_c, n_i = _mask_excess_samples(correct_samples, incorrect_samples, ratio)
        total_correct_after += n_c
        total_incorrect_after += n_i

    ratio_before = total_correct_before / total_incorrect_before if total_incorrect_before > 0 else float("inf")
    ratio_after = total_correct_after / total_incorrect_after if total_incorrect_after > 0 else float("inf")

    logger.info(
        f"[Filter Sample] "
        f"Before: correct={total_correct_before}, incorrect={total_incorrect_before}, "
        f"ratio={ratio_before:.4f} (total={total}). "
        f"After: correct={total_correct_after}, incorrect={total_incorrect_after}, "
        f"ratio={ratio_after:.4f} "
        f"(masked={total_correct_before - total_correct_after + total_incorrect_before - total_incorrect_after})."
    )

# --rollout-sample-filter-path
# rollout_sample_filter_path
def rollout_sample_balance(args, samples: list[list[Sample]]) -> None:
    """根据 response_correct 筛选样本，平衡正确/错误样本比例。

    Args:
        args: 训练参数。
        samples: list[list[Sample]]，外层是 prompt groups，内层是每个 prompt 的多个 rollout。

    通过设置 remove_sample=True 来 mask 掉多余的样本。
    """
    # 检查是否启用了筛选
    if not getattr(args, "filter_rollout_by_response_correct", False):
        logger.info("[Filter] filter_rollout_by_response_correct is False, skipping filter.")
        return

    # 对于 response_correct 为 None 的样本，尝试从 reward 中补充
    for group in samples:
        for s in group:
            if (
                s.response_correct is None
                and isinstance(s.reward, dict)
                and s.reward.get("response_correct") is not None
            ):
                s.response_correct = s.reward["response_correct"]

    # 检查是否所有 response_correct 都是 None（即没有正确性判断结果）
    all_none = all(s.response_correct is None for group in samples for s in group)
    if all_none:
        logger.warning("[Filter] All samples have response_correct=None, skipping filter.")
        return

    mode = getattr(args, "filter_rollout_mode", "global")
    ratio = getattr(args, "filter_rollout_ratio", 1.0)

    if ratio < 0:
        logger.warning(f"[Filter] filter_rollout_ratio={ratio} < 0, invalid value, skipping filter.")
        return

    if mode == "global":
        _filter_global(samples, ratio)
    elif mode == "sample":
        _filter_per_group(samples, ratio)
    else:
        logger.warning(f"[Filter] Unknown filter_rollout_mode='{mode}', skipping filter.")
