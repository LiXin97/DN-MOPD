import logging
from argparse import Namespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def _compute_gap(
    correct_advs: list[float],
    incorrect_advs: list[float],
    mode: str,
) -> float:
    """根据 mode 计算当前正/负样本集合之间的 gap。

    mode="mean":  mean(correct) - mean(incorrect)
    mode="minmax": min(correct) - max(incorrect)
    """
    if len(correct_advs) == 0 or len(incorrect_advs) == 0:
        return float("inf")  # 无法计算，视为已满足

    if mode == "mean":
        return sum(correct_advs) / len(correct_advs) - sum(incorrect_advs) / len(incorrect_advs)
    elif mode == "minmax":
        return min(correct_advs) - max(incorrect_advs)
    else:
        return float("inf")


def _compute_gap_after_remove_correct(
    correct_advs: list[float],
    remove_idx: int,
    incorrect_advs: list[float],
    mode: str,
) -> float:
    """计算移除某个正样本后的 gap。"""
    new_correct = [v for i, v in enumerate(correct_advs) if i != remove_idx]
    return _compute_gap(new_correct, incorrect_advs, mode)


def _compute_gap_after_remove_incorrect(
    correct_advs: list[float],
    incorrect_advs: list[float],
    remove_idx: int,
    mode: str,
) -> float:
    """计算移除某个负样本后的 gap。"""
    new_incorrect = [v for i, v in enumerate(incorrect_advs) if i != remove_idx]
    return _compute_gap(correct_advs, new_incorrect, mode)


def _greedy_mask_local(
    advantages: list[torch.Tensor],
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    mode: str,
    delta: float,
    min_retain_ratio: float,
    is_log_rank: bool = True,
    label: str = "Local",
) -> list[torch.Tensor]:
    """在本地样本上执行贪婪优势掩码。

    通过迭代剔除"最不具代表性"的样本（最差正样本或最好负样本），
    直到 margin 条件满足或达到最低存留比例。
    被剔除的样本的 advantages 全部置零。

    Args:
        advantages: token 级别 advantage 列表（会被原地修改）。
        mean_advs: 每个样本的平均 advantage。
        correct_mask: 正确样本的 bool mask。
        incorrect_mask: 错误样本的 bool mask。
        mode: "mean" 或 "minmax"。
        delta: 目标 margin。
        min_retain_ratio: 最低存留比例（0~1），防止过量剔除。
        is_log_rank: 是否打印日志。
        label: 日志标签。

    Returns:
        修改后的 advantages 列表。
    """
    n_total = len(advantages)
    correct_indices = correct_mask.nonzero(as_tuple=True)[0].tolist()
    incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0].tolist()

    n_correct_orig = len(correct_indices)
    n_incorrect_orig = len(incorrect_indices)

    if n_correct_orig == 0 or n_incorrect_orig == 0:
        if is_log_rank:
            logger.info(
                f"[OPD GreedyMask {label}] Skipping: n_correct={n_correct_orig}, n_incorrect={n_incorrect_orig}"
            )
        return advantages

    # 最低存留数量
    min_correct = max(1, int(n_correct_orig * min_retain_ratio))
    min_incorrect = max(1, int(n_incorrect_orig * min_retain_ratio))

    # 构建工作列表：(原始索引, mean_adv)
    # 正样本按 mean_adv 升序排列（最差的排前面，优先剔除）
    correct_work = sorted(
        [(idx, mean_advs[idx].item()) for idx in correct_indices],
        key=lambda x: x[1],
    )
    # 负样本按 mean_adv 降序排列（最好的排前面，优先剔除）
    incorrect_work = sorted(
        [(idx, mean_advs[idx].item()) for idx in incorrect_indices],
        key=lambda x: -x[1],
    )

    # 当前活跃的 adv 值列表
    active_correct_advs = [v for _, v in correct_work]
    active_incorrect_advs = [v for _, v in incorrect_work]

    # 记录被剔除的样本索引
    masked_indices: list[int] = []

    current_gap = _compute_gap(active_correct_advs, active_incorrect_advs, mode)

    if is_log_rank:
        logger.info(
            f"[OPD GreedyMask {label}] Initial: n_correct={n_correct_orig}, "
            f"n_incorrect={n_incorrect_orig}, gap={current_gap:.4f}, delta={delta}"
        )

    iteration = 0
    while current_gap < delta:
        # 检查是否还能剔除
        can_remove_correct = len(active_correct_advs) > min_correct
        can_remove_incorrect = len(active_incorrect_advs) > min_incorrect

        if not can_remove_correct and not can_remove_incorrect:
            if is_log_rank:
                logger.info(
                    f"[OPD GreedyMask {label}] Stopped: reached min_retain_ratio={min_retain_ratio}, "
                    f"remaining correct={len(active_correct_advs)}, "
                    f"remaining incorrect={len(active_incorrect_advs)}, "
                    f"gap={current_gap:.4f}"
                )
            break

        # 计算两种剔除方案的边际增益
        best_action = None  # ("correct", idx) 或 ("incorrect", idx)
        best_gap = current_gap

        if can_remove_correct:
            # 尝试剔除排序最前面的正样本（mean_adv 最低的）
            gap_after = _compute_gap_after_remove_correct(active_correct_advs, 0, active_incorrect_advs, mode)
            if gap_after > best_gap:
                best_gap = gap_after
                best_action = ("correct", 0)

        if can_remove_incorrect:
            # 尝试剔除排序最前面的负样本（mean_adv 最高的）
            gap_after = _compute_gap_after_remove_incorrect(active_correct_advs, active_incorrect_advs, 0, mode)
            if gap_after > best_gap:
                best_gap = gap_after
                best_action = ("incorrect", 0)

        if best_action is None:
            # 两种剔除都无法增大 gap，停止
            if is_log_rank:
                logger.info(f"[OPD GreedyMask {label}] Stopped: no beneficial removal found, gap={current_gap:.4f}")
            break

        # 执行剔除
        if best_action[0] == "correct":
            removed_idx, removed_val = correct_work.pop(0)
            active_correct_advs.pop(0)
            masked_indices.append(removed_idx)
        else:
            removed_idx, removed_val = incorrect_work.pop(0)
            active_incorrect_advs.pop(0)
            masked_indices.append(removed_idx)

        current_gap = best_gap
        iteration += 1

    # 将被剔除的样本的 advantages 置零
    for idx in masked_indices:
        advantages[idx] = torch.zeros_like(advantages[idx])

    if is_log_rank:
        n_masked = len(masked_indices)
        n_remain_correct = len(active_correct_advs)
        n_remain_incorrect = len(active_incorrect_advs)
        logger.info(
            f"[OPD GreedyMask {label}] Done: masked {n_masked}/{n_total} ({n_masked / n_total * 100:.1f}%) samples, "
            f"remaining correct={n_remain_correct}/{n_correct_orig} ({n_remain_correct / n_correct_orig * 100:.1f}%), "
            f"remaining incorrect={n_remain_incorrect}/{n_incorrect_orig} ({n_remain_incorrect / n_incorrect_orig * 100:.1f}%), "
            f"final_gap={current_gap:.4f}, iterations={iteration}"
        )

    return advantages


def _greedy_mask_global_distributed(
    advantages: list[torch.Tensor],
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    mode: str,
    delta: float,
    min_retain_ratio: float,
    dp_group: dist.ProcessGroup,
    is_log_rank: bool = True,
) -> list[torch.Tensor]:
    """通过 AllReduce 在所有 DP rank 上执行全局贪婪优势掩码。

    核心思路：
    1. 每个 rank 将本地样本的 (mean_adv, correct/incorrect, rank_id, local_idx) 通过
       AllGather 汇总到所有 rank。
    2. 所有 rank 在汇总后的全局视图上执行相同的贪婪算法（确定性的），得到一致的剔除决策。
    3. 每个 rank 根据决策将自己本地被剔除的样本 advantages 置零。

    Args:
        advantages: 本 rank 上的 token 级别 advantage 列表。
        mean_advs: 本 rank 上每个样本的平均 advantage。
        correct_mask: 本 rank 上正确样本的 bool mask。
        incorrect_mask: 本 rank 上错误样本的 bool mask。
        mode: "mean" 或 "minmax"。
        delta: 目标 margin。
        min_retain_ratio: 最低存留比例。
        dp_group: 数据并行通信组。
        is_log_rank: 是否打印日志。

    Returns:
        修改后的 advantages 列表。
    """
    device = mean_advs.device
    dp_world_size = dist.get_world_size(group=dp_group)
    dp_rank = dist.get_rank(group=dp_group)

    n_local = len(advantages)

    # 构建本地信息 tensor: [mean_adv, label]
    # label: 1.0 = correct, -1.0 = incorrect, 0.0 = unknown
    local_info = torch.zeros(n_local, 2, device=device, dtype=torch.float64)
    for i in range(n_local):
        local_info[i, 0] = mean_advs[i].double()
        if correct_mask[i]:
            local_info[i, 1] = 1.0
        elif incorrect_mask[i]:
            local_info[i, 1] = -1.0

    # AllGather 本地样本数量
    local_count = torch.tensor([n_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dp_world_size)]
    dist.all_gather(all_counts, local_count, group=dp_group)
    all_counts_list = [c.item() for c in all_counts]
    max_count = max(all_counts_list)

    # Pad 到相同长度后 AllGather
    padded_info = torch.zeros(max_count, 2, device=device, dtype=torch.float64)
    padded_info[:n_local] = local_info
    all_padded_info = [torch.zeros(max_count, 2, device=device, dtype=torch.float64) for _ in range(dp_world_size)]
    dist.all_gather(all_padded_info, padded_info, group=dp_group)

    # 重建全局样本列表: (mean_adv, label, rank_id, local_idx)
    global_samples: list[tuple[float, float, int, int]] = []
    for rank_id, (info, count) in enumerate(zip(all_padded_info, all_counts_list, strict=False)):
        for local_idx in range(count):
            adv_val = info[local_idx, 0].item()
            label = info[local_idx, 1].item()
            global_samples.append((adv_val, label, rank_id, local_idx))

    # 分离正/负样本
    # 正样本按 mean_adv 升序（最差的排前面）
    correct_samples = sorted(
        [(adv_val, rank_id, local_idx) for adv_val, label, rank_id, local_idx in global_samples if label > 0],
        key=lambda x: x[0],
    )
    # 负样本按 mean_adv 降序（最好的排前面）
    incorrect_samples = sorted(
        [(adv_val, rank_id, local_idx) for adv_val, label, rank_id, local_idx in global_samples if label < 0],
        key=lambda x: -x[0],
    )

    n_correct_orig = len(correct_samples)
    n_incorrect_orig = len(incorrect_samples)

    if n_correct_orig == 0 or n_incorrect_orig == 0:
        if is_log_rank:
            logger.info(f"[OPD GreedyMask Global] Skipping: n_correct={n_correct_orig}, n_incorrect={n_incorrect_orig}")
        return advantages

    min_correct = max(1, int(n_correct_orig * min_retain_ratio))
    min_incorrect = max(1, int(n_incorrect_orig * min_retain_ratio))

    # 当前活跃的 adv 值列表
    active_correct_advs = [s[0] for s in correct_samples]
    active_incorrect_advs = [s[0] for s in incorrect_samples]

    # 记录被剔除的 (rank_id, local_idx)
    masked_set: set[tuple[int, int]] = set()

    current_gap = _compute_gap(active_correct_advs, active_incorrect_advs, mode)

    if is_log_rank:
        logger.info(
            f"[OPD GreedyMask Global] Initial: n_correct={n_correct_orig}, "
            f"n_incorrect={n_incorrect_orig}, gap={current_gap:.4f}, delta={delta}"
        )

    # 维护指针，指向下一个待剔除的样本
    correct_ptr = 0
    incorrect_ptr = 0
    iteration = 0

    while current_gap < delta:
        can_remove_correct = (len(active_correct_advs) - correct_ptr) > min_correct - correct_ptr
        can_remove_incorrect = (len(active_incorrect_advs) - incorrect_ptr) > min_incorrect - incorrect_ptr

        # 重新计算：当前活跃列表是从 ptr 开始的
        remaining_correct = active_correct_advs[correct_ptr:]
        remaining_incorrect = active_incorrect_advs[incorrect_ptr:]

        can_remove_correct = len(remaining_correct) > min_correct
        can_remove_incorrect = len(remaining_incorrect) > min_incorrect

        if not can_remove_correct and not can_remove_incorrect:
            if is_log_rank:
                logger.info(
                    f"[OPD GreedyMask Global] Stopped: reached min_retain_ratio={min_retain_ratio}, "
                    f"remaining correct={len(remaining_correct)}, "
                    f"remaining incorrect={len(remaining_incorrect)}, "
                    f"gap={current_gap:.4f}"
                )
            break

        best_action = None
        best_gap = current_gap

        if can_remove_correct:
            gap_after = _compute_gap(remaining_correct[1:], remaining_incorrect, mode)
            if gap_after > best_gap:
                best_gap = gap_after
                best_action = "correct"

        if can_remove_incorrect:
            gap_after = _compute_gap(remaining_correct, remaining_incorrect[1:], mode)
            if gap_after > best_gap:
                best_gap = gap_after
                best_action = "incorrect"

        if best_action is None:
            if is_log_rank:
                logger.info(f"[OPD GreedyMask Global] Stopped: no beneficial removal found, gap={current_gap:.4f}")
            break

        if best_action == "correct":
            _, rank_id, local_idx = correct_samples[correct_ptr]
            masked_set.add((rank_id, local_idx))
            correct_ptr += 1
        else:
            _, rank_id, local_idx = incorrect_samples[incorrect_ptr]
            masked_set.add((rank_id, local_idx))
            incorrect_ptr += 1

        current_gap = best_gap
        iteration += 1

    # 将本 rank 被剔除的样本 advantages 置零
    n_masked_local = 0
    for local_idx in range(n_local):
        if (dp_rank, local_idx) in masked_set:
            advantages[local_idx] = torch.zeros_like(advantages[local_idx])
            n_masked_local += 1

    if is_log_rank:
        n_masked_global = len(masked_set)
        n_total_global = n_correct_orig + n_incorrect_orig
        n_remain_correct = n_correct_orig - correct_ptr
        n_remain_incorrect = n_incorrect_orig - incorrect_ptr
        logger.info(
            f"[OPD GreedyMask Global] Done: masked {n_masked_global}/{n_total_global} ({n_masked_global / n_total_global * 100:.1f}%) samples globally, "
            f"remaining correct={n_remain_correct}/{n_correct_orig} ({n_remain_correct / n_correct_orig * 100:.1f}%), "
            f"remaining incorrect={n_remain_incorrect}/{n_incorrect_orig} ({n_remain_incorrect / n_incorrect_orig * 100:.1f}%), "
            f"final_gap={current_gap:.4f}, iterations={iteration}"
        )

    return advantages


def _greedy_mask_group_distributed(
    advantages: list[torch.Tensor],
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    sample_group_index: list[int],
    mode: str,
    delta: float,
    min_retain_ratio: float,
    dp_group: dist.ProcessGroup,
    is_log_rank: bool = True,
) -> list[torch.Tensor]:
    """通过 AllGather 在所有 DP rank 上按 group 执行贪婪优势掩码。

    与 global 类似，但按 group 分别执行贪婪算法。

    Args:
        advantages: 本 rank 上的 token 级别 advantage 列表。
        mean_advs: 本 rank 上每个样本的平均 advantage。
        correct_mask: 本 rank 上正确样本的 bool mask。
        incorrect_mask: 本 rank 上错误样本的 bool mask。
        sample_group_index: 本 rank 上每个样本的 group id。
        mode: "mean" 或 "minmax"。
        delta: 目标 margin。
        min_retain_ratio: 最低存留比例。
        dp_group: 数据并行通信组。
        is_log_rank: 是否打印日志。

    Returns:
        修改后的 advantages 列表。
    """
    device = mean_advs.device
    dp_world_size = dist.get_world_size(group=dp_group)
    dp_rank = dist.get_rank(group=dp_group)

    n_local = len(advantages)

    # 构建本地信息 tensor: [mean_adv, label, group_id]
    local_info = torch.zeros(n_local, 3, device=device, dtype=torch.float64)
    for i in range(n_local):
        local_info[i, 0] = mean_advs[i].double()
        if correct_mask[i]:
            local_info[i, 1] = 1.0
        elif incorrect_mask[i]:
            local_info[i, 1] = -1.0
        local_info[i, 2] = float(sample_group_index[i])

    # AllGather
    local_count = torch.tensor([n_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dp_world_size)]
    dist.all_gather(all_counts, local_count, group=dp_group)
    all_counts_list = [c.item() for c in all_counts]
    max_count = max(all_counts_list) if all_counts_list else 0

    if max_count == 0:
        return advantages

    padded_info = torch.zeros(max_count, 3, device=device, dtype=torch.float64)
    padded_info[:n_local] = local_info
    all_padded_info = [torch.zeros(max_count, 3, device=device, dtype=torch.float64) for _ in range(dp_world_size)]
    dist.all_gather(all_padded_info, padded_info, group=dp_group)

    # 重建全局样本列表，按 group 分组
    # group_data[gid] = {"correct": [(adv, rank, local_idx), ...], "incorrect": [...]}
    group_data: dict[int, dict[str, list[tuple[float, int, int]]]] = {}
    for rank_id, (info, count) in enumerate(zip(all_padded_info, all_counts_list, strict=False)):
        for local_idx in range(count):
            adv_val = info[local_idx, 0].item()
            label = info[local_idx, 1].item()
            gid = int(info[local_idx, 2].item())
            if gid not in group_data:
                group_data[gid] = {"correct": [], "incorrect": []}
            if label > 0:
                group_data[gid]["correct"].append((adv_val, rank_id, local_idx))
            elif label < 0:
                group_data[gid]["incorrect"].append((adv_val, rank_id, local_idx))

    # 对每个 group 执行贪婪算法
    masked_set: set[tuple[int, int]] = set()

    for gid in sorted(group_data.keys()):
        gd = group_data[gid]
        correct_samples = sorted(gd["correct"], key=lambda x: x[0])  # 升序
        incorrect_samples = sorted(gd["incorrect"], key=lambda x: -x[0])  # 降序

        n_correct_orig = len(correct_samples)
        n_incorrect_orig = len(incorrect_samples)

        if n_correct_orig == 0 or n_incorrect_orig == 0:
            if is_log_rank:
                logger.info(
                    f"[OPD GreedyMask Group={gid}] Skipping: n_correct={n_correct_orig}, n_incorrect={n_incorrect_orig}"
                )
            continue

        min_correct = max(1, int(n_correct_orig * min_retain_ratio))
        min_incorrect = max(1, int(n_incorrect_orig * min_retain_ratio))

        active_correct_advs = [s[0] for s in correct_samples]
        active_incorrect_advs = [s[0] for s in incorrect_samples]

        correct_ptr = 0
        incorrect_ptr = 0
        current_gap = _compute_gap(active_correct_advs, active_incorrect_advs, mode)

        if is_log_rank:
            logger.info(
                f"[OPD GreedyMask Group={gid}] Initial: n_correct={n_correct_orig}, "
                f"n_incorrect={n_incorrect_orig}, gap={current_gap:.4f}, delta={delta}"
            )

        iteration = 0
        while current_gap < delta:
            remaining_correct = active_correct_advs[correct_ptr:]
            remaining_incorrect = active_incorrect_advs[incorrect_ptr:]

            can_remove_correct = len(remaining_correct) > min_correct
            can_remove_incorrect = len(remaining_incorrect) > min_incorrect

            if not can_remove_correct and not can_remove_incorrect:
                if is_log_rank:
                    logger.info(
                        f"[OPD GreedyMask Group={gid}] Stopped: reached min_retain_ratio, "
                        f"remaining correct={len(remaining_correct)}, "
                        f"remaining incorrect={len(remaining_incorrect)}, "
                        f"gap={current_gap:.4f}"
                    )
                break

            best_action = None
            best_gap = current_gap

            if can_remove_correct:
                gap_after = _compute_gap(remaining_correct[1:], remaining_incorrect, mode)
                if gap_after > best_gap:
                    best_gap = gap_after
                    best_action = "correct"

            if can_remove_incorrect:
                gap_after = _compute_gap(remaining_correct, remaining_incorrect[1:], mode)
                if gap_after > best_gap:
                    best_gap = gap_after
                    best_action = "incorrect"

            if best_action is None:
                if is_log_rank:
                    logger.info(f"[OPD GreedyMask Group={gid}] Stopped: no beneficial removal, gap={current_gap:.4f}")
                break

            if best_action == "correct":
                _, rank_id, local_idx = correct_samples[correct_ptr]
                masked_set.add((rank_id, local_idx))
                correct_ptr += 1
            else:
                _, rank_id, local_idx = incorrect_samples[incorrect_ptr]
                masked_set.add((rank_id, local_idx))
                incorrect_ptr += 1

            current_gap = best_gap
            iteration += 1

        if is_log_rank:
            n_masked_grp = correct_ptr + incorrect_ptr
            n_total_grp = n_correct_orig + n_incorrect_orig
            n_remain_correct = n_correct_orig - correct_ptr
            n_remain_incorrect = n_incorrect_orig - incorrect_ptr
            logger.info(
                f"[OPD GreedyMask Group={gid}] Done: masked "
                f"{n_masked_grp}/{n_total_grp} ({n_masked_grp / n_total_grp * 100:.1f}%), "
                f"remaining correct={n_remain_correct}/{n_correct_orig} ({n_remain_correct / n_correct_orig * 100:.1f}%), "
                f"remaining incorrect={n_remain_incorrect}/{n_incorrect_orig} ({n_remain_incorrect / n_incorrect_orig * 100:.1f}%), "
                f"final_gap={current_gap:.4f}, iterations={iteration}"
            )

    # 将本 rank 被剔除的样本 advantages 置零
    n_masked_local = 0
    for local_idx in range(n_local):
        if (dp_rank, local_idx) in masked_set:
            advantages[local_idx] = torch.zeros_like(advantages[local_idx])
            n_masked_local += 1

    if is_log_rank:
        logger.info(f"[OPD GreedyMask Group] Total masked on this rank: {n_masked_local}/{n_local}")

    return advantages


def opd_correctness_margin_mask(
    args: Namespace,
    advantages: list[torch.Tensor],
    rollout_data: RolloutBatch,
    parallel_state: ParallelState | None = None,
) -> list[torch.Tensor]:
    """基于 response 正确性对 OPD advantage 执行贪婪掩码。

    通过迭代剔除"最不具代表性"的样本（最差正样本或最好负样本），
    使得正确样本与错误样本之间的 advantage gap 满足 margin 条件。
    被剔除的样本的 advantages 全部置零，不参与梯度更新。

    支持 scope="global"（全局）和 scope="group"（按 group）两种模式。
    支持 mode="mean" 和 mode="minmax" 两种 margin 判定方式。
    支持分布式场景，通过 AllGather 汇总全局样本信息后执行确定性贪婪算法。

    Args:
        args: 包含以下参数：
            - opd_margin_scope: "global" 或 "group"
            - opd_margin_mode: "mean" 或 "minmax"
            - opd_margin_delta: margin 目标值
            - opd_mask_min_retain_ratio: 最低存留比例（默认 0.25）
        advantages: 每个样本的 token 级别 advantage 列表。
        rollout_data: 包含 response_correct, sample_group_index 等字段。
        parallel_state: 并行状态，用于跨 rank AllGather。

    Returns:
        调整后的 advantages 列表（被剔除样本的 advantages 置零）。
    """
    response_correct: list[bool | None] | None = rollout_data.get("response_correct", None)

    scope: str = args.opd_margin_scope
    mode: str = args.opd_margin_mode
    delta: float = args.opd_margin_delta
    min_retain_ratio: float = args.opd_mask_min_retain_ratio

    device = advantages[0].device
    dp_group: dist.ProcessGroup = parallel_state.dp_group if parallel_state is not None else None
    dp_rank = parallel_state.dp_rank if parallel_state is not None else 0
    tp_rank = parallel_state.tp_rank if parallel_state is not None else 0
    is_log_rank = dp_rank == 0 and tp_rank == 0

    # 注意：不能在此处做 early return，因为 AllGather 要求所有 rank 都参与。
    if response_correct is None or all(rc is None for rc in response_correct):
        if parallel_state is None:
            if response_correct is None:
                logger.warning("[OPD GreedyMask] response_correct not found in rollout_data, skipping.")
            else:
                logger.warning("[OPD GreedyMask] All samples have response_correct=None, skipping.")
            return advantages
        else:
            logger.warning(
                "[OPD GreedyMask] Local rank has no valid response_correct, "
                "but still participating in AllGather for consistency."
            )
            response_correct = [None] * len(advantages)

    # 计算每个样本的平均 advantage
    mean_advs = torch.stack([adv.mean() for adv in advantages])

    # 构建正确/错误 mask
    correct_mask = torch.tensor([rc is True for rc in response_correct], dtype=torch.bool, device=device)
    incorrect_mask = torch.tensor([rc is False for rc in response_correct], dtype=torch.bool, device=device)

    if scope == "global":
        if parallel_state is None:
            logger.warning(
                "[OPD GreedyMask] parallel_state is None for scope='global', falling back to local-only computation."
            )
            advantages = _greedy_mask_local(
                advantages=advantages,
                mean_advs=mean_advs,
                correct_mask=correct_mask,
                incorrect_mask=incorrect_mask,
                mode=mode,
                delta=delta,
                min_retain_ratio=min_retain_ratio,
                is_log_rank=is_log_rank,
                label="Global(local)",
            )
        else:
            advantages = _greedy_mask_global_distributed(
                advantages=advantages,
                mean_advs=mean_advs,
                correct_mask=correct_mask,
                incorrect_mask=incorrect_mask,
                mode=mode,
                delta=delta,
                min_retain_ratio=min_retain_ratio,
                dp_group=dp_group,
                is_log_rank=is_log_rank,
            )

    elif scope == "group":
        sample_group_index: list[int] | None = rollout_data.get("sample_group_index", None)
        if sample_group_index is None:
            logger.warning("[OPD GreedyMask] sample_group_index not found, falling back to global scope.")
            if parallel_state is None:
                advantages = _greedy_mask_local(
                    advantages=advantages,
                    mean_advs=mean_advs,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                    mode=mode,
                    delta=delta,
                    min_retain_ratio=min_retain_ratio,
                    is_log_rank=is_log_rank,
                    label="Global(fallback)",
                )
            else:
                advantages = _greedy_mask_global_distributed(
                    advantages=advantages,
                    mean_advs=mean_advs,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                    mode=mode,
                    delta=delta,
                    min_retain_ratio=min_retain_ratio,
                    dp_group=dp_group,
                    is_log_rank=is_log_rank,
                )
        else:
            if parallel_state is None:
                # 本地按 group 处理
                local_group_ids = set(sample_group_index)
                for gid in sorted(local_group_ids):
                    group_mask = torch.tensor([sg == gid for sg in sample_group_index], dtype=torch.bool, device=device)
                    g_correct = correct_mask & group_mask
                    g_incorrect = incorrect_mask & group_mask
                    advantages = _greedy_mask_local(
                        advantages=advantages,
                        mean_advs=mean_advs,
                        correct_mask=g_correct,
                        incorrect_mask=g_incorrect,
                        mode=mode,
                        delta=delta,
                        min_retain_ratio=min_retain_ratio,
                        is_log_rank=is_log_rank,
                        label=f"Group={gid}(local)",
                    )
            else:
                advantages = _greedy_mask_group_distributed(
                    advantages=advantages,
                    mean_advs=mean_advs,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                    sample_group_index=sample_group_index,
                    mode=mode,
                    delta=delta,
                    min_retain_ratio=min_retain_ratio,
                    dp_group=dp_group,
                    is_log_rank=is_log_rank,
                )
    else:
        logger.warning(f"[OPD GreedyMask] Unknown scope '{scope}', skipping.")

    return advantages
