# Modified by the DN-MOPD authors (2026): fixed the package imports (miles.miles.* -> miles.*); loss.py imports
# this module unconditionally.
import logging
from argparse import Namespace

import torch
import torch.distributed as dist
from miles.backends.training_utils.parallel import ParallelState
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def _apply_shift(
    advantages: list[torch.Tensor],
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    shift: float,
    direction: str,
) -> None:
    """根据 direction 将 shift 分配到正确/错误样本上（原地修改 advantages）。

    Args:
        advantages: token 级别 advantage 列表。
        correct_mask: 正确样本的 bool mask。
        incorrect_mask: 错误样本的 bool mask。
        shift: 需要施加的总偏移量（正值）。
        direction: 'correct_up' | 'incorrect_down' | 'both'。
    """
    if direction == "correct_up":
        for idx in correct_mask.nonzero(as_tuple=True)[0]:
            advantages[idx.item()] = advantages[idx.item()] + shift
    elif direction == "incorrect_down":
        for idx in incorrect_mask.nonzero(as_tuple=True)[0]:
            advantages[idx.item()] = advantages[idx.item()] - shift
    elif direction == "both":
        half = shift / 2.0
        for idx in correct_mask.nonzero(as_tuple=True)[0]:
            advantages[idx.item()] = advantages[idx.item()] + half
        for idx in incorrect_mask.nonzero(as_tuple=True)[0]:
            advantages[idx.item()] = advantages[idx.item()] - half
    else:
        logger.warning(f"[OPD Margin] Unknown direction '{direction}', defaulting to correct_up.")
        for idx in correct_mask.nonzero(as_tuple=True)[0]:
            advantages[idx.item()] = advantages[idx.item()] + shift
    return


def _margin_shift_for_indices(
    advantages: list[torch.Tensor],
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    mode: str,
    delta: float,
) -> list[torch.Tensor]:
    """对指定的正确/错误样本集合执行 margin shifting。

    Args:
        advantages: token 级别 advantage 列表（会被原地修改）。
        mean_advs: 每个样本的平均 advantage。
        correct_mask: 正确样本的 bool mask。
        incorrect_mask: 错误样本的 bool mask。
        mode: "mean" 或 "minmax"。
        delta: margin 值。

    Returns:
        修改后的 advantages 列表。
    """
    n_correct = correct_mask.sum().item()
    n_incorrect = incorrect_mask.sum().item()

    if n_correct == 0 or n_incorrect == 0:
        # 没有正确或没有错误样本，无法计算 margin，跳过
        logger.info(f"[OPD Margin] Skipping: n_correct={n_correct}, n_incorrect={n_incorrect}")
        return advantages

    correct_advs = mean_advs[correct_mask]
    incorrect_advs = mean_advs[incorrect_mask]

    if mode == "mean":
        # 要求 mean(correct) - mean(incorrect) > delta
        correct_stat = correct_advs.mean()
        incorrect_stat = incorrect_advs.mean()
    elif mode == "minmax":
        # 要求 min(correct) - max(incorrect) > delta
        correct_stat = correct_advs.min()
        incorrect_stat = incorrect_advs.max()
    else:
        logger.warning(f"[OPD Margin] Unknown mode '{mode}', skipping.")
        return advantages

    current_gap = (correct_stat - incorrect_stat).item()

    if current_gap >= delta:
        # 已满足 margin 条件，无需调整
        logger.info(
            f"[OPD Margin] No shift needed: mode={mode}, "
            f"correct_stat={correct_stat.item():.4f}, incorrect_stat={incorrect_stat.item():.4f}, "
            f"gap={current_gap:.4f} >= delta={delta}"
        )
        return advantages

    # 计算偏移量：shift = (incorrect_stat - correct_stat) + delta
    shift = (incorrect_stat - correct_stat).item() + delta
    logger.info(
        f"[OPD Margin] Shifting correct samples: mode={mode}, "
        f"correct_stat={correct_stat.item():.4f}, incorrect_stat={incorrect_stat.item():.4f}, "
        f"gap={current_gap:.4f} < delta={delta}, shift={shift:.4f}"
    )

    # 根据 direction 对正确/错误样本应用 shift（本地回退路径默认 correct_up）
    _apply_shift(advantages, correct_mask, incorrect_mask, shift, "correct_up")

    return advantages


def _compute_shift_via_allreduce(
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    mode: str,
    delta: float,
    dp_group: dist.ProcessGroup | None,
    is_log_rank: bool = True,
    label: str = "Global",
) -> float:
    """通过 AllReduce 在所有 DP rank 上聚合统计量并计算 margin shift。

    每个 rank 只持有部分样本，通过 AllReduce 聚合后计算 shift。
    可用于 global scope（所有样本）或 group scope（按 group 聚合）。

    对于 mode="mean":
        AllReduce 正确/错误样本的 sum 和 count，计算全局均值。
    对于 mode="minmax":
        AllReduce 正确样本的全局 min 和错误样本的全局 max。

    Args:
        mean_advs: 本 rank 上每个样本的平均 advantage。
        correct_mask: 本 rank 上正确样本的 bool mask。
        incorrect_mask: 本 rank 上错误样本的 bool mask。
        mode: "mean" 或 "minmax"。
        delta: margin 值。
        dp_group: 数据并行通信组。
        is_log_rank: 是否为负责打印日志的 rank（dp_rank==0 且 tp_rank==0），避免重复输出。
        label: 日志标签，用于区分 global/group 调用。

    Returns:
        shift 值。如果无需调整或无法计算，返回 0.0。
    """
    device = mean_advs.device
    n_correct_local = correct_mask.sum().item()
    n_incorrect_local = incorrect_mask.sum().item()

    if mode == "mean":
        # 聚合 [correct_sum, correct_count, incorrect_sum, incorrect_count]
        correct_sum = mean_advs[correct_mask].sum() if n_correct_local > 0 else torch.tensor(0.0, device=device)
        incorrect_sum = mean_advs[incorrect_mask].sum() if n_incorrect_local > 0 else torch.tensor(0.0, device=device)

        stats = torch.tensor(
            [correct_sum.item(), float(n_correct_local), incorrect_sum.item(), float(n_incorrect_local)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=dp_group)

        global_correct_sum, global_correct_count, global_incorrect_sum, global_incorrect_count = stats.tolist()

        if global_correct_count == 0 or global_incorrect_count == 0:
            if is_log_rank:
                logger.info(
                    f"[OPD Margin {label}] Skipping: global_correct_count={global_correct_count:.0f}, "
                    f"global_incorrect_count={global_incorrect_count:.0f}"
                )
            return 0.0

        correct_stat = global_correct_sum / global_correct_count
        incorrect_stat = global_incorrect_sum / global_incorrect_count

        if is_log_rank:
            logger.info(
                f"[OPD Margin {label}] Global stats: correct_sum={global_correct_sum:.4f}, "
                f"correct_count={global_correct_count:.0f}, incorrect_sum={global_incorrect_sum:.4f}, "
                f"incorrect_count={global_incorrect_count:.0f}, correct_mean={correct_stat:.4f}, "
                f"incorrect_mean={incorrect_stat:.4f}"
            )

    elif mode == "minmax":
        # 聚合 min(correct) 和 max(incorrect)
        # 没有对应样本的 rank 用 +inf / -inf 作为哨兵值，不影响全局 min/max
        local_correct_min = mean_advs[correct_mask].min().item() if n_correct_local > 0 else float("inf")
        local_incorrect_max = mean_advs[incorrect_mask].max().item() if n_incorrect_local > 0 else float("-inf")

        # AllReduce: min 用 MIN op，max 用 MAX op
        # 合并为一个 tensor：[correct_min, -incorrect_max]，统一用 MIN op
        # 因为 min(-incorrect_max) = -max(incorrect_max)
        stats = torch.tensor(
            [local_correct_min, -local_incorrect_max],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MIN, group=dp_group)

        correct_stat = stats[0].item()
        incorrect_stat = -stats[1].item()  # 还原 max

        # 检查是否全局都没有正确或错误样本（哨兵值未被替换）
        if correct_stat == float("inf") or incorrect_stat == float("-inf"):
            if is_log_rank:
                logger.info(
                    f"[OPD Margin {label}] Skipping: no correct or incorrect samples across all ranks. "
                    f"correct_min={correct_stat}, incorrect_max={incorrect_stat}"
                )
            return 0.0
        if is_log_rank:
            logger.info(
                f"[OPD Margin {label}] Global stats: correct_min={correct_stat:.4f}, incorrect_max={incorrect_stat:.4f}"
            )

    else:
        logger.warning(f"[OPD Margin {label}] Unknown mode '{mode}', skipping.")
        return 0.0

    current_gap: float = correct_stat - incorrect_stat

    if current_gap >= delta:
        if is_log_rank:
            logger.info(
                f"[OPD Margin {label}] No shift needed: mode={mode}, "
                f"correct_stat={correct_stat:.4f}, incorrect_stat={incorrect_stat:.4f}, "
                f"gap={current_gap:.4f} >= delta={delta}"
            )
        return 0.0

    shift = (incorrect_stat - correct_stat) + delta
    if is_log_rank:
        logger.info(
            f"[OPD Margin {label}] Shifting: mode={mode}, "
            f"correct_stat={correct_stat:.4f}, incorrect_stat={incorrect_stat:.4f}, "
            f"gap={current_gap:.4f} < delta={delta}, shift={shift:.4f}"
        )
    return shift


def _compute_group_shifts_via_allreduce(
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    sample_group_index: list[int],
    mode: str,
    delta: float,
    dp_group: dist.ProcessGroup | None,
    is_log_rank: bool = True,
) -> dict[int, float]:
    """通过 AllReduce 在所有 DP rank 上按 group 聚合统计量并计算每个 group 的 shift。

    同一个 group 的样本可能分散在不同 DP rank 上，因此需要通过 AllReduce 聚合。
    为了减少通信次数，将所有 group 的统计量打包到一个 tensor 中一次性 AllReduce。

    Args:
        mean_advs: 本 rank 上每个样本的平均 advantage。
        correct_mask: 本 rank 上正确样本的 bool mask。
        incorrect_mask: 本 rank 上错误样本的 bool mask。
        sample_group_index: 本 rank 上每个样本的 group id。
        mode: "mean" 或 "minmax"。
        delta: margin 值。
        dp_group: 数据并行通信组。
        is_log_rank: 是否为负责打印日志的 rank（dp_rank==0 且 tp_rank==0），避免重复输出。

    Returns:
        group_shifts: {group_id: shift_value} 字典。
    """
    device = mean_advs.device

    # 收集本 rank 上出现的所有 group id
    local_group_ids = set(sample_group_index)

    # 通过 AllReduce 获取全局所有 group id
    # 方法：每个 rank 创建一个足够大的 tensor 标记自己有哪些 group
    # 但 group id 可能很大，改用 AllGather 收集所有 rank 的 group id 集合
    # 为简化实现，先 AllGather group id 数量，再 AllGather group id 列表
    dp_world_size = dist.get_world_size(group=dp_group) if dp_group is not None else 1

    if dp_world_size <= 1:
        # 单 rank，直接本地计算
        group_shifts = {}
        for gid in local_group_ids:
            group_mask = torch.tensor([sg == gid for sg in sample_group_index], dtype=torch.bool, device=device)
            g_correct = correct_mask & group_mask
            g_incorrect = incorrect_mask & group_mask
            n_c = g_correct.sum().item()
            n_i = g_incorrect.sum().item()
            if n_c > 0 and n_i > 0:
                shift = _compute_shift_local(mean_advs, g_correct, g_incorrect, mode, delta, gid)
                if shift > 0:
                    group_shifts[gid] = shift
        return group_shifts

    # 多 rank 场景：通过 AllGather 收集所有 rank 的 group id
    local_gids_tensor = torch.tensor(sorted(local_group_ids), dtype=torch.long, device=device)
    local_count = torch.tensor([len(local_group_ids)], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dp_world_size)]
    dist.all_gather(all_counts, local_count, group=dp_group)

    max_count = max(c.item() for c in all_counts)
    # Pad 到相同长度
    padded_gids = torch.full((max_count,), -1, dtype=torch.long, device=device)
    padded_gids[: len(local_gids_tensor)] = local_gids_tensor
    all_padded_gids = [torch.zeros(max_count, dtype=torch.long, device=device) for _ in range(dp_world_size)]
    dist.all_gather(all_padded_gids, padded_gids, group=dp_group)

    # 合并所有 group id（去掉 padding 的 -1）
    all_group_ids = set()
    for gids in all_padded_gids:
        for gid in gids.tolist():
            if gid >= 0:
                all_group_ids.add(gid)
    all_group_ids = sorted(all_group_ids)
    num_groups = len(all_group_ids)
    gid_to_idx = {gid: idx for idx, gid in enumerate(all_group_ids)}

    if mode == "mean":
        # 每个 group 需要 4 个值：correct_sum, correct_count, incorrect_sum, incorrect_count
        stats = torch.zeros(num_groups, 4, device=device, dtype=torch.float64)
        for i, gid in enumerate(sample_group_index):
            idx = gid_to_idx.get(gid)
            if idx is None:
                continue
            if correct_mask[i]:
                stats[idx, 0] += mean_advs[i].double()
                stats[idx, 1] += 1.0
            elif incorrect_mask[i]:
                stats[idx, 2] += mean_advs[i].double()
                stats[idx, 3] += 1.0

        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=dp_group)

        group_shifts = {}
        for gid in all_group_ids:
            idx = gid_to_idx[gid]
            c_sum, c_count, i_sum, i_count = stats[idx].tolist()
            if c_count == 0 or i_count == 0:
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] Skipping: mode={mode}, "
                        f"n_correct={c_count:.0f}, n_incorrect={i_count:.0f} (need both > 0)"
                    )
                continue
            correct_stat = c_sum / c_count
            incorrect_stat = i_sum / i_count
            current_gap = correct_stat - incorrect_stat
            if current_gap < delta:
                shift = (incorrect_stat - correct_stat) + delta
                group_shifts[gid] = shift
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] Shifting: mode={mode}, "
                        f"correct_mean={correct_stat:.4f}, incorrect_mean={incorrect_stat:.4f}, "
                        f"n_correct={c_count:.0f}, n_incorrect={i_count:.0f}, "
                        f"gap={current_gap:.4f} < delta={delta}, shift={shift:.4f}"
                    )
            else:
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] No shift needed: mode={mode}, "
                        f"correct_mean={correct_stat:.4f}, incorrect_mean={incorrect_stat:.4f}, "
                        f"n_correct={c_count:.0f}, n_incorrect={i_count:.0f}, "
                        f"gap={current_gap:.4f} >= delta={delta}"
                    )
        return group_shifts

    elif mode == "minmax":
        # 每个 group 需要 2 个值：correct_min, -incorrect_max（统一用 MIN op）
        stats = torch.full((num_groups, 2), float("inf"), device=device, dtype=torch.float64)
        for i, gid in enumerate(sample_group_index):
            idx = gid_to_idx.get(gid)
            if idx is None:
                continue
            if correct_mask[i]:
                stats[idx, 0] = min(stats[idx, 0].item(), mean_advs[i].double().item())
            elif incorrect_mask[i]:
                # 存储 -max，这样用 MIN op 就能得到全局 -max(incorrect)
                stats[idx, 1] = min(stats[idx, 1].item(), -mean_advs[i].double().item())

        dist.all_reduce(stats, op=dist.ReduceOp.MIN, group=dp_group)

        # minmax 模式下无法直接从 stats 获取 count，需要额外统计
        # 通过 AllReduce SUM 获取每个 group 的 correct_count 和 incorrect_count
        count_stats = torch.zeros(num_groups, 2, device=device, dtype=torch.float64)
        for i, gid in enumerate(sample_group_index):
            idx = gid_to_idx.get(gid)
            if idx is None:
                continue
            if correct_mask[i]:
                count_stats[idx, 0] += 1.0
            elif incorrect_mask[i]:
                count_stats[idx, 1] += 1.0
        dist.all_reduce(count_stats, op=dist.ReduceOp.SUM, group=dp_group)

        group_shifts = {}
        for gid in all_group_ids:
            idx = gid_to_idx[gid]
            correct_min = stats[idx, 0].item()
            incorrect_max = -stats[idx, 1].item()  # 还原 max
            if correct_min == float("inf") or incorrect_max == float("-inf"):
                n_c, n_i = count_stats[idx].tolist()
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] Skipping: mode={mode}, "
                        f"n_correct={n_c:.0f}, n_incorrect={n_i:.0f} (need both > 0)"
                    )
                continue
            current_gap = correct_min - incorrect_max
            n_c, n_i = count_stats[idx].tolist()
            if current_gap < delta:
                shift = (incorrect_max - correct_min) + delta
                group_shifts[gid] = shift
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] Shifting: mode={mode}, "
                        f"correct_min={correct_min:.4f}, incorrect_max={incorrect_max:.4f}, "
                        f"n_correct={n_c:.0f}, n_incorrect={n_i:.0f}, "
                        f"gap={current_gap:.4f} < delta={delta}, shift={shift:.4f}"
                    )
            else:
                if is_log_rank:
                    logger.info(
                        f"[OPD Margin Group={gid}] No shift needed: mode={mode}, "
                        f"correct_min={correct_min:.4f}, incorrect_max={incorrect_max:.4f}, "
                        f"n_correct={n_c:.0f}, n_incorrect={n_i:.0f}, "
                        f"gap={current_gap:.4f} >= delta={delta}"
                    )
        return group_shifts

    else:
        logger.warning(f"[OPD Margin Group] Unknown mode '{mode}', skipping.")
        return {}


def _compute_shift_local(
    mean_advs: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
    mode: str,
    delta: float,
    gid: int,
) -> float:
    """本地计算单个 group 的 shift（单 rank 场景）。"""
    correct_advs = mean_advs[correct_mask]
    incorrect_advs = mean_advs[incorrect_mask]

    if mode == "mean":
        correct_stat = correct_advs.mean().item()
        incorrect_stat = incorrect_advs.mean().item()
    elif mode == "minmax":
        correct_stat = correct_advs.min().item()
        incorrect_stat = incorrect_advs.max().item()
    else:
        return 0.0

    current_gap = correct_stat - incorrect_stat
    if current_gap >= delta:
        return 0.0
    return (incorrect_stat - correct_stat) + delta


def opd_correctness_margin_shift(
    args: Namespace,
    advantages: list[torch.Tensor],
    rollout_data: RolloutBatch,
    parallel_state: ParallelState | None = None,
) -> list[torch.Tensor]:
    """基于 response 正确性对 OPD advantage 进行 margin shifting。

    对于每个样本 i，先计算其平均 advantage（token 级别均值）：
        A_KL_i = mean(advantages[i])

    然后根据 scope（global / group）和 mode（mean / minmax）计算偏移量 delta，
    使得正确样本的 advantage 统计量比错误样本高出至少 margin。

    scope="global" 时，通过 AllReduce 在所有 DP rank 上聚合全局统计量。
    scope="group" 时，通过 AllReduce 在所有 DP rank 上按 group 聚合统计量，
    保证即使同 group 的样本分散在不同 rank 上也能正确计算。

    Args:
        args: 包含 opd_margin_scope, opd_margin_mode,
              opd_margin_delta 等参数。
        advantages: 每个样本的 token 级别 advantage 列表。
        rollout_data: 包含 response_correct, sample_group_index 等字段。
        parallel_state: 并行状态，scope="global" 时需要用于跨 rank AllReduce。

    Returns:
        调整后的 advantages 列表。
    """
    response_correct: list[bool | None] | None = rollout_data.get("response_correct", None)

    scope: str = args.opd_margin_scope
    mode: str = args.opd_margin_mode
    delta: float = args.opd_margin_delta
    direction: str = args.opd_margin_direction

    device = advantages[0].device
    dp_group: dist.ProcessGroup = parallel_state.dp_group if parallel_state is not None else None
    dp_rank = parallel_state.dp_rank if parallel_state is not None else 0
    tp_rank = parallel_state.tp_rank if parallel_state is not None else 0
    is_log_rank = dp_rank == 0 and tp_rank == 0

    # 注意：不能在此处做 early return，因为 AllReduce 要求所有 rank 都参与。
    # 如果本 rank 没有 response_correct 数据，仍需参与 AllReduce（贡献零值），
    # 否则其他 rank 会 hang。
    # 只有在确定不需要 AllReduce（parallel_state is None）时才可以 early return。
    if response_correct is None or all(rc is None for rc in response_correct):
        if parallel_state is None:
            if response_correct is None:
                logger.warning("[OPD Margin] response_correct not found in rollout_data, skipping margin shifting.")
            else:
                logger.warning("[OPD Margin] All samples have response_correct=None, skipping margin shifting.")
            return advantages
        else:
            # 有 parallel_state 时，不能 early return，需要构造空 mask 参与 AllReduce
            logger.warning(
                "[OPD Margin] Local rank has no valid response_correct, "
                "but still participating in AllReduce for consistency."
            )
            response_correct = [None] * len(advantages)

    # 计算每个样本的平均 advantage（消除序列长度偏差）
    mean_advs = torch.stack([adv.mean() for adv in advantages])

    # 构建正确/错误 mask
    correct_mask = torch.tensor([rc is True for rc in response_correct], dtype=torch.bool, device=device)
    incorrect_mask = torch.tensor([rc is False for rc in response_correct], dtype=torch.bool, device=device)

    if scope == "global":
        # 通过 AllReduce 在所有 DP rank 上计算全局统计量
        if parallel_state is None:
            logger.warning(
                "[OPD Margin] parallel_state is None for scope='global', falling back to local-only computation (no AllReduce)."
            )
            # 回退到本地计算（单 rank 场景或测试场景）
            advantages = _margin_shift_for_indices(
                advantages=advantages,
                mean_advs=mean_advs,
                correct_mask=correct_mask,
                incorrect_mask=incorrect_mask,
                mode=mode,
                delta=delta,
            )
        else:
            # 跨 rank AllReduce 计算全局 shift
            shift = _compute_shift_via_allreduce(
                mean_advs=mean_advs,
                correct_mask=correct_mask,
                incorrect_mask=incorrect_mask,
                mode=mode,
                delta=delta,
                dp_group=dp_group,
                is_log_rank=is_log_rank,
                label="Global",
            )
            if shift > 0:
                _apply_shift(advantages, correct_mask, incorrect_mask, shift, direction)

    elif scope == "group":
        # 通过 AllReduce 按 group 聚合各 rank 上的统计量
        sample_group_index: list[int] | None = rollout_data.get("sample_group_index", None)
        if sample_group_index is None:
            logger.warning("[OPD Margin] sample_group_index not found, falling back to global scope.")
            if parallel_state is None:
                advantages = _margin_shift_for_indices(
                    advantages=advantages,
                    mean_advs=mean_advs,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                    mode=mode,
                    delta=delta,
                )
            else:
                shift = _compute_shift_via_allreduce(
                    mean_advs=mean_advs,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                    mode=mode,
                    delta=delta,
                    dp_group=dp_group,
                    is_log_rank=is_log_rank,
                    label="Global(fallback)",
                )
                if shift > 0:
                    _apply_shift(advantages, correct_mask, incorrect_mask, shift, direction)
        else:
            # 按 group 分别处理，通过 AllReduce 聚合各 rank 上同 group 的统计量
            if is_log_rank:
                logger.info(f"[OPD Margin] Applying group shift via AllReduce, direction={direction}.")
            group_shifts = _compute_group_shifts_via_allreduce(
                mean_advs=mean_advs,
                correct_mask=correct_mask,
                incorrect_mask=incorrect_mask,
                sample_group_index=sample_group_index,
                mode=mode,
                delta=delta,
                dp_group=dp_group,
                is_log_rank=is_log_rank,
            )
            # 对本 rank 上属于需要 shift 的 group 的样本应用 shift
            for i, gid in enumerate(sample_group_index):
                if gid not in group_shifts:
                    continue
                s = group_shifts[gid]
                if direction == "correct_up":
                    if correct_mask[i]:
                        advantages[i] = advantages[i] + s
                elif direction == "incorrect_down":
                    if incorrect_mask[i]:
                        advantages[i] = advantages[i] - s
                elif direction == "both":
                    half = s / 2.0
                    if correct_mask[i]:
                        advantages[i] = advantages[i] + half
                    elif incorrect_mask[i]:
                        advantages[i] = advantages[i] - half
                else:
                    if correct_mask[i]:
                        advantages[i] = advantages[i] + s
    else:
        logger.warning(f"[OPD Margin] Unknown scope '{scope}', skipping margin shifting.")

    return advantages
