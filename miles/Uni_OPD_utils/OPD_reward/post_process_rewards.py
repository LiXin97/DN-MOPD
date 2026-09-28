# Modified by the DN-MOPD authors (2026): fixed the package import (exps.OPD.* -> Uni_OPD_utils.*) and parse the
# all-teacher scores (reward["per_teacher"]) into sample.per_teacher_log_probs for Uni_OPD_utils.mopd_hook.
import logging

import torch

from Uni_OPD_utils.OPD_reward.get_reward import REWARD_FAILED_KEY
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# 失败样本的 teacher_log_probs 哨兵值
TEACHER_LOGP_FAILED_SENTINEL = -100.0


def _echoed_token_ids_match(input_token_logprobs: list, tokens: list[int], response_length: int) -> bool:
    """True when the token ids the teacher echoes for the response tail equal sample.tokens' tail.

    If token compatibility (maybe_convert_tokens_for_teacher_compat) removed a token inside the response, the
    [-response_length:] slice would be shifted and the teacher log-probs misaligned with the student's tokens;
    such a teacher is treated as failed for the sample.
    """
    tail = input_token_logprobs[-response_length:]
    echoed = [item[1] for item in tail]
    return echoed == list(tokens[-response_length:])


def _parse_response_log_probs(input_token_logprobs: list, response_length: int) -> torch.Tensor:
    """input_token_logprobs -> response log-probs, with the legacy slicing: drop the first position (no log-prob),
    then keep the last response_length. A shorter result is head-padded with the sentinel so that it stays
    position-aligned with the rollout log-probs."""
    t_log_probs = torch.tensor(
        [item[0] for item in input_token_logprobs[1:]],
        dtype=torch.float32,
    )
    t_log_probs = t_log_probs[-response_length:]
    if t_log_probs.numel() < response_length:
        pad = torch.full((response_length - t_log_probs.numel(),), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32)
        t_log_probs = torch.cat([pad, t_log_probs])
    return t_log_probs


def _maybe_set_per_teacher_log_probs(sample: Sample, reward: dict, response_length: int) -> None:
    """When reward["per_teacher"] exists (all-teacher scoring), parse it into
    sample.per_teacher_log_probs = {teacher_name: Tensor[response_length]}; a failed teacher gets a row of
    TEACHER_LOGP_FAILED_SENTINEL. The legacy fields (teacher_log_probs, ...) are not touched."""
    per_teacher = reward.get("per_teacher") if isinstance(reward, dict) else None
    if per_teacher is None:
        return

    sentinel_row = torch.full((response_length,), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32)
    per_teacher_log_probs: dict[str, torch.Tensor] = {}
    for teacher_name, info in per_teacher.items():
        input_token_logprobs = None if info.get("failed", False) else info.get("input_token_logprobs")
        if input_token_logprobs is None:
            per_teacher_log_probs[teacher_name] = sentinel_row.clone()
            continue
        try:
            if not _echoed_token_ids_match(input_token_logprobs, sample.tokens, response_length):
                logger.warning(
                    f"[post_process_rewards] per_teacher[{teacher_name}] echoed token ids do not match "
                    f"sample tokens tail (token-compat removal inside response region?), "
                    f"filling with sentinel {TEACHER_LOGP_FAILED_SENTINEL}."
                )
                info["failed"] = True
                per_teacher_log_probs[teacher_name] = sentinel_row.clone()
            else:
                per_teacher_log_probs[teacher_name] = _parse_response_log_probs(input_token_logprobs, response_length)
        except Exception as e:
            logger.warning(
                f"[post_process_rewards] per_teacher[{teacher_name}] logprob parse error: {e}, "
                f"filling with sentinel {TEACHER_LOGP_FAILED_SENTINEL}."
            )
            per_teacher_log_probs[teacher_name] = sentinel_row.clone()
        finally:
            # drop the raw payload after parsing (K copies would otherwise travel into the ray object store)
            info["input_token_logprobs"] = None

    sample.per_teacher_log_probs = per_teacher_log_probs


# 设置在 --custom-reward-post-process-path 当中
# 用于 miles/ray/rollout.py 的 _post_process_rewards 方法中，调用 custom_reward_post_process_func
def post_process_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """
    Post process rewards.
    若 teacher 调用失败，将该样本的 teacher_log_probs 全部填充为 TEACHER_LOGP_FAILED_SENTINEL(-100)，
    作为哨兵值在 loss.py 中识别并 mask 掉，保证训练流程不中断。

    Return:
        raw_rewards: list[float], the original rewards without any post processing, used for logging and metric checking
        rewards: list[float], the rewards after post processing, used for training
    """
    rewards: list[dict[str]] = [sample.reward for sample in samples]

    num_failed = 0
    teacher_log_probs_list: list[torch.Tensor] = []

    for i, (reward, sample) in enumerate(zip(rewards, samples, strict=False)):
        response_length: int = sample.response_length

        # all-teacher scoring: parse every teacher's scores (a failure of the routed teacher is not a failure of
        # the others)
        _maybe_set_per_teacher_log_probs(sample, reward, response_length)

        response_correct: bool = reward.get("response_correct", None)
        # 失败样本处理
        if reward.get(REWARD_FAILED_KEY, False):
            num_failed += 1
            # logger.warning(
            #     f"[post_process_rewards] sample[{i}] reward failed, "
            #     f"filling teacher_log_probs with sentinel {TEACHER_LOGP_FAILED_SENTINEL} "
            #     f"(response_length={response_length}). Will be masked in pg_loss."
            # )
            # 用哨兵值填充，shape 与 student 一致，不影响后续 cat
            t_log_probs = torch.full((response_length,), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32)
            sample.teacher_log_probs = t_log_probs
            sample.response_correct = response_correct
            teacher_log_probs_list.append(t_log_probs)
            continue

        # 正常样本处理
        try:
            t_log_probs = torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            # teacher_log_probs 变成仅 response 部分，长度 = response_length
            t_log_probs = t_log_probs[-response_length:]
        except Exception as e:
            # 解析失败也用哨兵值填充
            num_failed += 1
            logger.warning(
                f"[post_process_rewards] sample[{i}] logprob parse error: {e}, "
                f"filling with sentinel {TEACHER_LOGP_FAILED_SENTINEL}."
            )
            t_log_probs = torch.full((response_length,), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32)

        sample.response_correct = response_correct
        sample.teacher_log_probs = t_log_probs
        teacher_log_probs_list.append(t_log_probs)

    if num_failed > 0:
        logger.warning(
            f"[post_process_rewards] {num_failed}/{len(samples)} samples failed, "
            f"their pg_loss will be zeroed by sentinel mask in loss.py."
        )

    return teacher_log_probs_list, teacher_log_probs_list
