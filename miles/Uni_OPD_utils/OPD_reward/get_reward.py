# Modified by the DN-MOPD authors (2026): fixed the package imports (exps.OPD.* -> Uni_OPD_utils.*) and added
# the all-teacher scorer (_score_one_teacher / _get_reward_fanout) used by Uni_OPD_utils.mopd_hook.
import asyncio
import functools
import logging
import os
import random
import time
from argparse import Namespace
from collections.abc import Callable

import aiohttp

from Uni_OPD_utils.OPD_reward.reward_manager import RMSystemManager
from Uni_OPD_utils.OPD_reward.rule_base_reward import get_rule_based_reward  # noqa
from Uni_OPD_utils.OPD_reward.session_manager import RewardSessionManager
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY = 2.0  # 每次重试间隔（秒）


# 失败样本的哨兵标记键
REWARD_FAILED_KEY = "__opd_reward_failed__"

# 设置代理
os.environ["http_proxy"], os.environ["https_proxy"] = "", ""


# 设置在 --custom-rm-path 当中
# 用于 miles/rollout/rm_hub/__init__.py 的 async_rm 方法
async def get_reward(args: Namespace, sample: Sample, **kwargs) -> dict[str]:
    start_time = time.time()

    rm_manager = RMSystemManager(args)
    session_manager = RewardSessionManager()
    # 结果正确性验证（async：code 分支会通过 asyncio.to_thread 发起远程评测，不阻塞事件循环）
    response_correct, rule_based_metadata = await get_rule_based_reward(args, sample)
    # response_correct, rule_based_metadata = None, {}

    payload = rm_manager.build_payload(sample)

    last_exception = None
    for attempt in range(1, _MAX_RETRIES + 1):
        url = rm_manager.get_next_url(sample)

        try:
            async with session_manager.inflight_sem:
                session = await session_manager.get_session()
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    res = await resp.json()
                    # input_token_logprobs 是 teacher 对整个输入序列（prompt + response）的每个 token 的 logps
                    # res 结构示例：
                    # {
                    #     ...
                    #     "meta_info": {
                    #         "input_token_logprobs": [
                    #             [None,  token_id_0, None],     # 第一个 token 无 logprob
                    #             [logprob, token_id_1, None],   # 第二个 token 的 logprob
                    #             [logprob,  token_id_2, None],  # 第三个 token 的 logprob
                    #             ...
                    #         ],
                    #     },
                    # }

            res["reward_time"] = time.time() - start_time
            res["teacher_url"] = url
            res["response_correct"] = response_correct
            res["rule_based_metadata"] = rule_based_metadata
            return res

        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt}/{_MAX_RETRIES} failed for teacher={sample.teacher_model_name}, "
                f"url={url}: {type(e).__name__}: {e}"
            )
            is_bad_fd = isinstance(e, aiohttp.ClientOSError) and e.errno == 9
            is_session_closed = isinstance(e, RuntimeError) and "Session is closed" in str(e)
            is_fd_reuse = isinstance(e, RuntimeError) and "File descriptor" in str(e) and "used by transport" in str(e)
            if is_bad_fd or is_session_closed or is_fd_reuse:
                await session_manager.reset_session()
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Unexpected error on attempt {attempt}/{_MAX_RETRIES} for url={url}: {type(e).__name__}: {e}"
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))

    logger.error(
        f"All {_MAX_RETRIES} attempts failed. "
        f"last error: {type(last_exception).__name__}: {last_exception}. "
        f"Marking sample as failed (will be masked in loss)."
    )

    return {
        REWARD_FAILED_KEY: True,
        "response_correct": response_correct,
        "rule_based_metadata": rule_based_metadata,
        "reward_time": time.time() - start_time,
        "meta_info": {"input_token_logprobs": []},
    }


# ---------------------------------------------------------------------------
# All-teacher scoring (used by Uni_OPD_utils.mopd_hook: every registered teacher
# scores the sampled tokens of every response; the legacy fields still come from
# the teacher selected by the sample's domain label).
# ---------------------------------------------------------------------------


async def _score_one_teacher(
    session_manager: RewardSessionManager,
    teacher_name: str,
    get_url: Callable[[], str],
    payload: dict,
) -> dict:
    """Prefill-score one sample with one teacher (same retry semantics as the legacy path).

    The in-flight semaphore (OPD_RM_MAX_INFLIGHT) is held for each POST, so K concurrent
    coroutines each take one slot and the global limit still applies.

    Returns:
        {"teacher_name": str, "res": dict | None, "url": str | None,
         "failed": bool, "reward_time": float}
    """
    t0 = time.time()
    last_exception = None
    url: str | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            # get_url inside the try: a broken URL pool fails this teacher only, never the whole sample
            url = get_url()
            async with session_manager.inflight_sem:
                session = await session_manager.get_session()
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    res = await resp.json()

            # a response without meta_info.input_token_logprobs is unusable
            if res.get("meta_info", {}).get("input_token_logprobs") is None:
                raise RuntimeError("missing meta_info.input_token_logprobs in teacher response")

            return {
                "teacher_name": teacher_name,
                "res": res,
                "url": url,
                "failed": False,
                "reward_time": time.time() - t0,
            }

        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            last_exception = e
            logger.warning(
                f"[fanout] Attempt {attempt}/{_MAX_RETRIES} failed for teacher={teacher_name}, "
                f"url={url}: {type(e).__name__}: {e}"
            )
            is_bad_fd = isinstance(e, aiohttp.ClientOSError) and e.errno == 9
            is_session_closed = isinstance(e, RuntimeError) and "Session is closed" in str(e)
            is_fd_reuse = isinstance(e, RuntimeError) and "File descriptor" in str(e) and "used by transport" in str(e)
            if is_bad_fd or is_session_closed or is_fd_reuse:
                await session_manager.reset_session()
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))
        except Exception as e:
            last_exception = e
            logger.warning(
                f"[fanout] Unexpected error on attempt {attempt}/{_MAX_RETRIES} for "
                f"teacher={teacher_name}, url={url}: {type(e).__name__}: {e}"
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))

    logger.error(
        f"[fanout] All {_MAX_RETRIES} attempts failed for teacher={teacher_name}. "
        f"last error: {type(last_exception).__name__}: {last_exception}. "
        f"Marking this teacher as failed for the sample."
    )
    return {
        "teacher_name": teacher_name,
        "res": None,
        "url": url,
        "failed": True,
        "reward_time": time.time() - t0,
    }


async def _get_reward_fanout(
    args: Namespace,
    sample: Sample,
    rm_manager: RMSystemManager,
    session_manager: RewardSessionManager,
    start_time: float,
    response_correct,
    rule_based_metadata,
) -> dict:
    """Score one sample with EVERY registered teacher, concurrently.

    Backward compatible with the legacy reward dict: meta_info / teacher_url / REWARD_FAILED_KEY come from the
    teacher that the sample's label routes to (sample.teacher_model_name through the server map); only a failure of
    that teacher marks the whole sample REWARD_FAILED. Adds
        reward["per_teacher"]    = {teacher_name: {input_token_logprobs | None, reward_time, url, failed}}
        reward["routed_teacher"] = the routed teacher's registered name.
    """
    # An alias that the server map cannot resolve fails the routed teacher (the sample is REWARD_FAILED) while the
    # other registered teachers are still scored; the exception never escapes into the rollout step.
    try:
        routed_name: str | None = rm_manager.resolve_real_name(sample.teacher_model_name)
    except KeyError as e:
        logger.error(
            f"[fanout] cannot resolve routed teacher for teacher_model_name={sample.teacher_model_name!r}: {e}. "
            f"Marking the sample as REWARD_FAILED (the other registered teachers are still scored)."
        )
        routed_name = None
    teacher_names: list[str] = rm_manager.teacher_names()

    def _failed_result(name: str) -> dict:
        return {"teacher_name": name, "res": None, "url": None, "failed": True, "reward_time": 0.0}

    results: list[dict] = []
    task_names: list[str] = []
    tasks = []
    for name in teacher_names:
        # the payload is built per teacher (token compatibility depends on each teacher's hf_path)
        try:
            payload = rm_manager.build_payload_for_teacher(sample, name)
        except Exception as e:
            logger.error(
                f"[fanout] build_payload failed for teacher={name}: {type(e).__name__}: {e}. "
                f"Marking this teacher as failed for the sample."
            )
            results.append(_failed_result(name))
            continue
        get_url = functools.partial(rm_manager.get_next_url_for_teacher, name)
        task_names.append(name)
        tasks.append(_score_one_teacher(session_manager, name, get_url, payload))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    for name, raw in zip(task_names, raw_results, strict=True):
        if isinstance(raw, BaseException):
            logger.error(
                f"[fanout] scoring task raised unexpectedly for teacher={name}: "
                f"{type(raw).__name__}: {raw}. Marking this teacher as failed for the sample."
            )
            raw = _failed_result(name)
        results.append(raw)

    per_teacher: dict[str, dict] = {}
    routed_result: dict | None = None
    for r in results:
        meta = r["res"].get("meta_info", {}) if not r["failed"] else {}
        itl = meta.get("input_token_logprobs") if not r["failed"] else None
        per_teacher[r["teacher_name"]] = {
            "input_token_logprobs": itl,
            "reward_time": r["reward_time"],
            "url": r["url"],
            "failed": bool(r["failed"] or itl is None),
        }
        if r["teacher_name"] == routed_name:
            routed_result = r

    # the legacy fields come from the routed teacher; only its failure marks the whole sample REWARD_FAILED
    if routed_result is None or routed_result["failed"]:
        reward: dict = {
            REWARD_FAILED_KEY: True,
            "meta_info": {"input_token_logprobs": []},
        }
    else:
        reward = routed_result["res"]
    if routed_result is not None and not routed_result["failed"] and routed_result["url"] is not None:
        reward["teacher_url"] = routed_result["url"]

    reward["reward_time"] = time.time() - start_time
    reward["response_correct"] = response_correct
    reward["rule_based_metadata"] = rule_based_metadata
    reward["per_teacher"] = per_teacher
    reward["routed_teacher"] = routed_name
    return reward
