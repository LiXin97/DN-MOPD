import asyncio
import logging
import time
from argparse import Namespace

from miles.utils.types import Sample
from Uni_OPD_utils.outcome_reward.PRIME_code_server.server import CodeJudgeClient, CodeJudgeResponse

logger = logging.getLogger(__name__)

_code_judge_client: CodeJudgeClient = CodeJudgeClient()


def _compute_score_code(completion: str, test_cases) -> CodeJudgeResponse:
    return _code_judge_client.judge(completion, test_cases)


async def get_reward(args: Namespace, sample: Sample, **kwargs) -> dict[str]:
    start_time = time.time()

    response = sample.response
    test_cases = sample.label

    resp: CodeJudgeResponse = await asyncio.to_thread(_compute_score_code, response, test_cases)

    if resp.error:
        logger.warning(f"[CodeJudge] judge returned error: {resp.error}")

    reward: dict[str] = {}

    if isinstance(resp.success, bool):
        reward["score"] = 1.0 if resp.success else 0.0
    else:
        reward["score"] = float(resp.success)

    reward["metadata"] = resp.metadata
    reward["error"] = resp.error

    sample.reward_time = time.time() - start_time
    reward["reward_time"] = sample.reward_time

    return reward
