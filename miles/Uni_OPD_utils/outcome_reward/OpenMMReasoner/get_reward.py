import logging
import time
from argparse import Namespace

from miles.utils.types import Sample
from Uni_OPD_utils.outcome_reward.OpenMMReasoner.lmms_lab_recipe import compute_score

logger = logging.getLogger(__name__)


async def get_reward(args: Namespace, sample: Sample, **kwargs) -> dict[str]:
    start_time = time.time()

    use_format_reward = args.use_format_reward
    response = sample.response
    label = sample.label
    # logger.info(f"response: {response}, label: {label}, use_format_reward: {use_format_reward}")
    reward = compute_score(response, label, use_format_reward=use_format_reward)

    sample.reward_time = time.time() - start_time
    reward["reward_time"] = sample.reward_time
    return reward
