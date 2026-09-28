# Modified by the DN-MOPD authors (2026): fixed the package imports; the verifier is selected by
# sample.metadata["domain"] (falling back to the teacher label); IF responses are graded by the vendored
# constraint checkers; math grading runs in killable worker processes with a time limit (bounded_math_grade.py);
# a code-judge error or a verifier exception yields None ("no verdict") instead of a wrong-answer verdict; the
# code-judge client is created lazily.
import asyncio
import importlib.util
import json
import logging
import os
from argparse import Namespace

# from exps.RL.utils.reward.PRIME_code import compute_score as compute_score_code
from Uni_OPD_utils.outcome_reward.PRIME_code_server.server import CodeJudgeClient, CodeJudgeResponse

from miles.utils.types import Sample
from Uni_OPD_utils.OPD_reward.bounded_math_grade import grade_math_bounded

logger = logging.getLogger(__name__)

# 这里获取全局唯一实例，避免重复创建连接和服务器列表
# created lazily: math-only runs must not require the code-judge service at import time
_code_judge_client: CodeJudgeClient | None = None


def _get_code_judge_client() -> CodeJudgeClient:
    global _code_judge_client
    if _code_judge_client is None:
        _code_judge_client = CodeJudgeClient()
    return _code_judge_client


def _compute_score_code(completion: str, test_cases) -> tuple[float | bool | None, list | dict | None]:
    """通过远程 CodeJudgeClient 服务评测代码正确性，接口兼容原 compute_score。
    注意：此函数内部使用 requests.post（同步阻塞），必须通过 asyncio.to_thread 调用。
    """
    resp: CodeJudgeResponse = _get_code_judge_client().judge(completion, test_cases)
    if resp.error:
        logger.warning(f"[CodeJudge] judge returned error: {resp.error}")
        # an error is not a verdict: None means "unknown" (GRPO scores it 0.0; the injection trigger refuses it)
        return None, resp.metadata
    return resp.success, resp.metadata


# CodeJudgeClient.judge 内部使用 requests.post（同步阻塞）+ time.sleep，
# 必须放到线程池中执行，否则会阻塞整个 asyncio 事件循环，
# 导致 abort wait 等异步操作 hang 住。


_ifeval_score = None


def _get_ifeval_score():
    """The vendored instruction-following constraint checkers (ifeval_checkers.py beside this file)."""
    global _ifeval_score
    if _ifeval_score is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifeval_checkers.py")
        spec = importlib.util.spec_from_file_location("uni_opd_ifeval_checkers", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the vendored ifeval checkers from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ifeval_score = mod.score
    return _ifeval_score


def _verifier_domain(sample: Sample) -> str:
    """Domain that selects the correctness verifier: sample.metadata["domain"] (the dataset's true domain), falling
    back to the teacher-routing label sample.teacher_model_name when the metadata carries no domain."""
    meta = getattr(sample, "metadata", None) or {}
    dom = meta.get("domain")
    if isinstance(dom, str) and dom:
        return dom.lower()
    return (sample.teacher_model_name or "").lower()


async def get_rule_based_reward(args: Namespace, sample: Sample) -> tuple[bool | float | None, dict[str]]:
    verifier_domain = _verifier_domain(sample)
    try:
        if verifier_domain in ("vlm-puzzle", "vlm-math", "vlm-chart"):
            from exps.RL.utils.reward.OpenMMReasoner.get_reward import get_reward as get_reward_openmm

            grade_result: dict = await get_reward_openmm(args, sample)
            response_correct = grade_result["acc_score"] == 1.0
            grade_result = (response_correct, grade_result)
        elif verifier_domain in ("code", "code-a3b"):
            grade_result = await asyncio.to_thread(_compute_score_code, sample.response, sample.label)
        elif verifier_domain == "ifeval":
            # strict: every constraint of the prompt must hold
            constraints = sample.label
            if isinstance(constraints, str):
                constraints = json.loads(constraints)
            ok, detail = _get_ifeval_score()(sample.response or "", constraints)
            grade_result = (bool(ok), detail if isinstance(detail, dict) else {})
        else:  # Default math: grade_answer_verl in a killable worker with a time limit (bounded_math_grade.py)
            grade_result = await grade_math_bounded(sample.response, sample.label)

        if isinstance(grade_result, tuple):
            response_correct, metadata = grade_result
        elif isinstance(grade_result, bool):
            response_correct, metadata = grade_result, {}
        elif isinstance(grade_result, dict):
            response_correct, metadata = grade_result["correct"], grade_result
        else:
            response_correct, metadata = bool(grade_result), {}

    except Exception as e:
        logger.warning(f"[get_reward] grade_answer_verl raised an exception: {type(e).__name__}: {e}")
        # a verifier exception is not a verdict
        response_correct, metadata = None, {}

    return response_correct, metadata
