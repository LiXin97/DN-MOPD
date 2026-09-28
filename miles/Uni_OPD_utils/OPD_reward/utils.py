import ast
from argparse import Namespace
from pathlib import Path
import threading
import logging

mixed_short_to_pure_short_suffix = [151667, 271, 151668, 271]
compat_log_lock = threading.Lock()
compat_logged_keys: set[tuple[str, str]] = set()

logger = logging.getLogger(__name__)


def is_qwen3_mixed_short_student(args: Namespace) -> bool:
    """student: Qwen3 长短链混训 + enable_thinking=False（短链模式）"""
    if args is None:
        return False

    load_path = str(getattr(args, "load", "") or "")
    model_name = Path(load_path).name

    # 模型名含 Qwen3 且不含 Instruct-2507
    if "Qwen3" not in model_name or "Instruct-2507" in model_name:
        return False

    # 必须开启 apply_chat_template
    if not bool(getattr(args, "apply_chat_template", False)):
        return False

    kwargs = getattr(args, "apply_chat_template_kwargs", None)
    if isinstance(kwargs, str):
        try:
            kwargs = ast.literal_eval(kwargs)
        except Exception:
            kwargs = {}
    if not isinstance(kwargs, dict):
        kwargs = {}

    enable_thinking = kwargs.get("enable_thinking", None) is False
    return enable_thinking


def is_qwen3_pure_short_teacher(hf_path: str) -> bool:
    """teacher: Qwen3 Instruct-2507 纯短链"""
    p = str(hf_path)
    return ("Qwen3" in p) and ("Instruct-2507" in p)


def _remove_subsequence_all(tokens: list[int], pattern: list[int]) -> tuple[list[int], int]:
    """删除 tokens 中所有不重叠的 pattern，返回(新tokens, 删除次数)。"""
    if not pattern or len(tokens) < len(pattern):
        return tokens, 0

    out: list[int] = []
    i = 0
    n = len(pattern)
    removed = 0

    while i < len(tokens):
        if i + n <= len(tokens) and tokens[i : i + n] == pattern:
            i += n
            removed += 1
        else:
            out.append(tokens[i])
            i += 1

    return out, removed


def maybe_convert_tokens_for_teacher_compat(
    input_ids: list[int],
    teacher_hf_path: str,
    args: Namespace,
) -> list[int]:
    """
    若 student = Qwen3 混训短链 且 teacher = Qwen3-Instruct-2507 纯短链，
    则移除所有出现的 [151667, 271, 151668, 271] 后再请求 teacher。
    """
    if not is_qwen3_mixed_short_student(args):
        return input_ids
    if not is_qwen3_pure_short_teacher(teacher_hf_path):
        return input_ids

    out, removed_cnt = _remove_subsequence_all(input_ids, mixed_short_to_pure_short_suffix)

    if removed_cnt > 0:
        key = (str(getattr(args, "load", "")), teacher_hf_path)
        with compat_log_lock:
            if key not in compat_logged_keys:
                logger.info(
                    "[TeacherCompat] detected non-homogeneous pair: "
                    "student=Qwen3 mixed(short-mode), teacher=Qwen3 Instruct-2507 (pure short). "
                    f"Removed pattern {mixed_short_to_pure_short_suffix} x{removed_cnt} from input_ids."
                )
                compat_logged_keys.add(key)

    return out