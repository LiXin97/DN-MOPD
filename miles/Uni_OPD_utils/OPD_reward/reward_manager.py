# Modified by the DN-MOPD authors (2026): fixed the package imports (exps.OPD.* -> Uni_OPD_utils.*); the teacher
# server list/map paths can be set with OPD_TEACHER_SERVER_LIST / OPD_TEACHER_SERVER_MAP; added the per-teacher
# interface (teacher_names, resolve_real_name, get_next_url_for_teacher, build_payload_for_teacher) used by the
# all-teacher scorer.
import copy
import json
import logging
import os
import threading
from argparse import Namespace
from pathlib import Path

from PIL import Image

from Uni_OPD_utils.OPD_reward.teacher import TeacherState
from Uni_OPD_utils.OPD_reward.utils import maybe_convert_tokens_for_teacher_compat
from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_REWARD_DIR = Path(__file__).parent
# The paths can be overridden with OPD_TEACHER_SERVER_LIST / OPD_TEACHER_SERVER_MAP (the recipes write both per run).
_SERVER_LIST_PATH = _REWARD_DIR / "teacher_server_list.json"
_SERVER_MAP_PATH = _REWARD_DIR / "teacher_server_map.json"

DEFAULT_TEACHER_KEY = "default"


class RMSystemManager(metaclass=SingletonMeta):
    """RM 单例管理器。

    * 从 teacher_server_list.json 读取 teacher 模型的 URL + hf_path。
    * 从 teacher_server_map.json 读取别名 → 真实 model_name 的映射。
    * 每个 teacher 的 tokenizer/processor lazy 初始化，不在 init() 全部加载。
    * 支持多 teacher 并发使用时的线程安全。
    """

    def __init__(self, args: Namespace):
        self.args: Namespace = args
        self._server_map = {}
        self._teacher_states = {}
        self._states_lock = threading.Lock()
        self._default_payload = {  # 默认 payload，包含 logprob 打分，不生成新 token
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,  # The start location in the prompt for logprobs
        }

        server_list_path = Path(os.environ.get("OPD_TEACHER_SERVER_LIST", str(_SERVER_LIST_PATH)))
        server_map_path = Path(os.environ.get("OPD_TEACHER_SERVER_MAP", str(_SERVER_MAP_PATH)))
        server_list: dict = json.loads(server_list_path.read_text(encoding="utf-8"))
        self._server_map: dict = json.loads(server_map_path.read_text(encoding="utf-8"))

        # 预注册所有 teacher（不加载权重，只记录元信息）
        for model_name, info in server_list.items():
            self._teacher_states[model_name] = TeacherState(
                model_name=model_name,
                hf_path=info["path"],
                raw_urls=info["servers"],
            )

        logger.info(f"RMSystemManager init: {self._teacher_num()} teachers. server_map={self._server_map}")

    def _teacher_num(self) -> int:
        return len(self._teacher_states)

    # 解析 teacher_model_name → 真实 TeacherState
    def _resolve_state(self, teacher_model_name: str | None) -> TeacherState:
        # 1. 若为 None 或 "default"，走 server_map 的 default 键
        key: str = teacher_model_name if teacher_model_name else DEFAULT_TEACHER_KEY

        # 2. 通过 server_map 得到真实 model_name（如果 key 本身不在 map 里，就直接用 key）
        real_name = self._server_map.get(key, key)

        # 3. 取 TeacherState
        state = self._teacher_states.get(real_name)
        if state is None:
            raise KeyError(
                f"teacher_model_name={teacher_model_name!r} → real_name={real_name!r} "
                f"不在 teacher_server_list.json 中。可选: {list(self._teacher_states.keys())}"
            )
        # logger.info(f"Resolved teacher_name={teacher_model_name!r} to real_name={real_name!r} with URLs: {state.urls}")
        return state

    # 对外接口
    def get_next_url(self, sample: Sample) -> str:
        state = self._resolve_state(sample.teacher_model_name)
        return state.get_next_url()

    # ---------- per-teacher interface of the all-teacher scorer ----------
    def teacher_names(self) -> list[str]:
        """Registered teacher names (the keys of the server list)."""
        return list(self._teacher_states.keys())

    def resolve_real_name(self, teacher_model_name: str | None) -> str:
        """Alias (the dataset's teacher label) -> registered teacher name."""
        return self._resolve_state(teacher_model_name).model_name

    def get_next_url_for_teacher(self, teacher_name: str) -> str:
        """Next URL of a registered teacher, by its registered name (no alias resolution)."""
        return self._teacher_states[teacher_name].get_next_url()

    def build_payload_for_teacher(self, sample: Sample, teacher_name: str) -> dict:
        """Scoring payload for one registered teacher (built per teacher: token compatibility depends on each
        teacher's hf_path)."""
        return self._build_payload_for_state(sample, self._teacher_states[teacher_name])

    def build_payload(self, sample: Sample) -> dict:
        """根据 sample.teacher_model_name 选择对应的 teacher，构建请求 payload。

        * 多模态：processor 重新编码，提取 input_ids + image_data。
        * 纯文本：直接使用 sample.tokens。
        """
        state = self._resolve_state(sample.teacher_model_name)
        return self._build_payload_for_state(sample, state)

    def _build_payload_for_state(self, sample: Sample, state: TeacherState) -> dict:
        # 触发 lazy 初始化（如尚未加载 tokenizer/processor）
        processor = state.processor  # property 内部会 ensure_initialized

        payload: dict = copy.deepcopy(self._default_payload)  # 基础 payload（logprob 打分，不生成新 token）

        # 多模态
        if processor is not None and sample.multimodal_inputs is not None:
            # 用 processor 重新编码，提取 input_ids 和图片
            # processor_output = self._processor(text=sample.prompt, **sample.multimodal_inputs)
            # input_ids = processor_output["input_ids"][0]
            # payload["input_ids"] = input_ids

            payload["input_ids"] = sample.tokens  # 走到这一步已经是 prompt + response 的格式了

            # 将图片编码为 base64/url，传给 teacher server
            images: list[Image.Image] | None = sample.multimodal_inputs.get("images", None)
            if images:
                payload["image_data"] = [encode_image_for_rollout_engine(img) for img in images]

        # 纯文本：直接复用已有 token ids
        else:
            payload["input_ids"] = sample.tokens

        # 处理师生模型 diff
        payload["input_ids"] = maybe_convert_tokens_for_teacher_compat(payload["input_ids"], state.hf_path, self.args)
        return payload
