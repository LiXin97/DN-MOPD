import logging
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from miles.utils.processing_utils import load_processor, load_tokenizer

logger = logging.getLogger(__name__)


DEFAULT_TEACHER_KEY = "default"


def _normalize_url(raw: str) -> str:
    """统一将各种格式的 URL 规范化为 http://host:port/generate"""
    raw = raw.strip()
    # 补全 scheme
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw
    # 补全 /generate 路径
    if not raw.rstrip("/").endswith("/generate"):
        raw = raw.rstrip("/") + "/generate"
    return raw


class TeacherState:
    """
    每个 teacher model 的运行时状态

    某一个 teacher model 对应的 URL 池 + tokenizer/processor（lazy 初始化）。
    """

    def __init__(self, model_name: str, hf_path: str, raw_urls: list[str]):
        self.model_name = model_name
        self.hf_path = hf_path
        self.urls: list[str] = [_normalize_url(u) for u in raw_urls]

        # lazy 初始化字段
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._processor: ProcessorMixin | None = None
        self._initialized = False
        self._init_lock = threading.Lock()

        # URL 轮询状态
        self._poll_lock = threading.Lock()
        self._shuffled: list[str] = []
        self._poll_index = 0
        self._reshuffle()

        logger.info(f"[TeacherState] init: {self.model_name}  hf={self.hf_path}  urls={self.urls}")

    def _reshuffle(self) -> None:
        """
        URL 轮询

        调用方需持有 _poll_lock。
        """
        shuffled = list(self.urls)
        random.shuffle(shuffled)
        self._shuffled = shuffled
        self._poll_index = 0

    def get_next_url(self) -> str:
        with self._poll_lock:
            if self._poll_index >= len(self._shuffled) * 2:
                self._reshuffle()
            url = self._shuffled[self._poll_index % len(self._shuffled)]
            self._poll_index += 1
            return url

    # ---------- 服务器模型信息检查 ----------
    def _check_server_model_info(self) -> None:
        """并发请求所有 URL 的 /get_model_info，统计 model_path 并与 self.hf_path 比对。"""

        def _get_model_info(url: str) -> tuple[str, dict | None, str]:
            info_url = url.rsplit("/", 1)[0] + "/get_model_info"
            try:
                resp = requests.get(info_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return url, data, ""
                else:
                    return url, None, f"status={resp.status_code}"
            except Exception as e:
                return url, None, str(e)

        model_type_counter: Counter = Counter()
        model_path_counter: Counter = Counter()
        failed_urls: list[str] = []

        with ThreadPoolExecutor(max_workers=min(len(self.urls), 32)) as executor:
            futures = {executor.submit(_get_model_info, u): u for u in self.urls}
            for future in as_completed(futures):
                url, info, err = future.result()
                if info:
                    model_type_counter[info.get("model_type", "unknown")] += 1
                    model_path_counter[info.get("model_path", "unknown")] += 1
                else:
                    failed_urls.append(f"{url}({err})")

        if failed_urls:
            logger.warning(
                f"[TeacherState] {self.model_name}: "
                f"{len(failed_urls)}/{len(self.urls)} 个服务无法获取 model_info: {failed_urls}"
            )

        # 检查 model_path 是否与 self.hf_path 一致
        for path, cnt in model_path_counter.items():
            if path != self.hf_path:
                logger.warning(
                    f"[TeacherState] {self.model_name}: 服务端 model_path={path!r} (共{cnt}个) "
                    f"与配置的 hf_path={self.hf_path!r} 不一致!"
                )
            else:
                logger.info(f"[TeacherState] {self.model_name}: 服务端 model_path={path!r} (共{cnt}个) 与配置一致 ✅")

        # 打印 model_type 统计
        if model_type_counter:
            logger.info(
                f"[TeacherState] {self.model_name}: 服务端 model_type 统计: "
                + ", ".join(f"{t}={c}" for t, c in model_type_counter.most_common())
            )

    # ---------- Tokenizer / Processor lazy 初始化 ----------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            logger.info(f"[TeacherState] lazy init: {self.model_name}  hf={self.hf_path}")
            self._check_server_model_info()
            self._tokenizer = load_tokenizer(self.hf_path, trust_remote_code=True)
            self._processor = load_processor(self.hf_path, trust_remote_code=True)
            self._initialized = True
            logger.info(f"[TeacherState] init done: {self.model_name}  processor={'yes' if self._processor else 'no'}")

    @property
    def processor(self) -> ProcessorMixin | None:
        self._ensure_initialized()
        return self._processor

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        self._ensure_initialized()
        return self._tokenizer
