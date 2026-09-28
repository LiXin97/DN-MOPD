# Modified by the DN-MOPD authors (2026): the endpoint list path is CODE_JUDGE_ENDPOINTS or the file next to this
# module, instead of a path relative to the working directory.
import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests

# 服务器列表配置文件路径（后续可替换）
# The endpoint list is read from CODE_JUDGE_ENDPOINTS when set, else from available_endpoints.json next to this
# module (the upstream value was a path relative to the importing process's working directory).
CODE_JUDGE_SERVER_LIST = os.environ.get(
    "CODE_JUDGE_ENDPOINTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "available_endpoints.json"),
)
JUDGE_ROUTER = "/judge"
HEALTH_ROUTER = "/health"


def get_judge_servers(data_path=None) -> list:
    """获取代码评测服务器列表"""
    if data_path is None:
        with open(CODE_JUDGE_SERVER_LIST) as f:
            return json.load(f)
    return json.loads(data_path)


@dataclass
class CodeJudgeResponse:
    """代码评测响应"""

    success: float | bool
    metadata: list | dict | None = None
    error: str | None = None

    def __bool__(self):
        if isinstance(self.success, bool):
            return self.success
        return self.success >= 1.0


class BaseClient(ABC):
    """
    分布式服务客户端基类，实现轮询负载均衡

    每轮询完整 2 遍后，会重新 shuffle 服务器列表并从头开始轮询
    """

    _instances: dict[str, Any] = {}
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        class_name = cls.__name__
        if class_name not in cls._instances:
            with cls._lock:
                if class_name not in cls._instances:
                    cls._instances[class_name] = super().__new__(cls)
        return cls._instances[class_name]

    def __init__(self):
        # 避免重复初始化
        if hasattr(self, "_initialized"):
            return

        # 初始化轮询状态
        self._state = {"index": 0, "shuffled_servers": []}
        self._init_or_reshuffle_servers()
        self._initialized = True

    @abstractmethod
    def _get_server_list(self) -> list[str]:
        """获取服务器列表（子类实现）"""
        pass

    @abstractmethod
    def _build_server_url(self, host: str) -> str:
        """构建服务器 URL（子类实现）"""
        pass

    def _init_or_reshuffle_servers(self):
        """初始化或重新 shuffle 服务器列表"""
        hosts = self._get_server_list()
        self.servers = [self._build_server_url(host) for host in hosts]

        if not self.servers:
            self._state["shuffled_servers"] = []
            return

        shuffled = list(self.servers)
        random.shuffle(shuffled)
        self._state["index"] = 0
        self._state["shuffled_servers"] = shuffled

    def get_next_server(self) -> str | None:
        """获取下一个服务器，实现轮询负载均衡"""
        with self._lock:
            shuffled_servers = self._state["shuffled_servers"]

            if not shuffled_servers:
                return None

            current_index = self._state["index"]

            # 检查是否需要 reshuffle（完成2轮后）
            if current_index >= len(shuffled_servers) * 2:
                self._init_or_reshuffle_servers()
                shuffled_servers = self._state["shuffled_servers"]
                current_index = 0

            # 获取服务器
            server = shuffled_servers[current_index % len(shuffled_servers)]
            self._state["index"] = current_index + 1

            return server

    def _sleep_on_retry(self, attempt: int, max_retry: int):
        """根据重试次数决定 sleep 时间"""
        if attempt >= max_retry - 5:
            time.sleep(1.5)
        elif attempt >= max_retry // 2:
            time.sleep(1)
        else:
            time.sleep(0.5)


class CodeJudgeClient(BaseClient):
    """
    代码评测服务客户端

    输入（judge）：
        - completion: str, 生成的代码或包含代码的文本
        - test_cases: dict | str, 测试用例字典，包含 inputs 和 outputs 列表
        - continuous: bool, 是否进行连续评测（测试前10个样例）

    输出：
        CodeJudgeResponse 对象，包含：
        - success: float | bool, 评测成功率（0.0-1.0）或布尔值
        - metadata: list | dict | None, 详细的评测元数据
        - error: str | None, 错误信息（如果有）
    """

    def __init__(self):
        self.judge_router = JUDGE_ROUTER
        self.health_router = HEALTH_ROUTER
        super().__init__()

    def _get_server_list(self) -> list[str]:
        return get_judge_servers()

    def _build_server_url(self, host: str) -> str:
        return f"http://{host}"

    def judge(
        self,
        completion: str,
        test_cases: dict | str,
        continuous: bool = False,
        max_retry: int = 3,
        timeout: int = 20,
    ) -> CodeJudgeResponse:
        """
        调用远程服务评测代码正确性

        Args:
            completion: 生成的代码或包含代码的文本
            test_cases: 测试用例字典，包含 inputs 和 outputs 列表
            continuous: 是否进行连续评测（测试前10个样例）
            max_retry: 最大重试次数
            timeout: 单次请求超时时间（秒）

        Returns:
            CodeJudgeResponse: 评测结果
        """
        # 确保 test_cases 可序列化
        if isinstance(test_cases, str):
            payload_test_cases = test_cases
        else:
            payload_test_cases = test_cases  # dict 可直接 json 序列化

        payload = {
            "completion": completion,
            "test_cases": payload_test_cases,
            "continuous": continuous,
        }

        for attempt in range(max_retry):
            server = self.get_next_server()
            if server is None:
                continue

            url = server + self.judge_router
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return CodeJudgeResponse(
                        success=data.get("success", 0.0),
                        metadata=data.get("metadata"),
                        error=data.get("error"),
                    )
                else:
                    # 非 200 状态码，记录并重试
                    if attempt == max_retry - 1:
                        return CodeJudgeResponse(
                            success=0.0,
                            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        )
            except Exception as e:
                if attempt == max_retry - 1:
                    error_msg = f"Code judge service failed after {max_retry} attempts: {e}"
                    print(error_msg)
                    return CodeJudgeResponse(success=0.0, error=error_msg)

            self._sleep_on_retry(attempt, max_retry)

        return CodeJudgeResponse(success=0.0, error="Code judge service unreachable")

    def health_check(self, timeout: int = 5) -> bool:
        """检查任意一台服务器是否健康"""
        server = self.get_next_server()
        if server is None:
            return False
        try:
            resp = requests.get(server + self.health_router, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

