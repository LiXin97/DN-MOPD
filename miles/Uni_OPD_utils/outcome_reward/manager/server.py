import logging
import os
import random
import threading

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion

from Uni_OPD_utils.outcome_reward.manager.utils import encode_image, get_model_servers

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_SERVERS: dict = get_model_servers()


class ServerManager:
    """全局服务器管理器，实现轮询负载均衡

    每轮询完整 2 遍后，会重新 shuffle 服务器列表并从头开始轮询
    """

    _instance = None
    _lock = threading.Lock()
    _server_state = {}  # 存储每个 model_name 的状态: {model_name: {'index': int, 'shuffled_servers': list}}

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _init_or_reshuffle_servers(self, model_name):
        """初始化或重新 shuffle 服务器列表"""
        original_servers = self.get_server_list(model_name)
        if not original_servers:
            return []
        shuffled = list(original_servers)  # 创建副本
        random.shuffle(shuffled)
        self._server_state[model_name] = {"index": 0, "shuffled_servers": shuffled}
        return shuffled

    def get_server_list(self, model_name="Qwen2.5-VL-72B-Instruct"):
        """获取服务器列表"""
        return MODEL_SERVERS.get(model_name, [])

    def get_next_server(self, model_name="Qwen2.5-VL-72B-Instruct"):
        """获取下一个服务器，实现轮询负载均衡

        每轮询完整 2 遍后，会重新 shuffle 服务器列表并从头开始
        """
        original_servers = self.get_server_list(model_name)
        if not original_servers:
            return None

        with self._lock:
            # 初始化或获取当前状态
            if model_name not in self._server_state:
                self._init_or_reshuffle_servers(model_name)

            state = self._server_state[model_name]
            shuffled_servers = state["shuffled_servers"]
            current_index = state["index"]

            # 检查是否需要 reshuffle（完成2轮后）
            if current_index >= len(shuffled_servers) * 2:
                self._init_or_reshuffle_servers(model_name)
                state = self._server_state[model_name]
                shuffled_servers = state["shuffled_servers"]
                current_index = 0

            # 获取服务器
            server = shuffled_servers[current_index % len(shuffled_servers)]
            self._server_state[model_name]["index"] = current_index + 1

            return server

    def _request(self, server: str, prompt: str, img_path: str = None, **kwargs):
        # 从kwargs中获取timeout，如果没有则使用默认值
        timeout = kwargs.get("timeout", 400)

        client = OpenAI(
            base_url=f"http://{server}/v1",
            api_key="sk-xxx",
            http_client=httpx.Client(timeout=timeout),
        )

        extra_params = dict(
            temperature=0.2,
            # top_k=20,
            top_p=0.95,
            # repetition_penalty=1.0,
            # presence_penalty=1.5
            max_tokens=4096,
        )

        if kwargs.get("top_k"):
            top_k = kwargs["top_k"]
        else:
            top_k = 10  # noqa

        # top_p
        if kwargs.get("top_p"):
            extra_params.update({"top_p": kwargs["top_p"]})

        if kwargs.get("n_params"):
            extra_params.update({"n": kwargs["n_params"]})
            # if kwargs['n_params'] > 1:
            #     extra_params["temperature"] = None

        if kwargs.get("max_tokens"):
            extra_params.update({"max_tokens": kwargs["max_tokens"]})

        # system_prompt
        system_prompt = None
        if kwargs.get("system_prompt"):
            system_prompt = kwargs["system_prompt"]
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if img_path is not None and os.path.exists(img_path):
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encode_image(img_path)}}
            )
        completion: ChatCompletion = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=messages,
            **extra_params,
        )
        # print(f"completion.choices: {completion.choices}")

        if kwargs.get("n_params"):
            return [completion.choices[x].message.content for x in range(len(completion.choices))]
        else:
            return completion.choices[0].message.content

    def call(self, img_path: str = None, question: str = None, system_prompt=None, max_try=3) -> str:
        """调用自己 vllm 部署的模型，带重试和验证功能"""
        if not img_path and not question:
            raise ValueError("至少需要提供 img_path 或 question 之一")

        try_count = 0
        while try_count < max_try:
            try:
                server = self.get_next_server("Qwen2.5-VL-72B-Instruct")
                if server is None:
                    print("No available servers for Qwen2.5-VL-72B-Instruct")
                    return None

                resp = self._request(
                    server,
                    question,
                    img_path,
                    system_prompt=system_prompt,
                )
                return resp
            except Exception as e:
                print(f"Error calling server: {e}")
                try_count += 1
                print(f"Retrying... ({try_count}/{max_try})")
        return ""
