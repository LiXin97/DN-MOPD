import base64
import json
from pathlib import Path

import requests

_CURRENT_DIR = Path(__file__).parent
_MODEL_SERVER_LIST = _CURRENT_DIR / "model_server_list.json"


def encode_image(image_path):
    if image_path.startswith("http"):
        content = requests.get(image_path).content
        return base64.b64encode(content).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def get_model_servers(data_path=None) -> dict:
    """获取模型服务器列表"""
    if data_path is None:
        with open(_MODEL_SERVER_LIST) as f:
            return json.load(f)
    return json.loads(data_path)
