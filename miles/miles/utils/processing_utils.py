import base64
import copy
import io
import logging
import os
from io import BytesIO

import requests
from PIL import Image
from qwen_vl_utils.vision_process import (
    IMAGE_MAX_TOKEN_NUM,
    IMAGE_MIN_TOKEN_NUM,
    SPATIAL_MERGE_SIZE,
    smart_resize,
    to_rgb,
)
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 16


def load_tokenizer(name_or_path: str, chat_template_path: str = None, **kwargs) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if chat_template_path:
        assert os.path.isfile(chat_template_path), (
            f"chat_template_path not found: {chat_template_path}. "
            f"Ensure the path is accessible on this node (e.g. inside the miles repo or on a shared filesystem)."
        )
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Loaded custom chat template from %s", chat_template_path)
    return tokenizer


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def _fetch_image(ele: dict[str, str | Image.Image], image_patch_size: int = 14) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("data:") and "base64," in image:
        # 处理非 image MIME 类型的 base64 数据
        # 例如: data:application/octet-stream;base64,xxx
        _, base64_data = image.split("base64,", 1)
        data = base64.b64decode(base64_data)
        with BytesIO(data) as bio:
            image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor**2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor**2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def process_vision_info(prompt, processor: ProcessorMixin) -> dict[str, list[Image.Image] | None]:
    # temporary solution, will write image utils for miles later
    from qwen_vl_utils import process_vision_info, vision_process

    vision_process.fetch_image = _fetch_image

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image: Image.Image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
