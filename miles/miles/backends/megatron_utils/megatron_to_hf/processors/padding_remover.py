import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    支持纯文本模型（embedding.xxx）和 VLM 模型（language_model.embedding.xxx）的参数名。
    """
    stripped = strip_param_name_prefix(name)
    # 兼容 VLM 模型的 language_model. 前缀
    if stripped.startswith("language_model."):
        stripped = stripped[len("language_model.") :]
    if stripped in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
