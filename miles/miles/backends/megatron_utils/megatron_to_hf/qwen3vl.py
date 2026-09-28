from .qwen2 import convert_qwen2_to_hf

# Qwen3VL Megatron parameter name prefix
_LM_PREFIX = "module.module.language_model."
# Vision encoder prefixes (compatible with both 'visual' and 'vision_model' Megatron naming)
_VISUAL_PREFIXES = ("module.module.visual.", "module.module.vision_model.")
# Proxy prefix reused when converting via qwen2
_PROXY_PREFIX = "module.module."


def convert_qwen3vl_to_hf(args, name: str, param):
    """
    Convert Megatron parameter names of Qwen3VL (and similar VLM) models to HuggingFace format.

    Megatron VLM parameter name structure:
      - Text part:   module.module.language_model.{embedding/decoder/output_layer}.xxx
      - Vision part: module.module.visual.xxx  or  module.module.vision_model.xxx

    HuggingFace parameter name structure:
      - Text part:   model.language_model.xxx (insert "language_model." after "model." in the qwen2 conversion result)
      - Vision part: model.visual.xxx
    """
    # Handle the text part: strip the "language_model." prefix and delegate to the qwen2
    # converter, then replace the "model." prefix in its result with "model.language_model.".
    # e.g. module.module.language_model.embedding.word_embeddings.weight
    #   -> qwen2 conversion -> model.embed_tokens.weight
    #   -> final            -> model.language_model.embed_tokens.weight
    if name.startswith(_LM_PREFIX):
        rest = name[len(_LM_PREFIX) :]
        proxy_name = _PROXY_PREFIX + rest
        qwen2_results = convert_qwen2_to_hf(args, proxy_name, param)
        # Replace the "model." prefix returned by qwen2 with "model.language_model."
        patched = []
        for hf_name, tensor in qwen2_results:
            if hf_name.startswith("model."):
                hf_name = "model.language_model." + hf_name[len("model.") :]
            patched.append((hf_name, tensor))
        return patched

    # Handle the vision encoder part: uniformly map to model.visual.xxx
    # e.g. module.module.visual.patch_embed.proj.weight    -> model.visual.patch_embed.proj.weight
    # e.g. module.module.vision_model.blocks.0.attn.weight -> model.visual.blocks.0.attn.weight
    for prefix in _VISUAL_PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            hf_name = "model.visual." + rest
            return [(hf_name, param)]

    raise ValueError(f"Unknown Qwen3VL parameter name: {name}")
