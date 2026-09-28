# Modified by the DN-MOPD authors (2026): a passthrough export that keeps the origin repo's key names and dtype
# (needed for the multimodal Qwen3.5 checkpoints), a text-config retry and language-tower rename for the
# instantiate-then-load path, and a refusal to write a checkpoint whose keys match < 90% of the target.
import argparse
import glob
import json
import os
import pickle
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    predefined = [
        "model_state.model.",
        "model_state.",
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {k.removeprefix(prefix) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _origin_key_set(origin_hf_dir: str) -> set[str]:
    """The tensor names the ORIGIN repo actually ships, read from its own files."""
    index = os.path.join(origin_hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return set(json.load(f)["weight_map"].keys())
    keys: set[str] = set()
    for shard in sorted(glob.glob(os.path.join(origin_hf_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as h:
            keys |= set(h.keys())
    return keys


def _passthrough_save(tensor_items: dict[str, torch.Tensor], origin_hf_dir: str, output_dir: str) -> bool:
    """Write the checkpoint under the ORIGIN repo's own key names (the "passthrough" export).

    The instantiate-then-load path below asks `AutoModelForCausalLM.from_config` what the keys
    should be, and for Qwen3.5 that question has no good answer: the config is a multimodal
    wrapper (`Qwen3_5ForConditionalGeneration`), the class transformers builds from it does not
    use the checkpoint's key names, and vLLM registers only the wrapper.

    The FSDP checkpoint's keys, after prefix stripping, are the origin repo's keys (lm_head +
    model.language_model.* + model.visual.*; the origin additionally ships mtp.* tensors that the
    trainer never holds). So copy them out as they are and keep the origin config, which by
    construction describes them. No model is instantiated, nothing is renamed, and the
    checkpoint's own dtype survives (the from_config path builds fp32 params and would upcast
    bf16 weights into them).

    Returns True when it handled the conversion.
    """
    origin_keys = _origin_key_set(origin_hf_dir)
    if not origin_keys:
        return False
    prefix, match = _strip_best_prefix(list(tensor_items.keys()), origin_keys)
    if match < 0.9 * len(origin_keys):
        print(f"passthrough not applicable: only {match}/{len(origin_keys)} origin tensors matched")
        return False

    state = {}
    for k, v in tensor_items.items():
        stripped = k.removeprefix(prefix)
        if stripped in origin_keys:
            state[stripped] = v
    dropped = len(tensor_items) - len(state)
    print(f"passthrough: {len(state)}/{len(origin_keys)} origin tensors matched with prefix "
          f"'{prefix}'; {dropped} checkpoint tensors not in the origin repo were dropped")

    # Name what the ORIGIN has and the checkpoint does not: silently missing weights would give a
    # partially random model. For Qwen3.5 this legitimately reports the 15 `mtp.*` tensors: the
    # trainer does not train or save the multi-token-prediction head, and vLLM does not need it
    # unless speculative decoding is enabled. Any OTHER group appearing here is a bug.
    absent = origin_keys - set(state)
    if absent:
        groups: dict[str, int] = {}
        for k in absent:
            groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + 1
        print(f"   NOTE {len(absent)} origin tensors absent from the checkpoint, by top-level "
              f"group: {sorted(groups.items(), key=lambda x: -x[1])}")

    os.makedirs(output_dir, exist_ok=True)
    # Stale shards from an earlier conversion would be listed by no index and loaded by nothing,
    # or worse, listed by a stale index. Clear them.
    for stale in glob.glob(os.path.join(output_dir, "*.safetensors")) + [
        os.path.join(output_dir, "model.safetensors.index.json")
    ]:
        if os.path.isfile(stale):
            os.remove(stale)
    save_torch_state_dict(state, output_dir, max_shard_size="5GB")
    print(f"Model weights saved to {output_dir} (passthrough)")
    return True


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
) -> None:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    tensor_items = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    if _passthrough_save(tensor_items, origin_hf_dir, output_dir):
        return

    config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_config(config)
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    total_keys = len(tensor_items)

    # For a MULTIMODAL wrapper config (Qwen3_5ForConditionalGeneration), `from_config` builds
    # the wrapper -- keys `model.language_model.*` plus a vision tower -- while the trainer used
    # `AutoModelForCausalLM.from_pretrained`, which resolves to the text-only model with keys
    # `model.*`. With strict=False the mismatch would be silent, so retry against the text config
    # when the match is poor.
    if best_match < 0.9 * total_keys and hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
        if type(text_config).__name__ != type(config).__name__:
            print(f"low match {best_match}/{total_keys} with {type(hf_model).__name__}; "
                  f"retrying with text config {type(text_config).__name__}")
            alt_model = AutoModelForCausalLM.from_config(text_config)
            alt_keys = set(alt_model.state_dict().keys())
            alt_prefix, alt_match = _strip_best_prefix(list(tensor_items.keys()), alt_keys)
            if alt_match > best_match:
                hf_model, target_keys = alt_model, alt_keys
                best_prefix, best_match = alt_prefix, alt_match

    print(f"Using prefix '{best_prefix}' for key mapping. " f"Matched {best_match}/{total_keys} parameter keys.")

    model_state = {k.removeprefix(best_prefix): v for k, v in tensor_items.items()}

    # A multimodal-wrapper checkpoint stores the language tower under `model.language_model.*`
    # alongside `model.visual.*`, while the target text-only model expects `model.*`. Rename the
    # language tower and DROP the vision tower, which a text-only model does not have. Trigger on
    # a POOR match, not on zero overlap: exactly one key (lm_head.weight) matches by coincidence
    # in both layouts.
    if best_match < 0.9 * len(target_keys) and any(k.startswith("model.language_model.") for k in model_state):
        renamed = {}
        for k, v in model_state.items():
            if k.startswith("model.language_model."):
                renamed["model." + k[len("model.language_model."):]] = v
            elif k.startswith("model.visual.") or k.startswith("mtp."):
                continue  # not part of the text-only model
            else:
                renamed[k] = v
        n_hit = len(set(renamed) & target_keys)
        print(f"multimodal wrapper detected: renamed language tower, dropped vision/mtp; "
              f"now {n_hit}/{len(target_keys)} target keys covered")
        if n_hit > best_match:
            model_state, best_match = renamed, n_hit
            total_keys = len(target_keys)

    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    # A converter that writes a random model on a key mismatch is worse than one that crashes:
    # the artefact looks valid, loads, generates, and is wrong. Refuse instead. This check must
    # run AFTER the rename above.
    if best_match < 0.9 * len(target_keys):
        raise ValueError(
            f"key mapping matched only {best_match}/{len(target_keys)} target tensors with "
            f"model class {type(hf_model).__name__} and prefix '{best_prefix}'. Refusing to "
            f"write a partially-initialised checkpoint. Check that --origin-hf-dir matches the "
            f"architecture the trainer actually instantiated."
        )

    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir}")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(args.origin_hf_dir, model_dir, args.output_dir)
    copy_assets(args.origin_hf_dir, args.output_dir)
