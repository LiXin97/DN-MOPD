import itertools
import json
import logging
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import ray
import yaml
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, ProcessorMixin

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from miles.utils.processing_utils import process_vision_info
from miles.utils.ray_utils import Box
from miles.utils.timer import Timer
from miles.utils.types import MultimodalTypes, RolloutBatch, Sample

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path: str):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def read_yaml_config(yaml_path: str, *, input_key: str = "problem", label_key: str = "answer"):
    """
    读取 yaml 配置文件，返回 records 的迭代器。
    yaml 格式:
        data:
          - path: /some/file.jsonl
            teacher: code          # 可选，覆盖每行的 teacher_model
            input_key: prompt      # 可选，映射到全局 input_key
            label_key: label       # 可选，映射到全局 label_key
          - path: /some/file.parquet
    """
    with open(yaml_path, encoding="utf-8") as f:
        config: dict[str] = yaml.safe_load(f)

    if "data" in config:
        data_entries: list[dict] = config["data"]
    elif "train_data" in config:
        data_entries: list[dict] = config["train_data"]
    else:
        raise ValueError(f"YAML config '{yaml_path}' must contain 'data' or 'train_data' entries.")

    for entry in data_entries:
        file_path = entry.get("path")
        if not file_path:
            raise ValueError(f"YAML entry missing 'path': {entry}")

        teacher_override = entry.get("teacher", None)
        src_input_key = entry.get("input_key", None)
        src_label_key = entry.get("label_key", None)
        logger.info(
            "YAML config: loading '%s'%s",
            file_path,
            f" with teacher override='{teacher_override}'" if teacher_override else "",
        )

        count = 0
        for record in read_file(file_path):
            if teacher_override is not None:
                record["teacher_model"] = teacher_override
            if src_input_key and src_input_key != input_key and src_input_key in record:
                record[input_key] = record.pop(src_input_key)
            if src_label_key and src_label_key != label_key and src_label_key in record:
                record[label_key] = record.pop(src_label_key)
            count += 1
            yield record
        logger.info(f"  -> {file_path}: {count} samples")


def _parse_generalized_path(s: str):
    """
    支持一种带切片后缀的扩展路径格式
    <真实文件路径>@[<start>:<end>]

    返回 (path, slice)
    """
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(
    origin_samples: list[Sample],
    tokenizer,
    processor,
    max_length: int | None,
    num_proc: int = 1,
) -> list[Sample]:
    if max_length is None:
        return origin_samples

    logger.info(f"Filtering samples with prompt length > {max_length} tokens...")

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:

        def check_one(sample: Sample) -> bool:
            processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
            input_ids = processor_output["input_ids"][0]
            return len(input_ids) <= max_length

        keep_flags = [None] * len(origin_samples)
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            future_to_idx = {executor.submit(check_one, sample): idx for idx, sample in enumerate(origin_samples)}
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(origin_samples),
                desc="Filtering long prompts",
                mininterval=5,
                miniters=10,
            ):
                idx = future_to_idx[future]
                try:
                    keep_flags[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to check sample length at idx {idx}, error: {e}")
                    keep_flags[idx] = False

        filtered_samples = [sample for sample, keep in zip(origin_samples, keep_flags, strict=True) if keep]
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")
    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt: str | list = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]
    else:
        # prompt 已经是对话格式
        pass

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                raw = data.get(data_key)
                if raw is None:
                    continue
                multimodals[mt.placeholder] = (mt, list(raw))

        if multimodals:
            pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

            for message in prompt:
                if isinstance(message["content"], str):
                    content_list = []
                    for segment in re.split(pattern, message["content"]):
                        if not segment:
                            continue
                        if segment in multimodals:
                            mt, content = multimodals[segment]
                            content_list.append({"type": mt.name, mt.name: content.pop(0)})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list

                elif isinstance(message["content"], list):
                    # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                    # e.g.
                    #  "content": [
                    #     {
                    #         "type": "image",
                    #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    #     },
                    #     {"type": "text", "text": "Describe this image."},
                    # ],
                    logger.warning("message['content'] is a list of dicts, no processing will be done.")
                    continue
                else:
                    raise ValueError(
                        f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                    )

    return prompt


def _get_teacher_name(data: dict) -> str | None:
    for key in ["teacher_model", "teacher_model_name", "teacher"]:
        if key in data:
            return data[key]
    return None


class Dataset:
    def __init__(
        self,
        path,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin | None,
        max_length,  # rollout_max_prompt_len
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        is_OPD: bool = False,
        dataset_num_proc=32,
    ):
        origin_samples: list[Sample] = []

        if not label_key:
            logger.warning("label_key is not set, all samples will have label=None")

        if isinstance(path, str) and path.endswith(".yaml"):
            data_reader = read_yaml_config(path, input_key=prompt_key, label_key=label_key or "answer")
        else:
            data_reader = read_file(path)

        empty_teacher_num, error_num = 0, 0
        logger.info(f"Loading samples from {path}...")
        # 先把所有原始数据读入内存
        all_data: list[dict] = list(tqdm(data_reader, desc="Reading data", mininterval=5, miniters=10))

        def process_one(data: dict) -> tuple[Sample | None, bool]:
            try:
                as_conversation = apply_chat_template or (multimodal_keys is not None)
                prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

                metadata = data.get(metadata_key) or {}
                tools = None
                if tool_key is not None and tool_key in data:
                    tools = data[tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                    metadata["tools"] = tools

                if apply_chat_template:
                    output_prompt: str = tokenizer.apply_chat_template(
                        prompt,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=True,
                        **(apply_chat_template_kwargs or {}),
                    )
                else:
                    output_prompt: str = prompt

                has_mm = multimodal_keys and any(data.get(v) for v in multimodal_keys.values())
                if processor and has_mm:
                    assert isinstance(prompt, list), (
                        f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                    )
                    multimodal_inputs = process_vision_info(prompt, processor)
                else:
                    multimodal_inputs = None

                sample = Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )

                if is_OPD:
                    teacher_name = _get_teacher_name(data)
                    if teacher_name is None:
                        sample.teacher_model_name = "default"
                        return sample, True  # empty_teacher
                    else:
                        sample.teacher_model_name = teacher_name

                return sample, False
            except Exception as e:
                logger.warning(f"Failed to process sample, error: {e}")
                return None, False

        # 并行处理
        results = [None] * len(all_data)
        with ThreadPoolExecutor(max_workers=dataset_num_proc) as executor:
            future_to_idx = {executor.submit(process_one, data): idx for idx, data in enumerate(all_data)}
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(all_data),
                desc="Processing samples",
                mininterval=5,
                miniters=10,
            ):
                idx = future_to_idx[future]
                results[idx] = future.result()

        multimodal_num, text_only_num = 0, 0
        source_counter: dict[str, dict[str, int]] = {}
        for sample, is_empty_teacher in results:
            if sample is None:
                error_num += 1
            else:
                if is_empty_teacher:
                    empty_teacher_num += 1
                is_mm = bool(sample.multimodal_inputs)
                if is_mm:
                    multimodal_num += 1
                else:
                    text_only_num += 1
                key = getattr(sample, "teacher_model_name", "unknown") if is_OPD else "all"
                if key not in source_counter:
                    source_counter[key] = {"total": 0, "multimodal": 0, "text_only": 0}
                source_counter[key]["total"] += 1
                source_counter[key]["multimodal" if is_mm else "text_only"] += 1
                origin_samples.append(sample)

        if error_num > 0:
            logger.warning(f"Loaded {len(origin_samples)} samples from {path} with {error_num} errors.")
        else:
            logger.info(f"Loaded {len(origin_samples)} samples from {path}.")
        logger.info(
            "Data statistics:\n%s",
            "\n".join(
                f"  {name}: {c['total']} (multimodal: {c['multimodal']}, text_only: {c['text_only']})"
                for name, c in source_counter.items()
            ),
        )

        if is_OPD:
            teacher_counter = Counter(sample.teacher_model_name for sample in origin_samples)
            total: int = len(origin_samples)
            if empty_teacher_num > 0:
                logger.warning(f"{empty_teacher_num} samples have no 'teacher_model' field, set to default")
            logger.info(
                "Teacher model distribution:\n%s",
                "\n".join(f"  {name}: {cnt} ({cnt / total * 100:.1f}%)" for name, cnt in teacher_counter.most_common()),
            )

        if max_length is not None:
            self.origin_samples: list[Sample] = filter_long_prompt(
                origin_samples,
                tokenizer,
                processor,
                max_length,
                num_proc=dataset_num_proc,
            )
        else:
            self.origin_samples: list[Sample] = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref: list[Box], dp_rank: int, dp_size: int) -> RolloutBatch:
    assert len(rollout_data_ref) == dp_size

    # 1. 每个 DP rank 取自己对应的那份数据
    rollout_data: RolloutBatch = ray.get(rollout_data_ref[dp_rank].inner)

    # 2. 取出 partition 索引（用完即弃）
    partition = rollout_data.pop("partition")

    # 3. total_lengths 是完整传过来的，这里用 partition 做二次切分
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths  # 保存全局的 seq_lens 用于监控
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]  # 只保留本 rank 的

    return rollout_data
