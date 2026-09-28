# SPDX-License-Identifier: Apache-2.0
"""SeqKD SFT data "rollout" for miles: supervised fine-tuning of the student on frozen teacher answers.

The SFT row runs through the same miles FSDP trainer as the OPD students (--debug-train-only: no inference engines);
this function only tokenizes and masks. The stock miles.rollout.sft_rollout applies the chat template without
`enable_thinking=False`, so its prompt would lack the empty `<think>\\n\\n</think>\\n\\n` block that every OPD rollout
(--apply-chat-template-kwargs '{"enable_thinking": false}') and every evaluation uses. This module rebuilds that
layout exactly:

    tokens    = apply_chat_template(user_msgs, add_generation_prompt=True, enable_thinking=False)   # loss 0
              + tokenize(teacher_response)                                                         # loss 1
              + [<|im_end|>]  iff finish_reason == "stop"                                          # loss 1

i.e. loss on response tokens only, and the end token is learned only for teacher answers that terminated (a teacher
answer truncated at max_tokens trains without a terminal <|im_end|>). A response longer than
--rollout-max-response-len is cut to that length and trains without the end token.

Use:
    --rollout-function-path Uni_OPD_utils.seqkd.sft_rollout.generate_rollout
    --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only
    --prompt-data <seqkd targets .jsonl> --input-key messages

Input row contract (one JSON object per line):
    messages : [{role: user, ...}(, ...), {role: assistant, content: <teacher text>}]
    metadata : {finish_reason: "stop" | "length", ...}   (optional; default "stop")
"""

import logging

from miles.utils.processing_utils import load_tokenizer

__all__ = ["generate_rollout", "build_sft_tokens"]

logger = logging.getLogger(__name__)

TOKENIZER = None
EOS_ID = None
SAMPLE_PRINTED = False


def _get_tokenizer(args):
    global TOKENIZER, EOS_ID
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        eos_id = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id < 0:
            eos_id = TOKENIZER.eos_token_id
        EOS_ID = eos_id
        logger.info(
            "sft_rollout: terminal token id=%s (%r)",
            EOS_ID,
            TOKENIZER.convert_ids_to_tokens(EOS_ID),
        )
    return TOKENIZER


def build_sft_tokens(tokenizer, eos_id, messages, finish_reason, max_resp):
    """(prompt_ids, response_ids) for one SFT row; the loss mask is 1 on every response id."""
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=True, enable_thinking=False
    )
    # transformers 5.x returns a BatchEncoding here, not a list of ids
    if hasattr(prompt_ids, "keys"):
        prompt_ids = list(prompt_ids["input_ids"])
    response_text = messages[-1]["content"]
    # teacher text carries no special tokens (vLLM detokenizes without them)
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if max_resp and len(response_ids) >= max_resp:
        response_ids = response_ids[:max_resp]
        finish_reason = "length"  # no room / no right to a terminal token
    if finish_reason == "stop":
        response_ids = response_ids + [eos_id]
    return list(prompt_ids), list(response_ids)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """miles rollout entrypoint (data-only SFT). Mirrors miles.rollout.sft_rollout.

    Returns the list of sample groups pulled from the data buffer, with tokens / response_length / loss_mask / reward
    filled in.
    """
    assert not evaluation, "sft_rollout is train-only"
    assert args.rollout_global_dataset

    global SAMPLE_PRINTED
    tokenizer = _get_tokenizer(args)
    max_resp = int(getattr(args, "rollout_max_response_len", 0) or 0)

    groups = data_buffer.get_samples(args.rollout_batch_size)

    for i, group in enumerate(groups):
        (sample,) = group  # n_samples_per_prompt must be 1 for SFT
        messages = sample.prompt
        assert isinstance(messages, list) and messages and messages[-1].get("role") == "assistant", (
            f"sft_rollout expects a messages list ending in an assistant turn; got "
            f"{type(messages)} (input-key must be 'messages', WITHOUT --apply-chat-template)"
        )
        finish_reason = (sample.metadata or {}).get("finish_reason", "stop")
        prompt_ids, response_ids = build_sft_tokens(tokenizer, EOS_ID, messages, finish_reason, max_resp)

        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.reward = 0
        sample.loss_mask = [1] * len(response_ids)

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                "sft_rollout example: prompt_len=%d response_len=%d finish=%s prompt_tail=%r response_head=%r",
                len(prompt_ids),
                len(response_ids),
                finish_reason,
                tokenizer.decode(prompt_ids[-12:]),
                tokenizer.decode(response_ids[:12]),
            )
            SAMPLE_PRINTED = True

    return groups
