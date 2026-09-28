# SPDX-License-Identifier: Apache-2.0
"""The frozen evaluation protocol of the DN-MOPD paper (App. A.3 and A.5).

Every value here is the value used for the paper's numbers. Changing any of them produces numbers that are not
comparable with the paper; the command-line tools therefore expose none of them as options, except the generation
cap, which the paper reports at both 8,192 and 16,384 tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Answer instruction appended to every AIME and MATH-500 problem (a single user turn).
AIME_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'

CAPS: Tuple[int, ...] = (8192, 16384)
CONTEXT = 32768                       # max_model_len of the six-suite engine; prompt + cap must fit

# Sampling of every suite (vLLM SamplingParams). max_tokens is the cap.
SAMPLING = {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'presence_penalty': 0.0, 'seed': 42}
ENABLE_THINKING = False               # chat template: add_generation_prompt=True, enable_thinking=False

# vLLM engine of the six public suites (AIME25/26, LiveCodeBench v5/v6, IFEval, IFBench).
ENGINE = {'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.90, 'enforce_eager': True, 'seed': 42,
          'max_model_len': CONTEXT, 'max_num_seqs': 64}
# MATH-500 was run by a separate script with its own engine settings: all 500 prompts in one generate call, no
# max_model_len (the model's default), and responses stripped of surrounding whitespace before grading.
ENGINE_MATH500 = {'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.95, 'enforce_eager': True, 'seed': 42,
                  'max_num_seqs': 256}

VLLM_VERSION = '0.18.0'               # the version every paper cell was generated with


@dataclass(frozen=True)
class Suite:
    """One benchmark: its size, samples per question, problems per engine call and default shard count."""
    name: str
    domain: str
    problems: int
    n: int
    chunk: int
    shards: int
    grader: str


SUITES: Dict[str, Suite] = {
    'aime25': Suite('aime25', 'math', 30, 64, 4, 4, 'aime'),
    'aime26': Suite('aime26', 'math', 30, 64, 4, 4, 'aime'),
    'lcb_v5': Suite('lcb_v5', 'code', 167, 6, 16, 3, 'lcb'),
    'lcb_v6': Suite('lcb_v6', 'code', 175, 6, 16, 3, 'lcb'),
    'ifeval': Suite('ifeval', 'if', 541, 16, 32, 2, 'if'),
    'ifbench': Suite('ifbench', 'if', 300, 16, 32, 2, 'if'),
    'math500': Suite('math500', 'math500', 500, 16, 500, 1, 'math500'),
}
SIX: Tuple[str, ...] = ('aime25', 'aime26', 'lcb_v5', 'lcb_v6', 'ifeval', 'ifbench')
DOMAINS: Dict[str, Tuple[str, ...]] = {'math': ('aime25', 'aime26'), 'code': ('lcb_v5', 'lcb_v6'),
                                       'if': ('ifeval', 'ifbench'), 'total': SIX}

# sha256 of the complete question catalogs (messages included) the paper's cells were generated from. The released
# LiveCodeBench catalogs omit the problem statements, so their file hashes differ; eval/data/prepare.py rebuilds the
# complete catalogs from the public sources and must reproduce these hashes.
CATALOG_SHA256 = {
    'aime25': '0669f0ee3de6a6c7e66d214a37b4da316b85f4af3f5e4790a4110a27e62d0268',
    'aime26': '2a628293c7db5294c916ff48a810d174f3a81bde0b8a4a18efa1c6867e6faf5d',
    'lcb_v5': 'e528ee172aa6ef32f0fb59796e843679530173b82086f26fb2eb6202fa1bcb4b',
    'lcb_v6': '14777a823bade8a1d3480f6aafbedccfdecd3d3e2c69387683515d8b2dc92e5d',
    'ifeval': 'bd5be9acdfcc6f4019db1014422055bf4ecf265b3c877ff41d824fd6285a8d84',
    'ifbench': 'f3ef730c359be914d2d1654d178e94d22ed312a7afaf608a89cbfa31508f90e0',
}

# Graders.
LCB_TIMEOUT_SECONDS = 6               # per test case (official runner argument)
LCB_MEMORY_LIMIT_BYTES = 8 * 1024 ** 3
LCB_GRADE_WORKERS = 4                 # problems graded in parallel within one cell
IF_GRADING_REVISION = 'deterministic-official-v1-20260918'
IF_GRADING_RANDOM_SEED = 0

# Paired bootstrap (reproduce/reproduce.py): B replicates, a fresh generator with this seed per contrast.
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260923

# sha256 of the effective chat template of the paper's Qwen3.5 models (2B differs from 4B and 9B).
CHAT_TEMPLATE_SHA256 = {
    '2B': '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80',
    '4B/9B': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
}


def check_cap(cap: int) -> int:
    """Return cap if it is one of the paper's caps, else raise ValueError."""
    if cap not in CAPS:
        raise ValueError(f'max tokens must be one of {CAPS}, got {cap}')
    return cap
