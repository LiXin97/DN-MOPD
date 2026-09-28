# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU test of generate -> grade -> aggregate -> compare with a stand-in for vLLM and the tokenizer.

Checks that generation passes exactly the protocol's engine and sampling settings, renders the chat template with
add_generation_prompt=True and enable_thinking=False, writes resumable chunks, refuses a mixed-protocol resume, and
that grading and aggregation read what it wrote. The GPU path itself is exercised by eval/smoke_test.sh.
"""
import json
import shutil
import sys
import types

import pytest
from evaltest_helpers import needs_math_verify

import eval.generate as generate_module
from eval.common import released_dir
from eval.data.prepare import build_aime26
from eval.protocol import ENGINE, ENGINE_MATH500, SAMPLING

CALLS = {}


class FakeTokenizer:
    chat_template = '{% if enable_thinking is defined %}{% endif %}'

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return '|'.join(m['role'] + ':' + m['content'] for m in messages) + '|assistant:'

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))


class FakeLLM:
    def __init__(self, **kwargs):
        CALLS['engine'] = kwargs

    def generate(self, prompts, params, use_tqdm=True):
        CALLS.setdefault('batches', []).append(len(prompts))
        CALLS['sampling'] = params.kwargs
        n = params.kwargs['n']
        out = []
        for p in prompts:
            samples = [types.SimpleNamespace(text=(' \\boxed{70} ' if i % 2 == 0 else ' no answer '),
                                             token_ids=[0] * (5 + i), finish_reason='length' if i == 1 else 'stop')
                       for i in range(n)]
            out.append(types.SimpleNamespace(outputs=samples))
        return out


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: FakeTokenizer())))
    data = tmp_path / 'prepared'
    (data / 'catalog').mkdir(parents=True)
    from eval.data.prepare import build_math500
    from eval.common import digest, jsonl_text, write_text_atomic
    # A synthetic 30-question AIME-style catalog (the real AIME 2025 text is not redistributed); question 0's answer
    # is 70, every other answer differs. Its hash stands in for the evaluation catalog hash.
    synthetic = tmp_path / 'aime_synthetic.jsonl'
    synthetic.write_text(''.join(json.dumps({'id': i, 'problem': f'Synthetic problem {i}.', 'answer': str(70 + 11 * i)})
                                 + '\n' for i in range(30)))
    write_text_atomic(data / 'catalog' / 'aime25.jsonl', jsonl_text(build_aime26(synthetic, {})))
    monkeypatch.setitem(generate_module.CATALOG_SHA256, 'aime25', digest(data / 'catalog' / 'aime25.jsonl'))
    hf = tmp_path / 'm500.jsonl'
    hf.write_text(''.join(json.dumps({'problem': q['problem'], 'answer': q['answer'], 'unique_id': q['id']}) + '\n'
                          for q in map(json.loads, open(released_dir() / 'math500' / 'questions.jsonl'))))
    write_text_atomic(data / 'catalog' / 'math500.jsonl', jsonl_text(build_math500(hf, {})))
    (data / 'PREPARED.json').write_text(json.dumps({'math500': {'catalog_sha256':
                                                                digest(data / 'catalog' / 'math500.jsonl')}}))
    monkeypatch.setenv('DN_MOPD_EVAL_DATA', str(data))
    CALLS.clear()
    return tmp_path


@needs_math_verify
def test_generate_grade_aggregate(fake_backend, capsys):
    from eval import aggregate, compare, generate, grade
    root = fake_backend / 'runs'
    out = root / 'fake' / 'aime25' / 'cap8192'
    args = ['--model', 'fake/model', '--name', 'fake', '--suite', 'aime25', '--max-tokens', '8192', '--out', str(out),
            '--limit', '6']
    assert generate.main(args) == 0
    assert CALLS['engine'] == {'model': 'fake/model', **ENGINE}
    assert CALLS['sampling'] == {'n': 64, 'max_tokens': 8192, **SAMPLING}
    assert CALLS['batches'] == [4, 2]                          # AIME goes to the engine four questions at a time
    assert sorted(p.name for p in (out / 'raw').iterdir()) == ['0000.json', '0004.json']
    chunk = json.loads((out / 'raw' / '0000.json').read_text())
    assert chunk[0]['responses'][0] == ' \\boxed{70} '        # six-suite answers are kept exactly as generated
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    assert manifest['subset'] and manifest['questions'] == 6 and manifest['sampling']['max_tokens'] == 8192

    CALLS.clear()
    assert generate.main(args) == 0 and 'engine' not in CALLS   # resume: nothing left, no engine started
    with pytest.raises(SystemExit):
        generate.main(args[:-6] + ['16384', '--out', str(out), '--limit', '6'])   # another cap into the same dir

    assert grade.main(['--gen', str(out), '--workers', '2']) == 0
    scores = [json.loads(line) for line in (out / 'scores.jsonl').open()]
    assert [s['id'] for s in scores] == [str(i) for i in range(6)]
    # only question 0 has answer 70: even samples are correct there, nothing elsewhere
    assert scores[0]['correct'] == '10' * 32 and all(s['correct'] == '0' * 64 for s in scores[1:])
    assert scores[0]['cap_hits'] == 1 and scores[0]['mean_tokens'] == pytest.approx(5 + 63 / 2)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['mean_pass1'] == pytest.approx(0.5 / 6) and summary['cap_hit_rate'] == pytest.approx(1 / 64)

    capsys.readouterr()
    assert aggregate.main(['--records', str(root), '--cap', '8192']) == 0
    table = capsys.readouterr().out
    assert 'fake' in table and f'{100 * 0.5 / 6:.2f}' in table
    shutil.copytree(root / 'fake', root / 'fake2')
    assert compare.main(['fake', 'fake2', '--records', str(root), '--cap', '8192', '--domain', 'aime25', '--json']) == 0
    result = json.loads(capsys.readouterr().out)[0]
    assert result['delta_pp'] == 0.0 and result['ci95'] == [0.0, 0.0]


def test_math500_uses_its_own_engine_settings(fake_backend):
    from eval import generate
    out = fake_backend / 'm500'
    assert generate.main(['--model', 'fake/model', '--suite', 'math500', '--max-tokens', '16384', '--out', str(out),
                          '--limit', '3']) == 0
    assert CALLS['engine'] == {'model': 'fake/model', 'tokenizer': 'fake/model', **ENGINE_MATH500}
    assert CALLS['sampling'] == {'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 16384, 'n': 16, 'seed': 42}
    rows = json.loads((out / 'raw' / '0000.json').read_text())
    assert rows[0]['responses'][0] == '\\boxed{70}'           # MATH-500 answers were stripped


def test_catalog_hash_is_enforced(fake_backend):
    from eval import generate
    from eval.common import prepared_dir
    path = prepared_dir() / 'catalog' / 'aime25.jsonl'
    path.write_text(path.read_text().replace('Synthetic problem 3.', 'Synthetic problem 3!', 1))
    with pytest.raises(SystemExit, match='not the paper catalog'):
        generate.main(['--model', 'x', '--suite', 'aime25', '--max-tokens', '8192', '--out', str(fake_backend / 'o')])


def test_shards_cover_each_suite_once():
    from eval.generate import chunk_starts, shard_starts
    from eval.protocol import SUITES
    for s in SUITES.values():
        owned = [x for k in range(s.shards) for x in shard_starts(s.problems, s.chunk, s.shards, k)]
        assert owned == chunk_starts(s.problems, s.chunk)
