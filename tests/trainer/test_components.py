# SPDX-License-Identifier: Apache-2.0
"""SeqKD-SFT tokenization, weight merges, the HF passthrough export, the injection bank, the SeqKD corpus merge, the
code judge, and an import smoke test of the miles fork."""
import asyncio
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from conftest import MILES, RECIPES, Args, make_batch


# ------------------------------------------------------------------------------------------------ SeqKD-SFT rollout
class FakeTok:
    """Chat template: <u>content</u><a><think></think>; tokens = characters."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt and tokenize and enable_thinking is False
        text = "".join(f"<u>{m['content']}</u>" for m in messages) + "<a><think></think>"
        return {"input_ids": [ord(c) for c in text]}                   # transformers 5 returns a BatchEncoding

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_sft_tokens_mask_the_prompt_and_end_only_finished_answers():
    from Uni_OPD_utils.seqkd.sft_rollout import build_sft_tokens

    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "abc"}]
    p, r = build_sft_tokens(FakeTok(), 7, msgs, "stop", 16384)
    assert "".join(map(chr, p)) == "<u>q</u><a><think></think>" and r == [97, 98, 99, 7]
    p, r = build_sft_tokens(FakeTok(), 7, msgs, "length", 16384)
    assert r == [97, 98, 99]                                            # truncated teacher answer: no end token
    p, r = build_sft_tokens(FakeTok(), 7, msgs, "stop", 2)
    assert r == [97, 98]                                                # cut to the response cap, no end token


def test_sft_rollout_fills_samples(monkeypatch):
    from miles.utils.types import Sample
    from Uni_OPD_utils.seqkd import sft_rollout

    monkeypatch.setattr(sft_rollout, "TOKENIZER", FakeTok())
    monkeypatch.setattr(sft_rollout, "EOS_ID", 7)
    s = Sample(prompt=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "xy"}], index=0)
    s.metadata = {"finish_reason": "stop"}
    buf = SimpleNamespace(get_samples=lambda n: [[s]])
    args = SimpleNamespace(rollout_global_dataset=True, rollout_max_response_len=16384, rollout_batch_size=1)
    out = sft_rollout.generate_rollout(args, 0, buf)
    assert out[0][0].response_length == 3 and out[0][0].loss_mask == [1, 1, 1] and out[0][0].tokens[-1] == 7


# ------------------------------------------------------------------------------------------------ merges
def _load_merge():
    spec = importlib.util.spec_from_file_location("dn_merge", RECIPES / "merge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merge_tensor_rules():
    m = _load_merge()
    base = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    ts = [torch.tensor([2.0, 2.0], dtype=torch.bfloat16), torch.tensor([4.0, 5.0], dtype=torch.bfloat16),
          torch.tensor([0.0, 2.0], dtype=torch.bfloat16)]
    assert torch.equal(m.average(ts), torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    # 1 + (1 + 3 - 1) = 4 ; 2 + (0 + 3 + 0) = 5
    assert torch.equal(m.task_arithmetic(base, ts, 1.0), torch.tensor([4.0, 5.0], dtype=torch.bfloat16))
    with pytest.raises(SystemExit):
        m.check_lambda(1 / 3, 3)
    ints = torch.tensor([1, 2])
    assert torch.equal(m.task_arithmetic(ints, [ints.clone()] * 3, 1.0), ints)
    with pytest.raises(ValueError):
        m.task_arithmetic(ints, [ints, ints + 1, ints], 1.0)
    keys, extra = m.plan_keys({"a", "b", "mtp.0"}, [{"a", "b"}] * 3, "mtp.")
    assert keys == ["a", "b"] and extra == ["mtp.0"]
    with pytest.raises(SystemExit):
        m.plan_keys({"a", "b", "visual.x"}, [{"a", "b"}] * 3, "mtp.")
    with pytest.raises(SystemExit):
        m.plan_keys({"a", "b"}, [{"a", "b"}, {"a"}, {"a", "b"}], "mtp.")


def _hf_dir(path, tensors, aux="same"):
    path.mkdir(parents=True)
    save_file(tensors, str(path / "model.safetensors"))
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (path / f).write_text(aux)
    return str(path)


def test_merge_end_to_end(tmp_path, monkeypatch):
    m = _load_merge()
    w = {"w": torch.tensor([[1.0, 2.0]]), "mtp.x": torch.zeros(1)}
    base = _hf_dir(tmp_path / "base", w)
    teachers = {d: _hf_dir(tmp_path / d, {"w": torch.tensor([[1.0 + k, 2.0 * k]])})
                for k, d in enumerate(("math", "code", "ifeval"))}
    targs = [f"--teacher={d}={p}" for d, p in teachers.items()]
    m.main(["--kind", "avg", "--size", "2b", "--base", base, "--out", str(tmp_path / "avg"), *targs])
    m.main(["--kind", "ta", "--size", "2b", "--base", base, "--out", str(tmp_path / "ta"), *targs])
    for kind, want in (("avg", [[2.0, 2.0]]), ("ta", [[4.0, 2.0]])):     # teachers [1,0], [2,2], [3,4]; base [1,2]
        idx = json.load(open(tmp_path / kind / "model.safetensors.index.json"))["weight_map"]
        assert set(idx) == {"w"}                                        # mtp.* excluded
        with safe_open(str(tmp_path / kind / idx["w"]), "pt") as h:
            assert torch.allclose(h.get_tensor("w"), torch.tensor(want))
        prov = json.load(open(tmp_path / kind / "MERGE_PROVENANCE.json"))
        assert prov["excluded_base_only_keys"] == ["mtp.x"] and (prov["lambda"] == 1.0) == (kind == "ta")
    with pytest.raises(SystemExit, match="refuse to overwrite"):
        m.main(["--kind", "avg", "--size", "2b", "--base", base, "--out", str(tmp_path / "avg"), *targs])


# ------------------------------------------------------------------------------------------------ HF export
def _load_convert():
    spec = importlib.util.spec_from_file_location("dn_convert", MILES / "tools" / "convert_fsdp_to_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_passthrough_export_keeps_origin_names_and_dtype(tmp_path):
    c = _load_convert()
    origin = tmp_path / "origin"
    origin.mkdir()
    keys = ["lm_head.weight", "model.language_model.layers.0.w", "model.visual.patch.w", "mtp.fc.w"]
    json.dump({"weight_map": {k: "x.safetensors" for k in keys}}, open(origin / "model.safetensors.index.json", "w"))
    ckpt = {f"model_state.model.{k}": torch.full((2,), float(i), dtype=torch.bfloat16) for i, k in enumerate(keys[:3])}
    out = tmp_path / "out"
    assert c._passthrough_save(ckpt, str(origin), str(out)) is False    # 3 of 4 < 90 %: not applicable
    keys9 = [f"model.language_model.layers.{i}.w" for i in range(20)]
    json.dump({"weight_map": {k: "x" for k in keys9 + ["mtp.fc.w"]}}, open(origin / "model.safetensors.index.json", "w"))
    ckpt = {f"model_state.model.{k}": torch.full((2,), 1.0, dtype=torch.bfloat16) for k in keys9}
    assert c._passthrough_save(ckpt, str(origin), str(out)) is True
    files = list(out.glob("*.safetensors"))
    with safe_open(str(files[0]), "pt") as h:
        assert set(h.keys()) == set(keys9) and h.get_tensor(keys9[0]).dtype == torch.bfloat16


# ------------------------------------------------------------------------------------------------ injection bank
def _bank_lib():
    spec = importlib.util.spec_from_file_location("dn_bank", RECIPES / "lib" / "injection_bank.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EndTok:
    eos_token_id = 9

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 8, "<|endoftext|>": 9}[t]


def test_bank_keys_and_trajectory_normalisation(hook_module):
    b = _bank_lib()
    assert b.prompt_key("rendered prompt") == hook_module.prompt_key("rendered prompt")
    im_end, ends = b.end_ids(EndTok())
    assert b.response_ids([5, 6, 9, 8, 9], im_end, ends) == [5, 6, 8] and b.response_ids([5], im_end, ends) == [5, 8]


def test_bank_phase_c_seals_what_the_hook_reads(tmp_path, monkeypatch, hook_module, write_config):
    b = _bank_lib()
    samples = make_batch()
    firsts = {s.group_index: s for s in reversed(samples)}
    parts = tmp_path / "parts"
    parts.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("")
    for dom in ("math", "code", "ifeval"):
        grows, arows, brows = [], [], []
        for gi, s in firsts.items():
            if s.metadata["domain"] != dom:
                continue
            key = hook_module.prompt_key(s.prompt)
            plen = len(s.tokens) - s.response_length
            fin = "length" if gi == 5 else "stop"
            grows.append({"key": key, "idx": gi, "domain": dom, "prompt_ids": s.tokens[:plen],
                          "samples": [{"j": 0, "gen_ids": [4000 + gi, 4001, 9], "text": "t", "finish_reason": fin}]})
            if fin == "stop":
                arows.append({"key": key, "j": 0, "correct": gi != 4, "prompt_ids_match": True, "key_match": True})
                brows.append({"key": key, "teacher": "T", "logp": [-0.1, -0.2, -0.3]})
        for name, rows in ((f"G_{dom}_s0", grows), (f"A_{dom}", arows), (f"B_{dom}_s0", brows)):
            (parts / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (parts / f"{name}.done").write_text("x")
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: EndTok()))
    bank = tmp_path / "bank" / "inject_bank.pt"
    b.main(["C", "--prompts", str(prompts), "--parts", str(parts), "--tokenizer", "tok", "--nshards", "1",
            "--out", str(bank), "--teacher", "math=/t/m", "--teacher", "code=/t/c", "--teacher", "ifeval=/t/i"])
    sealed = torch.load(bank, weights_only=False)
    assert sealed["meta"]["per_domain"] == {"math": 2, "code": 1, "ifeval": 1}   # group 4 wrong, group 5 truncated
    assert (bank.parent / "BANK_META.json").exists()
    sha = (bank.parent / "inject_bank.pt.sha256").read_text().strip()
    hook = hook_module.MopdHook(write_config(inject_bank_path=str(bank), inject_bank_sha256=sha,
                                             inject_adv_const=1.0, inject_anneal_steps=40))
    hook.apply(Args(), samples)
    injected = [s for s in samples if s.inject_adv_const > 0]
    assert sorted(s.group_index for s in injected) == [0, 1, 2, 3]
    assert all(s.tokens[-3:] == [4000 + s.group_index, 4001, 8] for s in injected)


# ------------------------------------------------------------------------------------------------ SeqKD corpus merge
def test_seqkd_merge_orders_rows_and_reports_truncation(tmp_path):
    spec = importlib.util.spec_from_file_location("dn_seqkd", RECIPES / "lib" / "seqkd_generate.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    parts = []
    for dom, idxs in (("math", [0, 3]), ("code", [1]), ("ifeval", [2])):
        p = tmp_path / f"{dom}.jsonl"
        p.write_text("".join(json.dumps({"idx": i, "domain": dom, "response": "r", "metadata": {
            "finish_reason": "length" if i == 3 else "stop", "n_response_tokens": 10 + i, "teacher_model": dom}}) + "\n"
            for i in idxs))
        parts.append(str(p))
    g.main(["merge", "--out", str(tmp_path / "t.jsonl"), "--stats", str(tmp_path / "s.json"), "--n", "4", *parts])
    rows = [json.loads(line) for line in open(tmp_path / "t.jsonl")]
    stats = json.load(open(tmp_path / "s.json"))
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    assert stats["overall"]["truncation_rate"] == 0.25 and stats["per_domain"]["math"]["truncation_rate"] == 0.5
    with pytest.raises(SystemExit):
        g.main(["merge", "--out", str(tmp_path / "t2.jsonl"), "--stats", str(tmp_path / "s2.json"), "--n", "5", *parts])


# ------------------------------------------------------------------------------------------------ code judge
def test_code_judge_grades_correct_and_wrong_programs():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from Uni_OPD_utils.outcome_reward.PRIME_code_server import judge_app

    client = TestClient(judge_app.app)
    assert client.get("/health").json() == {"status": "ok"}
    cases = {"inputs": ["3\n"], "outputs": ["6\n"]}
    good = "```python\nn = int(input())\nprint(n * 2)\n```"
    bad = "```python\nn = int(input())\nprint(n + 2)\n```"
    r = client.post("/judge", json={"completion": good, "test_cases": cases}).json()
    assert r["success"] is True and r["error"] is None
    r = client.post("/judge", json={"completion": bad, "test_cases": json.dumps(cases)}).json()
    assert r["success"] is False


# ------------------------------------------------------------------------------------------------ imports
@pytest.mark.parametrize("module", [
    "Uni_OPD_utils.mopd_hook.hook", "Uni_OPD_utils.OPD_reward.get_reward", "Uni_OPD_utils.OPD_reward.post_process_rewards",
    "Uni_OPD_utils.OPD_reward.rule_base_reward", "Uni_OPD_utils.OPD_reward.grpo_rule_reward",
    "Uni_OPD_utils.anchor_group_filter", "Uni_OPD_utils.seqkd.sft_rollout", "Uni_OPD_utils.ray_launcher",
    "Uni_OPD_utils.margin_calibration.margin_shift", "miles.utils.types", "miles.utils.ppo_utils",
    "miles.backends.training_utils.loss", "miles.backends.training_utils.data", "miles.backends.training_utils.log_utils",
    "miles.backends.sglang_utils.arguments", "miles.backends.megatron_utils.lora_utils", "miles.ray.rollout",
    "miles.ray.rollout_logs", "miles.rollout.sglang_rollout", "miles.backends.fsdp_utils.actor",
    "miles.backends.fsdp_utils.checkpoint", "miles.utils.arguments",
])
def test_import_smoke(module):
    __import__(module)


def test_uni_opd_utils_holds_upstream_plus_the_paper_modules_only():
    subdirs = sorted(p.name for p in (MILES / "Uni_OPD_utils").iterdir() if p.is_dir() and p.name != "__pycache__")
    assert subdirs == ["OPD_reward", "margin_calibration", "mopd_hook", "online_balance", "outcome_reward", "scripts",
                       "seqkd"]
    offenders = []
    for p in (MILES / "Uni_OPD_utils").rglob("*.py"):
        text = p.read_text()
        for bad in ("from exps.OPD", "from miles.miles.", "from Math.generate"):
            if bad in text:
                offenders.append(f"{p.relative_to(MILES)}: {bad}")
    # the only remaining `exps.` import is upstream's lazy OpenMMReasoner import for VLM domains (not used here)
    assert offenders == [], offenders
