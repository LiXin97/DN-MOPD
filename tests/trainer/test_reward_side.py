# SPDX-License-Identifier: Apache-2.0
"""Reward side of the trainer: all-teacher scoring, per-teacher log-prob parsing, the rule-based verifiers, the GRPO
reward and the bounded dynamic-sampling filter."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch


# ------------------------------------------------------------------------------------------------ all-teacher scoring
class FakeRM:
    def __init__(self, names, routed):
        self.names, self.routed = names, routed

    def teacher_names(self):
        return list(self.names)

    def resolve_real_name(self, alias):
        if alias not in self.routed:
            raise KeyError(alias)
        return self.routed[alias]

    def build_payload_for_teacher(self, sample, name):
        return {"input_ids": list(sample.tokens), "teacher": name}

    def get_next_url_for_teacher(self, name):
        return f"http://127.0.0.1:1/{name}"


def _fanout(monkeypatch, failing=(), alias="math"):
    from Uni_OPD_utils.OPD_reward import get_reward as gr

    async def fake_score(session_manager, name, get_url, payload):
        if name in failing:
            return {"teacher_name": name, "res": None, "url": get_url(), "failed": True, "reward_time": 0.0}
        itl = [[None, payload["input_ids"][0], None]] + [[-0.1 * (i + 1), t, None]
                                                           for i, t in enumerate(payload["input_ids"][1:])]
        return {"teacher_name": name, "res": {"meta_info": {"input_token_logprobs": itl}, "from": name},
                "url": get_url(), "failed": False, "reward_time": 0.0}

    monkeypatch.setattr(gr, "_score_one_teacher", fake_score)
    rm = FakeRM(["math_teacher", "code_teacher", "ifeval_teacher"], {"math": "math_teacher", "code": "code_teacher"})
    sample = SimpleNamespace(tokens=[1, 2, 3, 4], teacher_model_name=alias)
    return gr, asyncio.run(gr._get_reward_fanout(args=None, sample=sample, rm_manager=rm, session_manager=None,
                                                 start_time=0.0, response_correct=True, rule_based_metadata={}))


def test_fanout_scores_every_teacher_and_keeps_the_routed_legacy_fields(monkeypatch):
    gr, r = _fanout(monkeypatch)
    assert set(r["per_teacher"]) == {"math_teacher", "code_teacher", "ifeval_teacher"}
    assert r["routed_teacher"] == "math_teacher" and r["from"] == "math_teacher"
    assert gr.REWARD_FAILED_KEY not in r and r["teacher_url"].endswith("math_teacher")
    assert r["response_correct"] is True


def test_fanout_failure_of_another_teacher_is_not_a_sample_failure(monkeypatch):
    gr, r = _fanout(monkeypatch, failing=("ifeval_teacher",))
    assert gr.REWARD_FAILED_KEY not in r
    assert r["per_teacher"]["ifeval_teacher"]["failed"] and r["per_teacher"]["ifeval_teacher"]["input_token_logprobs"] is None


def test_fanout_failure_of_the_routed_teacher_marks_the_sample(monkeypatch):
    gr, r = _fanout(monkeypatch, failing=("math_teacher",))
    assert r[gr.REWARD_FAILED_KEY] is True and "teacher_url" not in r
    gr, r = _fanout(monkeypatch, alias="unknown-alias")
    assert r[gr.REWARD_FAILED_KEY] is True and r["routed_teacher"] is None
    assert not any(v["failed"] for v in r["per_teacher"].values())


# ------------------------------------------------------------------------------------------------ per-teacher parsing
def _itl(tokens, lp):
    return [[None, tokens[0], None]] + [[v, t, None] for v, t in zip(lp, tokens[1:])]


def test_post_process_parses_every_teacher_and_the_legacy_route():
    from miles.utils.types import Sample
    from Uni_OPD_utils.OPD_reward import post_process_rewards as pp

    tokens = [10, 11, 12, 20, 21, 22]
    s = Sample(prompt="p", index=0, group_index=0)
    s.tokens, s.response_length = tokens, 3
    lp = {"a": [-0.1, -0.2, -0.3, -0.4, -0.5], "b": [-1.0, -1.1, -1.2, -1.3, -1.4]}
    bad_echo = list(tokens)
    bad_echo[-1] = 99
    s.reward = {"meta_info": {"input_token_logprobs": _itl(tokens, lp["a"])}, "response_correct": False,
                "per_teacher": {"a": {"input_token_logprobs": _itl(tokens, lp["a"]), "failed": False},
                                "b": {"input_token_logprobs": _itl(tokens, lp["b"]), "failed": False},
                                "c": {"input_token_logprobs": None, "failed": True},
                                "d": {"input_token_logprobs": _itl(bad_echo, lp["b"]), "failed": False}}}
    out, _ = pp.post_process_rewards(None, [s])
    assert torch.allclose(out[0], torch.tensor([-0.3, -0.4, -0.5]))
    assert torch.allclose(s.per_teacher_log_probs["a"], torch.tensor([-0.3, -0.4, -0.5]))
    assert torch.allclose(s.per_teacher_log_probs["b"], torch.tensor([-1.2, -1.3, -1.4]))
    assert torch.equal(s.per_teacher_log_probs["c"], torch.full((3,), pp.TEACHER_LOGP_FAILED_SENTINEL))
    assert torch.equal(s.per_teacher_log_probs["d"], torch.full((3,), pp.TEACHER_LOGP_FAILED_SENTINEL))
    assert s.reward["per_teacher"]["d"]["failed"] is True
    assert all(v["input_token_logprobs"] is None for v in s.reward["per_teacher"].values())   # raw payload dropped
    assert s.response_correct is False


def test_post_process_marks_a_failed_routed_teacher_with_the_sentinel():
    from miles.utils.types import Sample
    from Uni_OPD_utils.OPD_reward import get_reward as gr
    from Uni_OPD_utils.OPD_reward import post_process_rewards as pp

    s = Sample(prompt="p", index=0, group_index=0)
    s.tokens, s.response_length = [1, 2, 3, 4], 2
    s.reward = {gr.REWARD_FAILED_KEY: True, "response_correct": None, "meta_info": {"input_token_logprobs": []}}
    out, _ = pp.post_process_rewards(None, [s])
    assert torch.equal(out[0], torch.full((2,), pp.TEACHER_LOGP_FAILED_SENTINEL))
    assert s.per_teacher_log_probs is None                        # legacy route: no all-teacher scores


# ------------------------------------------------------------------------------------------------ verifiers
def _sample(domain, response, label, teacher=None):
    return SimpleNamespace(response=response, label=label, metadata={"domain": domain},
                           teacher_model_name=teacher or domain)


def test_ifeval_verifier_is_strict():
    from Uni_OPD_utils.OPD_reward.rule_base_reward import get_rule_based_reward

    label = json.dumps([{"type": "starts_with", "value": "In short:"}, {"type": "no_commas"}])
    ok, meta = asyncio.run(get_rule_based_reward(None, _sample("ifeval", "In short: yes indeed", label)))
    assert ok is True and meta["satisfied"] == 2
    ok, meta = asyncio.run(get_rule_based_reward(None, _sample("ifeval", "In short: yes, indeed", label)))
    assert ok is False and meta["satisfied"] == 1


def test_math_verifier_runs_bounded(monkeypatch):
    from Uni_OPD_utils.OPD_reward.rule_base_reward import get_rule_based_reward

    ok, _ = asyncio.run(get_rule_based_reward(None, _sample("math", "so \\boxed{42}", "42")))
    assert ok is True
    ok, _ = asyncio.run(get_rule_based_reward(None, _sample("math", "so \\boxed{41}", "42")))
    assert ok is False


def test_verifier_domain_comes_from_metadata():
    from Uni_OPD_utils.OPD_reward.rule_base_reward import _verifier_domain

    assert _verifier_domain(_sample("ifeval", "", "", teacher="code")) == "ifeval"
    assert _verifier_domain(SimpleNamespace(metadata={}, teacher_model_name="Code")) == "code"


def test_code_judge_error_is_no_verdict(monkeypatch):
    from Uni_OPD_utils.OPD_reward import rule_base_reward as rbr
    from Uni_OPD_utils.outcome_reward.PRIME_code_server.server import CodeJudgeResponse

    class Client:
        def __init__(self, resp):
            self.resp = resp

        def judge(self, completion, tests):
            return self.resp

    monkeypatch.setattr(rbr, "_get_code_judge_client", lambda: Client(CodeJudgeResponse(success=0.0, error="timeout")))
    ok, _ = asyncio.run(rbr.get_rule_based_reward(None, _sample("code", "x", "{}")))
    assert ok is None
    monkeypatch.setattr(rbr, "_get_code_judge_client", lambda: Client(CodeJudgeResponse(success=True, metadata={})))
    ok, _ = asyncio.run(rbr.get_rule_based_reward(None, _sample("code", "x", "{}")))
    assert ok is True


def test_code_judge_endpoints_path(monkeypatch, tmp_path):
    import importlib

    import Uni_OPD_utils.outcome_reward.PRIME_code_server.server as srv

    default = importlib.reload(srv).CODE_JUDGE_SERVER_LIST
    assert default.endswith("available_endpoints.json") and json.load(open(default)) == ["127.0.0.1:17580"]
    f = tmp_path / "ep.json"
    f.write_text('["127.0.0.1:18000"]')
    monkeypatch.setenv("CODE_JUDGE_ENDPOINTS", str(f))
    assert importlib.reload(srv).get_judge_servers() == ["127.0.0.1:18000"]
    monkeypatch.delenv("CODE_JUDGE_ENDPOINTS")
    importlib.reload(srv)


# ------------------------------------------------------------------------------------------------ GRPO reward
def test_grpo_reward_per_domain(monkeypatch):
    from Uni_OPD_utils.OPD_reward import grpo_rule_reward as g

    label = json.dumps([{"type": "all_lowercase"}])
    assert asyncio.run(g.grpo_rule_reward(None, _sample("ifeval", "all lower", label))) == 1.0
    assert asyncio.run(g.grpo_rule_reward(None, _sample("ifeval", "Not lower", label))) == 0.0
    with pytest.raises(KeyError):                                  # a malformed label is a data defect
        asyncio.run(g.grpo_rule_reward(None, _sample("ifeval", "x", json.dumps([{"type": "nope"}]))))
    assert asyncio.run(g.grpo_rule_reward(None, _sample("math", "\\boxed{7}", "7"))) == 1.0
    assert asyncio.run(g.grpo_rule_reward(None, _sample("math", "\\boxed{8}", "7"))) == 0.0

    for verdict, reward in ((None, 0.0), (True, 1.0), (False, 0.0), (0.5, 0.0), (1.0, 1.0)):
        async def fake(args, sample, v=verdict):
            return v, {}
        monkeypatch.setattr(g, "get_rule_based_reward", fake)
        assert asyncio.run(g.grpo_rule_reward(None, _sample("code", "x", "{}"))) == reward


# ------------------------------------------------------------------------------------------------ dynamic sampling
def _group(rewards):
    return [SimpleNamespace(get_reward_value=lambda args, r=r: r) for r in rewards]


def test_bounded_dynamic_sampling_filter(monkeypatch):
    from Uni_OPD_utils import anchor_group_filter as f

    monkeypatch.setenv("DYNAMIC_SAMPLING_MAX_GEN_BATCHES", "2")
    args = SimpleNamespace(rollout_batch_size=2)
    f.reset_state()
    assert not f.check_reward_nonzero_std_bounded(args, _group([0, 0, 0])).keep       # zero variance: dropped
    assert f.check_reward_nonzero_std_bounded(args, _group([0, 1, 0])).keep
    assert not f.check_reward_nonzero_std_bounded(args, _group([1, 1, 1])).keep
    assert not f.check_reward_nonzero_std_bounded(args, _group([0, 0, 0])).keep       # seen = 4 = budget
    out = f.check_reward_nonzero_std_bounded(args, _group([0, 0, 0]))                 # past 2 x 2 groups: released
    assert out.keep and f.get_state()["released"] == 1
    assert f.check_reward_nonzero_std_bounded(args, _group([1, 1])).keep is False     # kept == 2 -> new rollout
    assert f.get_state() == {"seen": 1, "kept": 0, "dropped": 1, "released": 0}
