# SPDX-License-Identifier: Apache-2.0
"""recipes/qwen3.5: every METHOD's trainer argv and hook config carry the paper's hyperparameters, and miles' own
argument parser accepts every argv the recipes build."""
import json
import subprocess
import sys

import pytest

import recipe
from conftest import RECIPES

METHODS = ("label", "label_hook", "dn_mopd", "single_math", "single_code", "single_if", "uniform_pool",
           "dynamic_router", "annealed_injection", "fixed_w", "fixed_math", "fixed_if", "frozen_update0", "observe")
SIZES = ("2b", "4b", "9b")


def argmap(argv):
    """{flag: value or True} of an argv list (a flag followed by another flag is a switch)."""
    out, i = {}, 0
    while i < len(argv):
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[argv[i]] = argv[i + 1]
            i += 2
        else:
            out[argv[i]] = True
            i += 1
    return out


def test_the_method_table_is_the_paper_set():
    assert tuple(sorted(recipe.methods())) == tuple(sorted(METHODS))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("method", METHODS)
def test_student_argv_is_the_paper_recipe(method, size):
    a = argmap(recipe.student_argv(method=method, size=size, seed=43, updates=80, student="S", data="D", save="C"))
    want = {"--train-backend": "fsdp", "--attn-implementation": "sdpa", "--actor-num-gpus-per-node": "4",
            "--rollout-num-gpus": "4", "--rollout-num-gpus-per-engine": "1", "--colocate": True,
            "--sglang-mem-fraction-static": "0.6", "--hf-checkpoint": "S", "--load": "C", "--save": "C",
            "--save-interval": "10", "--prompt-data": "D", "--input-key": "prompt", "--label-key": "label",
            "--apply-chat-template": True, "--apply-chat-template-kwargs": '{"enable_thinking": false}',
            "--rollout-shuffle": True, "--rollout-seed": "43", "--num-rollout": "80", "--rollout-batch-size": "64",
            "--n-samples-per-prompt": "8", "--global-batch-size": "512", "--micro-batch-size": "1",
            "--balance-data": True, "--rollout-max-prompt-len": "2048", "--rollout-max-response-len": "8192",
            "--rollout-temperature": "1", "--advantage-estimator": "on_policy_distillation",
            "--kl-loss-coef": "0", "--entropy-coef": "0", "--optimizer": "adam", "--lr": "1e-06",
            "--lr-decay-style": "constant", "--lr-warmup-iters": "5", "--weight-decay": "0.1",
            "--adam-beta1": "0.9", "--adam-beta2": "0.98", "--clip-grad": "1"}
    for k, v in want.items():
        assert a.get(k) == v, (k, a.get(k), v)
    assert "--seed" not in a                                   # --seed stays at the miles default (1234)
    assert "--custom-convert-samples-to-train-data-path" not in a
    route = recipe.method_spec(method)["route"]
    hook = "Uni_OPD_utils.mopd_hook.hook"
    legacy = ("Uni_OPD_utils.OPD_reward.get_reward.get_reward", "Uni_OPD_utils.OPD_reward.post_process_rewards.post_process_rewards")
    if route == "hook":
        assert (a["--custom-rm-path"], a["--custom-reward-post-process-path"]) == (f"{hook}.get_reward", f"{hook}.post_process_rewards")
    else:
        assert (a["--custom-rm-path"], a["--custom-reward-post-process-path"]) == legacy


EXPECTED_HOOK = {
    "label_hook": {"mode": "label"},
    "dn_mopd": {"mode": "label", "adv_norm": "per_domain_scale"},
    "uniform_pool": {"mode": "pool"},
    "dynamic_router": {"mode": "dynamic"},
    "annealed_injection": {"mode": "label", "inject_adv_const": 1.0, "inject_anneal_steps": 40},
    "fixed_w": {"mode": "label", "adv_norm": "fixed_domain_scale",
                "adv_norm_fixed": {"math": 2.0, "code": 1.0, "ifeval": 0.25}},
    "fixed_math": {"mode": "label", "adv_norm": "fixed_domain_scale",
                   "adv_norm_fixed": {"math": 2.0, "code": 1.0, "ifeval": 1.0}},
    "fixed_if": {"mode": "label", "adv_norm": "fixed_domain_scale",
                 "adv_norm_fixed": {"math": 1.0, "code": 1.0, "ifeval": 0.25}},
    "observe": {"mode": "label", "adv_norm": "observe_domain_scale"},
}
FROZEN = {"9b": {"math": 1.768, "code": 0.888, "ifeval": 0.25}, "4b": {"math": 2.009, "code": 0.811, "ifeval": 0.341},
          "2b": {"math": 2.102, "code": 0.822, "ifeval": 0.431}}
HOOK_KEYS = {"mode", "adv_norm", "adv_norm_fixed", "inject_adv_const", "inject_anneal_steps"}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("method", METHODS)
def test_hook_config_per_method(method, size, tmp_path):
    spec = recipe.method_spec(method)
    if spec["route"] == "legacy":
        with pytest.raises(recipe.RecipeError):
            recipe.hook_config(method=method, size=size, seed=42, updates=80, audit_dir="A")
        return
    bank = None
    if method == "annealed_injection":
        with pytest.raises(recipe.RecipeError, match="injection bank"):
            recipe.hook_config(method=method, size=size, seed=42, updates=80, audit_dir="A")
        bank = tmp_path / "bank.pt"
        bank.write_bytes(b"bank")
    cfg = recipe.hook_config(method=method, size=size, seed=42, updates=80, audit_dir="A",
                             bank=str(bank) if bank else None)
    want = dict(EXPECTED_HOOK.get(method) or {"mode": "label", "adv_norm": "fixed_domain_scale",
                                              "adv_norm_fixed": FROZEN[size]})
    assert {k: v for k, v in cfg.items() if k in HOOK_KEYS} == want
    assert cfg["n_samples"] == 8 and cfg["teachers"] == ["code_teacher", "ifeval_teacher", "math_teacher"]
    assert cfg["label_map"] == {"default": "code_teacher", "math": "math_teacher", "code": "code_teacher",
                                "ifeval": "ifeval_teacher"}
    if bank:
        assert cfg["inject_bank_path"] == str(bank) and len(cfg["inject_bank_sha256"]) == 64
    else:
        with pytest.raises(recipe.RecipeError, match="no injection bank"):
            recipe.hook_config(method=method, size=size, seed=42, updates=80, audit_dir="A", bank=__file__)


def test_hook_configs_are_accepted_by_the_hook(hook_module, tmp_path):
    for method in METHODS:
        if recipe.method_spec(method)["route"] != "hook" or method == "annealed_injection":
            continue
        cfg = recipe.hook_config(method=method, size="4b", seed=42, updates=80, audit_dir=str(tmp_path / method))
        p = tmp_path / f"{method}.json"
        p.write_text(json.dumps(cfg))
        hook = hook_module.MopdHook(p)
        assert hook.mode == cfg["mode"] and hook.adv_norm == cfg.get("adv_norm")


@pytest.mark.parametrize("method,target", [("label", None), ("single_math", "math_teacher"),
                                           ("single_code", "code_teacher"), ("single_if", "ifeval_teacher"),
                                           ("dn_mopd", None), ("uniform_pool", None)])
def test_server_files_route_or_collapse(method, target, tmp_path):
    lp, mp = recipe.server_files(method=method, teachers={"math": "/m", "code": "/c", "ifeval": "/i"},
                                 out_dir=str(tmp_path / method))
    listing, mapping = json.loads(lp.read_text()), json.loads(mp.read_text())
    assert listing == {"math_teacher": {"path": "/m", "servers": ["http://127.0.0.1:13141"]},
                       "code_teacher": {"path": "/c", "servers": ["http://127.0.0.1:13142"]},
                       "ifeval_teacher": {"path": "/i", "servers": ["http://127.0.0.1:13140"]}}
    if target is None:
        assert mapping == {"default": "code_teacher", "math": "math_teacher", "code": "code_teacher",
                           "ifeval": "ifeval_teacher"}
    else:
        assert set(mapping) == {"default", "math", "code", "ifeval"} and set(mapping.values()) == {target}


def test_teacher_servers_layout():
    assert recipe.teacher_servers() == [("math", 4, 13141, 0.35, "math_teacher"), ("code", 5, 13142, 0.35, "code_teacher"),
                                        ("ifeval", 6, 13140, 0.85, "ifeval_teacher")]


UPDATES = {"9b": {"math": 250, "code": 200, "ifeval": 400}, "4b": {"math": 320, "code": 300, "ifeval": 400},
           "2b": {"math": 400, "code": 400, "ifeval": 400}}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("domain", ("math", "code", "ifeval"))
def test_teacher_argv_is_the_paper_recipe(domain, size):
    a = argmap(recipe.teacher_argv(domain=domain, size=size, base="B", data="D", save="C"))
    want = {"--actor-num-gpus-per-node": "8", "--rollout-num-gpus": "8", "--colocate": True,
            "--sglang-mem-fraction-static": "0.6", "--num-rollout": str(UPDATES[size][domain]),
            "--rollout-batch-size": "128", "--n-samples-per-prompt": "8", "--global-batch-size": "256",
            "--micro-batch-size": "1", "--rollout-seed": "42", "--rollout-max-prompt-len": "2048",
            "--rollout-max-response-len": "8192", "--rollout-temperature": "1", "--advantage-estimator": "grpo",
            "--custom-rm-path": "Uni_OPD_utils.OPD_reward.grpo_rule_reward.grpo_rule_reward",
            "--dynamic-sampling-filter-path": "Uni_OPD_utils.anchor_group_filter.check_reward_nonzero_std_bounded",
            "--over-sampling-batch-size": "128", "--gradient-checkpointing": True, "--log-probs-chunk-size": "2048",
            "--recompute-loss-function": True, "--kl-loss-coef": "0", "--entropy-coef": "0", "--lr": "1e-06",
            "--lr-decay-style": "constant", "--lr-warmup-iters": "10", "--weight-decay": "0.1",
            "--adam-beta1": "0.9", "--adam-beta2": "0.98", "--clip-grad": "1", "--save-interval": "10"}
    for k, v in want.items():
        assert a.get(k) == v, (k, a.get(k), v)
    assert "--seed" not in a and recipe.teacher_updates(size, domain) == UPDATES[size][domain]
    with pytest.raises(recipe.RecipeError):
        recipe.teacher_argv(domain=domain, size=size, base="B", data="D", save="C", updates=401)


def test_sft_argv_is_the_paper_recipe():
    a = argmap(recipe.sft_argv(size="9b", student="S", data="T", save="C", n_rows=2700))
    want = {"--debug-train-only": True, "--actor-num-gpus-per-node": "8", "--input-key": "messages",
            "--rollout-function-path": "Uni_OPD_utils.seqkd.sft_rollout.generate_rollout",
            "--loss-type": "sft_loss", "--calculate-per-token-loss": True,
            "--disable-compute-advantages-and-returns": True, "--num-rollout": "84", "--rollout-batch-size": "128",
            "--n-samples-per-prompt": "1", "--global-batch-size": "128", "--micro-batch-size": "1",
            "--rollout-max-prompt-len": "8192", "--rollout-max-response-len": "16384", "--rollout-seed": "42",
            "--lr": "1e-05", "--lr-decay-style": "cosine", "--min-lr": "1e-06", "--lr-warmup-iters": "5",
            "--weight-decay": "0.1", "--adam-beta1": "0.9", "--adam-beta2": "0.98", "--clip-grad": "1",
            "--save-interval": "42", "--gradient-checkpointing": True, "--log-probs-chunk-size": "4096"}
    for k, v in want.items():
        assert a.get(k) == v, (k, a.get(k), v)


def test_generation_and_merge_settings():
    assert recipe.get_value("generation", "seqkd") == {
        "engine": "vllm", "gpus": [0, 1, 2], "temperature": 1.0, "top_p": 1.0, "n": 1, "seed": 42,
        "max_tokens": 16384, "max_model_len": 18432, "max_num_seqs": 256, "gpu_memory_utilization": 0.9,
        "enforce_eager": True, "chat_template_kwargs": {"enable_thinking": False}}
    merge = recipe.load("merge")
    assert merge["task_arithmetic_lambda"] == 1.0 and merge["base_only_prefix"] == "mtp."
    bank = recipe.load("injection_bank")["generation"]
    assert (bank["n"], bank["max_tokens"], bank["temperature"], bank["top_p"], bank["seed"]) == (4, 8192, 1.0, 1.0, 42)


def test_controls_match_the_paper():
    c = recipe.load("controls")
    assert c["frozen_update0_multipliers"] == FROZEN
    assert c["fixed_weights"] == {"fix21q": {"math": 2.0, "code": 1.0, "ifeval": 0.25},
                                  "fixmath": {"math": 2.0, "code": 1.0, "ifeval": 1.0},
                                  "fixif": {"math": 1.0, "code": 1.0, "ifeval": 0.25}}


def test_smoke_profile_only_shortens_the_run():
    paper = argmap(recipe.student_argv(method="dn_mopd", size="2b", seed=42, updates=2, student="S", data="D", save="C"))
    smoke = argmap(recipe.student_argv(method="dn_mopd", size="2b", seed=42, updates=2, student="S", data="D", save="C",
                                       profile="smoke"))
    changed = {k for k in paper if paper[k] != smoke.get(k)}
    assert changed == {"--rollout-batch-size", "--global-batch-size", "--rollout-max-response-len", "--save-interval"}
    assert paper["--lr-warmup-iters"] == smoke["--lr-warmup-iters"] == "1"   # warm-up is clamped below a 2-update run
    with pytest.raises(recipe.RecipeError):
        recipe.with_profile({"a": 1}, {"p": {"b": 2}}, "p")


def test_recipe_cli_check():
    out = subprocess.run([sys.executable, str(RECIPES / "lib" / "recipe.py"), "check"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("RECIPE_CHECK_OK"), out.stderr
    out = subprocess.run([sys.executable, str(RECIPES / "lib" / "recipe.py"), "student-route", "--method", "nope"],
                         capture_output=True, text=True)
    assert out.returncode == 2 and "unknown method" in out.stderr


def test_miles_parses_every_recipe_argv():
    """miles.utils.arguments.parse_args accepts the argv of every method, teacher and SFT recipe, with the paper's
    effective values (PPO clip 0.2/0.2, --seed 1234)."""
    parse_args = pytest.importorskip("miles.utils.arguments").parse_args
    old = sys.argv
    try:
        for method in METHODS:
            sys.argv = ["train.py"] + recipe.student_argv(method=method, size="2b", seed=44, updates=160,
                                                          student="/nonexistent/Qwen3.5-2B", data="/d.jsonl", save="/s")
            a = parse_args()
            assert (a.advantage_estimator, a.num_rollout, a.rollout_seed, a.seed) == ("on_policy_distillation", 160, 44, 1234)
            assert (a.eps_clip, a.eps_clip_high, a.lr, a.global_batch_size, a.n_samples_per_prompt) == (0.2, 0.2, 1e-6, 512, 8)
            assert a.rollout_max_response_len == 8192 and a.kl_loss_coef == 0.0 and a.colocate
        for domain in ("math", "code", "ifeval"):
            sys.argv = ["train.py"] + recipe.teacher_argv(domain=domain, size="4b", base="/nonexistent", data="/d",
                                                          save="/s")
            a = parse_args()
            assert (a.advantage_estimator, a.global_batch_size, a.rollout_batch_size, a.lr_warmup_iters) == ("grpo", 256, 128, 10)
            assert a.dynamic_sampling_filter_path.endswith("check_reward_nonzero_std_bounded") and a.seed == 1234
        sys.argv = ["train.py"] + recipe.sft_argv(size="2b", student="/nonexistent", data="/d.jsonl", save="/s",
                                                  n_rows=2700)
        a = parse_args()
        assert (a.loss_type, a.num_rollout, a.lr, a.min_lr, a.debug_train_only) == ("sft_loss", 84, 1e-5, 1e-6, True)
    finally:
        sys.argv = old
