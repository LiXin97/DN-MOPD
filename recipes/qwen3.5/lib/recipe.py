# SPDX-License-Identifier: Apache-2.0
"""Recipe configuration for the Qwen3.5 rows: turns configs/*.yaml into trainer argv, hook configs and teacher-server
files. The shell recipes call this module; the argv it prints is what miles receives, one argument per line.

    python recipe.py methods
    python recipe.py teacher-argv   --domain D --size S --base B --data F --save DIR [--updates N] [--profile smoke]
    python recipe.py student-argv   --method M --size S --seed N --updates U --student B --data F --save DIR
                                    [--profile smoke]
    python recipe.py student-route  --method M                 (prints: legacy|hook <rm path> <post-process path>)
    python recipe.py hook-config    --method M --size S --seed N --updates U --audit-dir DIR --out FILE
                                    [--bank FILE] [--profile smoke]
    python recipe.py server-files   --method M --math P --code P --ifeval P --out-dir DIR
    python recipe.py teacher-servers                           (prints: domain gpu port mem_fraction teacher_name)
    python recipe.py sft-argv       --size S --student B --data F --save DIR --n-rows N [--profile smoke]
    python recipe.py get            SECTION.KEY... [--file seqkd] [--profile smoke]
    python recipe.py check
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SIZES = ("2b", "4b", "9b")
DOMAINS = ("math", "code", "ifeval")
LEGACY_RM = "Uni_OPD_utils.OPD_reward.get_reward.get_reward"
LEGACY_POST = "Uni_OPD_utils.OPD_reward.post_process_rewards.post_process_rewards"
HOOK_RM = "Uni_OPD_utils.mopd_hook.hook.get_reward"
HOOK_POST = "Uni_OPD_utils.mopd_hook.hook.post_process_rewards"
GRPO_RM = "Uni_OPD_utils.OPD_reward.grpo_rule_reward.grpo_rule_reward"
DS_FILTER = "Uni_OPD_utils.anchor_group_filter.check_reward_nonzero_std_bounded"
SFT_ROLLOUT = "Uni_OPD_utils.seqkd.sft_rollout.generate_rollout"
CHAT_KWARGS = json.dumps({"enable_thinking": False})


class RecipeError(ValueError):
    """A request the paper recipes do not define."""


def load(name: str) -> dict:
    """configs/<name>.yaml (or .json)."""
    for ext in (".yaml", ".json"):
        path = CONFIG_DIR / f"{name}{ext}"
        if path.is_file():
            text = path.read_text()
            return json.loads(text) if ext == ".json" else yaml.safe_load(text)
    raise RecipeError(f"no config {name!r} under {CONFIG_DIR}")


def with_profile(section: dict, profiles: dict | None, profile: str | None) -> dict:
    """The section with the named profile's overrides applied (profile None = the paper values)."""
    out = copy.deepcopy(section)
    if profile:
        if not profiles or profile not in profiles:
            raise RecipeError(f"unknown profile {profile!r}")
        for key, value in profiles[profile].items():
            if key not in out:
                raise RecipeError(f"profile {profile!r} overrides unknown key {key!r}")
            out[key] = value
    return out


def _num(x) -> str:
    """Numbers as the trainer receives them (1e-06, 0.6, 128)."""
    if isinstance(x, bool):
        raise RecipeError("a boolean is not a numeric trainer argument")
    if isinstance(x, int):
        return str(x)
    return f"{float(x):g}"


def check_size(size: str) -> None:
    if size not in SIZES:
        raise RecipeError(f"size must be one of {SIZES}, got {size!r}")


def check_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise RecipeError(f"domain must be one of {DOMAINS}, got {domain!r}")


def methods() -> dict:
    return load("methods")["methods"]


def method_spec(method: str) -> dict:
    table = methods()
    if method not in table:
        raise RecipeError(f"unknown method {method!r}; one of {sorted(table)}")
    return table[method]


def _warmup(value: int, updates: int) -> int:
    """miles requires warm-up < decay steps; a run shorter than the warm-up keeps updates - 1 warm-up steps."""
    return value if value < updates else max(updates - 1, 0)


def _optimizer_args(r: dict, updates: int) -> list[str]:
    return ["--optimizer", r["optimizer"], "--lr", _num(r["lr"]), "--lr-decay-style", r["lr_decay_style"],
            "--lr-warmup-iters", _num(_warmup(int(r["lr_warmup_iters"]), updates)),
            "--weight-decay", _num(r["weight_decay"]), "--adam-beta1", _num(r["adam_beta1"]),
            "--adam-beta2", _num(r["adam_beta2"]), "--clip-grad", _num(r["clip_grad"])]


# ------------------------------------------------------------------------------------------------ GRPO teachers
def teacher_updates(size: str, domain: str) -> int:
    check_size(size)
    check_domain(domain)
    return int(load("teacher_grpo")["updates_used"][size][domain])


def teacher_argv(*, domain: str, size: str, base: str, data: str, save: str, updates: int | None = None,
                 profile: str | None = None) -> list[str]:
    cfg = load("teacher_grpo")
    r = with_profile(cfg["recipe"], cfg.get("profiles"), profile)
    n = int(updates) if updates is not None else teacher_updates(size, domain)
    if not 1 <= n <= int(r["max_updates"]):
        raise RecipeError(f"teacher updates must be in [1, {r['max_updates']}], got {n}")
    g = int(r["gpus"])
    argv = ["--train-backend", "fsdp", "--attn-implementation", "sdpa",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(g), "--num-gpus-per-node", str(g),
            "--colocate", "--rollout-num-gpus", str(g), "--rollout-num-gpus-per-engine", "1",
            "--sglang-mem-fraction-static", _num(r["sglang_mem_fraction_static"]),
            "--hf-checkpoint", base, "--load", save, "--save", save, "--save-interval", _num(r["save_interval"]),
            "--prompt-data", data, "--input-key", "prompt", "--label-key", "label",
            "--apply-chat-template", "--apply-chat-template-kwargs", CHAT_KWARGS,
            "--rollout-shuffle", "--rollout-seed", _num(r["rollout_seed"]),
            "--num-rollout", str(n), "--rollout-batch-size", _num(r["rollout_batch_size"]),
            "--n-samples-per-prompt", _num(r["n_samples_per_prompt"]),
            "--global-batch-size", _num(r["global_batch_size"]), "--micro-batch-size", _num(r["micro_batch_size"]),
            "--balance-data"]
    if r["dynamic_sampling"]:
        argv += ["--dynamic-sampling-filter-path", DS_FILTER,
                 "--over-sampling-batch-size", _num(r["over_sampling_batch_size"])]
    argv += [str(a) for a in r["memory_args"]]
    argv += ["--rollout-max-prompt-len", _num(r["max_prompt_len"]),
             "--rollout-max-response-len", _num(r["max_response_len"]),
             "--rollout-temperature", _num(r["temperature"]),
             "--advantage-estimator", r["advantage_estimator"],
             "--custom-rm-path", GRPO_RM,
             "--kl-loss-coef", _num(r["kl_loss_coef"]), "--entropy-coef", _num(r["entropy_coef"])]
    argv += _optimizer_args(r, n)
    return argv


# ------------------------------------------------------------------------------------------------ OPD students
def student_route(method: str) -> tuple[str, str, str]:
    spec = method_spec(method)
    if spec["route"] == "legacy":
        return "legacy", LEGACY_RM, LEGACY_POST
    if spec["route"] == "hook":
        return "hook", HOOK_RM, HOOK_POST
    raise RecipeError(f"{method}: unknown route {spec['route']!r}")


def student_argv(*, method: str, size: str, seed: int, updates: int, student: str, data: str, save: str,
                 profile: str | None = None) -> list[str]:
    check_size(size)
    cfg = load("student_opd")
    r = with_profile(cfg["recipe"], cfg.get("profiles"), profile)
    if int(updates) < 1:
        raise RecipeError(f"updates must be positive, got {updates}")
    _, rm, post = student_route(method)
    g = len(r["student_gpus"])
    argv = ["--train-backend", "fsdp", "--attn-implementation", "sdpa",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(g), "--num-gpus-per-node", str(g),
            "--colocate", "--rollout-num-gpus", str(g), "--rollout-num-gpus-per-engine", "1",
            "--sglang-mem-fraction-static", _num(r["sglang_mem_fraction_static"]),
            "--hf-checkpoint", student, "--load", save, "--save", save, "--save-interval", _num(r["save_interval"]),
            "--prompt-data", data, "--input-key", "prompt", "--label-key", "label",
            "--apply-chat-template", "--apply-chat-template-kwargs", CHAT_KWARGS,
            "--rollout-shuffle", "--rollout-seed", str(int(seed)),
            "--num-rollout", str(int(updates)), "--rollout-batch-size", _num(r["rollout_batch_size"]),
            "--n-samples-per-prompt", _num(r["n_samples_per_prompt"]),
            "--global-batch-size", _num(r["global_batch_size"]), "--micro-batch-size", _num(r["micro_batch_size"]),
            "--balance-data",
            "--rollout-max-prompt-len", _num(r["max_prompt_len"]),
            "--rollout-max-response-len", _num(r["max_response_len"]),
            "--rollout-temperature", _num(r["temperature"]),
            "--custom-rm-path", rm, "--custom-reward-post-process-path", post,
            "--advantage-estimator", r["advantage_estimator"],
            "--kl-loss-coef", _num(r["kl_loss_coef"]), "--entropy-coef", _num(r["entropy_coef"])]
    argv += _optimizer_args(r, int(updates))
    return argv


def _resolve_fixed(ref: str, size: str) -> dict:
    """'fixed_weights.fix21q' / 'frozen_update0_multipliers.SIZE' -> {math, code, ifeval} from controls.json."""
    controls = load("controls")
    section, _, key = ref.partition(".")
    key = size if key == "SIZE" else key
    try:
        value = controls[section][key]
    except KeyError as e:
        raise RecipeError(f"controls.json has no {section}.{key}") from e
    return {d: float(value[d]) for d in DOMAINS}


def hook_config(*, method: str, size: str, seed: int, updates: int, audit_dir: str, bank: str | None = None,
                profile: str | None = None) -> dict:
    """The run config Uni_OPD_utils.mopd_hook reads (MOPD_HOOK_CONFIG)."""
    check_size(size)
    spec = method_spec(method)
    if spec["route"] != "hook":
        raise RecipeError(f"{method} runs on the legacy route and takes no hook config")
    models = load("models")
    student = with_profile(load("student_opd")["recipe"], load("student_opd").get("profiles"), profile)
    cfg = {
        "method": method,
        "n_samples": int(student["n_samples_per_prompt"]),
        "teachers": sorted(models["domains"][d]["teacher_name"] for d in DOMAINS),
        "label_map": dict(models["label_map"]),
        "audit_dir": audit_dir,
        "extra_score_rounds": 3,
        "recipe": {"size": size, "seed": int(seed), "updates": int(updates), "profile": profile or "paper"},
    }
    for key, value in spec["hook"].items():
        if key == "adv_norm_fixed":
            value = _resolve_fixed(value["from_controls"], size)
        cfg[key] = value
    if spec.get("needs_injection_bank"):
        if not bank:
            raise RecipeError(f"{method} needs an injection bank (recipes/qwen3.5/build_injection_bank.sh)")
        cfg["inject_bank_path"] = bank
        cfg["inject_bank_sha256"] = hashlib.sha256(Path(bank).read_bytes()).hexdigest()
    elif bank:
        raise RecipeError(f"{method} takes no injection bank")
    return cfg


def server_files(*, method: str, teachers: dict, out_dir: str, host: str = "127.0.0.1") -> tuple[Path, Path]:
    """Write teacher_server_list.json ({name: {path, servers}}) and teacher_server_map.json (the router: the label map,
    or every key collapsed to one teacher for a single-teacher method)."""
    models = load("models")
    servers = load("student_opd")["teacher_servers"]
    names = {d: models["domains"][d]["teacher_name"] for d in DOMAINS}
    listing = {names[d]: {"path": str(teachers[d]), "servers": [f"http://{host}:{servers[d]['port']}"]}
               for d in DOMAINS}
    route = dict(models["label_map"])
    spec = method_spec(method)
    lm = spec.get("label_map", "route")
    if isinstance(lm, dict) and "collapse_to" in lm:
        check_domain(lm["collapse_to"])
        target = names[lm["collapse_to"]]
        route = {k: target for k in route}
    elif lm != "route":
        raise RecipeError(f"{method}: unknown label_map {lm!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    list_path, map_path = out / "teacher_server_list.json", out / "teacher_server_map.json"
    list_path.write_text(json.dumps(listing, indent=2) + "\n")
    map_path.write_text(json.dumps(route, indent=2) + "\n")
    return list_path, map_path


def teacher_servers() -> list[tuple[str, int, int, float, str]]:
    models = load("models")
    s = load("student_opd")["teacher_servers"]
    return [(d, int(s[d]["gpu"]), int(s[d]["port"]), float(s[d]["mem_fraction_static"]),
             models["domains"][d]["teacher_name"]) for d in DOMAINS]


# ------------------------------------------------------------------------------------------------ SeqKD-SFT
def sft_argv(*, size: str, student: str, data: str, save: str, n_rows: int, profile: str | None = None) -> list[str]:
    check_size(size)
    cfg = load("seqkd")
    r = with_profile(cfg["sft"], {k: {kk: vv for kk, vv in v.items() if kk in cfg["sft"]}
                                  for k, v in (cfg.get("profiles") or {}).items()}, profile)
    updates = int(r["epochs"]) * int(n_rows) // int(r["rollout_batch_size"])
    if updates < 1:
        raise RecipeError(f"{n_rows} rows give no SFT update at rollout batch {r['rollout_batch_size']}")
    g = int(r["gpus"])
    if int(r["global_batch_size"]) % g:
        raise RecipeError("the SFT global batch must be divisible by the number of GPUs")
    argv = ["--train-backend", "fsdp", "--attn-implementation", "sdpa",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(g), "--num-gpus-per-node", str(g),
            "--debug-train-only",
            "--hf-checkpoint", student, "--load", save, "--save", save,
            "--save-interval", _num(min(int(r["save_interval"]), updates)),
            "--prompt-data", data, "--input-key", "messages", "--label-key", "label",
            "--rollout-shuffle", "--rollout-seed", _num(r["rollout_seed"]),
            "--rollout-function-path", SFT_ROLLOUT,
            "--loss-type", r["loss_type"], "--calculate-per-token-loss", "--disable-compute-advantages-and-returns",
            "--num-rollout", str(updates), "--rollout-batch-size", _num(r["rollout_batch_size"]),
            "--n-samples-per-prompt", "1", "--global-batch-size", _num(r["global_batch_size"]),
            "--micro-batch-size", _num(r["micro_batch_size"]),
            "--rollout-max-prompt-len", _num(r["max_prompt_len"]),
            "--rollout-max-response-len", _num(r["max_response_len"]),
            "--min-lr", _num(r["min_lr"])]
    argv += _optimizer_args(r, updates)
    argv += [str(a) for a in r["memory_args"]]
    return argv


# ------------------------------------------------------------------------------------------------ generic access
def get_value(dotted: str, file: str, profile: str | None = None):
    cfg = load(file)
    head, _, rest = dotted.partition(".")
    if head not in cfg:
        raise RecipeError(f"{file} has no section {head!r}")
    node = cfg[head]
    if isinstance(node, dict) and profile:
        node = with_profile(node, {k: {kk: vv for kk, vv in v.items() if kk in node}
                                   for k, v in (cfg.get("profiles") or {}).items()}, profile)
    for part in [p for p in rest.split(".") if p]:
        if not isinstance(node, dict) or part not in node:
            raise RecipeError(f"{file}: no key {dotted!r}")
        node = node[part]
    return node


def check() -> list[str]:
    """Validate every config and every method; returns the list of checked items."""
    done = []
    models = load("models")
    assert set(models["sizes"]) == set(SIZES) and set(models["domains"]) == set(DOMAINS)
    assert set(models["label_map"]) == {"default", *DOMAINS}
    for size in SIZES:
        for d in DOMAINS:
            teacher_argv(domain=d, size=size, base="B", data="D", save="S")
            done.append(f"teacher:{size}:{d}")
        for m in methods():
            student_argv(method=m, size=size, seed=42, updates=80, student="B", data="D", save="S")
            if method_spec(m)["route"] == "hook":
                bank = None
                if method_spec(m).get("needs_injection_bank"):
                    bank = __file__          # any file: only its hash is taken here
                hook_config(method=m, size=size, seed=42, updates=80, audit_dir="A", bank=bank)
            done.append(f"student:{size}:{m}")
        sft_argv(size=size, student="B", data="D", save="S", n_rows=2700)
        done.append(f"sft:{size}")
    return done


def _print_lines(items) -> None:
    for x in items:
        if "\n" in str(x):
            raise RecipeError("an argument contains a newline")
        print(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("methods")
    p = sub.add_parser("teacher-argv")
    for k in ("--domain", "--size", "--base", "--data", "--save"):
        p.add_argument(k, required=True)
    p.add_argument("--updates", type=int)
    p.add_argument("--profile")
    p = sub.add_parser("teacher-updates")
    p.add_argument("--domain", required=True)
    p.add_argument("--size", required=True)
    p = sub.add_parser("student-argv")
    for k in ("--method", "--size", "--student", "--data", "--save"):
        p.add_argument(k, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--profile")
    p = sub.add_parser("student-route")
    p.add_argument("--method", required=True)
    p = sub.add_parser("hook-config")
    for k in ("--method", "--size", "--audit-dir", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--bank")
    p.add_argument("--profile")
    p = sub.add_parser("server-files")
    for k in ("--method", "--math", "--code", "--ifeval", "--out-dir"):
        p.add_argument(k, required=True)
    sub.add_parser("teacher-servers")
    p = sub.add_parser("sft-argv")
    for k in ("--size", "--student", "--data", "--save"):
        p.add_argument(k, required=True)
    p.add_argument("--n-rows", type=int, required=True)
    p.add_argument("--profile")
    p = sub.add_parser("get")
    p.add_argument("key")
    p.add_argument("--file", required=True)
    p.add_argument("--profile")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "methods":
            _print_lines(sorted(methods()))
        elif a.cmd == "teacher-argv":
            _print_lines(teacher_argv(domain=a.domain, size=a.size, base=a.base, data=a.data, save=a.save,
                                      updates=a.updates, profile=a.profile))
        elif a.cmd == "teacher-updates":
            print(teacher_updates(a.size, a.domain))
        elif a.cmd == "student-argv":
            _print_lines(student_argv(method=a.method, size=a.size, seed=a.seed, updates=a.updates,
                                      student=a.student, data=a.data, save=a.save, profile=a.profile))
        elif a.cmd == "student-route":
            print(" ".join(student_route(a.method)))
        elif a.cmd == "hook-config":
            cfg = hook_config(method=a.method, size=a.size, seed=a.seed, updates=a.updates, audit_dir=a.audit_dir,
                              bank=a.bank, profile=a.profile)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
            print(a.out)
        elif a.cmd == "server-files":
            lp, mp = server_files(method=a.method, teachers={"math": a.math, "code": a.code, "ifeval": a.ifeval},
                                  out_dir=a.out_dir)
            print(lp)
            print(mp)
        elif a.cmd == "teacher-servers":
            for row in teacher_servers():
                print(" ".join(str(x) for x in row))
        elif a.cmd == "sft-argv":
            _print_lines(sft_argv(size=a.size, student=a.student, data=a.data, save=a.save, n_rows=a.n_rows,
                                  profile=a.profile))
        elif a.cmd == "get":
            value = get_value(a.key, a.file, a.profile)
            if isinstance(value, (dict, list)):
                print(json.dumps(value))
            elif isinstance(value, bool):
                print("1" if value else "0")
            else:
                print(value)
        elif a.cmd == "check":
            items = check()
            print(f"RECIPE_CHECK_OK {len(items)} items")
    except RecipeError as e:
        print(f"recipe error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
