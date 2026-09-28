#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Hugging Face model cards of the released DN-MOPD models.

Every number in a card is read from ``hf/paper_results.json``, a verbatim copy of the paper's tables (Tables 1, 2
and 5, the DN-MOPD vs. Label contrasts and MATH-500). The script fails if a row it needs is missing; it never
computes or rounds a score itself.

Usage::

    python hf/make_cards.py                         # all models -> hf/cards/<name>.md
    python hf/make_cards.py --only DN-MOPD-Qwen3.5-9B
    python hf/make_cards.py --namespace MyOrg       # repo ids MyOrg/<name>
    python hf/make_cards.py --check                 # exit 1 if a card in hf/cards/ is out of date

Only the standard library is required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
DOMAIN_NAME = {"math": "mathematics", "code": "code", "if": "instruction-following (IF)"}
DOMAIN_SHORT = {"math": "Math", "code": "Code", "if": "IF"}
DOMAIN_TITLE = {"math": "Mathematics", "code": "Code", "if": "Instruction-following (IF)"}
EVAL_NOTE = (
    "Scores (%) from the paper; training seed 42; 16,384-token evaluation cap; non-thinking chat template; "
    "temperature 1.0, top-p 1.0, generation seed 42. AIME25/AIME26: avg@64. LiveCodeBench v5/v6 (167/175 disjoint "
    "problems): avg@6. IFEval/IFBench: strict prompt accuracy, avg@16. Total: mean of the six task scores."
)


class MissingRow(KeyError):
    """A paper row the card needs is not in the results file."""


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def row(table: dict, name: str, group_hint: Optional[str] = None) -> List[str]:
    """Cells of the row whose first cell is `name` (optionally within a group, e.g. 'Qwen3.5-9B' in Table 4/5)."""
    for r in table["rows"]:
        if r["cells"][0] == name and (group_hint is None or r.get("group") == group_hint):
            return r["cells"]
    raise MissingRow(f"row {name!r} (group {group_hint!r}) not in table {table.get('caption', '')[:60]!r}")


def signed(x: float) -> str:
    return f"{x:+.2f}"


def ci(lo: float, hi: float) -> str:
    return f"[{signed(lo)}, {signed(hi)}]"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def paper_name(m: dict) -> str:
    if m["kind"] == "teacher":
        return f"{DOMAIN_SHORT[m['domain']]} expert"
    return "DN-MOPD" if m["method"] == "DN-MOPD" else "Label (label-routed MOPD)"


def eval_section(m: dict, res: dict, repo_ids: Dict[str, str]) -> str:
    size = m["size"]
    parts: List[str] = []
    if m["updates"] in (80,) or m["kind"] == "teacher":
        # Main tables: this model, the initial student, and (for students) the other multi-teacher method.
        names = [(m["paper_row"], f"**{m['name']}** ({paper_name(m)})")]
        if m["kind"] == "student":
            other = "Label-routed MOPD" if m["method"] == "DN-MOPD" else "DN-MOPD (Ours)"
            names.append((other, "DN-MOPD" if other.startswith("DN-MOPD") else "Label (label-routed MOPD)"))
        names.append(("Initial student", f"Initial student ({size_base(size)})"))
        if size == "9B":
            t = res["table1_9b"]
            header = ["Model"] + t["columns"][1:]
            rows = [[label] + row(t, r)[1:] for r, label in names]
            parts.append("Paper Table 1 (Qwen3.5-9B):\n\n" + md_table(header, rows))
        else:
            t = res["table2_4b_2b"]
            cols = [c for c in t["columns"] if c.startswith(size + " ")]
            idx = [t["columns"].index(c) for c in cols]
            header = ["Model"] + [c.split(" ", 1)[1] for c in cols]
            rows = [[label] + [row(t, r)[i] for i in idx] for r, label in names]
            parts.append(f"Paper Table 2 (Qwen3.5-{size}; each domain averages its two tasks: AIME25/AIME26, "
                         "LiveCodeBench v5/v6, IFEval/IFBench):\n\n" + md_table(header, rows))
        parts.append(EVAL_NOTE)
    if m["kind"] == "student" and m["method"] == "DN-MOPD" and m["updates"] == 80:
        lines = []
        for r in res["dn_vs_label_total"]["rows"]:
            if r["size"] == size:
                lines.append(f"- {r['eval_cap'] // 1024}K cap: {signed(r['delta'])} {ci(*r['ci95'])} "
                             f"(DN-MOPD {r['dn_total']:.2f} vs. Label {r['label_total']:.2f})")
        seeds = [r for r in res["three_seeds"]["rows"] if r["size"] == size and r["eval_cap"] == 16384]
        m500 = [r for r in res["math500"]["rows"] if r["size"] == size]
        text = ("Six-task Total, DN-MOPD minus Label (pp), this checkpoint vs. the seed-42 Label checkpoint, with "
                "paired 95% bootstrap intervals (questions resampled within each task, B = 10,000):\n\n"
                + "\n".join(lines))
        if seeds:
            s = seeds[0]
            text += (f"\n\nAcross student seeds 42/43/44 (16K), the mean gain over Label is {signed(s['mean'])} "
                     f"{ci(*s['ci95'])}; only the seed-42 model is released.")
        if m500:
            by_cap = sorted(m500, key=lambda r: -r["eval_cap"])
            text += "\n\nMATH-500 (16 answers per question), DN-MOPD minus Label: " + "; ".join(
                f"{r['eval_cap'] // 1024}K {signed(r['delta'])} {ci(*r['ci95'])}" for r in by_cap) + "."
        parts.append(text)
    if m["kind"] == "student" and m["updates"] == 160:
        t = res["table5_budget"]
        group = f"Qwen3.5-{size}"
        rows = []
        for r in t["rows"]:
            if r["group"] == group:
                c = r["cells"]
                label = f"**{c[0]}**" if c[0] == m["budget_row"] else c[0]
                rows.append([label] + c[1:])
        if not rows:
            raise MissingRow(f"no {group} rows in table5_budget")
        intro = ("Paper Table 5 (six-task Total at 80 and 160 updates, 16K cap; changes use unrounded scores). "
                 "This checkpoint is the 160-update endpoint of the bold row.")
        parts.append(intro + "\n\n" + md_table(["Method (" + group + ")"] + t["columns"][1:], rows))
        parts.append(EVAL_NOTE)
    return "\n\n".join(parts)


def size_base(size: str) -> str:
    return f"Qwen3.5-{size}"


def details(m: dict, cfg: dict, repo_ids: Dict[str, str]) -> str:
    size = m["size"]
    rows = [["Base model", f"[{cfg['base_models'][size]}](https://huggingface.co/{cfg['base_models'][size]})"]]
    if m["kind"] == "teacher":
        rows += [["Role", f"{DOMAIN_TITLE[m['domain']]} expert; frozen teacher of the {size} students"],
                 ["Training", f"GRPO from the base model, {m['updates']} updates, seed {m['seed']}"]]
    else:
        teachers = ", ".join(f"[{d}]({'https://huggingface.co/' + repo_ids[k]})" if k in repo_ids else d
                             for d, k in (("math", f"DN-MOPD-Qwen3.5-{size}-teacher-math"),
                                          ("code", f"DN-MOPD-Qwen3.5-{size}-teacher-code"),
                                          ("IF", f"DN-MOPD-Qwen3.5-{size}-teacher-if")))
        method = ("DN-MOPD: label-routed on-policy distillation with per-domain advantage scaling "
                  "w_d = clip(σ_all / σ_d, 0.25, 4)" if m["method"] == "DN-MOPD"
                  else "Label: label-routed multi-teacher on-policy distillation (MOPD), every w_d = 1")
        rows += [["Method", method], ["Teachers (same size)", teachers],
                 ["Training", f"{m['updates']} updates from the base model, student seed {m['seed']}"]]
    rows += [["Precision", "bfloat16"], ["Chat format", "non-thinking (`enable_thinking=False`)"],
             ["License", "Apache-2.0 (same as the base model)"]]
    return md_table(["", ""], rows).replace("|---|:---:|", "|---|---|")


def recipe(m: dict) -> str:
    if m["kind"] == "teacher":
        return "\n".join([
            f"- **Algorithm:** GRPO on {DOMAIN_NAME[m['domain']]} prompts with a verifiable reward; "
            "no KL or entropy term.",
            "- **Batching:** 128 prompts per rollout, 8 responses per prompt, 256 responses per optimizer step. "
            "Dynamic sampling drops prompt groups without reward variation (at most 8 generation batches per rollout).",
            "- **Lengths:** prompt ≤ 2,048 tokens, response ≤ 8,192 tokens, temperature 1.0.",
            "- **Optimizer:** Adam, learning rate 1e-6 (constant after 10 warm-up updates), betas (0.9, 0.98), "
            "weight decay 0.1, gradient clipping 1.0.",
            f"- **Length of training:** {m['updates']} updates (runs were capped at 400), seed 42.",
        ])
    lines = [
        "- **Prompts:** 2,700 training prompts, 900 each for mathematics, code and instruction following; each "
        "prompt carries its domain label and is scored by that domain's expert (label routing).",
        "- **Advantage:** per sampled token, teacher log-probability minus the actor-recomputed student "
        "log-probability, used in the clipped policy-gradient OPD loss (ratio clip 0.2/0.2); no KL or entropy term.",
    ]
    if m["method"] == "DN-MOPD":
        lines.append("- **DN-MOPD scaling:** on every batch, σ_d is the population standard deviation of the "
                     "teacher–rollout log-ratios over the valid response tokens of domain d, and σ_all pools all "
                     "domains; each domain's advantages are multiplied by w_d = clip(σ_all / σ_d, 0.25, 4) "
                     "(w_d = 1 if a statistic is degenerate). Signs are preserved.")
    else:
        lines.append("- **Label baseline:** identical to DN-MOPD except that every domain multiplier is 1.")
    lines += [
        "- **Batching:** 64 prompts × 8 responses = 512 responses per update, one optimizer step per rollout batch.",
        "- **Lengths:** prompt ≤ 2,048 tokens, response ≤ 8,192 tokens, temperature 1.0.",
        "- **Optimizer:** Adam, learning rate 1e-6 (constant after 5 warm-up updates), betas (0.9, 0.98), "
        "weight decay 0.1, gradient clipping 1.0.",
        f"- **Length of training:** {m['updates']} updates"
        + (" (the 80-update run continued to 160)" if m["updates"] == 160 else "") + ", student seed 42.",
    ]
    return "\n".join(lines)


def summary(m: dict, repo_ids: Dict[str, str]) -> str:
    size = m["size"]
    if m["kind"] == "teacher":
        return (f"The Qwen3.5-{size} **{DOMAIN_NAME[m['domain']]} expert** used as a frozen teacher in the DN-MOPD "
                f"paper: Qwen3.5-{size} trained with GRPO on {DOMAIN_NAME[m['domain']]} prompts. It is one of three "
                f"same-size experts (math, code, IF) that the Qwen3.5-{size} students learn from.")
    dn = f"DN-MOPD-Qwen3.5-{size}"
    cont = " continued to 160 updates (paper Table 5)" if m["updates"] == 160 else ""
    if m["method"] == "DN-MOPD":
        return (f"A Qwen3.5-{size} student trained with **DN-MOPD** (Domain-Normalized Multi-Teacher On-Policy "
                f"Distillation){cont}. Three same-size RL experts (math, code, instruction following) teach one "
                "student on its own responses; each prompt is scored by the expert of its domain, and DN-MOPD "
                "rescales each domain's token-level feedback by its measured spread, "
                "w_d = clip(σ_all / σ_d, 0.25, 4), so that no domain dominates the shared update.")
    link = f"[{dn}](https://huggingface.co/{repo_ids[dn]})" if dn in repo_ids else dn
    return (f"The **Label** baseline of the DN-MOPD paper at Qwen3.5-{size}{cont}: multi-teacher on-policy "
            "distillation with label routing (each prompt is scored by the expert of its domain, every domain "
            f"multiplier is 1). Released for comparison with {link}; it is not the proposed method.")


def specialist_drops(m: dict, res: dict) -> List[str]:
    """Columns of the paper table in which this expert scores below the initial student (read, not computed)."""
    if m["size"] == "9B":
        t = res["table1_9b"]
        cols = t["columns"][1:-1]
        mine, init = row(t, m["paper_row"])[1:-1], row(t, "Initial student")[1:-1]
    else:
        t = res["table2_4b_2b"]
        cols = [c for c in t["columns"] if c.startswith(m["size"] + " ") and not c.endswith("Total")]
        idx = [t["columns"].index(c) for c in cols]
        mine = [row(t, m["paper_row"])[i] for i in idx]
        init = [row(t, "Initial student")[i] for i in idx]
        cols = [c.split(" ", 1)[1] for c in cols]
    return [c for c, a, b in zip(cols, mine, init) if float(a) < float(b)]


def limitations(m: dict, res: dict) -> str:
    items: List[str] = []
    if m["kind"] == "teacher":
        drops = specialist_drops(m, res)
        items.append("A specialist: it was trained for one domain only. "
                     + (f"In the table above it scores below the initial model on {', '.join(drops)}."
                        if drops else "Its gains concentrate in its own domain (see the table above)."))
    elif m["method"] == "DN-MOPD":
        items += [
            "DN-MOPD is not the best integration recipe overall: in the paper, SeqKD-SFT and task-arithmetic merging "
            "(ParamMerge-TA) reach higher Totals under their own recipes. DN-MOPD's gains are over label-routed "
            "multi-teacher OPD and, by a smaller margin whose intervals sometimes include zero, over the strongest "
            "single-teacher student.",
            "Gains are largest in mathematics; code and IF gains are smaller and less consistent (at 9B, code does "
            "not improve at the 16K cap). The IF multiplier usually sits at the 0.25 lower bound, and the clipping "
            "bounds were not tuned per size.",
        ]
    else:
        items.append("This is the baseline, not the proposed method. In the paper it does not outperform the "
                     "strongest single-teacher student at any size, and DN-MOPD improves on it at every size.")
    if m["kind"] == "student":
        items.append("Trained on 2,700 prompts in three domains with a single seed (42) and one expert pool per "
                     "size, within one model family; results for other teacher–student configurations may differ.")
    items.append("Trained with responses of at most 8,192 tokens and evaluated only in non-thinking mode; thinking "
                 "mode, multimodal inputs, other languages and safety behaviour were not evaluated beyond the base "
                 "model.")
    return "\n".join(f"- {x}" for x in items)


def render(template: str, m: dict, cfg: dict, res: dict, repo_ids: Dict[str, str]) -> str:
    size = m["size"]
    if m["kind"] == "teacher":
        tags = ["teacher", "grpo"]
    else:
        tags = ["student", "dn-mopd-student" if m["method"] == "DN-MOPD" else "label-routed-mopd"]
    thinking_note = ("" if size == "2B" else
                     f" Qwen3.5-{size}'s chat template enables thinking by default, so this argument is required.")
    values = {
        "name": m["name"],
        "repo_id": repo_ids[m["name"]],
        "base_model": cfg["base_models"][size],
        "extra_tags": "\n".join(f"- {t}" for t in tags),
        "summary": summary(m, repo_ids),
        "project_url": cfg["project_url"],
        "code_url": cfg["code_url"],
        "code_url_short": cfg["code_url"].replace("https://", ""),
        "details_table": details(m, cfg, repo_ids),
        "recipe": recipe(m),
        "thinking_note": thinking_note,
        "evaluation": eval_section(m, res, repo_ids),
        "limitations": limitations(m, res),
    }
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[a-z_]+\}\}", out)
    if left:
        raise ValueError(f"unfilled placeholders in {m['name']}: {sorted(set(left))}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", type=Path, default=HERE / "models.json")
    ap.add_argument("--results", type=Path, default=HERE / "paper_results.json")
    ap.add_argument("--template", type=Path, default=HERE / "model_card_template.md")
    ap.add_argument("--out", type=Path, default=HERE / "cards")
    ap.add_argument("--namespace", default=None, help="HF user/org (default: hf_namespace in models.json)")
    ap.add_argument("--only", nargs="*", default=None, help="model names to render")
    ap.add_argument("--core-only", action="store_true",
                    help="skip the optional 160-update continuations")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if any card differs from disk")
    args = ap.parse_args()

    cfg, res, template = load(args.models), load(args.results), args.template.read_text()
    ns = args.namespace or cfg["hf_namespace"]
    models = [m for m in cfg["models"] if not (args.core_only and m.get("priority") == "optional")]
    repo_ids = {m["name"]: f"{ns}/{m['name']}" for m in models}
    if args.only:
        unknown = set(args.only) - set(repo_ids)
        if unknown:
            ap.error(f"unknown model(s): {sorted(unknown)}")
        models = [m for m in models if m["name"] in args.only]

    stale = 0
    args.out.mkdir(parents=True, exist_ok=True) if not args.check else None
    for m in models:
        card = render(template, m, cfg, res, repo_ids)
        path = args.out / f"{m['name']}.md"
        if args.check:
            if not path.exists() or path.read_text() != card:
                print(f"STALE {path}")
                stale += 1
        else:
            path.write_text(card)
            print(f"wrote {path.relative_to(HERE.parent) if HERE.parent in path.parents else path}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
