"""Summarize the KTH evaluations next to the paper's Table 2 row.

Reads <out>/eval/<kind>/kth/<model>/<tag>.json (written by eval_model.py) and prints a
markdown table with:
  action  top-1 (original, obfuscated)   privacy top-1 (original, obfuscated)
  f_0.5 = 0.5 * (a + (1 - p)), a and p as fractions

The paper's KTH row averages 13 action models and 5 privacy models. We run one of each,
so the comparison is indicative, not exact. Each row states its model.

Usage: python summarize_kth.py <out_dir> [--write results.md]
"""

import argparse
import glob
import json
import os

PAPER_KTH = {
    "original": {"action": 90.15, "privacy": 91.87},
    "ours": {"action": 88.67, "privacy": 5.46, "f05": 0.92},
}


def load(out):
    rows = {}
    for path in glob.glob(os.path.join(out, "eval", "*", "kth", "*", "*.json")):
        d = json.load(open(path))
        key = (d["kind"], d["architecture"])
        rows.setdefault(key, {})[d["condition"]] = d["top1"]
    return rows


def f05(a_pct, p_pct):
    return 0.5 * (a_pct / 100.0 + 1.0 - p_pct / 100.0)


def ours_tag(conds):
    tags = [t for t in conds if t.startswith("obf_")]
    return tags[0] if tags else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()

    rows = load(args.out)
    action = {k[1]: v for k, v in rows.items() if k[0] == "action"}
    privacy = {k[1]: v for k, v in rows.items() if k[0] == "privacy"}

    lines = []
    lines.append("## KTH: this replication vs paper Table 2")
    lines.append("")
    lines.append("| Setting | Action top-1 | Privacy top-1 | f_0.5 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Paper, original (13 action / 5 privacy models, averaged) | "
        f"{PAPER_KTH['original']['action']:.2f} | {PAPER_KTH['original']['privacy']:.2f} | - |"
    )
    lines.append(
        f"| Paper, ours (same averaging) | {PAPER_KTH['ours']['action']:.2f} | "
        f"{PAPER_KTH['ours']['privacy']:.2f} | {PAPER_KTH['ours']['f05']:.2f} |"
    )

    a_model = next(iter(action), None)
    p_model = next(iter(privacy), None)
    a_orig = action.get(a_model, {}).get("original") if a_model else None
    p_orig = privacy.get(p_model, {}).get("original") if p_model else None
    a_tag = ours_tag(action.get(a_model, {})) if a_model else None
    p_tag = ours_tag(privacy.get(p_model, {})) if p_model else None
    a_ours = action[a_model][a_tag] if a_tag else None
    p_ours = privacy[p_model][p_tag] if p_tag else None

    if a_model and a_orig is not None:
        p_txt = f"{p_orig:.2f}" if p_orig is not None else "-"
        f_txt = f"{f05(a_orig, p_orig):.2f}" if p_orig is not None else "-"
        lines.append(f"| This run, original, action {a_model} | {a_orig:.2f} | {p_txt} | {f_txt} |")
    if a_model and a_ours is not None and p_ours is not None:
        lines.append(f"| This run, ours, action {a_model} + privacy {p_model} | {a_ours:.2f} | "
                     f"{p_ours:.2f} | {f05(a_ours, p_ours):.2f} |")
    elif a_model and a_ours is not None:
        lines.append(f"| This run, ours, action {a_model} | {a_ours:.2f} | - | - |")

    lines.append("")
    lines.append("Models evaluated:")
    lines.append("")
    for kind, models in (("action", action), ("privacy", privacy)):
        for name, conds in models.items():
            detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(conds.items()))
            lines.append(f"- {kind} `{name}`: {detail}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)), exist_ok=True)
        with open(args.write, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
