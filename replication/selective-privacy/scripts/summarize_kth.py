"""Summarize KTH evaluations next to the paper's Table 2 row, averaged across models.

Reads <out>/eval/<kind>/kth/<model>/<tag>.json (written by eval_model.py). Like the paper,
the headline row for each condition averages top-1 over every action model and every
privacy model evaluated under that condition. f_0.5 = 0.5 * (a + (1 - p)), a and p as
fractions. Per-model numbers follow, so partial coverage is visible.

Conditions: original, obf_arm_hair_leg_torso (default matcher),
obf_armT_hairT_legT_torsoT (use_targeted=True).

Usage: python summarize_kth.py <out_dir> [--write results.md]
"""

import argparse
import glob
import json
import os

PAPER = {"original": (90.15, 91.87, 13, 5), "ours": (88.67, 5.46, 13, 5)}
CONDITIONS = [
    ("original", "Original frames"),
    ("obf_arm_hair_leg_torso", "Obfuscated, default matcher"),
    ("obf_armT_hairT_legT_torsoT", "Obfuscated, use_targeted=True"),
]


def load(out):
    rows = {"action": {}, "privacy": {}}
    for path in glob.glob(os.path.join(out, "eval", "*", "kth", "*", "*.json")):
        d = json.load(open(path))
        rows[d["kind"]].setdefault(d["architecture"], {})[d["condition"]] = d["top1"]
    return rows


def f05(a_pct, p_pct):
    return 0.5 * (a_pct / 100.0 + 1.0 - p_pct / 100.0)


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(x):
    return f"{x:.2f}" if x is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    rows = load(args.out)

    lines = ["## KTH: this replication vs paper Table 2 (averaged over models)", "",
             "| Setting | Action top-1 | Privacy top-1 | f_0.5 | Models (action / privacy) |",
             "|---|---|---|---|---|"]
    for key, label in (("original", "Paper, original frames"), ("ours", "Paper, ours")):
        a, p, na, np_ = PAPER[key]
        lines.append(f"| {label} | {a:.2f} | {p:.2f} | {f05(a, p):.2f} | {na} / {np_} |")
    for tag, label in CONDITIONS:
        acts = [c[tag] for c in rows["action"].values() if tag in c]
        privs = [c[tag] for c in rows["privacy"].values() if tag in c]
        a, p = mean(acts), mean(privs)
        f = fmt(f05(a, p)) if a is not None and p is not None else "-"
        lines.append(f"| This run, {label.lower()} | {fmt(a)} | {fmt(p)} | {f} | {len(acts)} / {len(privs)} |")

    for kind in ("action", "privacy"):
        lines += ["", f"### {kind.capitalize()} models", "",
                  "| Model | " + " | ".join(lbl for _, lbl in CONDITIONS) + " |",
                  "|---|" + "---|" * len(CONDITIONS)]
        for arch in sorted(rows[kind]):
            c = rows[kind][arch]
            lines.append(f"| `{arch}` | " + " | ".join(fmt(c.get(t)) for t, _ in CONDITIONS) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)), exist_ok=True)
        with open(args.write, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
