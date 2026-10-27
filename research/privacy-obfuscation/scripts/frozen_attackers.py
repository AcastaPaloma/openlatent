"""Frozen vs retrained attackers, for every finished run.

Frozen ("weak" protocol, Ilic et al.): attacker models trained once on raw video, then tested on the obfuscated
test set without ever seeing an obfuscated frame. Retrained ("strong"): fresh attackers trained on the obfuscated
training set. Both come from the same run (result.json "weak" / "strong"); this script only tabulates them.

Writes results/frozen_attackers.md and results/figures/frozen_vs_retrained_<dataset>.png.
    python3 scripts/frozen_attackers.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FROZEN, RETRAINED = "#eb6834", "#2a78d6"  # validated pair (dataviz validate_palette.js, light surface)
INFO = {
    "kth": dict(name="KTH", task="person", metric="identity top-1 (%)", floor=4.0, floor_label="chance 4.0"),
    "ucf101_vp": dict(name="VP-UCF101", task="attr", metric="attribute cMAP (%)", floor=58.7, floor_label="blind floor 58.7"),
}


def best(d, task):
    v = [x for k, x in d.items() if k.startswith(task + "/")]
    return max(v) if v else None


def rows(ds):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*", "result.json"))):
        r = json.load(open(p))
        if r["name"].startswith("smoke") or r.get("dataset", "kth") != ds or not r.get("weak"):
            continue
        task = INFO[ds]["task"]
        w, s = r["weak"], r["strong"]
        out.append(dict(name=r["name"], split="s1" if ds == "ucf101_vp" else r.get("split"), anon=r["config"]["anonymizer"],
                        frozen_action=w.get("action/r2p1d"), retrained_action=s.get("action/r2p1d"),
                        frozen_frame=w.get(f"{task}/frame_r50"), frozen_video=w.get(f"{task}/r3d"),
                        frozen=best(w, task), retrained=best(s, task)))
    return sorted(out, key=lambda e: -e["retrained"])


def anon_str(a):
    if a["type"] == "chain":
        return " -> ".join(anon_str(s) for s in a["stages"])
    params = ", ".join(f"{k} {v}" for k, v in a.items() if k != "type")
    return a["type"] + (f" ({params})" if params else "")


def f1(v):
    return "-" if v is None else f"{v:.1f}"


def figure(ds, rs, path):
    info = INFO[ds]
    rs = [r for r in rs if r["split"] in ("ilic", "s1", None)]
    n = len(rs)
    fig, ax = plt.subplots(figsize=(8.2, 0.24 * n + 1.6))
    for i, r in enumerate(rs):
        y = n - 1 - i
        ax.plot([r["frozen"], r["retrained"]], [y, y], color="#c3c2b7", lw=1.2, zorder=1)
        ax.scatter(r["frozen"], y, s=34, color=FROZEN, marker="o", zorder=3, edgecolor="white", linewidth=1.2)
        ax.scatter(r["retrained"], y, s=34, color=RETRAINED, marker="s", zorder=3, edgecolor="white", linewidth=1.2)
    ax.axvline(info["floor"], color="#8b8b84", lw=1, ls=(0, (3, 3)), zorder=0)
    ax.text(info["floor"], n - 0.2, " " + info["floor_label"], fontsize=7.5, color="#5f5e57", va="bottom")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["name"] for r in reversed(rs)], fontsize=7.5, color="#3d3d3a")
    ax.set_ylim(-0.8, n + 0.6)
    ax.set_xlabel(f"{info['metric']} of the best attacker  (lower = more private)", fontsize=9, color="#3d3d3a")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors="#5f5e57", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    h1 = ax.scatter([], [], s=34, color=FROZEN, marker="o", label="frozen: trained on raw video only, never sees obfuscation")
    h2 = ax.scatter([], [], s=34, color=RETRAINED, marker="s", label="retrained on the obfuscated training set")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, frameon=False, fontsize=8)
    ax.set_title(f"{info['name']}: frozen vs retrained attackers, every run (sorted by retrained score)",
                 fontsize=10, color="#1a1a19", pad=36, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    lines = ["# Frozen vs retrained attackers", "",
             "Every finished run is scored twice by the same attacker architectures:", "",
             "- **frozen**: trained once on raw video, then tested on the obfuscated test set. It has never seen an "
             "obfuscated frame (the protocol in Ilic et al., \"Selective, Interpretable and Motion Consistent Privacy "
             "Attribute Obfuscation for Action Recognition\").",
             "- **retrained**: a fresh model of the same architecture trained on the obfuscated training set, as a real "
             "attacker who gets the released data would do.", "",
             "Privacy is the best attacker's score (frame ResNet-50 or video R3D-18; DINOv2 runs are attack-only and "
             "have no frozen score). Generated by `scripts/frozen_attackers.py` from `results/runs/*/result.json`.", ""]
    for ds, info in INFO.items():
        rs = rows(ds)
        if not rs:
            continue
        fig_path = os.path.join(ROOT, "results", "figures", f"frozen_vs_retrained_{ds}.png")
        figure(ds, rs, fig_path)
        non_raw = [r for r in rs if r["anon"]["type"] != "raw"]
        near_floor = [r for r in non_raw if r["frozen"] - info["floor"] < 0.25 * (r["retrained"] - info["floor"])]
        gaps = sorted(r["retrained"] - r["frozen"] for r in non_raw)
        lines += [f"## {info['name']} ({info['metric']}; {info['floor_label']})", "",
                  f"![frozen vs retrained](figures/frozen_vs_retrained_{ds}.png)", "",
                  f"- {len(non_raw)} obfuscated runs. Median gap retrained - frozen: **{gaps[len(gaps) // 2]:.1f} points**.",
                  f"- In {len(near_floor)} of {len(non_raw)} runs the frozen attacker recovers less than a quarter of what "
                  "the retrained one does (above the floor): under the frozen protocol they would look private.", "",
                  "| run | split | obfuscation | frozen action | frozen privacy (frame / video) | **frozen best** | **retrained best** | retrained action |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in rs:
            lines.append(f"| {r['name']} | {r['split'] or ''} | `{anon_str(r['anon'])}` | {f1(r['frozen_action'])} | "
                         f"{f1(r['frozen_frame'])} / {f1(r['frozen_video'])} | **{f1(r['frozen'])}** | **{f1(r['retrained'])}** | "
                         f"{f1(r['retrained_action'])} |")
        lines.append("")
    open(os.path.join(ROOT, "results", "frozen_attackers.md"), "w").write("\n".join(lines))
    print("wrote results/frozen_attackers.md")


if __name__ == "__main__":
    main()
