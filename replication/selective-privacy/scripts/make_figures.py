"""Figures for the KTH replication: side-by-sides, weight histograms, results bars.

The obfuscated frames here are blended in pixel space: (1 - w) * frame + w * afd, where w is
the mean of the four template similarity maps. Upstream blends after normalising frame and
AFD with their own dataset statistics, so colours differ slightly from what the classifier
sees, but the per-pixel weight is the same. That weight is what decides how much is hidden.

Usage (from the upstream checkout):
    python make_figures.py --out <dir> --runs <runs dir> [--partial-targeted ACTION PRIVACY]
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TEMPLATES = ["torso", "arm", "leg", "hair"]
PAPER = {"original": (90.15, 91.87), "ours": (88.67, 5.46)}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def weight(video, frame, suffix):
    maps = [np.asarray(Image.open(f"data/kth_{t}{suffix}_sim/{video}/{frame}").convert("L"), dtype=np.float32) / 255.0
            for t in TEMPLATES]
    return np.mean(maps, axis=0)


def complete(video, suffix):
    n = len(os.listdir(f"data/kth/{video}"))
    return all(os.path.isdir(f"data/kth_{t}{suffix}_sim/{video}") and
               len(os.listdir(f"data/kth_{t}{suffix}_sim/{video}")) == n for t in TEMPLATES)


def pick_clips(k=6):
    vids = sorted(v for v in os.listdir("data/kth_torso_sim")
                  if complete(v, "") and complete(v, "T") and os.path.isdir(f"data/kth_afd/{v}"))
    # one clip per action where possible, spread over people
    by_action = {}
    for v in vids:
        by_action.setdefault(v.split("_")[1], []).append(v)
    picks = [lst[len(lst) // 2] for _, lst in sorted(by_action.items())]
    return picks[:k]


def side_by_side(clips, out):
    cols = ["Original", "Weight, default", "Obfuscated, default", "Weight, targeted", "Obfuscated, targeted"]
    fig, axes = plt.subplots(len(clips), 5, figsize=(15, 2.6 * len(clips)))
    for r, v in enumerate(clips):
        frames = sorted(os.listdir(f"data/kth/{v}"))
        f = frames[len(frames) // 2]
        img, afd = load_rgb(f"data/kth/{v}/{f}"), load_rgb(f"data/kth_afd/{v}/{f}")
        w0, w1 = weight(v, f, ""), weight(v, f, "T")
        panels = [img, w0, (1 - w0[..., None]) * img + w0[..., None] * afd,
                  w1, (1 - w1[..., None]) * img + w1[..., None] * afd]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            if p.ndim == 2:
                ax.imshow(p, cmap="viridis", vmin=0, vmax=1)
                ax.set_xlabel(f"mean {p.mean():.2f}, max {p.max():.2f}", fontsize=8)
            else:
                ax.imshow(np.clip(p, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        axes[r, 0].set_ylabel(v.replace("_uncomp", ""), fontsize=8)
    fig.suptitle("KTH test frames: blend weight toward noise (0 = untouched, 1 = fully replaced)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def over_time(video, out, n=6):
    frames = sorted(os.listdir(f"data/kth/{video}"))
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 6.4))
    for c, i in enumerate(idx):
        f = frames[i]
        img, afd = load_rgb(f"data/kth/{video}/{f}"), load_rgb(f"data/kth_afd/{video}/{f}")
        w = weight(video, f, "T")
        for r, p in enumerate([img, afd, (1 - w[..., None]) * img + w[..., None] * afd]):
            axes[r, c].imshow(np.clip(p, 0, 1)); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[0, c].set_title(f"frame {i + 1}", fontsize=9)
    for r, lab in enumerate(["Original", "Motion-consistent noise (AFD)", "Obfuscated, targeted"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(video.replace("_uncomp", "") + ": the noise moves with the person", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def histograms(clips_all, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for suffix, label, color in [("", "default (use_targeted=False)", "#7a7a7a"), ("T", "targeted (use_targeted=True)", "#2a6fdb")]:
        vals = []
        for v in clips_all:
            frames = sorted(os.listdir(f"data/kth/{v}"))
            for f in frames[::15]:
                vals.append(weight(v, f, suffix).ravel())
        w = np.concatenate(vals)
        ax.hist(w, bins=100, range=(0, 1), density=True, alpha=0.6, color=color,
                label=f"{label}: median {np.median(w):.2f}, {100 * (w >= .5).mean():.1f}% of pixels >= 0.5")
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("blend weight toward noise, per pixel"); ax.set_ylabel("density (log)")
    ax.set_title("How much of each pixel gets replaced")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def results(runs, partial, out):
    def get(kind, tag):
        hits = glob.glob(f"{runs}/eval/{kind}/kth/*/{tag}.json")
        return json.load(open(hits[0]))["top1"] if hits else None
    a0, p0 = get("action", "original"), get("privacy", "original")
    a1, p1 = get("action", "obf_arm_hair_leg_torso"), get("privacy", "obf_arm_hair_leg_torso")
    a2, p2 = get("action", "obf_armT_hairT_legT_torsoT"), get("privacy", "obf_armT_hairT_legT_torsoT")
    t_label = "This run, ours (targeted)"
    if a2 is None and partial:
        a2, p2 = partial
        t_label += "\n(partial: 113/150 clips)"
    bars = [("Paper, original", *PAPER["original"]), ("Paper, ours", *PAPER["ours"]),
            ("This run, original", a0, p0), ("This run, ours\n(default matcher)", a1, p1), (t_label, a2, p2)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(bars)); wd = 0.38
    for off, j, lab, col in [(-wd / 2, 1, "Action top-1 (higher is better)", "#3b8a4f"),
                             (wd / 2, 2, "Privacy top-1 (lower is better)", "#c4473a")]:
        vals = [b[j] if b[j] is not None else 0 for b in bars]
        rects = ax.bar(x + off, vals, wd, label=lab, color=col)
        for rct, v in zip(rects, vals):
            ax.text(rct.get_x() + rct.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0, 108); ax.set_ylabel("top-1 %"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("KTH: action utility reproduces; privacy protection does not")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--partial-targeted", nargs=2, type=float, default=None, metavar=("ACTION", "PRIVACY"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clips = pick_clips()
    side_by_side(clips, f"{args.out}/1_side_by_side.png")
    over_time(clips[0], f"{args.out}/2_over_time.png")
    pool = [v for v in sorted(os.listdir("data/kth_torso_sim")) if complete(v, "") and complete(v, "T")]
    histograms(pool[::3], f"{args.out}/3_weight_histograms.png")
    results(args.runs, args.partial_targeted, f"{args.out}/4_results.png")
    print("clips:", clips)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
