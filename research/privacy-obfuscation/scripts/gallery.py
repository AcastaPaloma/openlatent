"""Obfuscation gallery: what every tested method actually releases, on the same test clips.

One row per finished run (fixed filters, controls, every learned checkpoint from HF), sorted by the
strong-protocol privacy score. Columns: the same frame of 4 test clips as released (8-bit), then two of
them contrast-stretched (1-99 percentile per frame, which exposes faint structure an attacker can still
use), then a 4-frame temporal strip of clip 1 (for methods whose leak is motion: diff, BDQ, tshuffle).
Each row is labelled with the run's action accuracy, privacy score and leakage.

Writes results/figures/gallery_<dataset>_<page>.png (12 rows per page).
Run from research/privacy-obfuscation on a GPU VM:
    python3 scripts/gallery.py --dataset kth --out ~/work/gallery
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("USE_TF", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppv import anonymizers as A  # noqa: E402
from ppv import kth, ucf101  # noqa: E402
from ppv.run import HF_REPO  # noqa: E402

DEV = "cuda"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROWS_PER_PAGE = 12


def finished_runs(dataset):
    """result.json of every finished, non-smoke, non-attack-only run on this dataset, best-hidden first."""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*", "result.json"))):
        r = json.load(open(p))
        c = r["config"]
        if r["name"].startswith("smoke") or c.get("anonymizer_from") or c.get("shuffle_train_privacy"):
            continue
        if r.get("dataset", "kth") != dataset:
            continue
        r["_priv"] = r.get("privacy_main", r.get("privacy_top1"))
        out.append(r)
    raw = next((r for r in out if r["config"]["anonymizer"]["type"] == "raw"), None)
    floor = {"kth": 4.0, "ucf101_vp": 58.7, "hmdb51_vp": 58.8, "charades": 100 / 90, "utdmhad": 12.5}[dataset]  # chance / blind cMAP
    for r in out:
        r["_leak"] = 100 * (r["_priv"] - floor) / (raw["_priv"] - floor) if raw else None
    return sorted(out, key=lambda r: -r["_priv"])


def build(r, in_ch):
    cfg = r["config"]["anonymizer"]
    f = A.build(cfg, in_ch=in_ch)
    if getattr(f, "learned", False) and r["config"].get("adv"):
        from huggingface_hub import hf_hub_download
        sd = torch.load(hf_hub_download(HF_REPO, f"runs/{r['name']}/anonymizer.pt"), map_location="cpu")
        missing, unexpected = f.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("vae.")]
        assert not missing and not unexpected, (r["name"], missing, unexpected)
    return f.to(DEV).eval()


@torch.no_grad()
def release(r, clips, videos, in_ch):
    """clips: list of (T, H, W[, 3]) uint8 windows; videos: full videos (for whole-video controls).
    Returns a list of (T, H, W[, 3]) float arrays in [0, 1], 8-bit quantised as released."""
    t = r["config"]["anonymizer"]["type"]
    if t in ("bg_median", "fg_only"):
        out = []
        for clip, (vid, s) in zip(clips, videos):
            med = np.median(vid, 0)
            if t == "bg_median":
                y = np.repeat(med[None], len(clip), 0)
            else:
                thr = r["config"]["anonymizer"].get("thresh", 20)
                y = np.where(np.abs(clip.astype(np.int16) - med) > thr, clip, 0)
            out.append(y.astype(np.float32) / 255)
        return out
    f = build(r, in_ch)
    out = []
    for clip in clips:
        x = torch.from_numpy(clip).to(DEV).float() / 255
        x = x[None, None] if x.dim() == 3 else x.permute(3, 0, 1, 2)[None]  # (1, C, T, H, W)
        y = f(x).float().clamp(0, 1)
        y = ((y * 255).round() / 255)[0]
        y = y[0] if y.shape[0] == 1 else y.permute(1, 2, 3, 0)  # (T, H, W[, 3])
        out.append(y.cpu().numpy())
    del f
    torch.cuda.empty_cache()
    return out


def stretch(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def label(r, metric):
    a = r.get("action_top1")
    leak = f"leak {r['_leak']:.0f}%" if r["_leak"] is not None else ""
    split = r.get("split")
    return (f"{r['name']}" + (f" [{split}]" if split not in (None, "ilic", "s1") else "")
            + f"\naction {'-' if a is None else f'{a:.1f}'}  {metric} {r['_priv']:.1f}\n{leak}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kth", choices=["kth", "ucf101_vp"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "kth":
        root = os.path.expanduser("~/work/data/kth")
        rows = kth.load_meta(root)
        _, te = kth.split(rows, "ilic")
        picks = []
        for a, p in (("walking", 2), ("boxing", 5), ("handwaving", 9), ("jogging", 21)):
            picks.append(next(r for r in te if r["action"] == a and r["person"] == p) if any(
                r["action"] == a and r["person"] == p for r in te) else next(r for r in te if r["action"] == a))
        frames = kth.load_frames(root, [r["video"] for r in picks])
        in_ch, stride, metric, cmap = 1, 2, "identity", "gray"
    else:
        _, te, frames = ucf101.load(os.path.expanduser("~/work/data/ucf101"))
        picks = [te[i] for i in np.linspace(0, len(te) - 1, 4).astype(int)]
        in_ch, stride, metric, cmap = 3, 1, "cMAP", None

    clips, videos = [], []
    for r in picks:
        arr = np.asarray(frames[r["video"]])
        a, b = r["start"] - 1, r["end"] - 1
        s = max(a, min((a + b) // 2 - 16, b - 32))
        clips.append(arr[s: s + 32: stride])
        videos.append((arr[a: b + 1], s))
    mid = len(clips[0]) // 2
    runs = finished_runs(args.dataset)
    print(f"{len(runs)} runs")

    cols = 4 + 2 + 4
    for page in range(0, len(runs), ROWS_PER_PAGE):
        chunk = runs[page: page + ROWS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), cols, figsize=(cols * 1.9 + 1.6, len(chunk) * 1.55), squeeze=False)
        for i, r in enumerate(chunk):
            try:
                ys = release(r, clips, videos, in_ch)
            except Exception as e:  # missing checkpoint etc.: leave the row blank with the reason
                print("skip", r["name"], e)
                ys = None
            for ax in axes[i]:
                ax.axis("off")
            axes[i, 0].text(-0.08, 0.5, label(r, metric), transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=7)
            if ys is None:
                axes[i, 1].set_title("checkpoint unavailable", fontsize=7)
                continue
            for k in range(4):
                axes[i, k].imshow(ys[k][mid], cmap=cmap, vmin=0, vmax=1)
            for k in range(2):
                axes[i, 4 + k].imshow(stretch(ys[k][mid]), cmap=cmap, vmin=0, vmax=1)
            for k, t in enumerate(np.linspace(0, len(ys[0]) - 1, 4).astype(int)):
                axes[i, 6 + k].imshow(ys[0][t], cmap=cmap, vmin=0, vmax=1)
            print("row", r["name"])
        heads = [f"clip {k + 1}" for k in range(4)] + ["clip 1 stretched", "clip 2 stretched"] + [f"clip 1, t{k}" for k in range(4)]
        for ax, h in zip(axes[0], heads):
            ax.set_title(h, fontsize=8)
        n = page // ROWS_PER_PAGE + 1
        fig.suptitle(f"{args.dataset}: every tested obfuscation as released (sorted by {metric}, most leaky first), page {n}", fontsize=10)
        fig.tight_layout(rect=(0.09, 0, 1, 0.98))
        path = os.path.join(args.out, f"gallery_{args.dataset}_{n}.png")
        fig.savefig(path, dpi=80)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
