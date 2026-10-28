"""Filmstrips of raw training data, as the models see it (after decoding and resizing), with labels.

One PNG per dataset: `n` test videos (one per action, spread over classes), 6 frames each spread over the video,
labelled with the action and the private label (KTH / Charades: person id; VP-UCF101 / VP-HMDB51: the five
attributes). Videos labelled with nudity are skipped.
    python3 scripts/raw_samples.py --out ~/work/raw_samples
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppv import hmdb51, kth, ucf101  # noqa: E402

ATTRS = ["face", "skin", "gender", "nudity", "relationship"]


def attr_str(p):
    return ", ".join(a for a, v in zip(ATTRS, p) if v) or "none"


def strip(rows, frames, label, title, path, cmap=None, n_frames=6):
    fig, axes = plt.subplots(len(rows), n_frames, figsize=(n_frames * 2.0, len(rows) * 1.75), squeeze=False)
    for i, r in enumerate(rows):
        arr = np.asarray(frames[r["video"]])
        a, b = r["start"] - 1, r["end"] - 1
        for j, t in enumerate(np.linspace(a, b, n_frames).astype(int)):
            ax = axes[i, j]
            ax.imshow(arr[t], cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label(r), fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.12, 0, 1, 0.96))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("wrote", path)


def spread(rows, key, n, rng):
    """One row per distinct key value, n rows, spread evenly over the key values."""
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    keys = sorted(by)
    pick = [keys[i] for i in np.linspace(0, len(keys) - 1, n).astype(int)]
    return [by[k][rng.integers(len(by[k]))] for k in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/work/raw_samples"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    W = os.path.expanduser("~/work/data")

    root = os.path.join(W, "kth")
    rows = kth.load_meta(root)
    _, te = kth.split(rows, "ilic")
    pick = spread(te, lambda r: r["action"], args.n, rng)
    frames = kth.load_frames(root, [r["video"] for r in pick])
    strip(pick, frames, lambda r: f"{r['action']}\nperson {r['person']}, d{r['scenario']}",
          "KTH: 25 people x 6 actions x 4 scenarios, 120x160 grayscale, 25 fps (private label: who)",
          os.path.join(args.out, "raw_kth.png"), cmap="gray")

    for name, mod, nice in (("ucf101", ucf101, "VP-UCF101"), ("hmdb51", hmdb51, "VP-HMDB51")):
        root = os.path.join(W, name)
        if not os.path.exists(os.path.join(root, "meta.json")):
            continue
        _, te, frames = mod.load(root)
        te = [r for r in te if not r["priv"][3]]  # no nudity-labelled videos
        pick = spread(te, lambda r: r["video"].split("/")[0], args.n, rng)
        strip(pick, frames, lambda r: f"{r['video'].split('/')[0]}\n[{attr_str(r['priv'])}]",
              f"{nice}: 112x150 RGB, 12.5 fps (private labels: face, skin colour, gender, nudity, relationship)",
              os.path.join(args.out, f"raw_{name}.png"))

    root = os.path.join(W, "charades")
    if os.path.exists(os.path.join(root, "meta.json")):
        from ppv import charades
        _, te, frames, _ = charades.load(root)
        pick = spread(te, lambda r: r["subject"], args.n, rng)
        strip(pick, frames, lambda r: f"subject {r['subject']}\n{r['scene'][:22]}",
              "Charades: people in their own homes, 112x150 RGB, 6.25 fps (private label: who)",
              os.path.join(args.out, "raw_charades.png"))


if __name__ == "__main__":
    main()
