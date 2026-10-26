"""Check what the "Ours" obfuscation actually does to KTH test inputs.

Two things:
  1. Distribution of the lerp weight. Upstream blends frame and AFD noise with
     torch.lerp(video, afd, sim), where sim is the mean of the template similarity maps
     scaled to [0, 1]. A pixel is fully replaced only where sim is near 1.
  2. Side-by-side images of the same test frame, original vs obfuscated, as the privacy
     classifier receives them (after upstream's test transform, un-normalised for display).

Usage (from the upstream checkout):
    python <repo>/replication/selective-privacy/scripts/inspect_obfuscation.py --out <dir>
"""

import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.getcwd()
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, HERE)

import compat  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from config import parser  # noqa: E402

parser.add_argument("--architecture", type=str, default="resnet18")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--n", type=int, default=8)
TEMPLATES = ["torso", "arm", "leg", "hair"]


def weight_stats(videos):
    """Mean-of-templates similarity per pixel, over every frame of the given clips."""
    vals = []
    for v in videos:
        names = sorted(os.listdir(os.path.join("data/kth_torso_sim", v)))
        for n in names[::10]:
            maps = [np.asarray(Image.open(os.path.join(f"data/kth_{t}_sim", v, n)).convert("L"), dtype=np.float32) / 255.0
                    for t in TEMPLATES]
            vals.append(np.mean(maps, axis=0).ravel())
    w = np.concatenate(vals)
    qs = np.percentile(w, [50, 90, 99, 99.9])
    print(f"lerp weight over {len(vals)} frames: mean {w.mean():.3f}, median {qs[0]:.3f}, "
          f"p90 {qs[1]:.3f}, p99 {qs[2]:.3f}, p99.9 {qs[3]:.3f}, max {w.max():.3f}")
    for thr in (0.25, 0.5, 0.75, 0.9):
        print(f"  pixels with weight >= {thr}: {100 * (w >= thr).mean():.2f}%")


def main():
    args = parser.parse_args()
    cfg = vars(args).copy()
    cfg.update(datasetname="kth", privacy=True, pretrained=True, train_backbone=True)
    from dataset.db_factory import DBfactory

    plain = DBfactory("kth", "test", dict(cfg, selectively_mask=False, obfuscate=[]))
    obf = DBfactory("kth", "test", dict(cfg, selectively_mask=True, obfuscate=TEMPLATES))

    with open("data/splits/kth/metadata.csv", newline="") as f:
        test = sorted(r["video"] for r in csv.DictReader(f) if r["set"] == "test")
    weight_stats(test[:: max(1, len(test) // 20)])

    rows = []
    for k, i in enumerate(range(0, len(plain), max(1, len(plain) // args.n))[: args.n]):
        random.seed(k); np.random.seed(k); torch.manual_seed(k)
        a = plain[i][0]
        random.seed(k); np.random.seed(k); torch.manual_seed(k)
        b = obf[i][0]
        a = plain.inverse_normalise(a.unsqueeze(1)).squeeze(1).clamp(0, 1)
        b = plain.inverse_normalise(b.unsqueeze(1)).squeeze(1).clamp(0, 1)
        rows += [a, b]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "kth_privacy_original_vs_obfuscated.png")
    save_image(make_grid(rows, nrow=2, padding=4), path)
    print("wrote", path, "(left: original, right: obfuscated, as the privacy model sees them)")


if __name__ == "__main__":
    main()
