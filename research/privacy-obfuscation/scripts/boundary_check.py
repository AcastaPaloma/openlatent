"""Does a learned anonymizer release near-raw frames? Runs the exact release path (train.materialize) on a few
test videos and, per output frame, measures the Pearson correlation with the raw input frame. Reports the
fraction of near-raw frames (corr > 0.9) overall and by position inside the 32-frame materialize window.

    python3 scripts/boundary_check.py --dataset ucf101_vp --runs p3_vae_a1,p3_vae_a1_ens
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("USE_TF", "0")

from ppv import anonymizers as A  # noqa: E402
from ppv import kth, train, ucf101  # noqa: E402
from ppv.run import HF_REPO  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def corr(a, b):
    a = a.reshape(len(a), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kth", choices=["kth", "ucf101_vp"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--videos", type=int, default=12)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    if args.dataset == "kth":
        root = os.path.expanduser("~/work/data/kth")
        rows = kth.load_meta(root)
        _, te = kth.split(rows, "ilic")
        vids = sorted({r["video"] for r in te})[:: max(1, len({r["video"] for r in te}) // args.videos)][: args.videos]
        frames = kth.load_frames(root, vids)
        in_ch = 1
    else:
        _, te, allf = ucf101.load(os.path.expanduser("~/work/data/ucf101"))
        vids = sorted({r["video"] for r in te})
        vids = vids[:: len(vids) // args.videos][: args.videos]
        frames = {v: np.asarray(allf[v]) for v in vids}
        in_ch = 3

    report = {}
    for name in args.runs.split(","):
        cfg = json.load(open(os.path.join(ROOT, "results", "runs", name, "result.json")))["config"]["anonymizer"]
        f = A.build(cfg, in_ch=in_ch)
        f.load_state_dict(torch.load(hf_hub_download(HF_REPO, f"runs/{name}/anonymizer.pt"), map_location="cpu"), strict=False)
        mb = 2 if cfg["type"] == "vae_latent" else 32
        out = train.materialize(f, frames, batch=mb)
        cs, pos = [], []
        for v in vids:
            raw = frames[v]
            y = out[v]
            if in_ch == 1 and y.ndim == 3:
                c = corr(y, raw)
            else:
                c = corr(y.astype(np.float32).mean(-1) if y.ndim == 4 else y, raw.astype(np.float32).mean(-1) if raw.ndim == 4 else raw)
            cs.append(c)
            pos.append(np.arange(len(c)) % 32)
        cs, pos = np.concatenate(cs), np.concatenate(pos)
        by_pos = {int(p): round(float((cs[pos == p] > 0.9).mean() * 100), 1) for p in range(32)}
        report[name] = dict(frames=int(len(cs)), near_raw_pct=round(float((cs > 0.9).mean() * 100), 2),
                            median_corr=round(float(np.median(cs)), 3), near_raw_pct_by_window_pos=by_pos)
        print(name, json.dumps(report[name]))
        del f
        torch.cuda.empty_cache()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
