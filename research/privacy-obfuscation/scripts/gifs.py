"""Animated obfuscation gallery: one looping GIF per finished run, the same 4 test clips in a 2x2 tile.

Motion is what the strongest attackers use (gait, silhouettes over time), so stills under-show the leak.
Frames go through the exact release path (train.materialize on whole videos, as the run's attackers saw them),
then the clip window is cut out at the attackers' temporal stride. 8-bit as released.

Writes results/gifs/<dataset>/<run>.gif and results/gifs/<dataset>/index.json (run, action, privacy, leakage).
Run from research/privacy-obfuscation on a GPU VM:
    python3 scripts/gifs.py --dataset kth
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402

from gallery import build, finished_runs  # noqa: E402
from ppv import kth, train, ucf101  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def released(r, frames, in_ch):
    """{video: uint8 (T, H, W[, 3])} exactly as the run released it."""
    t = r["config"]["anonymizer"]["type"]
    if t == "raw":
        return frames
    if t in ("bg_median", "fg_only"):
        out = {}
        for v, a in frames.items():
            med = np.median(a, 0)
            if t == "bg_median":
                out[v] = np.repeat(med.astype(np.uint8)[None], len(a), 0)
            else:
                thr = r["config"]["anonymizer"].get("thresh", 20)
                out[v] = np.where(np.abs(a.astype(np.int16) - med) > thr, a, 0).astype(np.uint8)
        return out
    f = build(r, in_ch)
    out = train.materialize(f, frames, batch=2 if t == "vae_latent" else 32)
    del f
    torch.cuda.empty_cache()
    return out


def tile(clips):
    """4 clips (T, H, W[, 3]) -> (T, 2H+2, 2W+2[, 3]) with a 2 px white gutter."""
    t = min(len(c) for c in clips)
    c = [x[:t] for x in clips]
    h, w = c[0].shape[1:3]
    shape = (t, 2 * h + 2, 2 * w + 2) + c[0].shape[3:]
    out = np.full(shape, 255, np.uint8)
    for k, x in enumerate(c):
        i, j = divmod(k, 2)
        out[:, i * (h + 2): i * (h + 2) + h, j * (w + 2): j * (w + 2) + w] = x
    return out


def save_gif(frames, path, ms):
    ims = []
    for f in frames:
        im = Image.fromarray(f)
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=128) if im.mode == "RGB" else im.convert("L"))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kth", choices=["kth", "ucf101_vp"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gifs"))
    ap.add_argument("--only", default=None, help="comma-separated run names (default: all finished runs)")
    args = ap.parse_args()
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "kth":
        root = os.path.expanduser("~/work/data/kth")
        rows = kth.load_meta(root)
        _, te = kth.split(rows, "ilic")
        picks = []
        for a, p in (("walking", 2), ("boxing", 5), ("handwaving", 9), ("jogging", 21)):
            m = [r for r in te if r["action"] == a and r["person"] == p] or [r for r in te if r["action"] == a]
            picks.append(m[0])
        frames = kth.load_frames(root, [r["video"] for r in picks])
        frames = {v: np.asarray(frames[v]) for v in {r["video"] for r in picks}}
        in_ch, stride, n, ms = 1, 2, 32, 80  # 25 fps video at the attackers' stride 2 -> 12.5 fps
    else:
        _, te, allf = ucf101.load(os.path.expanduser("~/work/data/ucf101"))
        picks = [te[i] for i in np.linspace(0, len(te) - 1, 4).astype(int)]
        frames = {r["video"]: np.asarray(allf[r["video"]]) for r in picks}
        in_ch, stride, n, ms = 3, 1, 32, 80  # stored at 12.5 fps

    windows = []
    for r in picks:
        a, b = r["start"] - 1, r["end"] - 1
        s = max(a, min((a + b) // 2 - n * stride // 2, b - n * stride))
        windows.append((r["video"], slice(s, s + n * stride, stride)))

    index_path = os.path.join(out_dir, "index.json")
    index = {e["run"]: e for e in json.load(open(index_path))["runs"]} if os.path.exists(index_path) else {}
    clip_info = [dict(video=r["video"], action=r.get("action") or r["video"].split("/")[0]) for r in picks]
    for r in finished_runs(args.dataset):
        if args.only and r["name"] not in args.only.split(","):
            continue
        if r["config"].get("anonymizer_from") is None and r["name"].startswith("p4_"):
            continue  # attack-only rerun of a fixed filter: same frames as its p1 run
        try:
            rel = released(r, frames, in_ch)
        except Exception as e:  # e.g. checkpoint never uploaded
            print("skip", r["name"], e)
            continue
        clips = [np.asarray(rel[v][sl]) for v, sl in windows]
        path = os.path.join(out_dir, f"{r['name']}.gif")
        save_gif(tile(clips), path, ms)
        index[r["name"]] = dict(run=r["name"], file=os.path.basename(path), split=r.get("split"),
                                action=r.get("action_top1"), privacy=r["_priv"], leakage=r["_leak"],
                                kb=os.path.getsize(path) // 1024)
        print("wrote", path, index[r["name"]]["kb"], "KB")
    json.dump(dict(clips=clip_info, runs=sorted(index.values(), key=lambda e: -e["privacy"])),
              open(index_path, "w"), indent=1)


if __name__ == "__main__":
    main()
