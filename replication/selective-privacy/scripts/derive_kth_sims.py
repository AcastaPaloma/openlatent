"""DINO similarity maps for KTH test-split clips, one folder per template, batched.

Upstream matcher/similarity_from_template_for_dataset.py runs DINO one frame at a time.
At stride 4, ViT-S/8 has a 56x56 token grid per frame, so one frame costs about 100 ms
on the A100. The full test split (73,756 frames) took hours per template that way. This
script runs the same steps on batches of frames: preprocess, extract_descriptors,
chunk_cosine_sim, resize to the original frame size, save. The ViT has no batch-dependent
layers, so batching changes float rounding only. `--check` compares the batched maps with
upstream's similarity_from_descriptor on one clip before any full run.

Output: data/kth_<template>_sim/<video>/frameNNNN.png, the same names upstream writes.
A clip is skipped when its map count already matches its frame count. After each
template, every test clip must have one map per frame, or the run fails.

Usage (from the upstream checkout):
    python <repo>/replication/selective-privacy/scripts/derive_kth_sims.py --check person01_boxing_d1_uncomp
    python <repo>/replication/selective-privacy/scripts/derive_kth_sims.py --templates torso arm leg hair --batch 32
"""

import argparse
import csv
import os
import sys
from glob import glob

UPSTREAM = os.getcwd()
sys.path.insert(0, UPSTREAM)

import PIL.Image  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from matcher.helpers import chunk_cosine_sim, similarity_from_descriptor  # noqa: E402
from matcher.extractor import ViTExtractor  # noqa: E402

META = os.path.join(UPSTREAM, "data/splits/kth/metadata.csv")
LOAD_SIZE = 224
LAYER = 11
STRIDE = 4
MODEL = "dino_vits8"  # upstream default; the descriptors were made with it


def test_videos():
    with open(META, newline="") as f:
        return sorted(r["video"] for r in csv.DictReader(f) if r["set"] == "test")


def load_descriptor(template, device):
    desc = torch.load(os.path.join("output/descriptors", f"{template}.pt"), map_location=device)
    return desc


def batched_sims(extractor, descriptor, paths, img_size, batch):
    """Similarity maps for `paths`, as CPU tensors shaped like the frames (H, W)."""
    resize = transforms.Resize(img_size, antialias=True)
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            x = torch.cat([extractor.preprocess(p, LOAD_SIZE)[0] for p in chunk]).cuda()
            descs = extractor.extract_descriptors(x, LAYER, "key", False, include_cls=False)
            num_patches = extractor.num_patches
            sim = chunk_cosine_sim(descriptor.to(descs.device)[None, None, None], descs)  # (B,1,1,t)
            for i in range(len(chunk)):
                s = sim[i].reshape(num_patches).unsqueeze(0)
                out.append(resize(s).squeeze().cpu())
    return out


def process_clip(extractor, descriptor, video, template, batch):
    src = os.path.join("data/kth", video)
    dst = os.path.join(f"data/kth_{template}_sim", video)
    frames = sorted(glob(os.path.join(src, "*.png")))
    if os.path.isdir(dst) and len(os.listdir(dst)) == len(frames):
        return "skipped"
    os.makedirs(dst, exist_ok=True)
    img_size = PIL.Image.open(frames[0]).size[::-1]
    maps = batched_sims(extractor, descriptor, frames, img_size, batch)
    for m, p in zip(maps, frames):
        save_image(m, os.path.join(dst, os.path.basename(p)))
    return "done"


def missing_maps(template, videos):
    """Clips whose similarity-map count differs from their frame count."""
    bad = []
    for video in videos:
        frames = [n for n in os.listdir(os.path.join("data/kth", video)) if n.endswith(".png")]
        sim_dir = os.path.join(f"data/kth_{template}_sim", video)
        have = len(os.listdir(sim_dir)) if os.path.isdir(sim_dir) else 0
        if have != len(frames):
            bad.append((video, have, len(frames)))
    return bad


def check(video, template, batch):
    """Batched maps vs upstream similarity_from_descriptor on one clip."""
    extractor = ViTExtractor(MODEL, STRIDE, device="cuda")
    desc = load_descriptor(template, "cuda")
    paths = sorted(glob(os.path.join("data/kth", video, "*.png")))[:8]
    img_size = PIL.Image.open(paths[0]).size[::-1]
    with torch.no_grad():
        ref = similarity_from_descriptor(desc, paths, LOAD_SIZE, LAYER, "key", False, STRIDE)
    new = batched_sims(extractor, desc, paths, img_size, batch)
    diffs = [(a.float().cpu() - b.float().cpu()).abs().max().item() for a, b in zip(ref, new)]
    print(f"check {video} / {template}: {len(paths)} frames, max |sim diff| = {max(diffs):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="+", default=["torso", "arm", "leg", "hair"])
    ap.add_argument("--batch", type=int, default=8, help="frames per DINO forward; each process holds ~6 GB at 8, ~13 GB at 16")
    ap.add_argument("--check", type=str, default=None)
    args = ap.parse_args()

    if args.check:
        check(args.check, args.templates[0], args.batch)
        return

    videos = test_videos()
    print(f"KTH test split: {len(videos)} clips; templates: {args.templates}; batch {args.batch}", flush=True)
    for t in args.templates:
        desc_path = os.path.join("output/descriptors", f"{t}.pt")
        if not os.path.exists(desc_path):
            raise SystemExit(f"missing descriptor {desc_path}")
        extractor = ViTExtractor(MODEL, STRIDE, device="cuda")
        desc = load_descriptor(t, "cuda")
        for i, video in enumerate(videos, 1):
            process_clip(extractor, desc, video, t, args.batch)
            if i % 25 == 0 or i == len(videos):
                print(f"[{t}] {i}/{len(videos)}", flush=True)

        bad = missing_maps(t, videos)
        if bad:
            for video, have, want in bad[:10]:
                print(f"[{t}] incomplete: {video} has {have} maps for {want} frames", flush=True)
            raise SystemExit(f"[{t}] {len(bad)} clips incomplete")
        print(f"[{t}] verified: one map per frame for all {len(videos)} clips", flush=True)


if __name__ == "__main__":
    main()
