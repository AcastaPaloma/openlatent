"""DINO similarity maps for KTH test-split clips, one DINO pass per frame for all templates.

Upstream matcher/similarity_from_template_for_dataset.py runs DINO once per template, so
it runs the ViT four times on every frame. The DINO features do not depend on the template.
This script runs DINO once per frame, then compares those features with each template's
descriptor and writes one map per template. The maps match upstream's per-template output
(see --check). Per frame: preprocess, extract_descriptors, chunk_cosine_sim against each
template, resize to the original frame size, save. Batching changes no values.

Output: data/kth_<template>_sim/<video>/frameNNNN.png, the same names upstream writes.
A clip is skipped for a template when that template's map count already matches the frame
count. After the run, every template must have one map per frame for every test clip,
or the run fails.

Usage (from the upstream checkout):
    python <repo>/replication/selective-privacy/scripts/derive_kth_sims.py --check person01_boxing_d1_uncomp
    python <repo>/replication/selective-privacy/scripts/derive_kth_sims.py --batch 8
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

from matcher.extractor import ViTExtractor  # noqa: E402
from matcher.helpers import chunk_cosine_sim, similarity_from_descriptor  # noqa: E402

META = os.path.join(UPSTREAM, "data/splits/kth/metadata.csv")
TEMPLATES = ["torso", "arm", "leg", "hair"]  # the paper's KTH template set
LOAD_SIZE = 224
LAYER = 11
STRIDE = 4
MODEL = "dino_vits8"  # upstream default; the descriptors were made with it


def test_videos():
    with open(META, newline="") as f:
        return sorted(r["video"] for r in csv.DictReader(f) if r["set"] == "test")


def load_descriptors(templates, device):
    return {t: torch.load(os.path.join("output/descriptors", f"{t}.pt"), map_location=device)
            for t in templates}


def maps_for_frames(extractor, descriptors, paths, img_size, batch):
    """{template: [map per path]}. One DINO forward per batch, shared by all templates."""
    resize = transforms.Resize(img_size, antialias=True)
    out = {t: [] for t in descriptors}
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            x = torch.cat([extractor.preprocess(p, LOAD_SIZE)[0] for p in chunk]).cuda()
            descs = extractor.extract_descriptors(x, LAYER, "key", False, include_cls=False)
            num_patches = extractor.num_patches
            for t, desc in descriptors.items():
                sim = chunk_cosine_sim(desc.to(descs.device)[None, None, None], descs)  # (B,1,1,t)
                for i in range(len(chunk)):
                    s = sim[i].reshape(num_patches).unsqueeze(0)
                    out[t].append(resize(s).squeeze().cpu())
    return out


def process_clip(extractor, descriptors, video, batch):
    src = os.path.join("data/kth", video)
    frames = sorted(glob(os.path.join(src, "*.png")))
    todo = []
    for t in descriptors:
        dst = os.path.join(f"data/kth_{t}_sim", video)
        if not (os.path.isdir(dst) and len(os.listdir(dst)) == len(frames)):
            todo.append(t)
    if not todo:
        return "skipped"

    img_size = PIL.Image.open(frames[0]).size[::-1]
    maps = maps_for_frames(extractor, {t: descriptors[t] for t in todo}, frames, img_size, batch)
    for t in todo:
        dst = os.path.join(f"data/kth_{t}_sim", video)
        os.makedirs(dst, exist_ok=True)
        for m, p in zip(maps[t], frames):
            save_image(m, os.path.join(dst, os.path.basename(p)))
    return "done " + ",".join(todo)


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


def check(video, batch):
    """Batched maps vs upstream similarity_from_descriptor, per template, on one clip."""
    extractor = ViTExtractor(MODEL, STRIDE, device="cuda")
    descs = load_descriptors(TEMPLATES, "cuda")
    paths = sorted(glob(os.path.join("data/kth", video, "*.png")))[:8]
    img_size = PIL.Image.open(paths[0]).size[::-1]
    new = maps_for_frames(extractor, descs, paths, img_size, batch)
    for t in TEMPLATES:
        with torch.no_grad():
            ref = similarity_from_descriptor(descs[t], paths, LOAD_SIZE, LAYER, "key", False, STRIDE)
        diffs = [(a.float().cpu() - b.float().cpu()).abs().max().item() for a, b in zip(ref, new[t])]
        print(f"check {video} / {t}: {len(paths)} frames, max |sim diff| vs upstream = {max(diffs):.6f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8, help="frames per DINO forward")
    ap.add_argument("--check", type=str, default=None, help="compare with upstream on this clip and exit")
    args = ap.parse_args()

    if args.check:
        check(args.check, args.batch)
        return

    for t in TEMPLATES:
        if not os.path.exists(os.path.join("output/descriptors", f"{t}.pt")):
            raise SystemExit(f"missing descriptor output/descriptors/{t}.pt")

    videos = test_videos()
    print(f"KTH test split: {len(videos)} clips; templates: {TEMPLATES}; batch {args.batch}", flush=True)
    extractor = ViTExtractor(MODEL, STRIDE, device="cuda")
    descriptors = load_descriptors(TEMPLATES, "cuda")
    for i, video in enumerate(videos, 1):
        status = process_clip(extractor, descriptors, video, args.batch)
        if i % 25 == 0 or i == len(videos):
            print(f"[{i}/{len(videos)}] {video}: {status}", flush=True)

    failed = False
    for t in TEMPLATES:
        bad = missing_maps(t, videos)
        if bad:
            failed = True
            for video, have, want in bad[:10]:
                print(f"[{t}] incomplete: {video} has {have} maps for {want} frames", flush=True)
            print(f"[{t}] {len(bad)} clips incomplete", flush=True)
        else:
            print(f"[{t}] verified: one map per frame for all {len(videos)} clips", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
