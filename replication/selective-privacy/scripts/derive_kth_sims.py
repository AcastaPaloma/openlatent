"""DINO similarity maps for KTH test-split clips, one folder per template.

Upstream matcher/similarity_from_template_for_dataset.py does this for every clip. Only
test-split clips are obfuscated at evaluation, so this script runs the same
process_video() on the 150 test clips. It writes data/kth_<template>_sim/<video>/ and
skips a video when its file count already matches (upstream's own integrity check).

Usage (from the upstream checkout, after frame extraction):
    python <repo>/replication/selective-privacy/scripts/derive_kth_sims.py --templates torso arm leg hair
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

UPSTREAM = os.getcwd()
sys.path.insert(0, UPSTREAM)

import matcher.similarity_from_template_for_dataset as sim  # noqa: E402

META = os.path.join(UPSTREAM, "data/splits/kth/metadata.csv")


def test_videos():
    with open(META, newline="") as f:
        return sorted(r["video"] for r in csv.DictReader(f) if r["set"] == "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="+", default=["torso", "arm", "leg", "hair"])
    args = ap.parse_args()

    videos = test_videos()
    print(f"KTH test split: {len(videos)} clips; templates: {args.templates}", flush=True)
    for t in args.templates:
        desc = os.path.join("output/descriptors", f"{t}.pt")
        if not os.path.exists(desc):
            raise SystemExit(f"missing descriptor {desc}")
        cfg = SimpleNamespace(
            descriptorpath=desc,
            load_size=224,
            stride=4,
            model_type="dino_vits8",  # upstream default; the descriptors were made with it
            facet="key",
            layer=11,
            bin=False,
            use_targeted=False,
            dataset="kth",
        )
        for i, video in enumerate(videos, 1):
            sim.process_video(cfg, os.path.join("data/kth", video), "kth")
            if i % 25 == 0 or i == len(videos):
                print(f"[{t}] {i}/{len(videos)}", flush=True)

        bad = missing_maps(t, videos)
        if bad:
            for video, have, want in bad[:10]:
                print(f"[{t}] incomplete: {video} has {have} maps for {want} frames", flush=True)
            raise SystemExit(f"[{t}] {len(bad)} clips incomplete")
        print(f"[{t}] verified: one map per frame for all {len(videos)} clips", flush=True)


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


if __name__ == "__main__":
    main()
