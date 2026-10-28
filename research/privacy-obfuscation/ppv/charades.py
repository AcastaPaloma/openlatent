"""Charades (Sigurdsson et al., ECCV 2016): people filming scripted activities in their own homes.

Why it is here: the home-monitoring setting privacy work is about, with a per-video `subject` id, so it is a
second identity benchmark next to KTH, with real homes instead of four studio scenarios.

Utility: 157 action classes, several per video (multilabel; mAP). Privacy: which person (subject id).
The official train/test split has disjoint subjects (214 / 53), which a closed-set identity attacker cannot
use, so we re-split like KTH's Ilic split: keep subjects with >= `min_videos` videos (20 -> 90 subjects,
9,072 videos) and send 30% of each subject's videos to test (fixed seed). Train and test never share a video,
but they do share subjects and, mostly, homes: like KTH, the scene is part of the identity signal, and that is
the realistic threat for footage from a person's own home.

Videos: AI2's public S3 (Charades_v1_480.zip, 16.3 GB; annotations Charades.zip).
Layout under <root> (default ~/work/data/charades): frames/<id>.npy uint8 (T, 112, 150, 3) at 6.25 fps
(Charades clips are ~30 s; 6.25 fps keeps the dataset at ~86 GB and a 16-frame clip spans ~2.6 s), meta.json.
"""

import csv
import io
import json
import os
import random
import subprocess
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .clips import LazyFrames

H, W, FPS = 112, 150, 6.25
N_ACTIONS = 157


def _decode(args):
    src, out = args
    if os.path.exists(out):
        return out, np.load(out, mmap_mode="r").shape[0]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + ".tmp.npy", arr)
    os.replace(out + ".tmp.npy", out)
    return out, len(arr)


def _annotations(raw_dir):
    with zipfile.ZipFile(os.path.join(raw_dir, "Charades.zip")) as z:
        rows = []
        for part in ("train", "test"):
            rows += list(csv.DictReader(io.StringIO(z.read(f"Charades/Charades_v1_{part}.csv").decode())))
    return rows


def prepare(raw_dir, root, min_videos=20, test_frac=0.3, seed=0, workers=24):
    vid_dir = os.path.join(raw_dir, "Charades_v1_480")
    if not os.path.isdir(vid_dir):
        with zipfile.ZipFile(os.path.join(raw_dir, "Charades_v1_480.zip")) as z:
            z.extractall(raw_dir)
    ann = _annotations(raw_dir)
    by_subject = defaultdict(list)
    for r in ann:
        by_subject[r["subject"]].append(r)
    subjects = sorted(s for s, rs in by_subject.items() if len(rs) >= min_videos)
    rng = random.Random(seed)
    keep, split = [], {}
    for s in subjects:
        rs = sorted(by_subject[s], key=lambda r: r["id"])
        rng.shuffle(rs)
        n_test = max(1, round(test_frac * len(rs)))
        for i, r in enumerate(rs):
            split[r["id"]] = "test" if i < n_test else "train"
        keep += rs
    jobs = [(os.path.join(vid_dir, r["id"] + ".mp4"), os.path.join(root, "frames", r["id"] + ".npy")) for r in keep
            if os.path.exists(os.path.join(vid_dir, r["id"] + ".mp4"))]
    missing = len(keep) - len(jobs)
    assert missing <= 20, f"{missing} Charades videos missing"
    with ProcessPoolExecutor(workers) as ex:
        lengths = dict(ex.map(_decode, jobs, chunksize=8))
    sid = {s: i for i, s in enumerate(subjects)}
    rows = []
    for r in keep:
        out = os.path.join(root, "frames", r["id"] + ".npy")
        n = lengths.get(out, 0)
        if n < 2:
            continue
        acts = [0] * N_ACTIONS
        for seg in filter(None, r["actions"].split(";")):
            acts[int(seg.split()[0][1:])] = 1
        rows.append(dict(video=r["id"], action_id=acts, priv=sid[r["subject"]], subject=r["subject"], scene=r["scene"],
                         start=1, end=n, split=split[r["id"]]))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(dict(subjects=subjects, rows=rows), f)
    print(f"Charades identity split: {len(subjects)} subjects, {sum(r['split'] == 'train' for r in rows)} train / "
          f"{sum(r['split'] == 'test' for r in rows)} test videos ({missing} missing)")
    return rows


def load(root):
    meta = json.load(open(os.path.join(root, "meta.json")))
    rows = meta["rows"]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    frames = LazyFrames({r["video"]: os.path.join(root, "frames", r["video"] + ".npy") for r in rows})
    return tr, te, frames, len(meta["subjects"])
