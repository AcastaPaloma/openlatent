"""KTH: decode to grayscale uint8 arrays, subsequence metadata, splits, and clip datasets.

Layout under <root> (default ~/work/data/kth):
    frames/<video>.npy   uint8 (T, 120, 160), every frame of the video
    meta.csv             one row per labelled subsequence (00sequences.txt):
                         video, person (1-25), action, scenario (1-4), seq (1-4), start, end

Frames are kept in RAM (about 6 GB for all of KTH), so loaders only slice arrays.
"""

import csv
import os
import re
import subprocess
import zipfile

import numpy as np

from .clips import Clips, worker_init  # noqa: F401  (re-exported for older imports)

ACTIONS = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
H, W = 120, 160


def decode_avi(path):
    """All frames of an AVI as uint8 (T, H, W) grayscale, via the ffmpeg CLI."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W)


def parse_sequences(path):
    """00sequences.txt -> {video: [(start, end), ...]} (1-based, inclusive)."""
    seqs = {}
    for line in open(path, encoding="latin-1"):
        m = re.match(r"\s*(person\d+_\w+_d\d)\s+frames\s+(.*)", line)
        if not m:
            continue
        ranges = [tuple(int(v) for v in r.split("-")) for r in re.findall(r"\d+\s*-\s*\d+", m.group(2))]
        seqs[m.group(1)] = [(a, b) for a, b in ranges]
    return seqs


def prepare(raw_dir, root):
    """Unzip + decode every KTH video and write meta.csv. Skips videos already decoded."""
    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    seqs = parse_sequences(os.path.join(raw_dir, "00sequences.txt"))
    rows = []
    for action in ACTIONS:
        with zipfile.ZipFile(os.path.join(raw_dir, f"{action}.zip")) as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".avi"):
                    continue
                video = os.path.basename(name).replace("_uncomp.avi", "")
                if video not in seqs:  # one archive video has no subsequence annotation
                    continue
                out = os.path.join(root, "frames", f"{video}.npy")
                if not os.path.exists(out):
                    tmp = os.path.join(root, f".{video}.avi")
                    with open(tmp, "wb") as f:
                        f.write(z.read(name))
                    frames = decode_avi(tmp)
                    os.remove(tmp)
                    np.save(out + ".tmp.npy", frames)
                    os.replace(out + ".tmp.npy", out)
                n = np.load(out, mmap_mode="r").shape[0]
                person, _, scen = video.split("_")
                for k, (a, b) in enumerate(seqs[video], 1):
                    b = min(b, n)
                    if b - a + 1 >= 16:
                        rows.append(dict(video=video, person=int(person[6:]), action=action,
                                         scenario=int(scen[1:]), seq=k, start=a, end=b))
    with open(os.path.join(root, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_meta(root):
    with open(os.path.join(root, "meta.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("person", "scenario", "seq", "start", "end"):
            r[k] = int(r[k])
        r["action_id"] = ACTIONS.index(r["action"])
        r["person_id"] = r["person"] - 1
        r["priv"] = r["person_id"]
    return rows


def split(rows, how):
    """Train/test split of subsequence rows. Every split keeps all 25 people and all 6
    actions on both sides, because identity is classified over the same 25 people.

    ilic     : Ilic et al. (CVPR 2024) video-level split, splits/kth_ilic.csv (copied from
               f-ilic/SelectivePrivacyPreservation data/splits/kth/metadata.csv): per person and
               action, one scenario is test; their val videos are folded into train. 450/150 videos.
    seq      : subsequence 4 of each video is test, 1-3 train (same clothes/scene on both sides)
    scenario : scenario d2 (scale variation) is test, d1/d3/d4 train (harder for identity:
               different recording conditions at test time)
    """
    if how == "ilic":
        path = os.path.join(os.path.dirname(__file__), "..", "splits", "kth_ilic.csv")
        with open(path) as f:
            sets = {r["video"].replace("_uncomp", ""): r["set"] for r in csv.DictReader(f)}
        test = lambda r: sets.get(r["video"]) == "test"  # noqa: E731
    elif how == "seq":
        test = lambda r: r["seq"] == 4  # noqa: E731
    elif how == "scenario":
        test = lambda r: r["scenario"] == 2  # noqa: E731
    else:
        raise ValueError(how)
    return [r for r in rows if not test(r)], [r for r in rows if test(r)]


def load_frames(root, videos, frames_dir="frames"):
    return {v: np.load(os.path.join(root, frames_dir, f"{v}.npy")) for v in sorted(set(videos))}
