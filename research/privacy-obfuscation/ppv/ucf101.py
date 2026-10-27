"""VP-UCF101: UCF101 videos with STPrivacy's five video-level privacy attributes.

Utility: 101 action classes, official split 1 (9,537 train / 3,783 test videos).
Privacy: 5 binary attributes per video from STPrivacy (Li et al., ICCV 2023), file
annotations/ucf101_privacy_attribute_label.pickle in github.com/ming1993li/stprivacy.
The repo does not document the column order; the paper lists the attributes as face, skin
color, gender, nudity, relationship, and we assume that order for per-attribute reports only
(cMAP, the headline metric, does not depend on it).

Layout under <root> (default ~/work/data/ucf101):
    frames/<Class>/<video>.npy   uint8 (T, 112, 150, 3), every 2nd frame (12.5 fps)
    meta.json                     rows: video, action_id, priv (5 ints), start, end, split1
"""

import json
import os
import pickle
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .clips import LazyFrames

ATTRS = ["face", "skin_color", "gender", "nudity", "relationship"]
H, W = 112, 150
N_VIDEOS = 13320


def safe_load_annotations(path):
    """Unpickle STPrivacy's labels allowing only numpy array reconstruction (the file is a
    plain {video: ndarray} dict; anything else is refused instead of executed)."""
    allow = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
             ("numpy", "ndarray"), ("numpy", "dtype"),
             ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    class NumpyOnly(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allow:
                raise pickle.UnpicklingError(f"refusing {module}.{name}")
            return getattr(__import__(module, fromlist=[name]), name)

    with open(path, "rb") as f:
        d = NumpyOnly(f).load()
    return {k.rsplit(".", 1)[0]: [int(v) for v in np.asarray(a).reshape(-1)] for k, a in d.items()}


def _decode(args):
    src, out = args
    if os.path.exists(out):
        return out, np.load(out, mmap_mode="r").shape[0]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps=12.5,scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + ".tmp.npy", arr)
    os.replace(out + ".tmp.npy", out)
    return out, len(arr)


def prepare(raw_dir, ann_path, root, workers=24):
    """Unzip UCF-101.zip, decode every video, check the count, write meta.json."""
    vid_dir = os.path.join(raw_dir, "UCF-101")
    if not os.path.isdir(vid_dir):
        with zipfile.ZipFile(os.path.join(raw_dir, "UCF-101.zip")) as z:
            z.extractall(raw_dir)
    with zipfile.ZipFile(os.path.join(raw_dir, "splits.zip")) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        read = lambda n: z.read(names[n]).decode().split("\n")  # noqa: E731
        classes = {line.split()[1]: int(line.split()[0]) - 1 for line in read("classInd.txt") if line.strip()}
        train = {line.split()[0].rsplit(".", 1)[0] for line in read("trainlist01.txt") if line.strip()}
        test = {line.strip().rsplit(".", 1)[0] for line in read("testlist01.txt") if line.strip()}
    videos = sorted(os.path.join(c, f) for c in os.listdir(vid_dir) for f in os.listdir(os.path.join(vid_dir, c)))
    assert len(videos) == N_VIDEOS, f"expected {N_VIDEOS} UCF101 videos, found {len(videos)}"
    attrs = safe_load_annotations(ann_path)
    jobs = [(os.path.join(vid_dir, v), os.path.join(root, "frames", v.rsplit(".", 1)[0] + ".npy")) for v in videos]
    with ProcessPoolExecutor(workers) as ex:
        lengths = dict(ex.map(_decode, jobs, chunksize=16))
    rows = []
    for v, (_, out) in zip(videos, jobs):
        key = v.rsplit(".", 1)[0]
        n = lengths[out]
        split = "train" if key in train else "test" if key in test else None
        if split is None or key not in attrs or n < 2:
            continue
        rows.append(dict(video=key, action_id=classes[key.split("/")[0]], priv=attrs[key], start=1, end=n, split1=split))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(rows, f)
    return rows


def load(root):
    rows = json.load(open(os.path.join(root, "meta.json")))
    tr = [r for r in rows if r["split1"] == "train"]
    te = [r for r in rows if r["split1"] == "test"]
    frames = LazyFrames({r["video"]: os.path.join(root, "frames", r["video"] + ".npy") for r in rows})
    return tr, te, frames
