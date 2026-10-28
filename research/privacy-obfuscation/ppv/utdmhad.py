"""UTD-MHAD (Chen, Jafari, Kehtarnavaz, ICIP 2015): 8 subjects x 27 actions x 4 trials, RGB, one lab, one background.

Why it is here: an identity benchmark with no scene cue. KTH identity is ~22% lighting/scenario and Charades
identity is mostly the home; here every subject is filmed in the same room against the same backdrop, so
identity must come from the person (body shape, clothing, motion).

Utility: 27 actions (top-1). Privacy: which of 8 subjects (chance 12.5%). Split: trials 1-3 train, trial 4 test
(648 / 216 videos); train and test share subjects and actions but never a video.

Videos: RGB.zip from the authors' page (personal.utdallas.edu/~kehtar/UTD-MHAD.html), files a<A>_s<S>_t<T>_color.avi.
Layout under <root> (default ~/work/data/utdmhad): frames/<name>.npy uint8 (T, 112, 150, 3) at 15 fps, meta.json.
"""

import json
import os
import re
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .clips import LazyFrames

H, W, FPS = 112, 150, 15
N_ACTIONS, N_SUBJECTS = 27, 8
URL = "https://personal.utdallas.edu/~kehtar/UTD-MAD/RGB.zip"


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


def prepare(raw_dir, root, workers=16):
    zpath = os.path.join(raw_dir, "RGB.zip")
    if not os.path.exists(zpath):
        os.makedirs(raw_dir, exist_ok=True)
        subprocess.run(["curl", "-sL", "--http1.1", "--retry", "3", "-o", zpath + ".part", URL], check=True)  # HTTP/2 streams from this host reset mid-file
        os.replace(zpath + ".part", zpath)
    vid_dir = os.path.join(raw_dir, "RGB")
    if not os.path.isdir(vid_dir):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw_dir)
    pat = re.compile(r"a(\d+)_s(\d+)_t(\d+)_color\.avi$")
    vids = []
    for dirpath, _, files in os.walk(raw_dir):
        for f in files:
            m = pat.search(f)
            if m:
                vids.append((os.path.join(dirpath, f), *map(int, m.groups())))
    assert len(vids) >= 850, f"expected ~861 UTD-MHAD RGB videos, found {len(vids)}"
    jobs = [(p, os.path.join(root, "frames", os.path.basename(p)[:-4] + ".npy")) for p, *_ in vids]
    with ProcessPoolExecutor(workers) as ex:
        lengths = dict(ex.map(_decode, jobs, chunksize=8))
    rows = []
    for (p, a, s, t), (_, out) in zip(vids, jobs):
        n = lengths[out]
        if n < 2:
            continue
        rows.append(dict(video=os.path.basename(p)[:-4], action_id=a - 1, priv=s - 1, subject=s, trial=t,
                         start=1, end=n, split="test" if t == 4 else "train"))
    rows.sort(key=lambda r: r["video"])
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(rows, f)
    print(f"UTD-MHAD: {sum(r['split'] == 'train' for r in rows)} train / {sum(r['split'] == 'test' for r in rows)} test videos")
    return rows


def load(root):
    rows = json.load(open(os.path.join(root, "meta.json")))
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    frames = LazyFrames({r["video"]: os.path.join(root, "frames", r["video"] + ".npy") for r in rows})
    return tr, te, frames
