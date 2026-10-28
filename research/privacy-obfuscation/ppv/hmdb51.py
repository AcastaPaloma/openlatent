"""VP-HMDB51: HMDB51 videos with STPrivacy's five video-level privacy attributes.

Utility: 51 action classes, official split 1 (70 train / 30 test videos per class: 3,570 / 1,530).
Privacy: 5 binary attributes per video from STPrivacy (Li et al., ICCV 2023), file
annotations/hmdb51_privacy_attribute_label.pickle in github.com/ming1993li/stprivacy (6,766 videos;
keys "<class>/<video>"), same attribute order as VP-UCF101 (see ppv/ucf101.py).

Videos: the official Serre-lab host serves HTML instead of the archives (2026-10-27), so download_data.sh
fetches the MichiganNLP/hmdb Hugging Face mirror (hmdb51_org.zip, test_train_splits.zip).

Layout under <root> (default ~/work/data/hmdb51): same as VP-UCF101 (frames at 12.5 fps, 112x150 RGB).
"""

import glob
import json
import os
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor

from .clips import LazyFrames
from .ucf101 import ATTRS, _decode, safe_load_annotations  # noqa: F401  (ATTRS re-exported)


def _extract(raw_dir):
    """hmdb51_org.zip -> <raw_dir>/videos/<class>/<video>.avi. The archive holds either the 51 per-class
    .rar files of the original release or the extracted videos; both are handled."""
    vid_dir = os.path.join(raw_dir, "videos")
    if os.path.isdir(vid_dir) and len(os.listdir(vid_dir)) == 51:
        return vid_dir
    stage = os.path.join(raw_dir, "org")
    if not os.path.isdir(stage):
        with zipfile.ZipFile(os.path.join(raw_dir, "hmdb51_org.zip")) as z:
            z.extractall(stage)
    os.makedirs(vid_dir, exist_ok=True)
    rars = glob.glob(os.path.join(stage, "**", "*.rar"), recursive=True)
    if rars:
        for rar in rars:
            cls = os.path.basename(rar)[:-4]
            os.makedirs(os.path.join(vid_dir, cls), exist_ok=True)
            subprocess.run(["unrar", "e", "-o+", "-inul", rar, os.path.join(vid_dir, cls) + "/"], check=True)
    else:
        for avi in glob.glob(os.path.join(stage, "**", "*.avi"), recursive=True):
            cls = os.path.basename(os.path.dirname(avi))
            os.makedirs(os.path.join(vid_dir, cls), exist_ok=True)
            os.replace(avi, os.path.join(vid_dir, cls, os.path.basename(avi)))
    return vid_dir


def _split1(raw_dir):
    """{"<class>/<video>": "train" | "test"} from the official split-1 files (id 1 train, 2 test, 0 unused)."""
    out = {}
    with zipfile.ZipFile(os.path.join(raw_dir, "test_train_splits.zip")) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if not base.endswith("_test_split1.txt"):
                continue
            cls = base[: -len("_test_split1.txt")]
            for line in z.read(n).decode().split("\n"):
                parts = line.split()
                if len(parts) == 2 and parts[1] in ("1", "2"):
                    out[f"{cls}/{parts[0].rsplit('.', 1)[0]}"] = "train" if parts[1] == "1" else "test"
    return out


def prepare(raw_dir, ann_path, root, workers=24):
    vid_dir = _extract(raw_dir)
    classes = sorted(os.listdir(vid_dir))
    assert len(classes) == 51, f"expected 51 HMDB51 classes, found {len(classes)}"
    split = _split1(raw_dir)
    attrs = safe_load_annotations(ann_path)
    videos = sorted(k for k in split if os.path.exists(os.path.join(vid_dir, k + ".avi")))
    missing = len(split) - len(videos)
    assert missing <= 10, f"{missing} split-1 videos missing from the archive"
    jobs = [(os.path.join(vid_dir, v + ".avi"), os.path.join(root, "frames", v + ".npy")) for v in videos]
    with ProcessPoolExecutor(workers) as ex:
        lengths = dict(ex.map(_decode, jobs, chunksize=16))
    rows = []
    for v, (_, out) in zip(videos, jobs):
        n = lengths[out]
        if v not in attrs or n < 2:
            continue
        rows.append(dict(video=v, action_id=classes.index(v.split("/")[0]), priv=attrs[v], start=1, end=n, split1=split[v]))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(rows, f)
    print(f"VP-HMDB51: {sum(r['split1'] == 'train' for r in rows)} train / {sum(r['split1'] == 'test' for r in rows)} test "
          f"videos with labels ({missing} split videos missing from the archive)")
    return rows


def load(root):
    rows = json.load(open(os.path.join(root, "meta.json")))
    tr = [r for r in rows if r["split1"] == "train"]
    te = [r for r in rows if r["split1"] == "test"]
    frames = LazyFrames({r["video"]: os.path.join(root, "frames", r["video"] + ".npy") for r in rows})
    return tr, te, frames
