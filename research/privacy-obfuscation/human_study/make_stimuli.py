"""Stimuli for the human re-identification study (KTH, raw vs fixed BDQ).

Each trial shows a "mystery person" clip (raw or BDQ) and a raw lineup of four people
(walking, d4, so the background never matches the query's d1-d3 scene). Only the AVIs
the trials need are read from the official KTH zips with HTTP range requests.

    uv run --with torch --with pillow python human_study/make_stimuli.py --cache <dir>

Writes human_study/stimuli/*.gif and human_study/answer_key.csv.
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ppv.anonymizers import Fixed  # noqa: E402
from ppv.kth import decode_avi, parse_sequences  # noqa: E402

BASE = "https://www.csc.kth.se/cvap/actions/"
OBFUSCATOR = Fixed("bdq_fixed", sigma=2.0, levels=8)  # = configs/p1/p1_bdqfixed_s2_l8.json
STRIDE, N_FRAMES, MS = 2, 32, 80  # the attackers' clip stride; 2.6 s at 12.5 fps
LINEUP = ("walking", 4)  # every lineup clip: raw, walking, d4

# Case = target person. Each case appears once raw and once obfuscated, with a different
# action, scenario, lineup subsequence, distractors and answer letter in the two trials.
# Order: trials 1-5 show every case once (3 obfuscated first, 2 raw first); 6-10 the
# other condition, 3-6 trials later.
# (trial, case, target, condition, query action, query scenario, query seq, lineup seq,
#  lineup persons A-D)
TRIALS = [
    (1, "C1", 3, "obf", "boxing", 2, 2, 1, [3, 9, 15, 21]),
    (2, "C2", 8, "raw", "handwaving", 3, 1, 1, [14, 8, 1, 19]),
    (3, "C3", 12, "obf", "walking", 3, 2, 2, [6, 20, 12, 4]),
    (4, "C4", 17, "raw", "running", 2, 3, 1, [10, 5, 23, 17]),
    (5, "C5", 22, "obf", "jogging", 2, 1, 1, [22, 13, 3, 7]),
    (6, "C2", 8, "obf", "running", 1, 2, 3, [24, 12, 8, 16]),
    (7, "C1", 3, "raw", "jogging", 1, 3, 3, [18, 3, 11, 2]),
    (8, "C4", 17, "obf", "handclapping", 1, 2, 3, [17, 1, 20, 22]),
    (9, "C3", 12, "raw", "boxing", 1, 1, 4, [9, 12, 17, 14]),
    (10, "C5", 22, "raw", "handclapping", 3, 2, 3, [5, 21, 24, 22]),
]
EXAMPLE = 25  # intro example (raw vs BDQ); never appears in a trial


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched with HTTP Range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch(cache, video):
    path = os.path.join(cache, video + "_uncomp.avi")
    if not os.path.exists(path):
        action = video.split("_")[1]
        with zipfile.ZipFile(io.BufferedReader(RangeFile(BASE + action + ".zip"), 1 << 20)) as z:
            name = next(n for n in z.namelist() if n.endswith(video + "_uncomp.avi"))
            with open(path + ".part", "wb") as f:
                f.write(z.read(name))
        os.replace(path + ".part", path)
    return decode_avi(path)


def clip(cache, seqs, person, action, d, seq, obf):
    video = f"person{person:02d}_{action}_d{d}"
    frames = fetch(cache, video)
    if obf:  # full frame rate, as train.materialize applies f
        x = torch.from_numpy(frames).float().div(255)[None, None]
        frames = (OBFUSCATOR(x).clamp(0, 1)[0, 0] * 255).round().byte().numpy()
    a, b = seqs[video][seq - 1]
    span = (N_FRAMES - 1) * STRIDE + 1
    s = max(a - 1, (a - 1 + b - span) // 2)  # centre of the subsequence
    idx = np.clip(np.arange(s, s + span, STRIDE), 0, len(frames) - 1)
    return frames[idx]


FONT = ImageFont.load_default(size=18)


def label(img, text):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=FONT)
    d.rectangle((4, 4, 4 + x1 - x0 + 10, 4 + y1 + 6), fill=255)
    d.text((9, 5), text, fill=0, font=FONT)


GREYS = 32  # fixed palette keeps GIFs small; BDQ's two levels (109, 146) stay distinct
PALETTE = [v for i in range(GREYS) for v in [i * 255 // (GREYS - 1)] * 3]


def save_gif(path, frames):
    imgs = []
    for f in frames:
        img = Image.fromarray((f.astype(np.uint16) * (GREYS - 1) // 255).astype(np.uint8), "P")
        img.putpalette(PALETTE)
        imgs.append(img)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=MS, loop=0, optimize=True)


def tile(c, scale):
    return np.stack([np.asarray(Image.fromarray(f).resize((160 * scale, 120 * scale), Image.NEAREST)) for f in c])


def grid(clips, labels, scale=2, gap=6, cols=2):
    tiles = [tile(c, scale) for c in clips]
    h, w = tiles[0].shape[1:]
    rows = -(-len(tiles) // cols)
    out = np.full((N_FRAMES, rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, cl = divmod(i, cols)
        out[:, r * (h + gap):r * (h + gap) + h, cl * (w + gap):cl * (w + gap) + w] = t
    frames = []
    for f in out:
        img = Image.fromarray(f)
        for i, text in enumerate(labels):
            r, cl = divmod(i, cols)
            sub = img.crop((cl * (w + gap), r * (h + gap), cl * (w + gap) + w, r * (h + gap) + h))
            label(sub, text)
            img.paste(sub, (cl * (w + gap), r * (h + gap)))
        frames.append(np.asarray(img))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="where downloaded KTH AVIs are kept")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    out_dir = os.path.join(HERE, "stimuli")
    os.makedirs(out_dir, exist_ok=True)
    seq_path = os.path.join(args.cache, "00sequences.txt")
    if not os.path.exists(seq_path):
        urllib.request.urlretrieve(BASE + "00sequences.txt", seq_path)
    seqs = parse_sequences(seq_path)

    raw = clip(args.cache, seqs, EXAMPLE, "handwaving", 1, 1, False)
    obf = clip(args.cache, seqs, EXAMPLE, "handwaving", 1, 1, True)
    save_gif(os.path.join(out_dir, "example.gif"), grid([raw, obf], ["Original", "Obfuscated"]))

    key = []
    for t, case, target, cond, action, d, seq, lseq, lineup in TRIALS:
        assert lineup.count(target) == 1 and len(set(lineup)) == 4
        q = clip(args.cache, seqs, target, action, d, seq, cond == "obf")
        save_gif(os.path.join(out_dir, f"t{t:02d}_query.gif"), tile(q, 2))
        cands = [clip(args.cache, seqs, p, *LINEUP, lseq, False) for p in lineup]
        save_gif(os.path.join(out_dir, f"t{t:02d}_lineup.gif"), grid(cands, list("ABCD")))
        key.append(dict(trial=t, case=case, condition=cond, target_person=target,
                        query=f"person{target:02d}_{action}_d{d} seq{seq}",
                        lineup=" ".join(f"{l}=p{p:02d}" for l, p in zip("ABCD", lineup)),
                        answer="ABCD"[lineup.index(target)]))
        print(f"trial {t}: {cond} {action} d{d} -> {key[-1]['answer']}")
    with open(os.path.join(HERE, "answer_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)


if __name__ == "__main__":
    main()
