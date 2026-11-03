"""Stimuli for the human re-identification study (KTH, raw vs learned BDQ).

Each trial shows a "mystery person" clip (raw or BDQ) and a raw lineup of four people
(walking, d4, so the background never matches the query's d1-d3 scene). Only the AVIs
the trials need are read from the official KTH zips with HTTP range requests.

    uv run --with torch --with pillow python human_study/make_stimuli.py --cache <dir> \
        --checkpoint <anonymizer.pt>

The obfuscator is the learned BDQ of run p2_bdq_a3_lr3e-3 (privacy-research branch; its
anonymizer.pt is in the HF checkpoint repo under runs/p2_bdq_a3_lr3e-3/).

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
from ppv.anonymizers import BDQ  # noqa: E402
from ppv.kth import decode_avi, parse_sequences  # noqa: E402

BASE = "https://www.csc.kth.se/cvap/actions/"
STRIDE, N_FRAMES, MS = 2, 32, 80  # the attackers' clip stride; 2.6 s at 12.5 fps
LINEUP = ("walking", 4)  # every lineup clip: raw, walking, d4

# Case = target person. Each case appears once raw and once obfuscated, with a different
# action, scenario, lineup subsequence, distractors and answer letter in the two trials.
# Order: trials 1-5 show every case once (3 obfuscated first, 2 raw first); 6-10 the
# other condition, 3-6 trials later.
# Queries are locomotion in both conditions (obfuscated ones in d1/d3): the learned BDQ erases boxing,
# handwaving and handclapping almost entirely (0.0-0.4% of pixels active) and the far
# d2 clips too, so those would test nothing. Raw and obfuscated get the same action mix.
# (trial, case, target, condition, query action, query scenario, lineup seq, lineup A-D)
TRIALS = [
    (1, "C1", 3, "obf", "running", 3, 1, [3, 9, 15, 21]),
    (2, "C2", 8, "raw", "walking", 3, 1, [14, 8, 1, 19]),
    (3, "C3", 12, "obf", "walking", 3, 2, [6, 20, 12, 4]),
    (4, "C4", 17, "raw", "running", 2, 1, [10, 5, 23, 17]),
    (5, "C5", 22, "obf", "jogging", 3, 1, [22, 13, 3, 7]),
    (6, "C2", 8, "obf", "running", 1, 3, [24, 12, 8, 16]),
    (7, "C1", 3, "raw", "jogging", 1, 3, [18, 3, 11, 2]),
    (8, "C4", 17, "obf", "jogging", 1, 3, [17, 1, 20, 22]),
    (9, "C3", 12, "raw", "jogging", 1, 4, [9, 12, 17, 14]),
    (10, "C5", 22, "raw", "running", 3, 3, [5, 21, 24, 22]),
]
EXAMPLE = (25, "jogging", 1)  # intro example (raw vs BDQ); person 25 is in no trial


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


SPAN = (N_FRAMES - 1) * STRIDE + 1


def obfuscate(obf, frames):
    """Whole video at full frame rate, as train.materialize applies f."""
    x = torch.from_numpy(frames.copy()).float().div(255)[None, None]
    with torch.no_grad():
        return (obf(x).clamp(0, 1)[0, 0] * 255).round().byte().numpy()


def window(frames, s):
    return frames[np.clip(np.arange(s, s + SPAN, STRIDE), 0, len(frames) - 1)]


def centre(seqs, video, seq):
    a, b = seqs[video][seq - 1]
    return seq, max(a - 1, (a - 1 + b - SPAN) // 2)


def activity(obfuscated):
    """Per-frame fraction of pixels off BDQ's static (background) level."""
    return (obfuscated != np.bincount(obfuscated.ravel()).argmax()).mean((1, 2))


def busiest(seqs, video, obfuscated):
    """(seq, start) of the window where the most pixels leave BDQ's static level. The same
    rule picks raw and obfuscated queries: the attacker's best case in both conditions."""
    active = activity(obfuscated)
    return max(((window(active, s).mean(), i, s) for i, (a, b) in enumerate(seqs[video], 1)
                for s in range(a - 1, max(a, b - SPAN + 1), 2)))[1:]


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
    ap.add_argument("--checkpoint", required=True, help="anonymizer.pt of p2_bdq_a3_lr3e-3")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    out_dir = os.path.join(HERE, "stimuli")
    os.makedirs(out_dir, exist_ok=True)
    seq_path = os.path.join(args.cache, "00sequences.txt")
    if not os.path.exists(seq_path):
        urllib.request.urlretrieve(BASE + "00sequences.txt", seq_path)
    seqs = parse_sequences(seq_path)
    obf = BDQ()
    obf.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    obf.eval()

    def query(person, action, d):
        video = f"person{person:02d}_{action}_d{d}"
        raw = fetch(args.cache, video)
        o = obfuscate(obf, raw)
        seq, s = busiest(seqs, video, o)
        return video, seq, window(raw, s), window(o, s)

    _, _, ex_raw, ex_obf = query(*EXAMPLE)
    save_gif(os.path.join(out_dir, "example.gif"), grid([ex_raw, ex_obf], ["Original", "Obfuscated"]))

    key = []
    for t, case, target, cond, action, d, lseq, lineup in TRIALS:
        assert lineup.count(target) == 1 and len(set(lineup)) == 4
        video, seq, q_raw, q_obf = query(target, action, d)
        q = q_obf if cond == "obf" else q_raw
        save_gif(os.path.join(out_dir, f"t{t:02d}_query.gif"), tile(q, 2))
        cands = []
        for p in lineup:
            v = f"person{p:02d}_{LINEUP[0]}_d{LINEUP[1]}"
            cands.append(window(fetch(args.cache, v), centre(seqs, v, lseq)[1]))
        save_gif(os.path.join(out_dir, f"t{t:02d}_lineup.gif"), grid(cands, list("ABCD")))
        key.append(dict(trial=t, case=case, condition=cond, target_person=target,
                        query=f"{video} seq{seq}", obf_active_pct=round(100 * activity(q_obf).mean(), 2),
                        lineup=" ".join(f"{l}=p{p:02d}" for l, p in zip("ABCD", lineup)),
                        answer="ABCD"[lineup.index(target)]))
        print(f"trial {t}: {cond} {action} d{d} -> {key[-1]['answer']}")
    with open(os.path.join(HERE, "answer_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)


if __name__ == "__main__":
    main()
