"""Stimuli for the human re-identification study (KTH; four obfuscators, every query masked).

Each trial shows an obfuscated "mystery person" clip and a lineup of four people (walking,
d4 indoors), each shown raw and, beneath, through the trial's filter: the human gets labelled
obfuscated examples of every candidate, like the strong-protocol attacker that trains on them. Queries come from d1/d2, where actors wear the same clothes as in d4,
so the background differs but the person does not: only the obfuscator hides identity.
Only the AVIs the trials need are read from the official KTH zips with HTTP range requests.

Needs the privacy-research branch's ppv (Pose, UNet3d, BDQ) on PYTHONPATH and its
checkpoints from the HF repo (runs/<run>/anonymizer.pt):

    PYTHONPATH=<privacy-research>/research/privacy-obfuscation \\
    uv run --no-project --with torch --with torchvision --with huggingface_hub \\
        --with pillow --with numpy python human_study/make_stimuli.py --cache <dir>

Writes human_study/stimuli/*.gif and human_study/answer_key.csv.
"""

import argparse
import csv
import io
import os
import random
import sys
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))  # PYTHONPATH (privacy-research's ppv) wins
from ppv.anonymizers import build, frozen_key  # noqa: E402
from ppv.kth import decode_avi, parse_sequences  # noqa: E402

BASE = "https://www.csc.kth.se/cvap/actions/"
HF_REPO = "KuanKuanKuan/openlatent-privacy"
STRIDE, N_FRAMES, MS = 2, 32, 80  # the attackers' clip stride; 2.6 s at 12.5 fps
SPAN = (N_FRAMES - 1) * STRIDE + 1
VISIBLE = 0.001  # a frame "shows something" when this fraction of pixels is off the background level
CTX = 8  # context frames each side, as train.materialize
LINEUP = ("walking", 4)  # every lineup clip: raw, walking, d4

# Condition -> (repo run, anonymizer config, has checkpoint, label shown in the intro).
# Strong-protocol machine identity (KTH, chance 4; raw 99.5): blur 95.0, unet 89.1, bdq 68.6, pose 29.1.
# Chosen from preview/ (every candidate on all six actions): these four keep every action visible.
# Not used: p2_bdq_a3_lr3e-3 (46.2) blanks boxing/waving/clapping; p11_kth_mask_k25_bdq_a1 shows the
# mask's patch grid, not the person; p11_kth_mask_k25_a1 pastes the real person; p2_vae_a1_ens
# releases a flat grey frame.
CONDS = {
    "blur": ("p1_blur8", {"type": "blur", "sigma": 8}, False, "Blur"),
    "unet": ("p2_unet_a1_ens", {"type": "unet3d"}, True, "Silhouette"),
    "bdq": ("p2_bdq_a3_ens", {"type": "bdq"}, True, "Motion patches"),
    "pose": ("p12_kth_pose_canon_e16", {"type": "pose", "every": 16, "width": 2, "canonical": 0.5}, False, "Stick figure"),
}
TARGETS = [2, 3, 6, 8, 11, 12, 15, 17, 20, 22]
EXAMPLE = (25, "jogging", 1)  # intro example; person 25 is in no trial
ACTIONS = ("boxing", "handclapping", "handwaving", "walking", "jogging", "running")
QUERY_CLIPS = [(a, d) for a in ACTIONS for d in (1, 2)]  # d1/d2: same clothes as the d4 lineup


def design(seed=0):
    """20 trials, every query obfuscated: 10 targets, each seen under two different filters.
    The 10 filter pairs are all 6 pairs of the 4 filters plus a 4-cycle, so every filter has
    5 trials and every pair of filters is compared within at least one person.
    Trials 1-10 show every target once, 11-20 the other filter (5+ trials later); each filter
    has 2-3 trials in each half; no filter twice in a row."""
    names = list(CONDS)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pairs += [(names[i], names[(i + 1) % 4]) for i in range(4)]
    rng = random.Random(seed)
    while True:
        flip = [rng.random() < 0.5 for _ in pairs]
        lead = [(b, a) if f else (a, b) for (a, b), f in zip(pairs, flip)]
        if not all(sum(p[0] == c for p in lead) in (2, 3) for c in names):
            continue
        order, order2 = rng.sample(range(10), 10), rng.sample(range(10), 10)
        conds = [lead[i][0] for i in order] + [lead[i][1] for i in order2]
        tgts = [TARGETS[i] for i in order] + [TARGETS[i] for i in order2]
        if all(a != b for a, b in zip(conds, conds[1:])) and all(tgts.index(t, 10) - tgts.index(t) >= 5 for t in TARGETS):
            break
    while True:  # a target's two clips differ in action and scenario; each filter gets >= 4 actions and both
        # scenarios; every action is used 3-4 times overall
        clips = [None] * 20
        for t in TARGETS:
            i, j = tgts.index(t), tgts.index(t, 10)
            while clips[i] is None or clips[i][0] == clips[j][0] or clips[i][1] == clips[j][1]:
                clips[i], clips[j] = rng.sample(QUERY_CLIPS, 2)
        per = {c: [clips[k] for k in range(20) if conds[k] == c] for c in names}
        counts = [sum(a == x for x, _ in clips) for a in ACTIONS]
        if all(len({a for a, _ in v}) >= 4 and len({d for _, d in v}) == 2 for v in per.values()) and min(counts) >= 3:
            break
    while True:  # 5 answers per letter; a target's two answers differ
        letters = rng.sample(list("AAAAABBBBBCCCCCDDDDD"), 20)
        if all(letters[tgts.index(t)] != letters[tgts.index(t, 10)] for t in TARGETS):
            break
    # distractors: least-used first (random tie-break), never repeated across a target's two lineups;
    # targets start at 2 uses, so "seen before" says nothing about who the target is
    used = {p: 2 if p in TARGETS else 0 for p in range(1, 25)}
    trials = []
    for k in range(20):
        t = tgts[k]
        avoid = {t}.union(*(x["lineup"] for x in trials if x["target"] == t))
        lineup = sorted((p for p in used if p not in avoid), key=lambda p: (used[p], rng.random()))[:3]
        for p in lineup:
            used[p] += 1
        lineup.insert("ABCD".index(letters[k]), t)
        lseq = k % 4 + 1
        if any(x["target"] == t and x["lineup_seq"] == lseq for x in trials):  # a target's two lineups differ
            lseq = lseq % 4 + 1
        trials.append(dict(trial=k + 1, target=t, condition=conds[k], action=clips[k][0], scenario=clips[k][1],
                           lineup_seq=lseq, lineup=lineup, answer=letters[k]))
    return trials


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


def load(cond):
    run, cfg, ckpt, _ = CONDS[cond]
    if cfg is None:
        return None
    f = build(cfg, in_ch=1).eval()
    if ckpt:
        from huggingface_hub import hf_hub_download
        sd = torch.load(hf_hub_download(HF_REPO, f"runs/{run}/anonymizer.pt"), map_location="cpu")
        missing, unexpected = f.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not frozen_key(k)] and not unexpected, (missing, unexpected)
    return f


def run(f, frames):
    x = torch.from_numpy(frames.copy()).float().div(255)[None, None]
    with torch.no_grad():
        return (f(x).clamp(0, 1)[0, 0] * 255).round().byte().numpy()


def window(frames, s):
    return frames[np.clip(np.arange(s, s + SPAN, STRIDE), 0, len(frames) - 1)]


def centre(seqs, video, seq):
    a, b = seqs[video][seq - 1]
    return max(a - 1, (a - 1 + b - SPAN) // 2)


def activity(frames):
    """Per-frame fraction of pixels off the most common (background) level, ignoring a 4-pixel
    border (the U-Net draws a thin line along the frame edge)."""
    inner = frames[:, 4:-4, 4:-4]
    return (inner != np.bincount(inner.ravel()).argmax()).mean((1, 2))


def ranked(seqs, video, motion):
    """(seq, start) of every window, most BDQ activity first. One motion measure ranks the
    windows in every condition: the attacker's best case."""
    act = activity(motion)
    return [w[1:] for w in sorted(((window(act, s).mean(), i, s) for i, (a, b) in enumerate(seqs[video], 1)
                                   for s in range(a - 1, max(a, b - SPAN + 1), 2)), reverse=True)]


GREYS = 32  # fixed palette keeps GIFs small
PALETTE = [v for i in range(GREYS) for v in [i * 255 // (GREYS - 1)] * 3]
FONT = ImageFont.load_default(size=18)


def label(img, text):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=FONT)
    d.rectangle((4, 4, 4 + x1 - x0 + 10, 4 + y1 + 6), fill=255)
    d.text((9, 5), text, fill=0, font=FONT)


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


def lineup_grid(pairs, letters, filter_label, scale=2, inner=2, outer=14, cols=2):
    """Candidates in a 2x2 grid; each cell is the raw clip with the filtered clip right beneath it."""
    h, w = 120 * scale, 160 * scale
    cell_h = 2 * h + inner
    rows = -(-len(pairs) // cols)
    out = np.full((N_FRAMES, rows * cell_h + (rows - 1) * outer, cols * w + (cols - 1) * outer), 255, np.uint8)
    boxes = []
    for i, (raw, filt) in enumerate(pairs):
        r, c = divmod(i, cols)
        y, x = r * (cell_h + outer), c * (w + outer)
        out[:, y:y + h, x:x + w] = tile(raw, scale)
        out[:, y + h + inner:y + cell_h, x:x + w] = tile(filt, scale)
        boxes += [(x, y, letters[i]), (x, y + h + inner, f"{letters[i]} {filter_label.lower()}")]
    frames = []
    for f in out:
        img = Image.fromarray(f)
        for x, y, text in boxes:
            sub = img.crop((x, y, x + w, y + h))
            label(sub, text)
            img.paste(sub, (x, y))
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
    fs = {c: load(c) for c in CONDS}

    def query(person, action, d, cond):
        """The highest-ranked window whose `cond` output shows something (>= 0.1% of pixels off the
        background level) in at least half its frames: the pose detector can miss a person on every
        keyframe of a window, a runner can leave the frame, the U-Net can shrink a far person to
        specks. Failing that, the top-40 window where the most frames show something."""
        video = f"person{person:02d}_{action}_d{d}"
        raw = fetch(args.cache, video)
        best = None
        for k, (seq, s) in enumerate(ranked(seqs, video, run(fs["bdq"], raw))[:40]):
            q = render(cond, raw, s)
            seen = (activity(q) >= VISIBLE).mean()
            if seen >= 0.5:
                return video, seq, s, raw, q, k, seen
            if best is None or seen > best[-1]:
                best = (video, seq, s, raw, q, k, seen)
        return best

    def render(cond, raw, s):
        """The window at full frame rate plus CTX frames of context each side (as materialize), then strided."""
        if fs[cond] is None:
            return window(raw, s)
        a, b = max(0, s - CTX), min(len(raw), s + SPAN + CTX)
        return window(run(fs[cond], raw[a:b]), s - a)

    _, _, s, raw, _, _, _ = query(*EXAMPLE, "pose")
    save_gif(os.path.join(out_dir, "example.gif"),
             grid([window(raw, s)] + [render(c, raw, s) for c in CONDS], ["Original"] + [CONDS[c][3] for c in CONDS], cols=3))

    key = []
    trials = design()
    for tr in trials:
        t, cond = tr["trial"], tr["condition"]
        # if no window of the assigned clip shows the person (the U-Net can erase a far runner),
        # try the target's other clips in the same scenario, never its other trial's action
        other = next(x["action"] for x in trials if x["target"] == tr["target"] and x is not tr)
        best = None
        for action in [tr["action"]] + [a for a in ACTIONS if a not in (tr["action"], other)]:
            res = query(tr["target"], action, tr["scenario"], cond)
            if best is None or res[-1] > best[-1]:
                best = res
            if res[-1] >= 0.5:
                break
        video, seq, s, raw, q, rank, seen = best
        save_gif(os.path.join(out_dir, f"t{t:02d}_query.gif"), tile(q, 2))
        # every candidate: the same walking window raw and through this trial's filter; if the filter
        # shows nothing in that window (a pose miss), the candidate's next subsequence
        pairs, used_seqs = [], []
        for p in tr["lineup"]:
            v = f"person{p:02d}_{LINEUP[0]}_d{LINEUP[1]}"
            lraw = fetch(args.cache, v)
            order = [tr["lineup_seq"]] + [x for x in range(1, len(seqs[v]) + 1) if x != tr["lineup_seq"]]
            for lseq in order:
                ls = centre(seqs, v, lseq)
                filt = render(cond, lraw, ls)
                if (activity(filt) >= VISIBLE).mean() >= 0.5:
                    break
            pairs.append((window(lraw, ls), filt))
            used_seqs.append(lseq)
        save_gif(os.path.join(out_dir, f"t{t:02d}_lineup.gif"), lineup_grid(pairs, "ABCD", CONDS[cond][3]))
        bdq_act = activity(render("bdq", raw, s)).mean()
        key.append(dict(trial=t, target_person=tr["target"], condition=cond, run=CONDS[cond][0],
                        query=f"{video} seq{seq} start{s}", window_rank=rank, bdq_active_pct=round(100 * bdq_act, 2),
                        lineup=" ".join(f"{l}=p{p:02d}" for l, p in zip("ABCD", tr["lineup"])),
                        lineup_seqs=" ".join(map(str, used_seqs)), answer=tr["answer"]))
        print(f"trial {t:2d}: p{tr['target']:02d} {cond:5s} {video} -> {tr['answer']}  (window rank {rank}, bdq {100 * bdq_act:.2f}%)", flush=True)
    with open(os.path.join(HERE, "answer_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)


if __name__ == "__main__":
    main()
