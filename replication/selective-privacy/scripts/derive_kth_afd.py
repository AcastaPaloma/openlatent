"""Motion-consistent noise (AFD) and optical flow for KTH test-split clips, with batched RAFT.

Why this exists. Upstream afd/generate_kth.py runs over every KTH clip and calls
util.flow_from_video, which runs RAFT on one frame pair at a time at 4x upsampling.
Measured here at about 0.2 s per pair, so all 289k frames would take several hours.
Obfuscation is applied only at test time, so only the 150 test-split clips
(73,756 frames) need AFD. This script:

  1. processes only the test split (metadata `set == test`),
  2. runs RAFT on batches of frame pairs instead of one pair at a time,
  3. keeps the per-pair math, the 301-frame chunking, the per-chunk set_seed() and the
     AFD warp (util.firstframe_warp) exactly as upstream has them.

Batching changes only float rounding, since RAFT's encoders normalise per sample.
`--check` compares the batched flow with upstream flow_from_video on one clip before
any full run.

Writes <upstream>/data/kth_afd/<video>/frameNNNN.png and kth_flow, the same names
upstream uses. A video is skipped when its output counts already match.

Usage (from the upstream checkout):
    python <repo>/replication/selective-privacy/scripts/derive_kth_afd.py --batch 8
    python <repo>/replication/selective-privacy/scripts/derive_kth_afd.py --check person01_boxing_d1_uncomp
"""

import argparse
import csv
import glob
import os
import sys
import argparse as _argparse

UPSTREAM = os.getcwd()
AFD_DIR = os.path.join(UPSTREAM, "afd")
sys.path.insert(0, AFD_DIR)
os.chdir(AFD_DIR)  # RAFT and util load their weights and modules relative to afd/

import cv2  # noqa: E402
import flowiz as fz  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.io import read_image  # noqa: E402

from RAFT.core.raft import RAFT  # noqa: E402
from RAFT.core.utils.utils import InputPadder  # noqa: E402
from util import chunker, firstframe_warp, flow_from_video, set_seed, upsample_flow  # noqa: E402

DATA = os.path.join(UPSTREAM, "data")
META = os.path.join(DATA, "splits/kth/metadata.csv")
CHUNK = 301  # upstream generate_kth.py chunksize
UPSAMPLE = 4  # upstream generate_kth.py calls flow_from_video(..., upsample_factor=4)


def load_raft():
    args = _argparse.Namespace(small=False, mixed_precision=True, alternate_corr=False, dropout=0)  # Namespace supports `in`, as RAFT needs
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("RAFT/models/raft-sintel.pth"))
    model = model.module
    model.cuda()
    model.eval()
    return model


def flow_batched(model, frames, upsample_factor=UPSAMPLE, batch=8):
    """Same per-pair computation as util.flow_from_video, run on batches of pairs."""
    n = len(frames) - 1
    _, h, w = frames[0].shape
    flows = []
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = list(range(start, min(start + batch, n)))
            image0 = torch.stack([frames[i] for i in idx]).cuda()
            image1 = torch.stack([frames[i + 1] for i in idx]).cuda()
            if upsample_factor != 1:
                up = nn.Upsample(scale_factor=upsample_factor, mode="bilinear")
                image0 = up(image0)
                image1 = up(image1)
            padder = InputPadder(image0.shape)
            image0, image1 = padder.pad(image0, image1)
            _, flow_up = model(image0, image1, iters=12, test_mode=True)
            flow_up = padder.unpad(flow_up).detach().cpu()  # (B, 2, H', W')
            for j in range(flow_up.shape[0]):
                fl = flow_up[j].permute(1, 2, 0)  # (H', W', 2), as upstream
                flows.append(upsample_flow(fl, h, w))
    return flows


def check_against_upstream(model, video, n_frames=10, batch=8):
    paths = sorted(glob.glob(os.path.join(DATA, "kth", video, "*.png")))[:n_frames]
    frames = [read_image(p).float() for p in paths]
    ref = flow_from_video(model, frames, upsample_factor=UPSAMPLE)
    new = flow_batched(model, frames, batch=batch)
    diffs = torch.stack([(a - b).abs().max() for a, b in zip(ref, new)])
    print(f"check {video}: {len(ref)} pairs, max |flow diff| = {diffs.max().item():.6f} px, "
          f"mean of per-pair max = {diffs.mean().item():.6f} px")
    afd_ref = firstframe_warp(ref, usecolor=True)
    set_seed()
    afd_new = firstframe_warp(new, usecolor=True)
    frac = (afd_ref != afd_new).float().mean().item()
    print(f"check {video}: AFD pixels that differ = {frac * 100:.3f}%")


def test_videos():
    with open(META, newline="") as f:
        return sorted(r["video"] for r in csv.DictReader(f) if r["set"] == "test")


def write_png(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, tensor.numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 9])


def process_video(model, video, batch):
    frame_paths = sorted(glob.glob(os.path.join(DATA, "kth", video, "*.png")))
    afd_dir = os.path.join(DATA, "kth_afd", video)
    flow_dir = os.path.join(DATA, "kth_flow", video)
    names = [os.path.basename(p) for p in frame_paths]
    have_afd = set(os.listdir(afd_dir)) if os.path.isdir(afd_dir) else set()
    have_flow = set(os.listdir(flow_dir)) if os.path.isdir(flow_dir) else set()
    if set(names) <= have_afd and set(names) <= have_flow:
        return "skipped"

    for chunk in chunker(frame_paths, CHUNK):
        set_seed()
        frame_list = [read_image(p).float() for p in chunk]
        flow = flow_batched(model, frame_list, batch=batch)

        rgbflows = torch.from_numpy(
            fz.video_flow2color(fz.video_normalized_flow(torch.stack(flow, dim=0)))
        )
        rgbflows = torch.cat([rgbflows, rgbflows[-1, ...].unsqueeze(0)], dim=0)
        afd = firstframe_warp(flow, usecolor=True)

        for im, p in zip(afd.unbind(0), chunk):
            write_png(os.path.join(afd_dir, os.path.basename(p)), im)
        for im, p in zip(rgbflows.unbind(0), chunk):
            write_png(os.path.join(flow_dir, os.path.basename(p)), im)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8, help="RAFT frame pairs per forward pass")
    ap.add_argument("--check", type=str, default=None, help="validate batched flow on this clip and exit")
    args = ap.parse_args()

    model = load_raft()
    if args.check:
        check_against_upstream(model, args.check, batch=args.batch)
        return

    videos = test_videos()
    print(f"KTH test split: {len(videos)} clips")
    for i, video in enumerate(videos, 1):
        status = process_video(model, video, args.batch)
        print(f"[{i}/{len(videos)}] {video}: {status}", flush=True)


if __name__ == "__main__":
    main()
