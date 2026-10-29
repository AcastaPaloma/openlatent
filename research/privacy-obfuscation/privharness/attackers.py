"""Attackers beyond ppv.models' pixel classifiers.

skeleton  a gait / body attacker: Keypoint R-CNN (COCO) finds the most confident person in every frame of the
          *obfuscated* footage; hip-centred, torso-scaled keypoint sequences feed a small temporal convolution
          network trained to predict the secret. Skeletons re-identify people well (Moon et al., AAAI 2023: ~80%
          re-identification from pose), so any obfuscator that leaves people detectable must face this attacker.
          On outputs where no person can be detected it falls to chance and the video attackers carry the weight.

The task-label attacker (the baseline leakage is measured from) is numpy-only and lives in scoring.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"


def keypoint_sequences(frames, items, t=16, stride=2, n_clips=2, log=print):
    """-> (n_items, n_clips, t, 17 * 3) normalised keypoints from obfuscated frames ({frames path: array})."""
    from .views import Detectors, sample_clip
    det = Detectors()
    out = np.zeros((len(items), n_clips, t, 17 * 3), np.float32)
    for i, it in enumerate(items):
        arr = frames[it["frames"]]
        n = it["end"] - it["start"] + 1
        span = (t - 1) * stride + 1
        starts = np.linspace(it["start"], max(it["start"], it["end"] - span + 1), n_clips).round().astype(int)
        clips = [sample_clip(arr, s, min(it["end"], s + span - 1), t, stride) if n >= span else sample_clip(arr, it["start"], it["end"], t, stride)
                 for s in starts]
        dets = det(np.concatenate(clips))
        for j in range(n_clips):
            for k in range(t):
                kp = dets[j * t + k]["kps"]
                if kp is None:
                    continue
                hip = kp[[11, 12], :2].mean(0)
                sho = kp[[5, 6], :2].mean(0)
                torso = max(float(np.linalg.norm(sho - hip)), 1.0)
                xy = (kp[:, :2] - hip) / torso
                conf = (kp[:, 2:3] > 2.0).astype(np.float32)
                out[i, j, k] = np.concatenate([xy * conf, conf], 1).ravel()
        if i % 500 == 0:
            log(f"    skeletons {i}/{len(items)}")
    return out


class TCN(nn.Module):
    def __init__(self, d_in, n_out, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, width, 3, padding=1), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1, dilation=1), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2), nn.BatchNorm1d(width), nn.GELU())
        self.head = nn.Linear(2 * width, n_out)

    def forward(self, x):  # (b, t, d)
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


def train_skeleton(X_tr, y_tr, X_te, kind, n_out, epochs=150, seed=0):
    """X: (n, clips, t, d). Trains on every clip, predicts each test item as the mean over its clips."""
    torch.manual_seed(seed)
    m = TCN(X_tr.shape[-1], n_out).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Xa = torch.tensor(X_tr.reshape(-1, *X_tr.shape[2:]), device=DEV)
    ya = np.repeat(y_tr, X_tr.shape[1], 0)
    Ya = torch.tensor(ya, device=DEV, dtype=torch.long if kind == "multiclass" else torch.float32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xa), device=DEV)
        for i in range(0, len(Xa), 256):
            b = perm[i:i + 256]
            x = Xa[b] + 0.02 * torch.randn_like(Xa[b])  # small jitter: keypoints are noisy
            logits = m(x)
            loss = F.cross_entropy(logits, Ya[b]) if kind == "multiclass" else F.binary_cross_entropy_with_logits(logits, Ya[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        Xb = torch.tensor(X_te.reshape(-1, *X_te.shape[2:]), device=DEV)
        logits = torch.cat([m(Xb[i:i + 1024]) for i in range(0, len(Xb), 1024)])
        p = logits.softmax(1) if kind == "multiclass" else logits.sigmoid()
    return p.view(len(X_te), X_te.shape[1], -1).mean(1).cpu().numpy()
