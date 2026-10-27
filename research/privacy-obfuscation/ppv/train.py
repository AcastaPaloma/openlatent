"""Training loops: classifiers, adversarial anonymizer training, and dataset materialisation.

Privacy labels come in two kinds:
    multiclass  one class per clip (KTH: 25 people); cross-entropy, metric top-1
    multilabel  several binary attributes (VP-UCF101: 5); BCE, metric class-mean AP (cMAP)
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import models
from .clips import Clips, worker_init

DEV = "cuda"


def to_input(clip):
    """uint8 (B, T, H, W[, C]) -> float (B, C, T, H, W) in [0, 1] on the GPU."""
    x = clip.to(DEV, non_blocking=True).float().div_(255)
    return x.unsqueeze(1) if x.dim() == 4 else x.permute(0, 4, 1, 2, 3)


def loader(rows, frames, train, batch, repeats=1, n_test=3, workers=8, seed=0, stride=2):
    ds = Clips(rows, frames, train, stride=stride, repeats=repeats, n_test=n_test, seed=seed)
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train, num_workers=workers,
                      pin_memory=True, persistent_workers=False, worker_init_fn=worker_init)


def cosine(opt, total, warmup):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))


def label_of(batch, task):
    return batch[1 if task == "action" else 2].to(DEV)


def kind_of(task, priv_kind):
    return "multiclass" if task == "action" else priv_kind


def loss_fn(logits, y, kind):
    if kind == "multilabel":
        return F.binary_cross_entropy_with_logits(logits.float(), y.float())
    return F.cross_entropy(logits.float(), y)


def n_correct(logits, y, kind):
    """Correct predictions (multiclass) or correct attribute decisions / n_attrs (multilabel)."""
    if kind == "multilabel":
        return ((logits > 0).float() == y).float().mean(1).sum().item()
    return (logits.argmax(1) == y).sum().item()


def frame_subsample(x, k):
    """k random frames of each clip (frame attackers learn per-frame identity; cheaper)."""
    if k is None or k >= x.shape[2]:
        return x
    idx = torch.randperm(x.shape[2], device=x.device)[:k].sort().values
    return x[:, :, idx]


def train_classifier(model, dl, task, epochs, lr=2e-4, wd=0.05, frames_per_clip=None, priv_kind="multiclass", log=print):
    kind = kind_of(task, priv_kind)
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = cosine(opt, epochs * len(dl), warmup=len(dl) // 2)
    for ep in range(epochs):
        t0, tot, correct, n = time.time(), 0.0, 0.0, 0
        for batch in dl:
            x, y = to_input(batch[0]), label_of(batch, task)
            if frames_per_clip:
                x = frame_subsample(x, frames_per_clip)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, y, kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += n_correct(logits.float(), y, kind)
            n += len(y)
        log(f"  [{task}] epoch {ep + 1}/{epochs} loss {tot / n:.3f} train_acc {100 * correct / n:.1f} ({time.time() - t0:.0f}s)")
    return model


def average_precision(scores, labels):
    """AP of one binary attribute (sklearn's definition, without the dependency)."""
    order = np.argsort(-scores)
    y = labels[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


@torch.no_grad()
def evaluate(model, dl, task, n_rows, priv_kind="multiclass"):
    """Scores over test rows; per-row scores are averaged over that row's clips.
    Returns {"main": top-1 or cMAP (percent), ...details}."""
    kind = kind_of(task, priv_kind)
    model.to(DEV).eval()
    agg, labels = None, None
    for batch in dl:
        x = to_input(batch[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float().cpu()
        y = label_of(batch, task).cpu()
        if agg is None:
            agg = torch.zeros(n_rows, logits.shape[1])
            labels = torch.zeros((n_rows,) + tuple(y.shape[1:]), dtype=y.dtype)
        probs = logits.sigmoid() if kind == "multilabel" else logits.softmax(1)
        agg.index_add_(0, batch[3], probs)
        labels[batch[3]] = y
    if kind == "multilabel":
        s, lab = agg.numpy(), labels.numpy()
        aps = [average_precision(s[:, j], lab[:, j]) for j in range(s.shape[1])]
        pred = (agg / dl.dataset.n_test > 0.5).float()
        tp = (pred * labels).sum(0)
        f1 = (2 * tp / (pred.sum(0) + labels.sum(0)).clamp(min=1)).mean().item()
        return {"main": 100 * float(np.nanmean(aps)), "ap": [100 * a for a in aps], "f1": 100 * f1,
                "prevalence": [100 * float(v) for v in lab.mean(0)]}
    return {"main": 100.0 * (agg.argmax(1) == labels).float().mean().item()}


def privacy_entropy(logits, kind):
    """Mean entropy of the adversary's prediction (nats); the anonymizer maximises it."""
    if kind == "multilabel":
        p = logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()).sum(1).mean()
    p = logits.float().log_softmax(1)
    return -(p.exp() * p).sum(1).mean()


def train_anonymizer(f, dl, cfg, n_actions, n_priv, priv_kind="multiclass", log=print):
    """Adversarial training (Wu et al. 2018/2020; BDQ 2022 use the same game):
        min over f, T : CE(T(f(x)), action) - alpha * H(B(f(x)))      [push B to max entropy]
        min over B    : privacy loss of B(f(x))
    B is re-initialised every `restart_every` epochs so f cannot overfit to one adversary.
    """
    T = models.build(cfg.get("utility", "r2p1d"), n_actions).to(DEV).train()
    new_B = lambda: models.build(cfg.get("adversary", "frame_r18"), n_priv).to(DEV).train()  # noqa: E731
    B = new_B()
    f.to(DEV).train()
    fparams = [p for p in f.parameters() if p.requires_grad]
    opt_fT = torch.optim.AdamW([{"params": fparams, "lr": cfg.get("lr_f", 1e-4)},
                                {"params": T.parameters(), "lr": cfg.get("lr_t", 1e-4)}], weight_decay=0.01)
    opt_B = torch.optim.AdamW(B.parameters(), lr=cfg.get("lr_b", 2e-4), weight_decay=0.01)
    alpha, epochs, restart = cfg.get("alpha", 1.0), cfg["epochs"], cfg.get("restart_every", 0)
    max_ent = n_priv * math.log(2) if priv_kind == "multilabel" else math.log(n_priv)
    hist = []
    for ep in range(epochs):
        if restart and ep and ep % restart == 0:
            B = new_B()
            opt_B = torch.optim.AdamW(B.parameters(), lr=cfg.get("lr_b", 2e-4), weight_decay=0.01)
            log(f"  adversary restarted at epoch {ep}")
        t0 = time.time()
        s = dict(la=0.0, ent=0.0, lb=0.0, acc_a=0.0, acc_b=0.0, n=0)
        for batch in dl:
            x, ya, yp = to_input(batch[0]), batch[1].to(DEV), batch[2].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = f(x)
                la_logits = T(z)
                pb = B(z)
                la = F.cross_entropy(la_logits.float(), ya)
                ent = privacy_entropy(pb, priv_kind)
                loss = la - alpha * ent
            opt_fT.zero_grad(set_to_none=True)
            loss.backward()
            opt_fT.step()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pb2 = B(z.detach())
                lb = loss_fn(pb2, yp, priv_kind)
            opt_B.zero_grad(set_to_none=True)
            lb.backward()
            opt_B.step()
            n = len(ya)
            s["la"] += la.item() * n
            s["ent"] += ent.item() * n
            s["lb"] += lb.item() * n
            s["acc_a"] += (la_logits.argmax(1) == ya).sum().item()
            s["acc_b"] += n_correct(pb2.float(), yp, priv_kind)
            s["n"] += n
        n = s["n"]
        row = dict(epoch=ep + 1, action_loss=s["la"] / n, adv_entropy=s["ent"] / n, adv_loss=s["lb"] / n,
                   train_action_acc=100 * s["acc_a"] / n, train_adv_acc=100 * s["acc_b"] / n)
        extra = ""
        if hasattr(f, "log_sigma"):
            row["sigma"] = f.log_sigma.exp().item()
            extra = f" sigma {row['sigma']:.2f}"
        hist.append(row)
        log(f"  [adv] epoch {ep + 1}/{epochs} action_acc {row['train_action_acc']:.1f} adv_acc {row['train_adv_acc']:.1f} "
            f"adv_entropy {row['adv_entropy']:.2f}/{max_ent:.2f}{extra} ({time.time() - t0:.0f}s)")
    return f, hist


@torch.no_grad()
def materialize(f, frames, window=32, ctx=2, batch=8, out_dir=None):
    """Apply f to every frame of every video; returns {video: uint8 (T, H, W[, 3])}.
    Windows carry `ctx` frames of left context so temporal ops (differences, 3D convs)
    see real neighbours at window boundaries; only the window's own frames are kept.
    With `out_dir`, each video is written to <out_dir>/<video>.npy and memory-mapped back
    (for datasets too large to hold a second copy in RAM)."""
    f.to(DEV).eval()
    out = {}
    for v, arr in frames.items():
        n = len(arr)
        starts = list(range(0, n, window))
        chunks, keep = [], []
        for s in starts:
            a = max(0, s - ctx)
            e = min(n, s + window)
            seg = np.asarray(arr[a:e])
            if len(seg) < window + ctx:  # pad by repeating the last frame
                seg = np.concatenate([seg, np.repeat(seg[-1:], window + ctx - len(seg), 0)])
            chunks.append(seg)
            keep.append((s - a, min(window, n - s)))
        res = []
        for i in range(0, len(chunks), batch):
            x = to_input(torch.from_numpy(np.stack(chunks[i:i + batch])))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = f(x).float()
            y = (y.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            for j, yj in enumerate(y):
                off, length = keep[i + j]
                yj = yj[:, off:off + length]  # (C, L, H, W)
                res.append(yj[0].numpy() if yj.shape[0] == 1 else yj.permute(1, 2, 3, 0).numpy())
        full = np.concatenate(res)
        if out_dir is None:
            out[v] = full
        else:
            path = os.path.join(out_dir, v + ".npy")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, full)
            out[v] = np.load(path, mmap_mode="r")
    return out
