"""Training loops: classifiers, adversarial anonymizer training, and dataset materialisation."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import models
from .kth import Clips, worker_init

DEV = "cuda"


def to_input(clip):
    """uint8 (B, T, H, W[, C]) -> float (B, C, T, H, W) in [0, 1] on the GPU."""
    x = clip.to(DEV, non_blocking=True).float().div_(255)
    return x.unsqueeze(1) if x.dim() == 4 else x.permute(0, 4, 1, 2, 3)


def loader(rows, frames, train, batch, repeats=1, n_test=3, workers=8, seed=0):
    ds = Clips(rows, frames, train, repeats=repeats, n_test=n_test, seed=seed)
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train, num_workers=workers,
                      pin_memory=True, persistent_workers=False, worker_init_fn=worker_init)


def cosine(opt, total, warmup):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))


def label_of(batch, task):
    return batch[1 if task == "action" else 2].to(DEV)


def frame_subsample(x, k):
    """k random frames of each clip (frame attackers learn per-frame identity; cheaper)."""
    if k is None or k >= x.shape[2]:
        return x
    idx = torch.randperm(x.shape[2], device=x.device)[:k].sort().values
    return x[:, :, idx]


def train_classifier(model, dl, task, epochs, lr=2e-4, wd=0.05, frames_per_clip=None, log=print):
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = cosine(opt, epochs * len(dl), warmup=len(dl) // 2)
    for ep in range(epochs):
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for batch in dl:
            x, y = to_input(batch[0]), label_of(batch, task)
            if frames_per_clip:
                x = frame_subsample(x, frames_per_clip)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        log(f"  [{task}] epoch {ep + 1}/{epochs} loss {tot / n:.3f} train_acc {100 * correct / n:.1f} ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def evaluate(model, dl, task, n_rows, anonymizer=None):
    """Top-1 over test rows; logits are summed over each row's clips."""
    model.to(DEV).eval()
    agg, labels = None, torch.full((n_rows,), -1, dtype=torch.long)
    for batch in dl:
        x = to_input(batch[0])
        if anonymizer is not None:
            x = anonymizer(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float().cpu()
        if agg is None:
            agg = torch.zeros(n_rows, logits.shape[1])
        agg.index_add_(0, batch[3], logits.softmax(1))
        labels[batch[3]] = label_of(batch, task).cpu()
    return 100.0 * (agg.argmax(1) == labels).float().mean().item()


def entropy(logits):
    p = logits.float().log_softmax(1)
    return -(p.exp() * p).sum(1).mean()


def train_anonymizer(f, dl, cfg, n_actions, n_people, log=print):
    """Adversarial training (Wu et al. 2018/2020; BDQ 2022 use the same game):
        min over f, T : CE(T(f(x)), action) - alpha * H(B(f(x)))      [push B to uniform]
        min over B    : CE(B(f(x)), person)
    B is re-initialised every `restart_every` epochs so f cannot overfit to one adversary.
    """
    T = models.build(cfg.get("utility", "r2p1d"), n_actions).to(DEV).train()
    new_B = lambda: models.build(cfg.get("adversary", "frame_r18"), n_people).to(DEV).train()  # noqa: E731
    B = new_B()
    f.to(DEV).train()
    fparams = [p for p in f.parameters() if p.requires_grad]
    opt_fT = torch.optim.AdamW([{"params": fparams, "lr": cfg.get("lr_f", 1e-4)},
                                {"params": T.parameters(), "lr": cfg.get("lr_t", 1e-4)}], weight_decay=0.01)
    opt_B = torch.optim.AdamW(B.parameters(), lr=cfg.get("lr_b", 2e-4), weight_decay=0.01)
    alpha, epochs, restart = cfg.get("alpha", 1.0), cfg["epochs"], cfg.get("restart_every", 0)
    hist = []
    for ep in range(epochs):
        if restart and ep and ep % restart == 0:
            B = new_B()
            opt_B = torch.optim.AdamW(B.parameters(), lr=cfg.get("lr_b", 2e-4), weight_decay=0.01)
            log(f"  adversary restarted at epoch {ep}")
        t0 = time.time()
        s = dict(la=0.0, ent=0.0, lb=0.0, acc_a=0, acc_b=0, n=0)
        for batch in dl:
            x, ya, yp = to_input(batch[0]), batch[1].to(DEV), batch[2].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = f(x)
                la_logits = T(z)
                pb = B(z)
                la = F.cross_entropy(la_logits.float(), ya)
                ent = entropy(pb)
                loss = la - alpha * ent
            opt_fT.zero_grad(set_to_none=True)
            loss.backward()
            opt_fT.step()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pb2 = B(z.detach())
                lb = F.cross_entropy(pb2.float(), yp)
            opt_B.zero_grad(set_to_none=True)
            lb.backward()
            opt_B.step()
            n = len(ya)
            s["la"] += la.item() * n
            s["ent"] += ent.item() * n
            s["lb"] += lb.item() * n
            s["acc_a"] += (la_logits.argmax(1) == ya).sum().item()
            s["acc_b"] += (pb2.argmax(1) == yp).sum().item()
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
            f"adv_entropy {row['adv_entropy']:.2f}/{math.log(n_people):.2f}{extra} ({time.time() - t0:.0f}s)")
    return f, hist


@torch.no_grad()
def materialize(f, frames, window=32, ctx=2, batch=8):
    """Apply f to every frame of every video; returns {video: uint8 (T, H, W[, 3])}.
    Windows carry `ctx` frames of left context so temporal ops (differences, 3D convs)
    see real neighbours at window boundaries; only the window's own frames are kept."""
    f.to(DEV).eval()
    out = {}
    for v, arr in frames.items():
        n = len(arr)
        starts = list(range(0, n, window))
        chunks, keep = [], []
        for s in starts:
            a = max(0, s - ctx)
            e = min(n, s + window)
            seg = arr[a:e]
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
        out[v] = np.concatenate(res)
    return out
