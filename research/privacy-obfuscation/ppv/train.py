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

from . import anonymizers, models
from .clips import Clips, LazyFrames, worker_init

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


# Action labels are one class per clip on KTH / UCF101 / HMDB51; Charades has several actions per video
# (multilabel, scored by mAP). run.py sets this per dataset; each run is its own process.
ACTION_KIND = "multiclass"


def kind_of(task, priv_kind):
    return ACTION_KIND if task == "action" else priv_kind


def loss_fn(logits, y, kind):
    if kind == "multilabel":
        return F.binary_cross_entropy_with_logits(logits.float(), y.float())
    return F.cross_entropy(logits.float(), y)


def positive_recall(logits, y):
    """Sum over the batch of each clip's recall on its positive labels (multilabel). Per-label accuracy is useless for
    the collapse guard on Charades: predicting no action at all scores ~96% there (6 of 157 actions per video)."""
    pos = y.sum(1).clamp(min=1)
    return (((logits > 0).float() * y).sum(1) / pos).sum().item()


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
    lr = lr * getattr(model, "lr_scale", 1.0)  # e.g. DINOv2: probe x5, fine-tune x0.1
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
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
    agg, labels, actions = None, None, torch.zeros(n_rows, dtype=torch.long)
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
        if batch[1].dim() == 1:
            actions[batch[3]] = batch[1]
    if kind == "multilabel":
        s, lab = agg.numpy(), labels.numpy()
        aps = [average_precision(s[:, j], lab[:, j]) for j in range(s.shape[1])]
        pred = (agg / dl.dataset.n_test > 0.5).float()
        tp = (pred * labels).sum(0)
        f1 = (2 * tp / (pred.sum(0) + labels.sum(0)).clamp(min=1)).mean().item()
        out = {"main": 100 * float(np.nanmean(aps)), "ap": [100 * a for a in aps], "f1": 100 * f1,
               "prevalence": [100 * float(v) for v in lab.mean(0)]}
        if kind != task and (actions > 0).any():  # privacy attributes with one action per clip (VP-UCF101 / VP-HMDB51)
            # Attribute labels are largely implied by the action (an attacker that only knows the action class reaches
            # ~76-78 cMAP on VP-UCF101). The action-conditional cMAP asks what the video reveals beyond its action: AP of
            # each attribute within each action class where it varies, averaged; its blind floor is the mean within-class
            # prevalence over the same (class, attribute) cells.
            a_np = actions.numpy()
            cond, floor = [], []
            for j in range(s.shape[1]):
                cj, fj = [], []
                for a in np.unique(a_np):
                    sel = a_np == a
                    yj = lab[sel, j]
                    if 0 < yj.sum() < len(yj):
                        cj.append(average_precision(s[sel, j], yj))
                        fj.append(float(yj.mean()))
                if cj:
                    cond.append(float(np.mean(cj)))
                    floor.append(float(np.mean(fj)))
            if cond:
                out.update(cond_main=100 * float(np.mean(cond)), cond_ap=[100 * c for c in cond],
                           cond_floor=100 * float(np.mean(floor)))
        return out
    correct = (agg.argmax(1) == labels).float()
    # per-action accuracy: separates e.g. identity from locomotion clips (person crosses the
    # frame) from stationary ones (boxing, hand gestures), where scene and person mix differently
    by_action = {int(a): 100.0 * correct[actions == a].mean().item() for a in actions.unique()}
    return {"main": 100.0 * correct.mean().item(), "by_action": by_action}


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
    `adversary` may be a list (e.g. ["frame_r18", "r3d"]): every adversary trains on f's output and
    f maximises their mean entropy, so it cannot hide appearance from a frame model while leaving
    gait for a video model.
    """
    T = models.build(cfg.get("utility", "r2p1d"), n_actions).to(DEV).train()
    adv_names = cfg.get("adversary", "frame_r18")
    adv_names = [adv_names] if isinstance(adv_names, str) else list(adv_names)

    def new_advs():
        Bs = [models.build(a, n_priv).to(DEV).train() for a in adv_names]
        opts = [torch.optim.AdamW(B.parameters(), lr=cfg.get("lr_b", 2e-4), weight_decay=0.01) for B in Bs]
        return Bs, opts

    Bs, opts_B = new_advs()
    f.to(DEV).train()
    fparams = [p for p in f.parameters() if p.requires_grad]
    opt_fT = torch.optim.AdamW([{"params": fparams, "lr": cfg.get("lr_f", 1e-4)},
                                {"params": T.parameters(), "lr": cfg.get("lr_t", 1e-4)}], weight_decay=0.01)
    alpha, epochs, restart = cfg.get("alpha", 1.0), cfg["epochs"], cfg.get("restart_every", 0)
    b_steps, b_warmup = cfg.get("b_steps", 1), cfg.get("b_warmup", 0)
    max_ent = n_priv * math.log(2) if priv_kind == "multilabel" else math.log(n_priv)
    keep_margin, collapse_drop = cfg.get("keep_margin", 5.0), cfg.get("collapse_drop", 30.0)
    collapse_patience, min_action = cfg.get("collapse_patience", 3), cfg.get("min_action", 40.0)
    best_act, kept, kept_epoch, collapsed = 0.0, None, 0, 0
    hist = []
    def warm_up_adversaries():
        """Train the adversaries on f's current (frozen, 8-bit) output for `b_warmup` batches (-1 = one
        epoch), so f always faces a converged attacker, as the evaluation's fresh attackers are."""
        n_batches = len(dl) if b_warmup < 0 else b_warmup
        f.eval()
        for bi, batch in enumerate(dl):
            if bi >= n_batches:
                break
            x, yp = to_input(batch[0]), batch[2].to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zd = f(x)
                zd = torch.round(zd.float().clamp(0, 1) * 255) / 255
            for B, opt_B in zip(Bs, opts_B):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lb = loss_fn(B(zd), yp, priv_kind)
                opt_B.zero_grad(set_to_none=True)
                lb.backward()
                opt_B.step()
        f.train()
        log(f"  adversaries warmed up for {min(n_batches, len(dl))} batches on frozen f")

    if b_warmup:
        warm_up_adversaries()
    for ep in range(epochs):
        if restart and ep and ep % restart == 0:
            Bs, opts_B = new_advs()
            log(f"  adversary restarted at epoch {ep}")
            if b_warmup:
                warm_up_adversaries()
        t0 = time.time()
        s = dict(la=0.0, ent=0.0, lb=0.0, acc_a=0.0, acc_b=[0.0] * len(Bs), n=0)
        for batch in dl:
            x, ya, yp = to_input(batch[0]), batch[1].to(DEV), batch[2].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = f(x)
                # Released videos are 8-bit. Round to 1/255 here (straight-through gradient) so f
                # can't hide utility below one quantisation step: p2_unet_a3's epoch-2 output let the
                # co-trained T reach 74% action, but a fresh model on the uint8 output got 16.7%.
                z = z + (torch.round(z.float().clamp(0, 1) * 255) / 255 - z).detach()
                la_logits = T(z)
                if ACTION_KIND == "multilabel":
                    # Charades: ~6 of 157 actions per video. Plain BCE lets the in-game action model predict "nothing"
                    # (positive recall 0 from epoch 2 in p8_bdq_a3_r18_g), so f gets no utility signal and the privacy
                    # term erases the video. Weight positives; the attackers and the evaluation keep plain BCE.
                    la = F.binary_cross_entropy_with_logits(la_logits.float(), ya.float(),
                                                            pos_weight=torch.full_like(ya[0].float(), cfg.get("pos_weight", 20.0)))
                else:
                    la = loss_fn(la_logits, ya, ACTION_KIND)
                ent = torch.stack([privacy_entropy(B(z), priv_kind) for B in Bs]).mean()
                loss = la - alpha * ent
            opt_fT.zero_grad(set_to_none=True)
            loss.backward()
            opt_fT.step()
            zd = z.detach()
            lbs = []
            # b_steps adversary updates per f update keep B close to a converged attacker. With one
            # step, f learned to fool a short-trained B: p2_vae_a1_vadv's in-training R3D sat at ~4%
            # while a fresh R3D trained on its output reached 75.6%.
            for i, (B, opt_B) in enumerate(zip(Bs, opts_B)):
                for _ in range(b_steps):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pb2 = B(zd)
                        lb = loss_fn(pb2, yp, priv_kind)
                    opt_B.zero_grad(set_to_none=True)
                    lb.backward()
                    opt_B.step()
                lbs.append(lb.item())
                s["acc_b"][i] += n_correct(pb2.float(), yp, priv_kind)
            n = len(ya)
            s["la"] += la.item() * n
            s["ent"] += ent.item() * n
            s["lb"] += sum(lbs) / len(lbs) * n
            # the collapse guard needs a score that drops when f erases the action: top-1 for one action per clip,
            # recall on the positive labels for several (Charades)
            s["acc_a"] += (positive_recall(la_logits.float(), ya) if ACTION_KIND == "multilabel"
                           else n_correct(la_logits.float(), ya, ACTION_KIND))
            s["n"] += n
        n = s["n"]
        accs = [100 * a / n for a in s["acc_b"]]
        row = dict(epoch=ep + 1, action_loss=s["la"] / n, adv_entropy=s["ent"] / n, adv_loss=s["lb"] / n,
                   train_action_acc=100 * s["acc_a"] / n, train_adv_acc=max(accs),
                   train_adv_accs=dict(zip(adv_names, accs)))
        extra = ""
        if hasattr(f, "log_sigma"):
            row["sigma"] = f.log_sigma.exp().item()
            extra = f" sigma {row['sigma']:.2f}"
        hist.append(row)
        if len(accs) > 1:
            extra += " per-adversary " + " ".join(f"{k}={v:.1f}" for k, v in row["train_adv_accs"].items())
        log(f"  [adv] epoch {ep + 1}/{epochs} action_acc {row['train_action_acc']:.1f} adv_acc {row['train_adv_acc']:.1f} "
            f"adv_entropy {row['adv_entropy']:.2f}/{max_ent:.2f}{extra} ({time.time() - t0:.0f}s)")
        # The game can collapse: the privacy term wins and f erases everything, action included
        # (seen in p2_unet_a1: action 96 -> 16 between epochs 10 and 14). Keep the latest epoch
        # whose action accuracy is within keep_margin of the best so far; stop on a collapse.
        # A single bad epoch (e.g. right after an adversary restart) can recover, so stopping needs
        # `collapse_patience` consecutive collapsed epochs.
        acc = row["train_action_acc"]
        best_act = max(best_act, acc)
        if acc >= best_act - keep_margin:
            kept = {k: v.detach().clone() for k, v in f.state_dict().items() if not anonymizers.frozen_key(k)}
            kept_epoch = ep + 1
        collapsed = collapsed + 1 if acc < best_act - collapse_drop else 0
        if ep + 1 >= 3 and best_act < min_action:  # never learned the task (p2_unet_a3_vadv: 17% from epoch 1)
            log(f"  utility never reached {min_action:.0f} (best {best_act:.1f}) in {ep + 1} epochs; stopping")
            break
        if collapsed >= collapse_patience:
            log(f"  collapse: action {acc:.1f} vs best {best_act:.1f} for {collapsed} epochs; stopping, keeping epoch {kept_epoch}")
            break
    if kept is not None and kept_epoch != len(hist):
        f.load_state_dict(kept, strict=False)
        log(f"  restored anonymizer from epoch {kept_epoch}")
    for row in hist:
        row["kept"] = row["epoch"] == kept_epoch
    return f, hist


def _windows(frames, window, ctx):
    """(video, n_windows, index, segment, (offset, length)) for every window of every video.
    Each segment carries `ctx` frames of context on both sides and is padded to window + 2 ctx frames.
    Context must cover the anonymizer's temporal receptive field: with too little, frames at window edges
    see zero padding, and a learned 3D-conv edit can leave them nearly unedited (near-raw release)."""
    for v in frames:
        arr = frames[v]
        n = len(arr)
        n_win = (n + window - 1) // window
        for wi, s in enumerate(range(0, n, window)):
            a, e = max(0, s - ctx), min(n, s + window + ctx)
            seg = np.asarray(arr[a:e])
            if s - a < ctx:  # video start: repeat the first frame so every window has the same left context
                seg = np.concatenate([np.repeat(seg[:1], ctx - (s - a), 0), seg])
                a -= ctx - (s - a)
            if len(seg) < window + 2 * ctx:  # pad by repeating the last frame
                seg = np.concatenate([seg, np.repeat(seg[-1:], window + 2 * ctx - len(seg), 0)])
            yield v, n_win, wi, seg, (s - a, min(window, n - s))


@torch.no_grad()
def materialize(f, frames, window=32, ctx=8, batch=32, out_dir=None):
    """Apply f to every frame of every video; returns {video: uint8 (T, H, W[, 3])}.
    Windows carry `ctx` frames of context on each side so temporal ops (differences, 3D convs)
    see real neighbours at window boundaries; only the window's own frames are kept.
    ctx=8 covers the VAE edit's +-6-frame receptive field (ctx=2 released near-raw edge frames
    for the VP-UCF101 VAE runs; see scripts/boundary_check.py).
    Batches mix windows from different videos (UCF101 videos are 2-4 windows long), and
    with `out_dir` each finished video is written to <out_dir>/<video>.npy by a writer thread
    and returned memory-mapped on access (datasets too large for a second copy in RAM)."""
    from concurrent.futures import ThreadPoolExecutor

    f.to(DEV).eval()
    out, paths, parts, writes = {}, {}, {}, []
    writer = ThreadPoolExecutor(4)

    def finish(v):
        full = np.concatenate([parts[v][i] for i in range(len(parts[v]))])
        del parts[v]
        if out_dir is None:
            out[v] = full
        else:
            path = os.path.join(out_dir, v + ".npy")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writes.append(writer.submit(np.save, path, full))
            paths[v] = path

    def run(pending):
        x = to_input(torch.from_numpy(np.stack([p[3] for p in pending])))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = f(x).float()
        y = (y.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        for (v, n_win, wi, _, (off, length)), yj in zip(pending, y):
            yj = yj[:, off:off + length]  # (C, L, H, W)
            parts.setdefault(v, {})[wi] = yj[0].numpy() if yj.shape[0] == 1 else yj.permute(1, 2, 3, 0).numpy()
            if len(parts[v]) == n_win:
                finish(v)

    pending = []
    for item in _windows(frames, window, ctx):
        pending.append(item)
        if len(pending) == batch:
            run(pending)
            pending = []
    if pending:
        run(pending)
    for w in writes:
        w.result()
    writer.shutdown()
    return out if out_dir is None else LazyFrames(paths)
