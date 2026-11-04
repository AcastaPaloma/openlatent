"""Representations, retrained attackers/probes and metrics for Focus C (CREMA-D identity-free emotion).

Design notes (see research/fields/results_notes/C.md for the protocol):
  * A released representation `Rep` is `pooled` [N,P] (what linear/MLP/kNN see) plus an optional ragged sequence
    `seq` (what the transformer sees).  Utterance-level latents have pooled = z and seq = None.
  * Every attacker/probe is trained FROM SCRATCH on the released representation.  Hyper-parameters and the number of
    epochs are chosen on a held-out validation subset (sentences for identity, actors for emotion), then the model is
    refit on fit+val and scored once on the sealed test sets.
  * Metrics keep per-actor counts so that the intervals can be bootstrapped over actors (independent unit).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------------- ragged sequences
class Ragged:
    """Flat [Nfr, d] tensor + offsets [N+1]; builds padded batches on the device."""

    def __init__(self, flat, offsets, dev=DEV):
        self.flat = flat.to(dev)
        self.off = torch.as_tensor(np.asarray(offsets), dtype=torch.long, device=dev)
        self.len = self.off[1:] - self.off[:-1]
        self.dev = dev
        self.n = len(self.len)

    def batch(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.dev)
        L = self.len[idx]
        T = int(L.max())
        ar = torch.arange(T, device=self.dev)
        mask = ar[None] < L[:, None]
        pos = torch.where(mask, self.off[idx][:, None] + ar[None], torch.zeros_like(mask, dtype=torch.long))
        x = self.flat[pos]
        x = x * mask.view(*mask.shape, *([1] * (x.dim() - 2))).to(x.dtype)
        return x, mask

    def pooled_stats(self, bs=256):
        out = []
        for s in range(0, self.n, bs):
            x, m = self.batch(np.arange(s, min(s + bs, self.n)))
            x = x.float()
            mf = m.float()[..., None]
            n = mf.sum(1).clamp(min=1)
            mu = (x * mf).sum(1) / n
            var = (((x - mu[:, None]) ** 2) * mf).sum(1) / n
            out.append(torch.cat([mu, var.clamp(min=0).sqrt()], 1))
        return torch.cat(out, 0)


class Rep:
    def __init__(self, pooled, seq=None, name=""):
        self.pooled = pooled.float().to(DEV)
        self.seq = seq
        self.name = name

    @property
    def n(self):
        return self.pooled.shape[0]


def constant_rep(n, p=8, T=10):
    return Rep(torch.zeros(n, p), Ragged(torch.zeros(n * T, p), np.arange(n + 1) * T), "constant")


def length_rep(lengths):
    lengths = np.asarray(lengths)
    return Rep(torch.log(torch.as_tensor(lengths, dtype=torch.float32))[:, None],
               Ragged(torch.zeros(int(lengths.sum()), 1), np.concatenate([[0], np.cumsum(lengths)])), "length_only")


# --------------------------------------------------------------------------------------------- metrics
def uar(y, p, ncls):
    rec = [float((p[y == c] == c).mean()) for c in range(ncls) if (y == c).any()]
    return float(np.mean(rec))


def actor_counts(actors, correct):
    """dict actor -> [n_correct, n_total]"""
    out = {}
    for a, c in zip(actors.tolist(), correct.tolist()):
        o = out.setdefault(int(a), [0, 0]); o[0] += int(c); o[1] += 1
    return out


def actor_confusions(actors, y, p, ncls):
    out = {}
    for a in np.unique(actors):
        m = actors == a
        C = np.zeros((ncls, ncls), dtype=np.int64)
        np.add.at(C, (y[m], p[m]), 1)
        out[int(a)] = C.tolist()
    return out


# --------------------------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, din, dout, hidden=512, p=0.2, depth=2):
        super().__init__()
        layers, d = [nn.Dropout(p)], din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p)]
            d = hidden
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SeqTransformer(nn.Module):
    """Small transformer attacker over a z sequence: conv front-end (local order) + 2 layers + mean pool."""

    def __init__(self, din, dout, d=192, layers=2, heads=4, p=0.1):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, dout)

    def forward(self, x, mask):
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.tr(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        mf = mask.float()[..., None]
        return self.out((h * mf).sum(1) / mf.sum(1).clamp(min=1))


# --------------------------------------------------------------------------------------------- training helpers
def _loss(out, y, task):
    return F.cross_entropy(out, y) if task == "cls" else F.mse_loss(out.squeeze(-1), y)


def _score(out, y, task, ncls):
    """higher is better; classification = accuracy (UAR for emotion handled by caller), regression = -mse"""
    if task == "cls":
        return float((out.argmax(1) == y).float().mean())
    return -float(F.mse_loss(out.squeeze(-1), y))


def _fit_net(make, X, y, evals, epochs, lr, l2, bs, seed, task, ncls, val=None, seq=None):
    """Train from scratch.  X is pooled [N,P] (or None when seq given: X are clip indices into `seq`).
    val=(Xv,yv) -> returns per-epoch val score curve.  evals: list of eval inputs -> logits at the end."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    net = make().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    n = len(y)
    steps_per = max(1, math.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per, pct_start=0.1)

    def fwd(inp, train):
        net.train(train)
        if seq is not None:
            idx = inp
            xb, mb = seq.batch(idx)
            return net(xb.float(), mb)
        return net(inp)

    def predict(inp, chunk=512):
        outs = []
        with torch.no_grad():
            for s in range(0, len(inp), chunk):
                outs.append(fwd(inp[s:s + chunk], False))
        return torch.cat(outs, 0)

    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            out = fwd(X[b.to(X.device)], True)
            loss = _loss(out, y[b.to(y.device)], task)
            if l2 > 0 and seq is None:
                loss = loss + l2 * sum((p ** 2).sum() for nme, p in net.named_parameters() if nme.endswith("weight"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
        if val is not None:
            o = predict(val[0])
            curve.append(_score(o, val[1], task, ncls))
    return curve, [predict(e) for e in evals]


def _knn(Xtr, ytr, Xev_list, k, ncls):
    Xn = F.normalize(Xtr, dim=1)
    outs = []
    for Xe in Xev_list:
        Xe = F.normalize(Xe, dim=1)
        preds = []
        for s in range(0, len(Xe), 1024):
            sim = Xe[s:s + 1024] @ Xn.T
            v, i = sim.topk(min(k, Xn.shape[0]), dim=1)
            votes = torch.zeros(len(v), ncls, device=Xe.device)
            votes.scatter_add_(1, ytr[i], torch.exp(v * 10))  # similarity-weighted vote
            preds.append(votes)
        outs.append(torch.cat(preds, 0))
    return outs


def _ridge(Xtr, ytr, Xev_list, alpha):
    A = Xtr.T @ Xtr + alpha * torch.eye(Xtr.shape[1], device=Xtr.device)
    ym = ytr.mean()
    w = torch.linalg.solve(A, Xtr.T @ (ytr - ym))
    return [(Xe @ w + ym)[:, None] for Xe in Xev_list]


GRIDS = dict(linear=[dict(l2=1e-5), dict(l2=1e-4), dict(l2=1e-3)], mlp=[dict(l2=0.0), dict(l2=1e-4)],
             knn=[dict(k=1), dict(k=5), dict(k=15)], ridge=[dict(alpha=a) for a in (1e-1, 1, 10, 100, 1000)],
             transformer=[dict(l2=0.0)])
import os as _os
EPOCHS = dict(linear=60, mlp=60, transformer=30) if not _os.environ.get("C_FAST") else dict(linear=6, mlp=6, transformer=3)


def attack(rep, y, fit_idx, val_idx, train_idx, eval_sets, kinds, task="cls", ncls=91, seed=0, metric=None):
    """Retrain every attacker in `kinds` on `rep`.

    y: np array of labels (int for cls, float for reg).  fit/val/train/eval idx: np int arrays over clips.
    Hyper-parameters (and epoch count) are chosen by fitting on fit_idx and scoring on val_idx; the final model is refit
    on train_idx and applied to every set in eval_sets {name: idx}.
    Returns {kind: dict(preds={name: np.array}, hp=..., val=float)}.
    metric: optional callable(y_np, pred_np) -> float used for model selection (default accuracy / -mse).
    """
    res = {}
    yt = torch.as_tensor(y, dtype=torch.long if task == "cls" else torch.float32, device=DEV)
    for kind in kinds:
        if kind == "transformer" and rep.seq is None:
            continue
        best = None
        P = rep.pooled
        if kind in ("linear", "mlp", "knn", "ridge"):
            mu, sd = P[train_idx].mean(0), P[train_idx].std(0) + 1e-5
            X = (P - mu) / sd
        for hp in GRIDS[kind]:
            if kind == "knn":
                out_v = _knn(X[fit_idx], yt[fit_idx], [X[val_idx]], hp["k"], ncls)[0]
                curve, ep = [_score(out_v, yt[val_idx], task, ncls)], 1
                sc = curve[0]
            elif kind == "ridge":
                out_v = _ridge(X[fit_idx], yt[fit_idx], [X[val_idx]], hp["alpha"])[0]
                sc, ep = _score(out_v, yt[val_idx], task, ncls), 1
            else:
                make, kw = _make(kind, rep, X if kind != "transformer" else None, ncls, task)
                inp_fit = X[fit_idx] if kind != "transformer" else torch.as_tensor(fit_idx, device=DEV)
                inp_val = X[val_idx] if kind != "transformer" else torch.as_tensor(val_idx, device=DEV)
                curve, _ = _fit_net(make, inp_fit, yt[fit_idx], [], EPOCHS[kind], kw["lr"], hp["l2"], kw["bs"], seed, task,
                                    ncls, val=(inp_val, yt[val_idx]), seq=rep.seq if kind == "transformer" else None)
                ep = int(np.argmax(curve)) + 1
                sc = float(np.max(curve))
            if best is None or sc > best[0]:
                best = (sc, hp, ep)
        sc, hp, ep = best
        evn = list(eval_sets)
        if kind == "knn":
            outs = _knn(X[train_idx], yt[train_idx], [X[eval_sets[n]] for n in evn], hp["k"], ncls)
        elif kind == "ridge":
            outs = _ridge(X[train_idx], yt[train_idx], [X[eval_sets[n]] for n in evn], hp["alpha"])
        else:
            make, kw = _make(kind, rep, X if kind != "transformer" else None, ncls, task)
            if kind == "transformer":
                inp_tr = torch.as_tensor(train_idx, device=DEV)
                evs = [torch.as_tensor(eval_sets[n], device=DEV) for n in evn]
            else:
                inp_tr, evs = X[train_idx], [X[eval_sets[n]] for n in evn]
            _, outs = _fit_net(make, inp_tr, yt[train_idx], evs, ep, kw["lr"], hp["l2"], kw["bs"], seed + 1, task, ncls,
                               seq=rep.seq if kind == "transformer" else None)
        preds = {n: (o.argmax(1).cpu().numpy() if task == "cls" else o.squeeze(-1).cpu().numpy()) for n, o in zip(evn, outs)}
        res[kind] = dict(preds=preds, hp=dict(hp, epochs=ep), val=sc)
    return res


def _make(kind, rep, X, ncls, task):
    dout = ncls if task == "cls" else 1
    if kind == "linear":
        return (lambda: nn.Linear(X.shape[1], dout)), dict(lr=3e-3, bs=256)
    if kind == "mlp":
        return (lambda: MLP(X.shape[1], dout)), dict(lr=1e-3, bs=256)
    if kind == "transformer":
        d = rep.seq.flat.shape[-1]
        return (lambda: _SeqWrap(SeqTransformer(d, dout), rep)), dict(lr=5e-4, bs=64)
    raise ValueError(kind)


class _SeqWrap(nn.Module):
    """Standardises frames with global (train-agnostic) stats of this rep before the transformer."""

    def __init__(self, net, rep):
        super().__init__()
        self.net = net
        f = rep.seq.flat
        samp = f[torch.randperm(len(f), device=f.device)[:200000]].float()
        self.register_buffer("mu", samp.mean(0)); self.register_buffer("sd", samp.std(0) + 1e-5)

    def forward(self, x, mask):
        return self.net((x - self.mu) / self.sd, mask)


def suite_for(rep, extra=()):
    import os
    k = ["linear", "mlp", "knn"]
    if rep.seq is not None:
        k.append("transformer")
    only = os.environ.get("C_KINDS")  # dev/smoke only
    if only:
        k = [x for x in k if x in only.split(",")]
    return k + list(extra)


# --------------------------------------------------------------------------------------------- bootstrap
def boot_idx(actors, B=1000, seed=0):
    """Multiplicity matrix [B, n_actors] of actor-level bootstrap resamples."""
    rs = np.random.RandomState(seed)
    n = len(actors)
    return np.stack([np.bincount(rs.randint(0, n, n), minlength=n) for _ in range(B)])
