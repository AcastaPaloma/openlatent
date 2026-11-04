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


class TDNN(nn.Module):
    """x-vector style attacker: dilated 1-D convs over the z sequence + statistics pooling (mean/std) + 2 FC layers."""

    def __init__(self, din, dout, ch=192, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(din, ch, 5, padding=2)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(ch, ch, 3, padding=3, dilation=3)
        self.c4 = nn.Conv1d(ch, 2 * ch, 1)
        self.n = nn.ModuleList([nn.LayerNorm(ch), nn.LayerNorm(ch), nn.LayerNorm(ch), nn.LayerNorm(2 * ch)])
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(4 * ch, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, dout))

    def forward(self, x, mask):
        h = x.transpose(1, 2)
        for conv, ln in zip((self.c1, self.c2, self.c3, self.c4), self.n):
            h = torch.relu(conv(h)).transpose(1, 2)
            h = ln(h).transpose(1, 2)
        h = h.transpose(1, 2)
        mf = mask.float()[..., None]
        n = mf.sum(1).clamp(min=1)
        mu = (h * mf).sum(1) / n
        sd = ((((h - mu[:, None]) ** 2) * mf).sum(1) / n).clamp(min=1e-6).sqrt()
        return self.fc(torch.cat([mu, sd], -1))


# --------------------------------------------------------------------------------------------- training helpers
def cosine_sched(opt, total, warm_frac=0.1):
    """linear warm-up over the first 10% of steps, then cosine decay to 0 (safe for very short runs)."""
    warm = max(1, int(warm_frac * total))
    f = lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, total) / max(total, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


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
    sched = cosine_sched(opt, epochs * steps_per)

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


def _lgbm(Xtr, ytr, Xev_list, ncls, seed, val=None, rounds=None, max_rounds=80):
    """LightGBM multiclass (held-out attacker family: trees, not a neural net).  Returns (best_rounds, [probs])."""
    import lightgbm as lgb
    params = dict(objective="multiclass", num_class=ncls, learning_rate=0.15, num_leaves=7, min_data_in_leaf=8,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=63, verbose=-1,
                  num_threads=int(_os.environ.get("LGBM_THREADS", 8)), seed=seed)
    Xt, yt_ = Xtr.cpu().numpy(), ytr.cpu().numpy()
    if val is not None:
        dv = lgb.Dataset(val[0].cpu().numpy(), val[1].cpu().numpy())
        b = lgb.train(params, lgb.Dataset(Xt, yt_), num_boost_round=max_rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(8, verbose=False)])
        rounds = max(1, b.best_iteration)
        pv = torch.as_tensor(b.predict(val[0].cpu().numpy(), num_iteration=rounds))
        return rounds, [pv.to(DEV)]
    b = lgb.train(params, lgb.Dataset(Xt, yt_), num_boost_round=rounds)
    return rounds, [torch.as_tensor(b.predict(e.cpu().numpy())).to(DEV) for e in Xev_list]


def _lda(Xtr, ytr, Xev_list, alpha, ncls):
    """Shrinkage LDA (equal priors): the classic few-shot speaker-ID attacker on pooled embeddings."""
    X = Xtr.double(); y = ytr
    cnt = torch.zeros(ncls, dtype=torch.float64, device=X.device).index_add_(0, y, torch.ones(len(y), dtype=torch.float64, device=X.device))
    mu = torch.zeros(ncls, X.shape[1], dtype=torch.float64, device=X.device).index_add_(0, y, X) / cnt.clamp(min=1)[:, None]
    R = X - mu[y]
    S = R.T @ R / max(len(y) - int((cnt > 0).sum()), 1)
    S = (1 - alpha) * S + alpha * (torch.trace(S) / S.shape[0]) * torch.eye(S.shape[0], dtype=torch.float64, device=X.device)
    W = torch.linalg.solve(S, mu.T)  # [d, C]
    b = -0.5 * (mu * W.T).sum(1)
    b = torch.where(cnt > 0, b, torch.full_like(b, -1e9))
    return [(Xe.double() @ W + b).float() for Xe in Xev_list]


def _ridge(Xtr, ytr, Xev_list, alpha):
    A = Xtr.T @ Xtr + alpha * torch.eye(Xtr.shape[1], device=Xtr.device)
    ym = ytr.mean()
    w = torch.linalg.solve(A, Xtr.T @ (ytr - ym))
    return [(Xe @ w + ym)[:, None] for Xe in Xev_list]


GRIDS = dict(linear=[dict(l2=1e-5), dict(l2=1e-4), dict(l2=1e-3)], mlp=[dict(l2=0.0), dict(l2=1e-4)],
             knn=[dict(k=1), dict(k=5), dict(k=15)], ridge=[dict(alpha=a) for a in (1e-1, 1, 10, 100, 1000)],
             transformer=[dict(l2=0.0)], tdnn=[dict(l2=0.0)], lgbm=[dict()],
             lda=[dict(alpha=a) for a in (0.02, 0.1, 0.4, 0.8)])
import os as _os
EPOCHS = dict(linear=60, mlp=60, transformer=30, tdnn=30) if not _os.environ.get("C_FAST") else dict(linear=6, mlp=6, transformer=3, tdnn=3)
SEQ_KINDS = ("transformer", "tdnn")


def attack(rep, y, fit_idx, val_idx, train_idx, eval_sets, kinds, task="cls", ncls=91, seed=0, metric=None):
    """Retrain every attacker in `kinds` on `rep`.

    y: np array of labels (int for cls, float for reg).  fit/val/train/eval idx: np int arrays over clips.
    Hyper-parameters (and epoch / boosting-round count) are chosen by fitting on fit_idx and scoring on val_idx; the final
    model is refit on train_idx and applied to every set in eval_sets {name: idx}.
    Returns {kind: dict(preds={name: np.array}, hp=..., val=float)}.
    kinds: linear, mlp, knn, ridge (pooled), lgbm (trees, pooled), transformer / tdnn (need rep.seq).
    """
    res = {}
    yt = torch.as_tensor(y, dtype=torch.long if task == "cls" else torch.float32, device=DEV)
    fit_idx, val_idx, train_idx = (np.asarray(i) for i in (fit_idx, val_idx, train_idx))
    for kind in kinds:
        seqk = kind in SEQ_KINDS
        if (seqk and rep.seq is None) or (kind in ("lgbm", "lda") and task != "cls"):
            continue
        best = None
        P = rep.pooled
        if not seqk:
            mu, sd = P[train_idx].mean(0), P[train_idx].std(0) + 1e-5
            X = (P - mu) / sd
        for hp in GRIDS[kind]:
            if kind == "knn":
                out_v = _knn(X[fit_idx], yt[fit_idx], [X[val_idx]], hp["k"], ncls)[0]
                sc, ep = _score(out_v, yt[val_idx], task, ncls), 1
            elif kind == "ridge":
                out_v = _ridge(X[fit_idx], yt[fit_idx], [X[val_idx]], hp["alpha"])[0]
                sc, ep = _score(out_v, yt[val_idx], task, ncls), 1
            elif kind == "lda":
                out_v = _lda(X[fit_idx], yt[fit_idx], [X[val_idx]], hp["alpha"], ncls)[0]
                sc, ep = _score(out_v, yt[val_idx], task, ncls), 1
            elif kind == "lgbm":
                ep, (out_v,) = _lgbm(X[fit_idx], yt[fit_idx], [], ncls, seed, val=(X[val_idx], yt[val_idx]))
                sc = _score(out_v, yt[val_idx], task, ncls)
            else:
                make, kw = _make(kind, rep, None if seqk else X, ncls, task)
                inp_fit = torch.as_tensor(fit_idx, device=DEV) if seqk else X[fit_idx]
                inp_val = torch.as_tensor(val_idx, device=DEV) if seqk else X[val_idx]
                curve, _ = _fit_net(make, inp_fit, yt[fit_idx], [], EPOCHS[kind], kw["lr"], hp["l2"], kw["bs"], seed, task,
                                    ncls, val=(inp_val, yt[val_idx]), seq=rep.seq if seqk else None)
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
        elif kind == "lda":
            outs = _lda(X[train_idx], yt[train_idx], [X[eval_sets[n]] for n in evn], hp["alpha"], ncls)
        elif kind == "lgbm":
            _, outs = _lgbm(X[train_idx], yt[train_idx], [X[eval_sets[n]] for n in evn], ncls, seed + 1, rounds=ep)
        else:
            make, kw = _make(kind, rep, None if seqk else X, ncls, task)
            if seqk:
                inp_tr = torch.as_tensor(train_idx, device=DEV)
                evs = [torch.as_tensor(eval_sets[n], device=DEV) for n in evn]
            else:
                inp_tr, evs = X[train_idx], [X[eval_sets[n]] for n in evn]
            _, outs = _fit_net(make, inp_tr, yt[train_idx], evs, ep, kw["lr"], hp["l2"], kw["bs"], seed + 1, task, ncls,
                               seq=rep.seq if seqk else None)
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
    if kind == "tdnn":
        d = rep.seq.flat.shape[-1]
        return (lambda: _SeqWrap(TDNN(d, dout), rep)), dict(lr=1e-3, bs=64)
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
    k = ["linear", "mlp", "knn", "lda", "lgbm"]
    if rep.seq is not None:
        k += ["transformer", "tdnn"]
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
