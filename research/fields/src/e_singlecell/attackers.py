"""Batch attackers and label probes, always retrained from scratch on the released embedding.

Attackers (secret = batch id, classes = the fold's held-out batches): linear (class-balanced softmax, L-BFGS, L2 from an inner split),
mlp (2x256, early-stopping epoch from the inner split), knn (cosine, k from the inner split), lgbm (early stopping on the inner split).
Probes (utility = fine cell type, trained on training batches): class-balanced linear softmax (L2 1e-2) and cosine kNN (k=15).
Features are standardised with the attacker's/probe's own training statistics.  Adapted from a_pathorob/attackers.py (same standards).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KINDS = ("linear", "mlp", "knn", "lgbm")


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEV)


def bacc(y, p):
    cl = np.unique(y)
    return float(np.mean([(p[y == k] == k).mean() for k in cl]))


def cw_of(y, K):
    cnt = torch.bincount(y, minlength=K).float()
    return torch.where(cnt > 0, cnt.sum() / (K * cnt.clamp(min=1)), torch.zeros_like(cnt))


class Std:
    def fit(self, X):
        self.mu = X.mean(0, keepdim=True)
        s = X.std(0, keepdim=True)
        self.sd = torch.where(s < 1e-6, torch.ones_like(s), s)
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def inner_split(u, y, seed, frac=0.2):
    """unit-disjoint inner split when every class has >= 2 units, else a random cell split."""
    rng = np.random.RandomState(seed)
    uu = np.unique(u)
    ycl = np.unique(y)
    nu = {k: len(np.unique(u[y == k])) for k in ycl}
    if min(nu.values()) >= 2:
        for _ in range(30):
            va = set()
            for k in ycl:
                us = np.unique(u[y == k])
                va |= set(rng.choice(us, size=max(1, int(round(frac * len(us)))), replace=False).tolist())
            m = np.isin(u, list(va))
            if len(np.unique(y[m])) == len(ycl) and len(np.unique(y[~m])) == len(ycl):
                return ~m, m
    m = rng.rand(len(u)) < frac
    return ~m, m


def fit_linear(X, y, K, wd, iters=60):
    W = torch.zeros(X.shape[1], K, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    cw = cw_of(y, K)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y, weight=cw) + 0.5 * wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Z: Z @ W + b


class MLP(nn.Module):
    def __init__(self, d, K, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, K))

    def forward(self, x):
        return self.net(x)


def fit_mlp(X, y, K, epochs, seed, val=None, lr=2e-3, wd=1e-2, bs=512):
    torch.manual_seed(seed)
    m = MLP(X.shape[1], K).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    cw = cw_of(y, K)
    n = X.shape[0]
    best = (1, -1.0)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(m(X[idx]), y[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            m.eval()
            with torch.no_grad():
                a = bacc(val[1].cpu().numpy(), m(val[0]).argmax(1).cpu().numpy())
            if a > best[1] + 1e-9:
                best = (ep, a)
    if val is not None:
        return best[0]
    m.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.cat([m(Z[i:i + 4096]) for i in range(0, Z.shape[0], 4096)])
    return pred


def knn_logits(Xtr, ytr, Xte, K, k):
    A = F.normalize(Xtr, dim=1)
    cw = cw_of(ytr, K)
    out = []
    for i in range(0, Xte.shape[0], 2048):
        s = F.normalize(Xte[i:i + 2048], dim=1) @ A.T
        nn_idx = s.topk(min(k, A.shape[0]), dim=1).indices
        out.append(F.one_hot(ytr[nn_idx], K).float().sum(1) * cw)
    return torch.cat(out)


def _lgbm(Xtr, ytr, Xva, yva, n_iter, seed, n_jobs=8):
    import lightgbm as lgb
    kw = dict(learning_rate=0.1, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
              class_weight="balanced", verbose=-1, n_jobs=n_jobs, random_state=seed)
    if Xva is not None:
        m = lgb.LGBMClassifier(n_estimators=200, **kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(15, verbose=False)])
        return max(int(m.best_iteration_ or 30), 15)
    m = lgb.LGBMClassifier(n_estimators=n_iter, **kw)
    m.fit(Xtr, ytr)
    return m


def run_attackers(Xa, ya, ua, Xb, K, kinds=KINDS, seed=0, epochs=30, shuffle=False):
    """Train every attacker kind on (Xa, ya) from scratch, predict the rows of Xb.  ya in 0..K-1 (numpy), ua = unit ids of Xa (inner split).
    shuffle=True permutes ya over cells (control: must give chance).  Returns {kind: int predictions for Xb}."""
    ya = np.asarray(ya).copy()
    if shuffle:
        ya = np.random.RandomState(777 + seed).permutation(ya)
    Xa_t, Xb_t = _t(Xa), _t(Xb)
    ya_t = _t(ya, torch.long)
    tr, va = inner_split(ua, ya, seed)
    st_full = Std().fit(Xa_t)
    trm, vam = _t(tr, torch.bool), _t(va, torch.bool)
    st_in = Std().fit(Xa_t[trm])
    Xtr_i, ytr_i, Xva_i, yva_i = st_in(Xa_t[trm]), ya_t[trm], st_in(Xa_t[vam]), ya_t[vam]
    yva_np = yva_i.cpu().numpy()
    Xf, Xbf = st_full(Xa_t), st_full(Xb_t)
    out = {}
    for kind in kinds:
        torch.manual_seed(seed)
        if kind == "linear":
            best = (-1, 1e-2)
            for wd in (1e-3, 1e-1):
                a = bacc(yva_np, fit_linear(Xtr_i, ytr_i, K, wd)(Xva_i).argmax(1).cpu().numpy())
                if a > best[0] + 1e-9:
                    best = (a, wd)
            out[kind] = fit_linear(Xf, ya_t, K, best[1])(Xbf).argmax(1).cpu().numpy()
        elif kind == "mlp":
            ep = fit_mlp(Xtr_i, ytr_i, K, epochs, seed, val=(Xva_i, yva_i))
            out[kind] = fit_mlp(Xf, ya_t, K, max(ep, 3), seed)(Xbf).argmax(1).cpu().numpy()
        elif kind == "knn":
            best = (-1, 15)
            for k in (10, 30):
                a = bacc(yva_np, knn_logits(Xtr_i, ytr_i, Xva_i, K, k).argmax(1).cpu().numpy())
                if a > best[0] + 1e-9:
                    best = (a, k)
            out[kind] = knn_logits(Xf, ya_t, Xbf, K, best[1]).argmax(1).cpu().numpy()
        elif kind == "lgbm":
            n_it = _lgbm(Xtr_i.cpu().numpy(), ytr_i.cpu().numpy(), Xva_i.cpu().numpy(), yva_np, None, seed)
            m = _lgbm(Xf.cpu().numpy(), ya, None, None, n_it, seed)
            out[kind] = m.predict(Xbf.cpu().numpy()).astype(np.int64)
        else:
            raise ValueError(kind)
    return out


def run_probes(Xtr, ytr, Xte, K, seed=0):
    """fine-label probes trained on training-batch cells.  Returns {'lin': preds, 'knn': preds} (int, in 0..K-1)."""
    Xtr_t, Xte_t = _t(Xtr), _t(Xte)
    ytr_t = _t(ytr, torch.long)
    st = Std().fit(Xtr_t)
    A, B = st(Xtr_t), st(Xte_t)
    torch.manual_seed(seed)
    return {"lin": fit_linear(A, ytr_t, K, 1e-2)(B).argmax(1).cpu().numpy(),
            "knn": knn_logits(A, ytr_t, B, K, 15).argmax(1).cpu().numpy()}
