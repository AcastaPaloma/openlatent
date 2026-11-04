"""Attacker / probe suite, always retrained from scratch on the released representation.

Four families (all torch, GPU if available):  linear (softmax regression, L-BFGS, L2 chosen on a unit-disjoint inner
validation split), mlp (2x512, early stopping epoch from the inner split), knn (cosine, k chosen on the inner split) and
tfm (FT-Transformer style: the vector is cut into <=32 tokens, 2 pre-norm layers).  Features are standardised with the
attacker's own training statistics.  Every model is refit on the full attacker-train set with the selected setting.
Class-balanced losses and balanced accuracy throughout.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KINDS = ("linear", "mlp", "knn", "tfm")


def bacc(y, p, K=None):
    y = np.asarray(y)
    p = np.asarray(p)
    cl = np.unique(y)
    return float(np.mean([(p[y == k] == k).mean() for k in cl]))


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEV)


class Std:
    def fit(self, X):
        self.mu = X.mean(0, keepdim=True)
        s = X.std(0, keepdim=True)
        self.sd = torch.where(s < 1e-6, torch.ones_like(s), s)
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _cw(y, K):
    cnt = torch.bincount(y, minlength=K).float()
    w = torch.where(cnt > 0, cnt.sum() / (K * cnt.clamp(min=1)), torch.zeros_like(cnt))
    return w


def inner_split(u, y, seed, frac=0.25):
    """unit-disjoint inner train/val split of the attacker-train set; every class kept in both parts when possible."""
    rng = np.random.RandomState(seed)
    uu = np.unique(u)
    for _ in range(50):
        va = set(rng.choice(uu, size=max(1, int(round(frac * len(uu)))), replace=False).tolist())
        m = np.array([x in va for x in u])
        if len(np.unique(y[m])) == len(np.unique(y)) and len(np.unique(y[~m])) == len(np.unique(y)):
            return ~m, m
    m = rng.rand(len(u)) < frac        # fallback (very few units): patch level
    return ~m, m


# ----------------------------------------------------------------------------- models
def fit_linear(X, y, K, wd, iters=60):
    d = X.shape[1]
    W = torch.zeros(d, K, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    cw = _cw(y, K)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y, weight=cw) + 0.5 * wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = W.detach(), b.detach()

    def pred(Z):
        return Z @ W + b
    pred.W, pred.b = W, b
    return pred


class MLP(nn.Module):
    def __init__(self, d, K, h=512, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(h, K))

    def forward(self, x):
        return self.net(x)


class Tok(nn.Module):
    def __init__(self, d, K, dim=64, layers=2, heads=4, max_tok=32):
        super().__init__()
        self.g = int(np.ceil(d / max_tok))
        self.T = int(np.ceil(d / self.g))
        self.pad = self.T * self.g - d
        self.emb = nn.Linear(self.g, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.T + 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.pos, std=0.02)
        L = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(L, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, K)

    def forward(self, x):
        n = x.shape[0]
        if self.pad:
            x = F.pad(x, (0, self.pad))
        t = self.emb(x.view(n, self.T, self.g))
        t = torch.cat([self.cls.expand(n, -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t)[:, 0]))


def fit_nn(make, X, y, K, epochs, lr, wd, bs=256, val=None, seed=0):
    """train; if val=(Xv,yv) given, return (best_epoch, best_acc) without keeping the model, else return predictor."""
    torch.manual_seed(seed)
    m = make().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    cw = _cw(y, K)
    n = X.shape[0]
    best = (0, -1.0)
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
                p = m(val[0]).argmax(1).cpu().numpy()
            a = bacc(val[1].cpu().numpy(), p)
            if a > best[1] + 1e-9:
                best = (ep, a)
    if val is not None:
        return best
    m.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.cat([m(Z[i:i + 2048]) for i in range(0, Z.shape[0], 2048)])
    return pred


def knn_logits(Xtr, ytr, Xte, K, k):
    A = F.normalize(Xtr, dim=1)
    cw = _cw(ytr, K)
    out = []
    for i in range(0, Xte.shape[0], 2048):
        s = F.normalize(Xte[i:i + 2048], dim=1) @ A.T
        nn_idx = s.topk(min(k, A.shape[0]), dim=1).indices
        votes = F.one_hot(ytr[nn_idx], K).float().sum(1) * cw
        out.append(votes)
    return torch.cat(out)


# ----------------------------------------------------------------------------- suite
def run_suite(Xb, yb, ub, evals, K, kinds=KINDS, seed=0, epochs_nn=40, secret_shuffle=False):
    """Train every attacker kind on (Xb, yb) from scratch and predict each array in `evals` (dict name->X).
    Returns {kind: {eval_name: int predictions}}.  Xb numpy float32, yb int, ub unit ids (for the inner split).
    secret_shuffle: permute yb at the unit level (control: must give chance)."""
    yb = np.asarray(yb).copy()
    if secret_shuffle:
        rng = np.random.RandomState(777 + seed)
        uu = np.unique(ub)
        # each unit gets the label of another random unit
        lab = {u: yb[ub == u][0] for u in uu}
        perm = rng.permutation(len(uu))
        yb = np.array([lab[uu[perm[np.searchsorted(uu, u)]]] for u in ub])
    Xb_t = _t(Xb)
    yb_t = _t(yb, torch.long)
    tr, va = inner_split(ub, yb, seed)
    out = {}
    ev = {k: _t(v) for k, v in evals.items()}
    # full-set and inner-split standardisers
    st_full = Std().fit(Xb_t)
    st_in = Std().fit(Xb_t[_t(tr, torch.bool)])
    Xtr_i, ytr_i = st_in(Xb_t[_t(tr, torch.bool)]), yb_t[_t(tr, torch.bool)]
    Xva_i, yva_i = st_in(Xb_t[_t(va, torch.bool)]), yb_t[_t(va, torch.bool)]
    Xf = st_full(Xb_t)
    evf = {k: st_full(v) for k, v in ev.items()}
    yva_np = yva_i.cpu().numpy()
    for kind in kinds:
        torch.manual_seed(seed)
        if kind == "linear":
            best = (-1, 1e-2)
            for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                f = fit_linear(Xtr_i, ytr_i, K, wd)
                a = bacc(yva_np, f(Xva_i).argmax(1).cpu().numpy())
                if a > best[0] + 1e-9:
                    best = (a, wd)
            f = fit_linear(Xf, yb_t, K, best[1])
            out[kind] = {k: f(v).argmax(1).cpu().numpy() for k, v in evf.items()}
        elif kind in ("mlp", "tfm"):
            d = Xf.shape[1]
            make = (lambda: MLP(d, K)) if kind == "mlp" else (lambda: Tok(d, K))
            lr, wd = (1e-3, 1e-2) if kind == "mlp" else (5e-4, 1e-2)
            ep, _ = fit_nn(make, Xtr_i, ytr_i, K, epochs_nn, lr, wd, val=(Xva_i, yva_i), seed=seed)
            f = fit_nn(make, Xf, yb_t, K, max(ep, 3), lr, wd, seed=seed)
            out[kind] = {k: f(v).argmax(1).cpu().numpy() for k, v in evf.items()}
        elif kind == "knn":
            best = (-1, 15)
            for k in (5, 15, 50):
                a = bacc(yva_np, knn_logits(Xtr_i, ytr_i, Xva_i, K, k).argmax(1).cpu().numpy())
                if a > best[0] + 1e-9:
                    best = (a, k)
            out[kind] = {k: knn_logits(Xf, yb_t, v, K, best[1]).argmax(1).cpu().numpy() for k, v in evf.items()}
        else:
            raise ValueError(kind)
    return out
