"""Linear probes on frozen features (GPU): how much a feature view predicts a target, with a bootstrap CI.

A probe is a softmax (multiclass) or per-class sigmoid (multilabel) linear layer on standardised features, trained
full-batch with AdamW. Weight decay is chosen on a 20% validation part of the training items (split by recording so
footage of one recording never sits on both sides), then the probe is refit on all training items.
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import scoring

DEV = "cuda"
WDS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _fit(X, Y, kind, n_out, wd, epochs=300, lr=1e-2):
    w = torch.zeros(X.shape[1], n_out, device=DEV, requires_grad=True)
    b = torch.zeros(n_out, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        logits = X @ w + b
        loss = (F.cross_entropy(logits, Y) if kind == "multiclass" else F.binary_cross_entropy_with_logits(logits, Y))
        loss = loss + wd * (w ** 2).sum() / X.shape[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def _predict(X, w, b, kind):
    logits = X @ w + b
    return (logits.softmax(1) if kind == "multiclass" else logits.sigmoid()).cpu().numpy()


def probe(X_tr, y_tr, X_te, y_te, kind, n_out, groups_tr=None, cond_te=None, seed=0):
    """-> dict(score, lo, hi, probs) on the test items. y: int array (multiclass) or (n, n_out) 0/1 (multilabel)."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=DEV)
    Xte = torch.tensor((X_te - mu) / sd, dtype=torch.float32, device=DEV)
    Ytr = torch.tensor(y_tr, device=DEV, dtype=torch.long if kind == "multiclass" else torch.float32)
    # validation split by recording for the weight decay
    rng = np.random.default_rng(seed)
    g = np.asarray(groups_tr) if groups_tr is not None else np.arange(len(y_tr))
    uniq = np.unique(g)
    val_groups = set(rng.choice(uniq, max(1, len(uniq) // 5), replace=False).tolist())
    val = np.array([x in val_groups for x in g])
    best_wd, best = WDS[0], -np.inf
    if val.sum() >= 5 and (~val).sum() >= 5:
        for wd in WDS:
            w, b = _fit(Xtr[~val], Ytr[~val], kind, n_out, wd)
            s = scoring.metric(kind, _predict(Xtr[val], w, b, kind), y_tr[val])
            if np.isfinite(s) and s > best:
                best, best_wd = s, wd
    w, b = _fit(Xtr, Ytr, kind, n_out, best_wd)
    probs = _predict(Xte, w, b, kind)
    score, lo, hi = scoring.bootstrap(kind, probs, y_te, cond_te)
    return dict(score=score, lo=lo, hi=hi, wd=best_wd, probs=probs)
