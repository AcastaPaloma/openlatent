"""F6, the HELD-OUT view (T.md 3.5): LightGBM on hand-crafted pair statistics.  Never used for selection, margins, nomination or design decisions.

Per-image descriptor (no learned weights): row and column intensity profiles (64 bins), 64x64 mean-intensity grid (8x8 cells), unsigned gradient-orientation
histograms on an 8x8 cell grid (HOG-lite, 9 bins), 32-bin intensity histogram.  Pair features: profile correlations, L1 profile distances, HOG cosine, chi2
histogram distance, cell-grid L1 and the 64 absolute cell differences; LightGBM (300 trees, early stopping on V).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def descriptors(X: torch.Tensor, size: int = 64, bs: int = 512) -> dict:
    rows, cols, grid, hog, hist = [], [], [], [], []
    for i in range(0, X.shape[0], bs):
        x = F.interpolate(X[i:i + bs].float().unsqueeze(1), size=(size, size), mode="area")
        rows.append(x.mean(3).squeeze(1)); cols.append(x.mean(2).squeeze(1))
        grid.append(F.adaptive_avg_pool2d(x, 8).flatten(1))
        gx = F.conv2d(x, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), padding=1)
        gy = F.conv2d(x, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), padding=1)
        mag = (gx ** 2 + gy ** 2).sqrt(); ang = (torch.atan2(gy, gx) % np.pi) / np.pi * 9
        b = ang.floor().clamp(max=8).long()
        oh = F.one_hot(b.squeeze(1), 9).permute(0, 3, 1, 2).float() * mag
        hog.append(F.adaptive_avg_pool2d(oh, 8).flatten(1))
        hist.append(torch.stack([torch.histc(xi, bins=32, min=0, max=1) for xi in x]) / (size * size))
    cat = lambda l: torch.cat(l).numpy()
    return dict(row=cat(rows), col=cat(cols), grid=cat(grid), hog=cat(hog), hist=cat(hist))


def _corr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def pair_stats(Da: dict, Db: dict, ia, ib) -> np.ndarray:
    g = lambda D, k: D[k][ia] if D is Da else D[k][ib]
    ra, rb, ca, cb = Da["row"][ia], Db["row"][ib], Da["col"][ia], Db["col"][ib]
    ha, hb = Da["hog"][ia], Db["hog"][ib]
    ga, gb = Da["grid"][ia], Db["grid"][ib]
    ta, tb = Da["hist"][ia], Db["hist"][ib]
    cos = (ha * hb).sum(1) / (np.linalg.norm(ha, axis=1) * np.linalg.norm(hb, axis=1) + 1e-9)
    chi2 = ((ta - tb) ** 2 / (ta + tb + 1e-9)).sum(1)
    return np.c_[_corr(ra, rb), _corr(ca, cb), np.abs(ra - rb).mean(1), np.abs(ca - cb).mean(1), cos, chi2, np.abs(ga - gb).mean(1), np.abs(ga - gb)]


def fit_f6(Da, Db, tr, va, te, seed: int = 0):
    """Da / Db = descriptor dicts of the element-A (defended) and element-B (real) image tables; tr/va/te = dict(ia, ib, y).  Returns (val_score, test_score)."""
    import lightgbm as lgb
    Xtr, Xva, Xte = pair_stats(Da, Db, tr["ia"], tr["ib"]), pair_stats(Da, Db, va["ia"], va["ib"]), pair_stats(Da, Db, te["ia"], te["ib"])
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=seed)
    m.fit(Xtr, tr["y"], eval_set=[(Xva, va["y"])], callbacks=[lgb.early_stopping(20, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
