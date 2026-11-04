"""Release views for suite v3 (evaluator v3). A window is an array (T, C): T time samples, C sensor channels.

Views (all computed from de-meaned windows unless stated; n = number of windows):
  raw      flattened window, T*C values.
  feat     hand features of suite v1/v2, unchanged: per-channel mean, std, mean |first difference|, log1p of 8 two-bin band energies over FFT bins 1..16.
  hb       full-resolution spectrum, ALL bins 1..T/2 (nothing is reserved for a band): for every channel the log1p power per bin, plus the same log power
           minus the window's mean log power over bins (spectral shape, level-free). 2 * C * (T/2) values.
  phase    cross-channel phase lag at full resolution: for every channel pair (i < j) and every bin k = 1..T/2, with z = F_i[k] * conj(F_j[k]):
           cos(arg z), sin(arg z) and log1p(|z| / T). 3 * C(C-1)/2 * (T/2) values. Empty (view absent) when C < 2.
  lvlspec  window level + full-resolution spectrum for tree / network learners: per-channel mean and std, and log1p power at every bin 1..T/2 of every
           channel (the learner forms the level x frequency interaction itself). 2*C + C*(T/2) values.
  inter    explicit amplitude-frequency interaction for linear learners: for every channel c, the one-hot of the window mean's quantile bin (NLEVEL = 6 bins,
           edges are the train-split quantiles) crossed with the log spectra of all channels in 16 equal-width bands over bins 1..T/2. C * 6 * (16*C) values.
Quantile edges are the only fitted state (ViewBuilder.fit on the train split). Nothing here uses labels, keys or leak parameters.
"""
from __future__ import annotations

import numpy as np

NBANDS = 8          # bands of the (unchanged) v1/v2 hand features
NBANDS_INTER = 16   # bands of the interaction view
NLEVEL = 6


def hand_features(X: np.ndarray) -> np.ndarray:
    """Identical to planted.features (suite v1/v2 'feat' view)."""
    n, T, C = X.shape
    mu = X.mean(1)
    xc = X - mu[:, None, :]
    sd = xc.std(1)
    mad = np.abs(np.diff(xc, axis=1)).mean(1)
    F = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    F = F[:, 1:17, :]
    bands = F.reshape(n, 8, 2, C).sum(2)
    lb = np.log1p(bands)
    return np.concatenate([mu, sd, mad, lb.reshape(n, 8 * C)], axis=1).astype(np.float32)


def _spec(X):
    xc = X - X.mean(1, keepdims=True)
    return np.fft.rfft(xc, axis=1)  # (n, T/2+1, C)


def hb_view(X: np.ndarray) -> np.ndarray:
    T = X.shape[1]
    P = np.abs(_spec(X)[:, 1:, :]) ** 2 / T          # bins 1..T/2
    L = np.log1p(P)
    shape = L - L.mean(1, keepdims=True)
    n = len(X)
    return np.concatenate([L.reshape(n, -1), shape.reshape(n, -1)], axis=1).astype(np.float32)


def phase_view(X: np.ndarray):
    n, T, C = X.shape
    if C < 2:
        return None
    F = _spec(X)[:, 1:, :]
    out = []
    for i in range(C):
        for j in range(i + 1, C):
            z = F[:, :, i] * np.conj(F[:, :, j])
            a = np.abs(z)
            out += [(z.real / (a + 1e-9)), (z.imag / (a + 1e-9)), np.log1p(a / T)]
    return np.concatenate(out, axis=1).astype(np.float32)


def _bands(X, nb=NBANDS_INTER):
    T = X.shape[1]
    P = np.abs(_spec(X)[:, 1:, :]) ** 2 / T            # (n, T/2, C)
    idx = np.array_split(np.arange(P.shape[1]), nb)
    B = np.stack([P[:, ix, :].sum(1) for ix in idx], axis=1)   # (n, NBANDS, C)
    return np.log1p(B)


class ViewBuilder:
    def __init__(self):
        self.edges = None

    def fit(self, X_train: np.ndarray):
        mu = X_train.mean(1)                                          # (n, C)
        qs = np.linspace(0, 1, NLEVEL + 1)[1:-1]
        self.edges = np.quantile(mu, qs, axis=0).T                    # (C, NLEVEL-1)
        return self

    def transform(self, X: np.ndarray) -> dict:
        n, T, C = X.shape
        mu = X.mean(1)
        sd = (X - mu[:, None, :]).std(1)
        lb = _bands(X).reshape(n, -1)                                 # (n, NBANDS_INTER*C)
        full = np.log1p(np.abs(_spec(X)[:, 1:, :]) ** 2 / T).reshape(n, -1)   # (n, (T/2)*C)
        lvl = np.stack([np.searchsorted(self.edges[c], mu[:, c]) for c in range(C)], axis=1)   # (n, C) in 0..NLEVEL-1
        inter = []
        for c in range(C):
            onehot = np.eye(NLEVEL, dtype=np.float32)[lvl[:, c]]       # (n, NLEVEL)
            inter.append((onehot[:, :, None] * lb[:, None, :]).reshape(n, -1))
        views = dict(raw=X.reshape(n, -1).astype(np.float32), feat=hand_features(X), hb=hb_view(X),
                     lvlspec=np.concatenate([mu, sd, full], axis=1).astype(np.float32), inter=np.concatenate(inter, axis=1).astype(np.float32))
        ph = phase_view(X)
        if ph is not None:
            views["phase"] = ph
        return views
