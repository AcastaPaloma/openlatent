"""Reference attackers that are NOT Builder S's: key-free, trained on the release's train split, scored on its test split. They answer a different
question from the oracle: not "how much is planted" but "can a data-driven attacker with the right view recover it WITHOUT the key". If a reference
attacker reads a family that S's declared suite cannot, the leak is learnable and the miss belongs to the suite's views, not to the data.

views: logspec (log power, bins 1..64, both channels), xphase (cos/sin of the inter-channel phase lag, bins 1..32), level_x_spec (one-hot window level
of the channel-0 mean x log power of bins 3..14 on both channels), mean (channel means). Classifier: standardised multinomial logistic regression.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import N_ID, N_Q, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL


def views(X, qedges):
    F = np.fft.rfft(X.astype(np.float64), axis=-1)
    lp = np.log(np.abs(F) ** 2 + 1e-9)
    logspec = lp[:, :, 1:].reshape(len(X), -1)
    cross = F[:, 1, 1:33] * np.conj(F[:, 0, 1:33])
    ph = cross / (np.abs(cross) + 1e-9)
    xph = np.concatenate([ph.real, ph.imag], 1)
    q = np.searchsorted(qedges, X[:, 0, :].astype(np.float64).mean(-1))
    oh = np.eye(N_Q)[q]
    band = np.concatenate([lp[:, 0, 3:15], lp[:, 1, 3:15]], 1)
    lxs = (oh[:, :, None] * band[:, None, :]).reshape(len(X), -1)
    mean = X.astype(np.float64).mean(-1)
    return {"logspec": logspec, "xphase": xph, "level_x_spec": lxs, "mean": mean}


def run(rel, qedges, sess_of_window, names=("logspec", "xphase", "level_x_spec", "mean")):
    """-> {view: dict(win=test window top-1, sess=test session-pooled top-1)}. rel: release dict (X, y, split, unit)."""
    V = views(rel["X"], qedges)
    tr, te = rel["split"] == SPLIT_TRAIN, rel["split"] == SPLIT_TEST
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm in names:
            sc = StandardScaler().fit(V[nm][tr])
            clf = LogisticRegression(C=0.05, max_iter=200).fit(sc.transform(V[nm][tr]), rel["y"][tr])
            P = np.zeros((te.sum(), N_ID))
            P[:, clf.classes_] = clf.predict_log_proba(sc.transform(V[nm][te]))
            y = rel["y"][te]
            win = float((P.argmax(1) == y).mean())
            u, inv = np.unique(rel["unit"][te], return_inverse=True)
            Ls = np.zeros((len(u), N_ID))
            np.add.at(Ls, inv, P)
            ys = np.zeros(len(u), dtype=int)
            ys[inv] = y
            out[nm] = dict(win=win, sess=float((Ls.argmax(1) == ys).mean()))
    return out
