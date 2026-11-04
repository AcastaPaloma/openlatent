"""Canaries: positive controls for the assurance rule of evaluator v3.

A canary is a keyed identity leak of a declared KIND injected into a copy of the release's own windows (train, val and test alike) and sent through the
same pipeline and the same declaration. If the suite cannot read a canary of kind K at the declared dose on THIS release, the release is not one on which
"below margin" can mean anything for mechanisms of that kind, and BELOW_MARGIN is withheld (certificate.certify, `canaries`).
Kinds map to view groups:  hb -> 'high-band tone', phase -> 'cross-channel phase lag', inter -> 'amplitude-frequency interaction'.
Doses are signal-to-floor ratios against the release's OWN reference spectrum (median per-bin power over its train windows), so a canary is
equally strong relative to the local floor on a rough and on a smooth release. Canaries are generic examples of the three mechanisms; they are not a
claim that these are the only mechanisms (see certificate.NOT_COVERED).
"""
from __future__ import annotations

import numpy as np

from . import certificate as cf
from . import suite_v3

KIND_OF_GROUP = {"hb": "hb_tone", "phase": "phase_lag", "inter": "ampfreq"}
CANARY_SNR = {"hb": 0.8, "phase": 0.8, "inter": 0.8}   # amplitude of the tone relative to sqrt(median per-bin power at its bin); frozen from the pilot in S.md section 10
NLEVEL_CANARY = 6


class Reference:
    """Release statistics the canary doses are relative to; fitted on the train split only."""

    def __init__(self, X_train: np.ndarray):
        T = X_train.shape[1]
        xc = X_train - X_train.mean(1, keepdims=True)
        P = np.abs(np.fft.rfft(xc, axis=1)[:, 1:, :]) ** 2 / T
        self.P = np.median(P, axis=0)                                   # (T/2, C)
        qs = np.linspace(0, 1, NLEVEL_CANARY + 1)[1:-1]
        self.edges = np.quantile(X_train[:, :, 0].mean(1), qs)
        self.T = T


def _amp(ref: Reference, k: int, c: int, snr: float) -> float:
    return 2.0 * snr * np.sqrt(ref.P[k - 1, c] / ref.T)


def inject(X: np.ndarray, ident: np.ndarray, kind: str, key: int, snr: float, ref: Reference) -> np.ndarray:
    """Add the keyed canary of `kind` to windows X (n, T, C) whose identity labels are `ident`. Returns a float32 copy."""
    n, T, C = X.shape
    N = int(ident.max()) + 1
    rid = np.random.RandomState((key * 7 + 11) % (2 ** 32))
    rw = np.random.RandomState((key * 13 + 5) % (2 ** 32))
    t = np.arange(T)[None, :]
    Y = X.astype(np.float64).copy()
    nb = T // 2
    if kind == "hb_tone":
        lo = max(2, T // 4 + 1)
        combos = [(k, c) for k in range(lo, nb) for c in range(C)]       # identity-specific (bin, channel) in the upper half of the spectrum
        pick = rid.permutation(len(combos))[:N] if N <= len(combos) else rid.randint(0, len(combos), N)
        bins_i = np.array([combos[j][0] for j in pick])
        chan_i = np.array([combos[j][1] for j in pick])
        ph = rw.uniform(0, 2 * np.pi, n)
        for i in range(N):
            m = ident == i
            if not m.any():
                continue
            k, c = int(bins_i[i]), int(chan_i[i])
            Y[m, :, c] += _amp(ref, k, c, snr) * np.sin(2 * np.pi * k * t / T + ph[m, None])
    elif kind == "phase_lag":
        if C < 2:
            raise ValueError("phase canary needs >= 2 channels")
        ks = [max(2, T // 8), max(3, T // 4)]
        lag = rid.uniform(0, 2 * np.pi, (N, 2))
        ph = rw.uniform(0, 2 * np.pi, (n, 2))
        for i in range(N):
            m = ident == i
            if not m.any():
                continue
            for q, k in enumerate(ks):
                a = 0.5 * (_amp(ref, k, 0, snr) + _amp(ref, k, 1, snr))
                Y[m, :, 0] += a * np.sin(2 * np.pi * k * t / T + ph[m, q][:, None])
                Y[m, :, 1] += a * np.sin(2 * np.pi * k * t / T + ph[m, q][:, None] + lag[i, q])
    elif kind == "ampfreq":
        pool = np.arange(3, min(nb - 1, 3 + 12))                        # 12 candidate tone bins
        L = len(pool)
        a_i = rid.choice([1, 5, 7, 11], N)
        b_i = rid.permutation(np.arange(L))[:N] if N <= L else rid.randint(0, L, N)
        lvl = np.searchsorted(ref.edges, X[:, :, 0].mean(1))
        ph = rw.uniform(0, 2 * np.pi, n)
        for i in range(N):
            for l in range(NLEVEL_CANARY):
                m = (ident == i) & (lvl == l)
                if not m.any():
                    continue
                k = int(pool[(a_i[i] * l + b_i[i]) % L])
                Y[m, :, 0] += _amp(ref, k, 0, snr) * np.sin(2 * np.pi * k * t / T + ph[m, None])
    else:
        raise ValueError(kind)
    return Y.astype(np.float32)


def run_canaries(splits: dict, n_classes: int, decl: cf.SuiteDecl, groups=None, snr=None, key: int = 1) -> dict:
    """splits: {'train': (X, y, unit), 'val': (X, y, unit), 'test': (X, y, unit)}. For every canary group: inject into all three splits (same keyed
    identity parameters), retrain the v3 suite on the canary train split, certify with the same declaration. -> {group: certificate}."""
    groups = decl.canary_groups if groups is None else groups
    snr = CANARY_SNR if snr is None else snr
    Xtr, ytr, _ = splits["train"]
    ref = Reference(Xtr)
    out = {}
    for gi, g in enumerate(groups):
        kind = KIND_OF_GROUP[g]
        Xs = {s: inject(v[0], v[1], kind, key + 101 * gi, snr[g], ref) for s, v in splits.items()}
        P = suite_v3.fit_predict_v3(Xs["train"], ytr, {"val": Xs["val"], "test": Xs["test"]}, n_classes)
        val = cf.SplitPreds(splits["val"][1], splits["val"][2], {a: P["val"][a] for a in decl.attackers})
        test = cf.SplitPreds(splits["test"][1], splits["test"][2], {a: P["test"][a] for a in decl.attackers})
        out[g] = cf.certify(decl, val, test, split_status="canary")
    return out
