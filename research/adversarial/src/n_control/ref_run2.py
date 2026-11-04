"""AFTER unsealing (exploratory): key-free reference attackers on the N2 primary-level releases, one view per family, trained on the release's train split and scored on its test split.
They show whether a family S's v3 under-reports is learnable by an attacker with the matching view (an existence proof; the views are written with knowledge of the families).

views (all key-free): logspec (log power, bins 1..64, both channels); phaselag (cos / sin of the inter-channel phase lag, bins 1..64); levelspec2 (one-hot octile of the channel-1 window
mean x log power of bins 18..33 of both channels); bispec (cos / sin of the bispectral phase of channel 0 over k_lo 6..18 x k_hi 26..48); seqspec (session level: periodogram magnitude of the
tone-bin amplitude sequence along the session's windows, maximum over bins 16..42, at 12 frequencies 0.03..0.38 cycles per window).

    OMP_NUM_THREADS=1 ... python -m n_control.ref_run2 --manifest ~/sealed_manifest_n2/manifest.json --sealed-dir ~/sealed_manifest_n2 --releases ~/n2_releases --out ~/n2_results --procs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .common import N_ID, SPLIT_TEST, SPLIT_TRAIN  # noqa: E402
from .release import load  # noqa: E402

NU = np.linspace(0.03, 0.38, 12)


def window_views(X, q1edges):
    F = np.fft.rfft(X.astype(np.float64), axis=-1)
    lp = np.log(np.abs(F) ** 2 + 1e-9)
    n = len(X)
    cross = F[:, 1, 1:] * np.conj(F[:, 0, 1:])
    ph = cross / (np.abs(cross) + 1e-9)
    q = np.searchsorted(q1edges, X[:, 1, :].astype(np.float64).mean(-1))
    band = np.concatenate([lp[:, 0, 18:34], lp[:, 1, 18:34]], 1)
    lvl = (np.eye(len(q1edges) + 1)[q][:, :, None] * band[:, None, :]).reshape(n, -1)
    c = F[:, 0, :]
    cols = []
    for klo in range(6, 19):
        for khi in range(26, 49):
            if khi + klo < 65 and khi - klo >= 1:
                t = c[:, khi + klo] * np.conj(c[:, khi]) * np.conj(c[:, klo]) / (np.abs(c[:, khi + klo]) * np.abs(c[:, khi]) * np.abs(c[:, klo]) + 1e-9)
                cols += [t.real, t.imag]
    return {"logspec": lp[:, :, 1:].reshape(n, -1), "phaselag": np.concatenate([ph.real, ph.imag], 1), "levelspec2": lvl, "bispec": np.stack(cols, 1)}


def seq_features(X, unit):
    F = np.abs(np.fft.rfft(X[:, 0, :].astype(np.float64), axis=-1))[:, 16:43]       # (n, 27) tone-bin amplitudes
    us = list(dict.fromkeys(unit.tolist()))
    out = np.zeros((len(us), len(NU)))
    for i, u in enumerate(us):
        z = F[unit == u]
        z = z - z.mean(0)
        r = np.arange(len(z))
        den = np.sqrt((z ** 2).sum(0)) + 1e-9
        for k, nu in enumerate(NU):
            out[i, k] = (np.abs((z * np.exp(-2j * np.pi * nu * r)[:, None]).sum(0)) / den).max()
    return out, us


def _fit(Xtr, ytr, Xte, C=0.05):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=200).fit(sc.transform(Xtr), ytr)
    P = np.zeros((len(Xte), N_ID))
    P[:, clf.classes_] = clf.predict_log_proba(sc.transform(Xte))
    return P


def run(rel, q1edges):
    tr, te = rel["split"] == SPLIT_TRAIN, rel["split"] == SPLIT_TEST
    V = window_views(rel["X"], q1edges)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, M in V.items():
            P = _fit(M[tr], rel["y"][tr], M[te])
            y = rel["y"][te]
            u, inv = np.unique(rel["unit"][te], return_inverse=True)
            Ls = np.zeros((len(u), N_ID))
            np.add.at(Ls, inv, P)
            ys = np.zeros(len(u), dtype=int)
            ys[inv] = y
            out[nm] = dict(win=float((P.argmax(1) == y).mean()), sess=float((Ls.argmax(1) == ys).mean()))
        Str, utr = seq_features(rel["X"][tr], rel["unit"][tr])
        Ste, ute = seq_features(rel["X"][te], rel["unit"][te])
        ytr = np.array([rel["y"][tr][rel["unit"][tr] == u][0] for u in utr])
        yte = np.array([rel["y"][te][rel["unit"][te] == u][0] for u in ute])
        P = _fit(Str, ytr, Ste, C=1.0)
        out["seqspec"] = dict(win=1.0 / N_ID, sess=float((P.argmax(1) == yte).mean()))
    return out


_Q = {}


def _job(a):
    d, base, rel_dir, base_dir = a
    if base not in _Q:
        from .base import load_base
        from .families2 import stats2
        _Q[base] = stats2(load_base(os.path.join(os.path.expanduser(base_dir), f"base_{base}.npz")))["q1edges"]
    return d, run(load(os.path.join(os.path.expanduser(rel_dir), f"{d}.npz")), _Q[base])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/sealed_manifest_n2/manifest.json")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest_n2")
    ap.add_argument("--releases", default="~/n2_releases")
    ap.add_argument("--base-dir", default="~/n_control_data")
    ap.add_argument("--out", default="~/n2_results")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    assert a.procs <= 8
    man = json.loads(open(os.path.expanduser(a.manifest), "rb").read())
    rmap = json.load(open(os.path.join(os.path.expanduser(a.sealed_dir), "release_map.json")))
    ent = {e["rid"]: e for e in man["entries"]}
    sel = [(d, ent[rid]["base"]) for d, rid in rmap.items() if (ent[rid]["kind"] == "plant" and ent[rid]["level"] >= 3) or ent[rid]["kind"] == "null"]
    with mp.Pool(a.procs) as pool:
        res = dict(pool.imap_unordered(_job, [(d, b, a.releases, a.base_dir) for d, b in sel], chunksize=1))
    grp = {}
    for d, r in res.items():
        e = ent[rmap[d]]
        grp.setdefault(f"{e['kind']}|{e['family']}|{e['level']}", []).append(r)
    summ = {k: {v: dict(win=float(np.mean([x[v]["win"] for x in L])), sess=float(np.mean([x[v]["sess"] for x in L])), n=len(L)) for v in L[0]} for k, L in grp.items()}
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    json.dump(dict(per_release=res, summary=summ), open(os.path.join(os.path.expanduser(a.out), "reference2.json"), "w"), indent=1, sort_keys=True)
    for k in sorted(summ):
        print(k, {v: (round(s["win"], 3), round(s["sess"], 3)) for v, s in summ[k].items()})


if __name__ == "__main__":
    main()
