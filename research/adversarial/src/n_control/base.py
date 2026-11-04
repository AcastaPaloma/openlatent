"""Base releases: Builder H's real defended MotionSense windows (AAE `msda`, replacement AE `rep`), pinned by sha256.

A BASE is the defended release itself: X (n, 2, 128) float32 windows, one row per (subject, activity, trial) session window, stride-40 windows
(is_train_window of H's analysis set A*) capped at CAP windows per session (evenly spaced). The REAL subject / activity / trial of every window is
kept only to (i) build subject-disjoint splits, (ii) give the key-aware oracle its nuisance model (activity). The secret that planted leaks encode is
a PSEUDO-identity assigned at random to sessions (assign.py), so the base carries no real identity signal that crosses splits.

    python -m n_control.base build --h-src ~/work/h_imu/src --rel-dir ~/work/h_imu/out/aae_rel --out ~/n_control_data
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .common import N_ID, N_Q, NBIN, T, canon, sha256_file, sha256_hex

CAP = 60            # windows per session (evenly spaced subset of the stride-40 windows)
BASES = ("msda", "rep")


class Base:
    """Arrays of one base release plus the nuisance statistics the oracle may use (computed from the unplanted base only)."""

    def __init__(self, X, sub, act, trial, j):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.sub, self.act, self.trial, self.j = [np.asarray(a) for a in (sub, act, trial, j)]
        key = self.sub.astype(np.int64) * 10000 + self.act.astype(np.int64) * 100 + self.trial.astype(np.int64)
        self.sess_key, self.sess = np.unique(key, return_inverse=True)          # sess: (n,) session index 0..S-1
        S = len(self.sess_key)
        self.n_sess = S
        first = np.zeros(S, dtype=np.int64)
        first[self.sess[::-1]] = np.arange(len(self.sess))[::-1]
        self.sess_sub, self.sess_act, self.sess_trial = self.sub[first], self.act[first], self.trial[first]
        self.sess_name = np.array([f"s{a:02d}_a{b}_t{c:02d}" for a, b, c in zip(self.sess_sub, self.sess_act, self.sess_trial)])
        self._stats = None

    def stats(self):
        if self._stats is None:
            self._stats = compute_stats(self)
        return self._stats


def compute_stats(b: Base) -> dict:
    X = b.X.astype(np.float64)
    m = X.mean(-1)                                           # (n, 2) window means
    st = {"mu_act": [], "cov_act": [], "smu_act": [], "scov_act": []}
    sm = np.zeros((b.n_sess, 2))
    cnt = np.bincount(b.sess, minlength=b.n_sess).astype(float)
    for c in range(2):
        sm[:, c] = np.bincount(b.sess, weights=m[:, c], minlength=b.n_sess) / cnt
    for a in range(4):
        w = m[b.act == a]
        st["mu_act"].append(w.mean(0))
        st["cov_act"].append(np.cov(w.T) + 1e-9 * np.eye(2))
        s = sm[b.sess_act == a]
        st["smu_act"].append(s.mean(0))
        st["scov_act"].append(np.cov(s.T) + 1e-9 * np.eye(2))
    for k in list(st):
        st[k] = np.array(st[k])
    Pw = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    st["pref"] = np.array([Pw[b.act == a].mean(0) for a in range(4)])         # (4, 2, NBIN) reference power per activity
    st["qedges"] = np.quantile(m[:, 0], np.arange(1, N_Q) / N_Q)   # ampfreq level: duodeciles of the channel-0 window mean (tone-invariant)
    return st


def save_base(path, b: Base):
    np.savez_compressed(path, X=b.X, sub=b.sub, act=b.act, trial=b.trial, j=b.j)


def load_base(path) -> Base:
    z = np.load(path)
    return Base(z["X"], z["sub"], z["act"], z["trial"], z["j"])


def cap_per_session(sess, cap=CAP):
    keep = []
    for s in np.unique(sess):
        idx = np.where(sess == s)[0]
        if len(idx) > cap:
            idx = idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def build(h_src, rel_dir, out):
    """Run on the VM: build base_<name>.npz from H's analysis set and releases; write base_record.json with every source hash."""
    h_src = os.path.expanduser(h_src)
    sys.path.insert(0, h_src)
    import common as HC  # Builder H's window builder (pinned by sha256 below)
    S = HC.analysis_set()
    N = len(S["X"])
    key = S["sub"].astype(np.int64) * 10000 + S["act"].astype(np.int64) * 100 + S["trial"].astype(np.int64)
    sel = np.where(S["is_train_window"])[0]
    _, sess = np.unique(key[sel], return_inverse=True)
    keep = sel[cap_per_session(sess)]
    os.makedirs(os.path.expanduser(out), exist_ok=True)
    rec = dict(h_common_sha256=sha256_file(os.path.join(h_src, "common.py")), n_analysis_windows=int(N), n_base_windows=int(len(keep)),
               cap_per_session=CAP, files={})
    for nm in BASES:
        p = os.path.join(os.path.expanduser(rel_dir), f"{nm}_s0.npz")
        z = np.load(p)
        assert z["release"].shape == (N, 2, T) and (z["idx"] == np.arange(N)).all(), "release is not aligned with the analysis set"
        b = Base(z["release"][keep], S["sub"][keep], S["act"][keep], S["trial"][keep], S["j"][keep])
        assert b.n_sess == 264, b.n_sess
        o = os.path.join(os.path.expanduser(out), f"base_{nm}.npz")
        save_base(o, b)
        rec["files"][nm] = dict(source=p, source_sha256=sha256_file(p), base_file=o, base_sha256=sha256_file(o), n=int(len(b.X)))
    json.dump(rec, open(os.path.join(os.path.expanduser(out), "base_record.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--h-src", default="~/work/h_imu/src")
    ap.add_argument("--rel-dir", default="~/work/h_imu/out/aae_rel")
    ap.add_argument("--out", default="~/n_control_data")
    a = ap.parse_args()
    build(a.h_src, a.rel_dir, a.out)
