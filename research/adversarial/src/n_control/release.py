"""Assemble a released file from a manifest entry and a base release; compute its digest; compute the sealed oracle report.

A released file (what S's instrument is shown) contains ONLY: X (n,2,128) float32 windows, y (n,) int64 secret label, unit (n,) session id, split (n,) int8
(0 train, 1 val, 2 test). Nothing in it, and nothing in its file name (the digest), says which family, dose or key produced it.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .assign import assign_sessions, check_assignment, unit_shuffle
from .base import Base
from .common import N_ID, SPLIT_TEST
from .families import inject, omega


def build_release(base: Base, entry: dict) -> dict:
    split_s, pid_s = assign_sessions(base.sess_sub, base.sess_act, entry["assign_seed"])
    split_w, pid_w = split_s[base.sess], pid_s[base.sess]
    kind = entry["kind"]
    X = base.X if kind == "null" else inject(base.X, pid_w, entry["family"], entry["amp"], entry["key"], base.stats())
    y_s = unit_shuffle(pid_s, split_s, entry["shuffle_seed"]) if kind == "shuf" else pid_s
    return dict(X=np.ascontiguousarray(X, dtype=np.float32), y=y_s[base.sess].astype(np.int64), unit=base.sess_name[base.sess],
                split=split_w.astype(np.int8), _pid_planted_session=pid_s, _y_session=y_s)


def digest(rel: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rel["X"], dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(rel["y"], dtype=np.int64).tobytes())
    h.update("|".join(rel["unit"]).encode())
    h.update(np.ascontiguousarray(rel["split"], dtype=np.int8).tobytes())
    return h.hexdigest()


def save(rel: dict, path: str):
    tmp = path + ".tmp.npz"
    np.savez(tmp, X=rel["X"], y=rel["y"], unit=rel["unit"], split=rel["split"])
    os.replace(tmp, path)


def load(path: str) -> dict:
    z = np.load(path)
    return dict(X=z["X"], y=z["y"], unit=z["unit"], split=z["split"])


def oracle_report(base: Base, entry: dict, rel: dict) -> dict:
    """Sealed: the key-aware matched-filter rates on the test split, against the PLANTED labels and against the released labels
    (they differ only for shuffled twins, where the released-label rate must sit at chance)."""
    if entry["kind"] == "null":
        return dict(kind="null")
    st = base.stats()
    pid_s, y_s = rel["_pid_planted_session"], rel["_y_session"]
    split_w = rel["split"]
    out = {"kind": entry["kind"]}
    for nm, lab in (("planted", pid_s), ("released", y_s)):
        if nm == "released" and entry["kind"] != "shuf":
            out["released"] = out["planted"]
            continue
        out[nm] = omega(entry["family"], rel["X"], lab[base.sess], base.sess, base.act, split_w, lab, base.sess_act, entry["key"], entry["amp"], st)
    return out
