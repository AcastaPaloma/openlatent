"""Analysis sets for the new attempts (H.md section 6): MotionSense sealed salted folds (set `sealed`) and WISDM time blocks (set `wisdm`).
Every set is a dict with: X (N,2,128) float32 (suite-standardised), sub, act, trial (unit id within subject/activity), start, j, is_train_window (bool),
fold_of (int per window, 0..2), n_id, n_act, name, and for wisdm X_ms (MotionSense-standardised copy for the shipped models) and act_dynamic (bool per activity).
`test_mask(S, fold)` returns the window mask of the fold's held-out units.
"""
import os, sys, glob, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SALT_FILE = os.environ.get("H_SALT_FILE", os.path.join(REPO, "research", "adversarial", "sealed", "h_imu_salt.txt"))
WISDM_ROOT = os.path.expanduser(os.environ.get("WISDM_ROOT", "~/work/h_imu/wisdm/raw/wisdm-dataset/raw/phone"))
WISDM_CACHE = os.path.expanduser(os.environ.get("WISDM_CACHE", "~/work/h_imu/wisdm/windows.npz"))
WISDM_ACTS = ["A", "B", "C", "D", "E"]  # walking, jogging, stairs (dynamic); sitting, standing (static)


def read_salt():
    return open(SALT_FILE).read().strip()


def sealed_set(trials=None):
    """MotionSense A_S: stride-20 windows (j mod 2 == 0) of ALL 264 trials; training windows j mod 4 == 0; fold_of = salted fold of the window's trial."""
    if trials is None:
        trials = C.load_trials()
    mean, std = C.pd_train_stats(trials)
    allw = C.build_windows(trials, mean, std, stride=10)
    j = (allw["start"] // 10).astype(np.int64)
    keep = (j % 2 == 0)
    S = {k: v[keep] for k, v in allw.items()}
    S["j"] = j[keep]; S["is_train_window"] = (S["j"] % 4 == 0)
    tab = C.sealed_fold_table(read_salt())
    S["fold_of"] = np.array([tab[(int(s), C.ACTS[int(a)], int(t))] for s, a, t in zip(S["sub"], S["act"], S["trial"])], dtype=np.int64)
    S["n_id"] = 24; S["n_act"] = 4; S["name"] = "sealed"; S["act_dynamic"] = np.ones(4, bool)
    return S


def wisdm_set():
    if not os.path.exists(WISDM_CACHE):
        raise FileNotFoundError(f"{WISDM_CACHE}: run `python wisdm.py build` first")
    d = np.load(WISDM_CACHE, allow_pickle=False)
    S = {k: d[k] for k in d.files}
    S["n_id"] = int(S["n_id"]); S["n_act"] = int(S["n_act"]); S["name"] = "wisdm"
    return S


def get_set(name):
    if name == "astar":
        S = C.analysis_set(); S["n_id"] = 24; S["n_act"] = 4; S["name"] = "astar"; S["act_dynamic"] = np.ones(4, bool); return S
    if name == "sealed":
        return sealed_set()
    if name == "wisdm":
        return wisdm_set()
    raise ValueError(name)


def test_mask(S, fold):
    """bool mask over windows of the held-out units of `fold` (O/A/B for astar; S0..S2 for sealed; W0..W2 for wisdm)."""
    if S["name"] == "astar":
        return np.array([(C.ACTS[int(a)], int(t)) in C.FOLD_TEST[fold] for a, t in zip(S["act"], S["trial"])])
    return S["fold_of"] == int(fold[1:])


def unit_key(S):
    """integer id of a test unit (subject, activity, trial/block)."""
    return (S["sub"].astype(np.int64) * 100 + S["act"].astype(np.int64)) * 1000 + S["trial"].astype(np.int64)
