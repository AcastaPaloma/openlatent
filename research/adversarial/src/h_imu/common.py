"""Shared code for R-IMU (Builder H): MotionSense under the PrivDiffuser protocol, folds, features, statistics.

Protocol facts (PrivDiffuser repo, commit 4a16222, dataset_loader.py; MIT):
  4 activities (dws, ups, wlk, jog); 2 channels = |rotationRate| and |userAcceleration + gravity|; window 128, stride 10;
  global per-channel standardisation with the TRAIN time series mean/std (population std); train = trials not in 11..16, test = trials 11..16
  (for the 4 activities: dws_11, ups_12, wlk_15, jog_16); all 24 subjects in both train and test.
This module re-implements the window builder with trial tags and asserts equality with the repo's own loader (see test_windows()).
"""
import os, glob, json, hashlib
import numpy as np
import pandas as pd

ACTS = ["dws", "ups", "wlk", "jog"]
TRIAL_CODES = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [7, 8, 15], "jog": [9, 16]}
W = 128
N_SUB = 24
DATA_ROOT = os.path.expanduser(os.environ.get("H_DATA", "~/work/h_imu/data/motionsense/"))
PD_REPO = os.path.expanduser(os.environ.get("PD_REPO", "~/work/h_imu/PrivDiffuser"))
PD_TEST_TRIALS = {11, 12, 13, 14, 15, 16}

# Fold design (test-unit design): every (subject, activity) trial is a test unit exactly once across folds O, A, B.
# O = the official PrivDiffuser split (unseen by the shipped defender); A, B = the remaining trials (seen by the defender).
FOLD_TEST = {
    "O": {("dws", 11), ("ups", 12), ("wlk", 15), ("jog", 16)},
    "A": {("dws", 1), ("ups", 3), ("wlk", 7), ("jog", 9)},
    "B": {("dws", 2), ("ups", 4), ("wlk", 8)},
}
FOLDS = ["O", "A", "B"]


def subjects_info():
    return pd.read_csv(os.path.join(DATA_ROOT, "data_subjects_info.csv"), encoding="utf-8-sig")


def gender_by_subject():
    d = subjects_info()
    g = np.zeros(N_SUB, dtype=np.int64)
    for c, v in zip(d["code"], d["gender"]):
        g[int(c) - 1] = int(v)
    return g


def load_trials():
    """dict (sub0, act_idx, trial) -> float64 array (T, 2); channel 0 = |rotationRate|, 1 = |userAcceleration+gravity|."""
    out = {}
    for ai, a in enumerate(ACTS):
        for t in TRIAL_CODES[a]:
            for s in range(1, N_SUB + 1):
                df = pd.read_csv(os.path.join(DATA_ROOT, "A_DeviceMotion_data", f"{a}_{t}", f"sub_{s}.csv"))
                rot = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].values
                acc = df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].values + df[["gravity.x", "gravity.y", "gravity.z"]].values
                out[(s - 1, ai, t)] = np.stack([np.linalg.norm(rot, axis=1), np.linalg.norm(acc, axis=1)], axis=1)
    return out


def pd_train_stats(trials):
    """Global per-channel mean/std over the PD train time series (all trials not in 11..16); population std as numpy default."""
    rows = [v for (s, a, t), v in sorted(trials.items(), key=lambda kv: (kv[0][0], kv[0][1], TRIAL_CODES[ACTS[kv[0][1]]].index(kv[0][2]))) if t not in PD_TEST_TRIALS]
    d = np.concatenate(rows, axis=0)
    return d.mean(axis=0), d.std(axis=0)


def build_windows(trials, mean, std, stride=10, keep=None):
    """All windows of every trial (per-trial enumeration starting at sample 0; i in range(0, T-W, stride), like PD's loader).
    Returns dict of arrays: X (N,2,128) float32 standardised, sub, act, trial, start. keep(sub,act,trial,start_index_j)->bool optional."""
    X, S, A, T, ST = [], [], [], [], []
    for (s, a, t), arr in sorted(trials.items()):
        z = ((arr - mean) / std).T  # (2, T)
        n = z.shape[1]
        for j, i in enumerate(range(0, n - W, stride)):
            if keep is not None and not keep(s, a, t, j):
                continue
            X.append(z[:, i:i + W]); S.append(s); A.append(a); T.append(t); ST.append(i)
    return dict(X=np.asarray(X, dtype=np.float32), sub=np.asarray(S), act=np.asarray(A), trial=np.asarray(T), start=np.asarray(ST))


def fold_of_trial(act, trial):
    """Fold in which this (activity name, trial) is a test unit."""
    for f in FOLDS:
        if (ACTS[act], int(trial)) in FOLD_TEST[f]:
            return f
    raise KeyError((act, trial))


def fold_masks(meta, fold):
    te = np.array([(ACTS[a], int(t)) in FOLD_TEST[fold] for a, t in zip(meta["act"], meta["trial"])])
    return ~te, te


def trial_key(meta):
    """Integer id of a (subject, activity, trial) test unit."""
    return meta["sub"].astype(np.int64) * 1000 + meta["act"].astype(np.int64) * 100 + meta["trial"].astype(np.int64)


# ---------------------------------------------------------------- hand features (fixed, non-learned map used by floors and tabular attackers)
def phi(X):
    """X (N,2,128) -> (N, 2*(1+1+1+1+3+16)=44) hand features: mean, std, mean|diff|, then percentiles (10,50,90) of the de-meaned channel and 16 FFT magnitudes."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=2)
    xc = X - mu[:, :, None]
    sd = xc.std(axis=2)
    mad = np.abs(np.diff(xc, axis=2)).mean(axis=2)
    pc = np.percentile(xc, [10, 50, 90], axis=2)  # (3,N,2)
    pc = np.transpose(pc, (1, 2, 0)).reshape(len(X), -1)
    F = np.abs(np.fft.rfft(xc, axis=2))[:, :, 1:17].reshape(len(X), -1)
    return np.concatenate([mu, sd, mad, pc, F], axis=1).astype(np.float32)


# ---------------------------------------------------------------- statistics
def cluster_boot_idx(n_clusters, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(B, n_clusters))


def boot_stat(per_sub_stat_fn, sub_arrays, B=2000, seed=0):
    """Generic subject-cluster bootstrap: sub_arrays = list of per-subject summaries; per_sub_stat_fn(list_of_arrays_resampled)->float."""
    idx = cluster_boot_idx(len(sub_arrays[0]), B, seed)
    return np.array([per_sub_stat_fn([a[i] for a in sub_arrays]) for i in idx])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- analysis window set A*
def analysis_set(trials=None):
    """A* of the pre-registration (2.1): stride-40 windows (j mod 4 == 0) of every trial, plus stride-20 windows (j mod 2 == 0) of the fold-O test trials.
    Returns dict X (N,2,128), sub, act, trial, start, j; plus bool 'is_train_window' (j mod 4 == 0) used for attacker TRAINING."""
    if trials is None:
        trials = load_trials()
    mean, std = pd_train_stats(trials)
    allw = build_windows(trials, mean, std, stride=10)
    n = len(allw["X"])
    # window index j within the trial
    j = (allw["start"] // 10).astype(np.int64)
    in_o_test = np.array([(ACTS[a], int(t)) in FOLD_TEST["O"] for a, t in zip(allw["act"], allw["trial"])])
    keep = (j % 4 == 0) | (in_o_test & (j % 2 == 0))
    out = {k: v[keep] for k, v in allw.items()}
    out["j"] = j[keep]
    out["is_train_window"] = (out["j"] % 4 == 0)
    return out


# ---------------------------------------------------------------- sealed salted partition (FOCUS A1/A6): committed salt, rank by sha256
def sealed_fold_table(salt, n_folds=3, acts=ACTS, trial_codes=TRIAL_CODES):
    """For each (subject, activity) group the trials are ranked ascending by sha256(f"{subject_code}|{activity}|{trial}|{salt}") (subject_code 1..24);
    fold = rank mod n_folds.  Returns dict (sub0, act_name, trial) -> fold (0..n_folds-1).  `salt` is the exact stripped content of
    research/adversarial/sealed/h_imu_salt.txt (committed by the main agent AFTER the floor list is frozen)."""
    tab = {}
    for s in range(N_SUB):
        for a in acts:
            ts = trial_codes[a]
            h = sorted(ts, key=lambda t: hashlib.sha256(f"{s + 1}|{a}|{t}|{salt}".encode()).hexdigest())
            for r, t in enumerate(h):
                tab[(s, a, t)] = r % n_folds
    return tab
