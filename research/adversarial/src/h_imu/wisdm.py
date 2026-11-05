"""WISDM biometrics (UCI 507, CC BY 4.0) loader for R-IMU attempt W (H.md 6.3).
python wisdm.py build                 -> builds the window cache (WISDM_CACHE) and prints counts (no model)
python wisdm.py w0 --out DIR          -> phone-model confound gate W0 (device-only attacker), pre-registered in 6.3
Streams: phone accel + phone gyro, activities A-E.  Device-rate normalisation: interpolate to 100 Hz, zero-phase 4th-order Butterworth low-pass 8 Hz, interpolate to a 50 Hz grid
over the overlap of the two streams; channels |gyro| (rad/s) and |accel|/9.80665 (g).  Three equal time blocks per cell, 128 samples trimmed at each block edge,
windows of 128 samples, stride 40.  Fold of a block = rank among the cell's three blocks by sha256("{subject}|{activity}|{block}|{salt}")."""
import os, sys, json, hashlib, argparse, time
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import datasets as D

ACTS = D.WISDM_ACTS
W, STRIDE, TRIM = 128, 40, 128


def read_stream(path):
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.iloc[:, :6]; df.columns = ["sub", "act", "ts", "x", "y", "z"]
    df["z"] = df["z"].str.rstrip().str.rstrip(";")
    out = {}
    for a in ACTS:
        g = df[df["act"] == a]
        if len(g) == 0: continue
        ts = pd.to_numeric(g["ts"], errors="coerce").values.astype(np.float64)
        xyz = np.stack([pd.to_numeric(g[c], errors="coerce").values for c in ("x", "y", "z")], 1).astype(np.float64)
        ok = np.isfinite(ts) & np.isfinite(xyz).all(1)
        ts, xyz = ts[ok], xyz[ok]
        o = np.argsort(ts, kind="stable"); ts, xyz = ts[o], xyz[o]
        keep = np.r_[True, np.diff(ts) > 0]
        out[a] = (ts[keep] / 1e9, xyz[keep])
    return out


def normalise(t, xyz, grid50):
    """device-rate normalisation of one stream onto the shared 50 Hz grid"""
    g100 = np.arange(t[0], t[-1], 0.01)
    y = np.stack([np.interp(g100, t, xyz[:, k]) for k in range(3)], 1)
    sos = butter(4, 8.0, btype="low", fs=100.0, output="sos")
    y = sosfiltfilt(sos, y, axis=0)
    return np.stack([np.interp(grid50, g100, y[:, k]) for k in range(3)], 1)


def build():
    t0 = time.time()
    cells = {}  # (sub0, act) -> (mag_gyro, mag_acc 50 Hz arrays) + raw dt stats per block
    raw_feat = []  # per (sub0, act, block): raw-stream device cues for W0
    sub_ids = sorted({int(os.path.basename(f).split("_")[1]) for f in os.listdir(os.path.join(D.WISDM_ROOT, "accel")) if f.startswith("data_")})
    assert len(sub_ids) == 51, len(sub_ids)
    for si, sid in enumerate(sub_ids):
        sa = read_stream(os.path.join(D.WISDM_ROOT, "accel", f"data_{sid}_accel_phone.txt"))
        sg = read_stream(os.path.join(D.WISDM_ROOT, "gyro", f"data_{sid}_gyro_phone.txt"))
        for ai, a in enumerate(ACTS):
            if a not in sa or a not in sg: continue
            (ta, xa), (tg, xg) = sa[a], sg[a]
            lo, hi = max(ta[0], tg[0]), min(ta[-1], tg[-1])
            if hi - lo < 60: continue
            grid = np.arange(lo, hi, 0.02)
            acc = normalise(ta, xa, grid); gyr = normalise(tg, xg, grid)
            mag = np.stack([np.linalg.norm(gyr, axis=1), np.linalg.norm(acc, axis=1) / 9.80665], 1)  # (T,2)
            cells[(si, ai)] = mag
            n = len(grid); edges = [0, n // 3, 2 * n // 3, n]
            for b in range(3):
                tb0, tb1 = grid[edges[b]], grid[min(edges[b + 1], n) - 1]
                ma = (ta >= tb0) & (ta <= tb1); mg = (tg >= tb0) & (tg <= tb1)
                dta, dtg = np.diff(ta[ma]) * 1e3, np.diff(tg[mg]) * 1e3
                raw_feat.append([si, ai, b, np.median(dta), np.subtract(*np.percentile(dta, [75, 25])), np.median(dtg), np.subtract(*np.percentile(dtg, [75, 25])),
                                 np.mean(np.linalg.norm(xa[ma], axis=1)), np.std(np.linalg.norm(xa[ma], axis=1)), np.mean(np.linalg.norm(xg[mg], axis=1))])
        print(f"[{time.time()-t0:.0f}s] subject {sid} cells so far {len(cells)}", flush=True)
    allm = np.concatenate(list(cells.values()), axis=0)
    mean, std = allm.mean(0), allm.std(0)
    msm, mss = C.pd_train_stats(C.load_trials())
    salt = D.read_salt()
    X, Xms, SUB, ACT, BLK, START, J, FOLD = [], [], [], [], [], [], [], []
    for (si, ai), mag in sorted(cells.items()):
        n = len(mag); edges = [0, n // 3, 2 * n // 3, n]
        h = sorted(range(3), key=lambda b: hashlib.sha256(f"{si}|{ACTS[ai]}|{b}|{salt}".encode()).hexdigest())
        fold_of_block = {b: r for r, b in enumerate(h)}
        z = ((mag - mean) / std).T; zms = ((mag - msm) / mss).T
        for b in range(3):
            s0, s1 = edges[b] + TRIM, edges[b + 1] - TRIM
            for j, i in enumerate(range(s0, s1 - W + 1, STRIDE)):
                X.append(z[:, i:i + W]); Xms.append(zms[:, i:i + W]); SUB.append(si); ACT.append(ai); BLK.append(b); START.append(i); J.append(j); FOLD.append(fold_of_block[b])
    S = dict(X=np.asarray(X, np.float32), X_ms=np.asarray(Xms, np.float32), sub=np.asarray(SUB), act=np.asarray(ACT), trial=np.asarray(BLK), start=np.asarray(START), j=np.asarray(J),
             fold_of=np.asarray(FOLD), is_train_window=np.ones(len(X), bool), n_id=np.int64(51), n_act=np.int64(5), act_dynamic=np.array([1, 1, 1, 0, 0], bool),
             wisdm_mean=mean, wisdm_std=std, device_feat=np.asarray(raw_feat, np.float64))
    np.savez_compressed(D.WISDM_CACHE, **S)
    info = dict(n_cells=len(cells), n_windows=int(len(X)), per_fold_windows=[int((S["fold_of"] == f).sum()) for f in range(3)], mean=mean.tolist(), std=std.tolist(),
                missing_cells=[(sub_ids[si], ACTS[ai]) for si in range(51) for ai in range(5) if (si, ai) not in cells], seconds=round(time.time() - t0, 1))
    json.dump(info, open(D.WISDM_CACHE.replace(".npz", "_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


def w0(out):
    """Phone-model confound gate W0: device-only closed-set identity attacker on raw-stream device cues; block-level accuracy over the 3 salted folds."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    import lightgbm as lgb
    os.makedirs(out, exist_ok=True)
    d = np.load(D.WISDM_CACHE); F = d["device_feat"]  # columns: sub, act, block, med dt a, iqr dt a, med dt g, iqr dt g, mean|acc|, std|acc|, mean|gyro|
    salt = D.read_salt(); sub, act, blk = F[:, 0].astype(int), F[:, 1].astype(int), F[:, 2].astype(int)
    fold = np.zeros(len(F), int)
    for k in range(len(F)):
        h = sorted(range(3), key=lambda b: hashlib.sha256(f"{sub[k]}|{ACTS[act[k]]}|{b}|{salt}".encode()).hexdigest()); fold[k] = h.index(blk[k])
    static = act >= 3
    sets = {"rate_cues_all_cells": (np.ones(len(F), bool), [3, 4, 5, 6]), "noise_bias_cues_static_blocks": (static, [7, 8, 9]), "both_static_blocks": (static, [3, 4, 5, 6, 7, 8, 9])}
    res = {}
    rng = np.random.default_rng(0)
    for nm, (mask, cols) in sets.items():
        per = {}
        for mname in ("lgbm", "rf", "knn"):
            ok = np.zeros(len(F), bool); valid = np.zeros(len(F), bool)
            for f in range(3):
                tr = mask & (fold != f); te = mask & (fold == f)
                if te.sum() == 0: continue
                A, B = F[tr][:, cols], F[te][:, cols]
                mu, sd = A.mean(0), A.std(0) + 1e-9; A, B = (A - mu) / sd, (B - mu) / sd
                m = {"lgbm": lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=8, verbose=-1, random_state=0),
                     "rf": lambda: RandomForestClassifier(300, min_samples_leaf=1, n_jobs=8, random_state=0), "knn": lambda: KNeighborsClassifier(5)}[mname]().fit(A, sub[tr])
                ok[te] = (m.predict(B) == sub[te]); valid[te] = True
            # subject-cluster bootstrap of the block-level accuracy
            c = np.bincount(sub[valid], weights=ok[valid], minlength=51); n = np.bincount(sub[valid], minlength=51).astype(float)
            idx = rng.integers(0, 51, size=(2000, 51)); Wt = np.stack([np.bincount(r, minlength=51) for r in idx]).astype(float)
            bs = (Wt @ c) / np.maximum(Wt @ n, 1)
            per[mname] = dict(acc=float(c.sum() / n.sum()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], n_blocks=int(valid.sum()))
        best = max(per, key=lambda k: per[k]["acc"]); res[nm] = dict(per_attacker=per, best=best, best_acc=per[best]["acc"])
    res["chance"] = 1 / 51
    both = res["both_static_blocks"]["best_acc"]; rate = res["rate_cues_all_cells"]["best_acc"]; nb = res["noise_bias_cues_static_blocks"]["best_acc"]
    worst = max(both, rate, nb)
    res["gate_value_max_of_three"] = worst
    res["decision"] = "STOP (device fingerprint dominates)" if worst >= 0.50 else ("PROCEED flag=device-confounded" if worst >= 0.15 else "PROCEED clean")
    json.dump(res, open(os.path.join(out, "w0_device_gate.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a in ("best", "best_acc")}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["build", "w0"]); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build() if a.mode == "build" else w0(a.out)
