"""Diagnostics (H.md 2.10 a, b, g): split inflation for identity, gender under PD's split vs subject-disjoint, static vs dynamic identity premise.
No defender is involved except the optional release file for the gender diagnostic.  CPU, <= 8 threads.
python diag.py ident_split|gender|static --out DIR [--release NPZ --name NAME]"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from suite import _fit_predict_tab
from sklearn.model_selection import GroupKFold


def acc(P, y): return float((P.argmax(1) == y).mean())


def boot_subject(correct_w, sub, B=2000, seed=0):
    """cluster bootstrap over subjects of a window-level accuracy"""
    rng = np.random.default_rng(seed); c = np.bincount(sub, weights=correct_w, minlength=24); n = np.bincount(sub, minlength=24).astype(float)
    idx = rng.integers(0, 24, size=(B, 24)); W = np.stack([np.bincount(r, minlength=24) for r in idx]).astype(float)
    b = (W @ c) / (W @ n); return float(c.sum() / n.sum()), [float(v) for v in np.percentile(b, [2.5, 97.5])]


def ident_split(a):
    trials = C.load_trials(); mean, std = C.pd_train_stats(trials)
    allw = C.build_windows(trials, mean, std, stride=10); X = allw["X"]; PHI = C.phi(X)
    sub, act, trial = allw["sub"], allw["act"], allw["trial"]
    j = (allw["start"] // 10)
    res = {}
    names = ["linear", "knn", "rf", "lgbm"]
    # T1: trial-held-out (fold O, PD split), training windows j%4==0, test windows = all stride-10 windows of the test trials
    te_o = np.array([(C.ACTS[x], int(t)) in C.FOLD_TEST["O"] for x, t in zip(act, trial)])
    tr_o = (~te_o) & (j % 4 == 0)
    for nm in names:
        P = _fit_predict_tab(nm, PHI[tr_o], sub[tr_o], PHI[te_o], 24, 0, 8)
        res.setdefault("T1_trial_heldout_foldO", {})[nm] = boot_subject((P.argmax(1) == sub[te_o]).astype(float), sub[te_o])
        # trial-level
        tk = C.trial_key(dict(sub=sub[te_o], act=act[te_o], trial=trial[te_o])); u = np.unique(tk)
        ok = [(np.mean(P[tk == k], 0).argmax() == k // 1000) for k in u]
        res["T1_trial_heldout_foldO"][nm + "_trial_level"] = float(np.mean(ok))
    # T2: random window split (80/20) of ALL windows; stride 10 (max overlap) and stride 40
    rng = np.random.default_rng(0)
    for stride_tag, keep in [("stride10", np.ones(len(X), bool)), ("stride40", j % 4 == 0)]:
        idx = np.where(keep)[0]; perm = rng.permutation(idx); ntr = int(0.8 * len(perm)); tr, te = perm[:ntr], perm[ntr:]
        for nm in names:
            P = _fit_predict_tab(nm, PHI[tr], sub[tr], PHI[te], 24, 0, 8)
            res.setdefault(f"T2_window_random_{stride_tag}", {})[nm] = boot_subject((P.argmax(1) == sub[te]).astype(float), sub[te])
    res["n_windows_all_stride10"] = int(len(X))
    json.dump(res, open(os.path.join(a.out, "diag_ident_split.json"), "w"), indent=1); print(json.dumps(res, indent=1))


def gender(a):
    trials = C.load_trials(); mean, std = C.pd_train_stats(trials)
    S = C.analysis_set(); N = len(S["X"])
    if a.release:
        d = np.load(a.release); R = np.zeros((N, 2, 128), np.float32); R[d["idx"]] = d["release"]
    else:
        R = S["X"]
    PHI = C.phi(R); sub = S["sub"]; g = C.gender_by_subject()[sub]
    res = dict(name=a.name, majority_window=float(max(g.mean(), 1 - g.mean())))
    names = ["linear", "knn", "rf", "lgbm"]
    te_o = np.array([(C.ACTS[x], int(t)) in C.FOLD_TEST["O"] for x, t in zip(S["act"], S["trial"])]); tr_o = (~te_o) & S["is_train_window"]
    from sklearn.metrics import balanced_accuracy_score
    for nm in names:
        P = _fit_predict_tab(nm, PHI[tr_o], g[tr_o], PHI[te_o], 2, 0, 8); p = P.argmax(1)
        res.setdefault("window_level_PD_split_all_users_in_train_and_test", {})[nm] = dict(acc=acc(P, g[te_o]), bacc=float(balanced_accuracy_score(g[te_o], p)))
    # subject-disjoint 6 folds x 4 subjects (hash-free fixed grouping: subject index mod 6), all trials of the held-out subjects are test
    fold = (sub % 6)
    for nm in names:
        pred = np.zeros(N, int); ok = np.zeros(N, bool)
        for f in range(6):
            te = (fold == f); tr = (~te) & S["is_train_window"]
            P = _fit_predict_tab(nm, PHI[tr], g[tr], PHI[te], 2, 0, 8); pred[te] = P.argmax(1); ok[te] = True
        res.setdefault("subject_disjoint_6fold", {})[nm] = dict(acc=float((pred[ok] == g[ok]).mean()), bacc=float(balanced_accuracy_score(g[ok], pred[ok])))
    # subject-level (24 units): majority vote of window predictions per subject
    json.dump(res, open(os.path.join(a.out, f"diag_gender_{a.name}.json"), "w"), indent=1); print(json.dumps(res, indent=1))


def static(a):
    """F7 replication: static (sit 5/13, std 6/14) vs dynamic identity after de-meaning, RF, trial-held-out (last trial)."""
    import pandas as pd
    def series(act, t, s):
        df = pd.read_csv(os.path.join(C.DATA_ROOT, "A_DeviceMotion_data", f"{act}_{t}", f"sub_{s}.csv"))
        rot = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].values; acc_ = df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].values + df[["gravity.x", "gravity.y", "gravity.z"]].values
        return np.stack([np.linalg.norm(rot, axis=1), np.linalg.norm(acc_, axis=1)], 1)
    trials = C.load_trials(); mean, std = C.pd_train_stats(trials)
    def feats(arr, demean_only):
        z = ((arr - mean) / std).T; W = []
        for i in range(0, z.shape[1] - 128, 40): W.append(z[:, i:i + 128])
        X = np.asarray(W, np.float32); F = C.phi(X)
        return F[:, 2:] if demean_only else F  # drop the per-channel means (first 2 columns)
    out = {}
    for stratum, spec in [("static", [("sit", [5, 13], 13), ("std", [6, 14], 14)]), ("dynamic", [("dws", [1, 2, 11], 11), ("ups", [3, 4, 12], 12), ("wlk", [7, 8, 15], 15), ("jog", [9, 16], 16)])]:
        Xtr, ytr, Xte, yte, tkey = [], [], [], [], []
        for act, ts, test_t in spec:
            for t in ts:
                for s in range(24):
                    arr = series(act, t, s + 1) if stratum == "static" else trials[(s, C.ACTS.index(act), t)]
                    F = feats(arr, True)
                    if t == test_t: Xte.append(F); yte += [s] * len(F); tkey += [(s, act)] * len(F)
                    else: Xtr.append(F); ytr += [s] * len(F)
        Xtr, Xte = np.concatenate(Xtr), np.concatenate(Xte); ytr, yte = np.array(ytr), np.array(yte)
        res = {}
        for nm in ["rf", "lgbm"]:
            P = _fit_predict_tab(nm, Xtr, ytr, Xte, 24, 0, 8)
            keys = {}
            for p, y, k in zip(P, yte, tkey): keys.setdefault(k, []).append((p, y))
            ok = [float(np.mean([x[0] for x in v], 0).argmax() == v[0][1]) for v in keys.values()]
            sub_ = np.array([k[0] for k in keys]); cw = np.array(ok)
            m, ci = boot_subject_trials(cw, sub_)
            res[nm] = dict(trial_acc=m, ci=ci, n_trials=len(ok), window_acc=acc(P, yte))
        out[stratum] = res
    json.dump(out, open(os.path.join(a.out, "diag_static.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def boot_subject_trials(correct, sub, B=2000, seed=0):
    rng = np.random.default_rng(seed); c = np.bincount(sub, weights=correct, minlength=24); n = np.bincount(sub, minlength=24).astype(float)
    idx = rng.integers(0, 24, size=(B, 24)); W = np.stack([np.bincount(r, minlength=24) for r in idx]).astype(float)
    b = (W @ c) / (W @ n); return float(c.sum() / n.sum()), [float(v) for v in np.percentile(b, [2.5, 97.5])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["ident_split", "gender", "static"]); ap.add_argument("--out", required=True)
    ap.add_argument("--release", default=None); ap.add_argument("--name", default="raw")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"ident_split": ident_split, "gender": gender, "static": static}[a.mode](a)
