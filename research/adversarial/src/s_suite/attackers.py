"""CPU attacker pool for planted-leak data and tabular windows (sklearn + LightGBM). The ORDER is the declaration: the first 12 entries are
the pre-declared SUITE12 used by the certificate; the full list of 30 is the pool for the selection-bias study (K = 1..30).

Each attacker is retrained on the released representation (train split), standardised on train only. Two views: 'raw' (flattened window) and
'feat' (hand features: mean, std, mean abs difference, log band energies). No attacker sees val/test labels.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _lgbm(**kw):
    import lightgbm as lgb
    return lgb.LGBMClassifier(verbosity=-1, n_jobs=1, random_state=0, **kw)


def pool_spec():
    """Ordered list of (name, view, factory). Index < 12 is SUITE12."""
    P = []
    add = lambda n, v, f: P.append((n, v, f))  # noqa: E731
    # --- SUITE12 (declared first) ---
    add("lr_raw_c0.1", "raw", lambda: LogisticRegression(C=0.1, max_iter=300))
    add("lda_raw_auto", "raw", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    add("knn5_raw", "raw", lambda: KNeighborsClassifier(5))
    add("rf_raw_150", "raw", lambda: RandomForestClassifier(150, n_jobs=1, random_state=0))
    add("lr_feat_c1", "feat", lambda: LogisticRegression(C=1.0, max_iter=300))
    add("lda_feat_auto", "feat", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    add("knn5_feat", "feat", lambda: KNeighborsClassifier(5))
    add("rf_feat_300", "feat", lambda: RandomForestClassifier(300, n_jobs=1, random_state=0))
    add("et_feat_300", "feat", lambda: ExtraTreesClassifier(300, n_jobs=1, random_state=0))
    add("lgbm_feat", "feat", lambda: _lgbm(n_estimators=60, num_leaves=8, learning_rate=0.1))
    add("mlp_feat_64", "feat", lambda: MLPClassifier((64,), max_iter=200, random_state=0))
    add("knn1_raw", "raw", lambda: KNeighborsClassifier(1))
    # --- 18 more, pool for K up to 30 ---
    add("lr_raw_c0.01", "raw", lambda: LogisticRegression(C=0.01, max_iter=300))
    add("lr_raw_c1", "raw", lambda: LogisticRegression(C=1.0, max_iter=300))
    add("lr_feat_c0.01", "feat", lambda: LogisticRegression(C=0.01, max_iter=300))
    add("lr_feat_c0.1", "feat", lambda: LogisticRegression(C=0.1, max_iter=300))
    add("lr_feat_c10", "feat", lambda: LogisticRegression(C=10.0, max_iter=300))
    add("lda_raw_s0.3", "raw", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3))
    add("lda_feat_s0.3", "feat", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3))
    add("lda_feat_s0.7", "feat", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.7))
    add("knn3_raw", "raw", lambda: KNeighborsClassifier(3))
    add("knn15_raw", "raw", lambda: KNeighborsClassifier(15))
    add("knn1_feat", "feat", lambda: KNeighborsClassifier(1))
    add("knn15_feat", "feat", lambda: KNeighborsClassifier(15))
    add("rf_feat_d4", "feat", lambda: RandomForestClassifier(200, max_depth=4, n_jobs=1, random_state=1))
    add("rf_feat_d8", "feat", lambda: RandomForestClassifier(200, max_depth=8, n_jobs=1, random_state=2))
    add("et_raw_200", "raw", lambda: ExtraTreesClassifier(200, n_jobs=1, random_state=3))
    add("lgbm_feat_l4", "feat", lambda: _lgbm(n_estimators=40, num_leaves=4, learning_rate=0.15))
    add("mlp_feat_32", "feat", lambda: MLPClassifier((32,), max_iter=200, random_state=1))
    add("mlp_feat_128", "feat", lambda: MLPClassifier((128,), max_iter=200, random_state=2))
    assert len(P) == 30
    return P


POOL = pool_spec()
POOL_NAMES = [p[0] for p in POOL]
SUITE12 = tuple(POOL_NAMES[:12])


def fit_predict_pool(views_train: dict, y_train, views_eval: dict, names=None, n_classes=None):
    """views_*: {'raw': array, 'feat': array}; views_eval: {split: {'raw':..,'feat':..}}. -> {split: {attacker: probs (n, C)}}.
    Retrains every attacker on the train split of the released representation."""
    scalers = {v: StandardScaler().fit(views_train[v]) for v in views_train}
    Xtr = {v: scalers[v].transform(views_train[v]) for v in views_train}
    y_train = np.asarray(y_train)
    C = n_classes or int(y_train.max() + 1)
    out = {s: {} for s in views_eval}
    sel = set(names) if names is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, view, make in POOL:
            if sel is not None and name not in sel:
                continue
            clf = make().fit(Xtr[view], y_train)
            for s in views_eval:
                P = clf.predict_proba(scalers[view].transform(views_eval[s][view]))
                full = np.zeros((len(P), C), dtype=np.float32)
                full[:, np.asarray(clf.classes_, dtype=int)] = P
                out[s][name] = full
    return out
