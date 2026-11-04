"""Suite v3 (evaluator v3): the 12 attackers of v1/v2 plus three view groups, declared and frozen.

Order is the declaration. base = SUITE12 (views raw, feat; defined in attackers.py, unchanged). New members, each retrained on the train split of the
release, standardised on train only:
  hb     lr_hb_c0.05, lda_hb_auto, rf_hb_200, knn5_hb                         (full-resolution log spectrum + spectral shape)
  phase  lr_phase_c0.05, lda_phase_auto, rf_phase_200, knn5_phase            (cross-channel phase lag; absent when C < 2)
  inter  lr_inter_c0.05, lgbm_lvlspec, rf_lvlspec_200, mlp_lvlspec_64         (amplitude-frequency interaction)
K = 24 (C >= 2) or 20 (C = 1). The Bonferroni cost of the extra members is printed by the certificate (resolution floor).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .attackers import SUITE12, _LGBM, fit_predict_pool
from .views import ViewBuilder

GROUPS = {
    "hb": ("lr_hb_c0.05", "lda_hb_auto", "rf_hb_200", "knn5_hb"),
    "phase": ("lr_phase_c0.05", "lda_phase_auto", "rf_phase_200", "knn5_phase"),
    "inter": ("lr_inter_c0.05", "lgbm_lvlspec", "rf_lvlspec_200", "mlp_lvlspec_64"),
}


def _members():
    return [
        ("lr_hb_c0.05", "hb", lambda: LogisticRegression(C=0.05, max_iter=300)),
        ("lda_hb_auto", "hb", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("rf_hb_200", "hb", lambda: RandomForestClassifier(200, n_jobs=1, random_state=0)),
        ("knn5_hb", "hb", lambda: KNeighborsClassifier(5, algorithm="brute")),
        ("lr_phase_c0.05", "phase", lambda: LogisticRegression(C=0.05, max_iter=300)),
        ("lda_phase_auto", "phase", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("rf_phase_200", "phase", lambda: RandomForestClassifier(200, n_jobs=1, random_state=0)),
        ("knn5_phase", "phase", lambda: KNeighborsClassifier(5, algorithm="brute")),
        ("lr_inter_c0.05", "inter", lambda: LogisticRegression(C=0.05, max_iter=300)),
        ("lgbm_lvlspec", "lvlspec", lambda: _LGBM(n_estimators=60, num_leaves=8, learning_rate=0.1)),
        ("rf_lvlspec_200", "lvlspec", lambda: RandomForestClassifier(200, n_jobs=1, random_state=0)),
        ("mlp_lvlspec_64", "lvlspec", lambda: MLPClassifier((64,), max_iter=200, random_state=0)),
    ]


MEMBERS = _members()


def suite_v3(n_channels: int) -> tuple:
    """The declared attacker names for windows with `n_channels` channels (phase members only when n_channels >= 2)."""
    new = [n for n, v, _ in MEMBERS if not (v == "phase" and n_channels < 2)]
    return tuple(SUITE12) + tuple(new)


def active_groups(n_channels: int) -> tuple:
    return tuple(g for g in GROUPS if not (g == "phase" and n_channels < 2))


def fit_predict_v3(X_train: np.ndarray, y_train, X_eval: dict, n_classes: int):
    """Windows (n, T, C) -> {split: {attacker: probs (n, n_classes)}} for the declared suite. Retrains every member on the train split."""
    C = X_train.shape[2]
    vb = ViewBuilder().fit(X_train)
    vtr = vb.transform(X_train)
    vev = {s: vb.transform(x) for s, x in X_eval.items()}
    out = fit_predict_pool({k: vtr[k] for k in ("raw", "feat")}, y_train, {s: {k: v[k] for k in ("raw", "feat")} for s, v in vev.items()},
                           names=SUITE12, n_classes=n_classes)
    y_train = np.asarray(y_train)
    scalers = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, view, make in MEMBERS:
            if view == "phase" and C < 2:
                continue
            if view not in scalers:
                scalers[view] = StandardScaler().fit(vtr[view])
            clf = make().fit(scalers[view].transform(vtr[view]), y_train)
            cols = np.asarray(clf.classes_, dtype=int)
            for s in vev:
                P = clf.predict_proba(scalers[view].transform(vev[s][view]))
                full = np.zeros((len(P), n_classes), dtype=np.float32)
                full[:, cols] = P
                out[s][name] = full
    return out


def declaration(n_classes: int, n_channels: int, name: str = "suite-v3", margin: float = 0.02, seed: int = 0, n_boot: int = 2000):
    """The frozen v3 declaration for windows with `n_channels` channels: attackers, view groups read, canary groups required for assurance."""
    from . import certificate as cf
    groups = active_groups(n_channels)
    return cf.SuiteDecl(name, suite_v3(n_channels), "accuracy", 0.05, n_boot, seed, margin, n_classes,
                        views=("raw", "feat") + tuple(groups), canary_groups=tuple(groups))
