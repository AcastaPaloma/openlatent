"""F3 frozen-feature heads, F5 demographic attacker, F6 patient-ID metric learner (HELD OUT).  Operate on feature matrices / image tensors.

Pair features for F3: [|fa - fb|, fa * fb] on standardised frozen features (CLS + mean patch token of DINOv2-B/14 at 224 px).
F3a logistic regression (C chosen on validation), F3b MLP 2x256, F3c retrieval scores with no training: -cosine, -L2, -Canberra distance
(Canberra: Bajar et al. 2610.20494 found it beats cosine on every encoder tested; cosine alone would be a weak deterministic scorer).
F5 demographic: ridge age + logistic sex + logistic view fitted on defended TRAIN images with patient-grouped OUT-OF-FOLD predictions (so the pair
model never sees in-sample estimates), then a gradient-boosted classifier on [age_hat_a, age_hat_b, |diff|, sex probs, view probs].
"""
from __future__ import annotations

import numpy as np


def std_fit(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def pair_matrix(F, ia, ib):
    return np.concatenate([np.abs(F[ia] - F[ib]), F[ia] * F[ib]], axis=1)


def retrieval_scores(F, ia, ib, metric: str) -> np.ndarray:
    a, b = F[ia].astype(np.float64), F[ib].astype(np.float64)
    if metric == "cosine":
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    if metric == "l2":
        return -np.linalg.norm(a - b, axis=1)
    if metric == "canberra":
        return -(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)).sum(1)
    raise ValueError(metric)


def fit_f3(Ftr, tr, Fva, va, Fte, te, seed=0, stats_from=None) -> dict:
    """tr/va/te = dict(ia, ib, y).  Returns {name: (val_score, test_score)} for the learned heads and the no-training retrieval scores.
    `stats_from` = feature rows used for the standardisation statistics (the A-role images only: no V / T image may inform anything)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mu, sd = std_fit(Ftr if stats_from is None else stats_from)
    S = lambda F: (F - mu) / sd
    Ftr, Fva, Fte = S(Ftr), S(Fva), S(Fte)
    Xtr, Xva, Xte = pair_matrix(Ftr, tr["ia"], tr["ib"]), pair_matrix(Fva, va["ia"], va["ib"]), pair_matrix(Fte, te["ia"], te["ib"])
    out = {}
    best = (-1, None)
    for C in (1e-3, 1e-2, 1e-1):
        m = LogisticRegression(C=C, max_iter=300).fit(Xtr, tr["y"])
        a = roc_auc_score(va["y"], m.decision_function(Xva))
        if a > best[0]:
            best = (a, m)
    out["f3a_logreg"] = (best[1].decision_function(Xva), best[1].decision_function(Xte))
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier((256, 256), alpha=1e-2, early_stopping=True, validation_fraction=0.1, max_iter=60, random_state=seed).fit(Xtr, tr["y"])
    out["f3b_mlp"] = (mlp.predict_proba(Xva)[:, 1], mlp.predict_proba(Xte)[:, 1])
    for metric in ("cosine", "l2", "canberra"):
        out[f"f3c_{metric}"] = (retrieval_scores(Fva, va["ia"], va["ib"], metric), retrieval_scores(Fte, te["ia"], te["ib"], metric))
    return out


def _logit(Xa, ya, Xb, C=0.01):
    """P(y=1) from a logistic fit; a constant when the training labels contain one class only (robust for tiny folds / synthetic tests)."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(ya)) < 2:
        return np.full(len(Xb), float(ya[0]))
    return LogisticRegression(C=C, max_iter=300).fit(Xa, ya).predict_proba(Xb)[:, 1]


def oof_demographics(F, pid, age, male, pa, folds: int = 4, seed: int = 0):
    """patient-grouped out-of-fold age (ridge), sex and view (logistic) estimates for every image of the TRAIN split."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    n = len(F)
    ah, sh, vh = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in GroupKFold(folds).split(F, groups=pid):
        mu, sd = std_fit(F[tr])
        A, B = (F[tr] - mu) / sd, (F[te] - mu) / sd
        ah[te] = Ridge(alpha=100.0).fit(A, age[tr]).predict(B)
        sh[te] = _logit(A, male[tr], B)
        vh[te] = _logit(A, pa[tr], B)
    return ah, sh, vh


def demo_pair_features(ah, sh, vh, ia, ib):
    return np.c_[ah[ia], ah[ib], np.abs(ah[ia] - ah[ib]), ah[ib] - ah[ia], sh[ia], sh[ib], np.abs(sh[ia] - sh[ib]), vh[ia], vh[ib], np.abs(vh[ia] - vh[ib])]


def fit_f5(F, dem, mask, tr, va, te, seed=0, true_rows=None) -> tuple[np.ndarray, np.ndarray]:
    """F = features of ALL image rows (defended-A table stacked on the real-B table); dem = dict(pid, age, male, pa) per row; mask = rows of A-role images (the
    only rows the regressors may be fitted on).  Estimates for masked rows are patient-grouped OUT-OF-FOLD, for all other rows they come from regressors fitted on
    the masked rows only (so no V / T image or label ever informs the demographic models).  Pair model: gradient boosting on A pairs.  Returns (val, test) scores."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    Fm = F[mask]
    oa, os_, ov = oof_demographics(Fm, dem["pid"][mask], dem["age"][mask], dem["male"][mask], dem["pa"][mask], seed=seed)
    mu, sd = std_fit(Fm)
    Z = (F - mu) / sd
    A = Z[mask]
    ah, sh, vh = Ridge(alpha=100.0).fit(A, dem["age"][mask]).predict(Z), _logit(A, dem["male"][mask], Z), _logit(A, dem["pa"][mask], Z)
    ah[mask], sh[mask], vh[mask] = oa, os_, ov
    if true_rows is not None:
        ah[true_rows], sh[true_rows], vh[true_rows] = dem["age"][true_rows], dem["male"][true_rows].astype(float), dem["pa"][true_rows].astype(float)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=seed).fit(demo_pair_features(ah, sh, vh, tr["ia"], tr["ib"]), tr["y"])
    return m.predict_proba(demo_pair_features(ah, sh, vh, va["ia"], va["ib"]))[:, 1], m.predict_proba(demo_pair_features(ah, sh, vh, te["ia"], te["ib"]))[:, 1]


def fit_f3_xl(Ftr, tr, Fva, va, Fte, te, seed=0, stats_from=None) -> dict:
    """T.md section 12, labels14 R5: a larger MLP head (512, 256, up to 200 iterations, early stopping) on [|fa - fb|, fa * fb]."""
    from sklearn.neural_network import MLPClassifier
    mu, sd = std_fit(Ftr if stats_from is None else stats_from)
    S_ = lambda F: (F - mu) / sd
    Ftr, Fva, Fte = S_(Ftr), S_(Fva), S_(Fte)
    Xtr, Xva, Xte = pair_matrix(Ftr, tr["ia"], tr["ib"]), pair_matrix(Fva, va["ia"], va["ib"]), pair_matrix(Fte, te["ia"], te["ib"])
    m = MLPClassifier((512, 256), alpha=1e-3, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, max_iter=200, random_state=seed).fit(Xtr, tr["y"])
    return {"f3x_mlp": (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])}
