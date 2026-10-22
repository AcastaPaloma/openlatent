"""Convex probes: L2-regularized logistic regression and ridge regression.

Both objectives are strictly convex, so for a given regularization strength the
probe is the unique optimum, independent of initialization or seeds. The strength
is picked from a fixed grid by patient-grouped K-fold CV on the fit set (folds are
deterministic), so a probe cannot under-report leakage just because it overfit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class ProbeConfig:
    # Both grids run from most to least regularized; CV ties keep the first.
    Cs: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    alphas: tuple[float, ...] = (1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 1e-1)
    cv_folds: int = 3
    max_iter: int = 5000
    tol: float = 1e-6


def standardize(fit: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both sets with the fit set's statistics; constant columns stay zero."""
    fit = np.asarray(fit, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std == 0] = 1.0
    return (fit - mean) / std, (test - mean) / std


def _logistic(C: float, cfg: ProbeConfig) -> LogisticRegression:
    return LogisticRegression(C=C, solver="lbfgs", max_iter=cfg.max_iter, tol=cfg.tol)


def _fit(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(x, y)


def _auc(model, x, y) -> float | None:
    return float(roc_auc_score(y, model.decision_function(x))) if len(np.unique(y)) == 2 else None


def _r2(model, x, y) -> float:
    return float(r2_score(y, model.predict(x)))


def _select(grid, make, score: Callable, x, y, groups, folds: int):
    """Best grid value by mean CV score over valid folds; ties go to the earliest
    (most regularized) value. Returns (value, cv_score)."""
    splits = list(GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(x, y, groups))
    best, best_score = grid[0], -np.inf
    for value in grid:
        scores = []
        for tr, va in splits:
            if len(np.unique(y[tr])) < 2:
                continue
            s = score(_fit(make(value), x[tr], y[tr]), x[va], y[va])
            if s is not None:
                scores.append(s)
        if scores and np.mean(scores) > best_score:
            best, best_score = value, float(np.mean(scores))
    return best, best_score


def _converged(model, cfg: ProbeConfig) -> bool:
    return bool(np.all(model.n_iter_ < cfg.max_iter))


def binary_probe(x_fit, y_fit, groups_fit, x_test, y_test, cfg: ProbeConfig = ProbeConfig()) -> dict:
    """ROC AUC of a logistic probe for a binary attribute (chance 0.5)."""
    y_fit, y_test = np.asarray(y_fit), np.asarray(y_test)
    result = {"chance": 0.5, "n_fit": len(y_fit), "n_test": len(y_test)}
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_test)) < 2:
        return {**result, "auc": None, "C": None, "converged": None}
    x_fit, x_test = standardize(x_fit, x_test)
    C, _ = _select(cfg.Cs, lambda c: _logistic(c, cfg), _auc, x_fit, y_fit, np.asarray(groups_fit), cfg.cv_folds)
    model = _fit(_logistic(C, cfg), x_fit, y_fit)
    return {**result, "auc": _auc(model, x_test, y_test), "C": C, "converged": _converged(model, cfg)}


def multilabel_probe(x_fit, y_fit, groups_fit, x_test, y_test, names, cfg: ProbeConfig = ProbeConfig()) -> dict:
    """One logistic probe per label; mean AUC over labels with both classes in fit and test."""
    x_fit, x_test = standardize(x_fit, x_test)
    y_fit, y_test, groups_fit = np.asarray(y_fit), np.asarray(y_test), np.asarray(groups_fit)
    per_label, chosen, all_converged = {}, {}, True
    for j, name in enumerate(names):
        if len(np.unique(y_fit[:, j])) < 2 or len(np.unique(y_test[:, j])) < 2:
            per_label[name] = chosen[name] = None
            continue
        C, _ = _select(cfg.Cs, lambda c: _logistic(c, cfg), _auc, x_fit, y_fit[:, j], groups_fit, cfg.cv_folds)
        model = _fit(_logistic(C, cfg), x_fit, y_fit[:, j])
        all_converged &= _converged(model, cfg)
        per_label[name], chosen[name] = _auc(model, x_test, y_test[:, j]), C
    scored = [v for v in per_label.values() if v is not None]
    return {
        "mean_auc": float(np.mean(scored)) if scored else None,
        "chance": 0.5,
        "per_label": per_label,
        "C": chosen,
        "converged": all_converged,
        "n_fit": len(y_fit),
        "n_test": len(y_test),
    }


def ridge_probe(x_fit, y_fit, groups_fit, x_test, y_test, cfg: ProbeConfig = ProbeConfig()) -> dict:
    """Closed-form ridge regression for a continuous attribute such as age (chance R^2 = 0)."""
    x_fit, x_test = standardize(x_fit, x_test)
    y_fit, y_test = np.asarray(y_fit, dtype=np.float64), np.asarray(y_test, dtype=np.float64)
    make = lambda a: Ridge(alpha=a, solver="cholesky")  # noqa: E731
    alpha, _ = _select(cfg.alphas, make, _r2, x_fit, y_fit, np.asarray(groups_fit), cfg.cv_folds)
    pred = make(alpha).fit(x_fit, y_fit).predict(x_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "alpha": alpha,
        "chance": 0.0,
        "n_fit": len(y_fit),
        "n_test": len(y_test),
    }
