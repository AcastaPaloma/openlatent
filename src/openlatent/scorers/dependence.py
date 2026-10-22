"""Attacker-free dependence statistics: linear CKA, distance correlation, and
normalized HSIC against an identity kernel. All are closed-form."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def _center_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0)


def _center_gram(k: np.ndarray) -> np.ndarray:
    return k - k.mean(axis=0) - k.mean(axis=1)[:, None] + k.mean()


def _alignment(kc: np.ndarray, lc: np.ndarray) -> float:
    denom = np.linalg.norm(kc) * np.linalg.norm(lc)
    return float(np.sum(kc * lc) / denom) if denom > 0 else 0.0


def linear_cka(x: np.ndarray, z: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between two representations of the same rows.

    1 means z is an isotropic rescaling/rotation of x; 0 means no linear relation.
    """
    x, z = _center_columns(x), _center_columns(z)
    n, dx, dz = len(x), x.shape[1], z.shape[1]
    if n * n <= dx * dz:  # Gram form is cheaper for wide features
        return _alignment(x @ x.T, z @ z.T)
    cross = np.linalg.norm(z.T @ x) ** 2
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(z.T @ z)
    return float(cross / denom) if denom > 0 else 0.0


def distance_correlation(x: np.ndarray, z: np.ndarray) -> float:
    """Distance correlation (Szekely et al., 2007), the statistic NoPeek minimizes.

    O(n^2) memory; subsample before calling on large sets.
    """
    a = _center_gram(cdist(x, x))
    b = _center_gram(cdist(z, z))
    dcov2 = max(np.mean(a * b), 0.0)
    dvar = np.sqrt(np.mean(a * a) * np.mean(b * b))
    return float(np.sqrt(dcov2 / dvar)) if dvar > 0 else 0.0


def rbf_gram(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel with the median pairwise distance as bandwidth."""
    d = cdist(x, x, "sqeuclidean")
    distances = pdist(x)
    positive = distances[distances > 0]
    if positive.size == 0:
        return np.ones_like(d)
    sigma2 = np.median(positive) ** 2
    return np.exp(-d / (2.0 * sigma2))


def unbiased_hsic(k: np.ndarray, l: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012). Needs n >= 4.

    The plain biased estimator rewards the matching diagonals of any two kernels,
    so random features score well above 0 at small n; this one is centred on 0.
    """
    n = len(k)
    k = k - np.diag(np.diag(k))
    l = l - np.diag(np.diag(l))
    ones_k, ones_l = k.sum(axis=0), l.sum(axis=0)
    trace = np.sum(k * l)
    return float(
        (trace + ones_k.sum() * ones_l.sum() / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * ones_k @ ones_l)
        / (n * (n - 3))
    )


def identity_alignment(z: np.ndarray, ids: np.ndarray, kernel: str = "rbf") -> float:
    """Normalized unbiased HSIC between a kernel on z and the same-identity kernel 1[id_i == id_j].

    About 0 when z carries no identity structure (can dip slightly below); near 1
    when z clusters perfectly by identity.
    """
    ids = np.asarray(ids)
    z = np.asarray(z, dtype=np.float64)
    k = rbf_gram(z) if kernel == "rbf" else z @ z.T
    l = (ids[:, None] == ids[None, :]).astype(np.float64)
    denom = unbiased_hsic(k, k) * unbiased_hsic(l, l)
    return unbiased_hsic(k, l) / np.sqrt(denom) if denom > 0 else 0.0
