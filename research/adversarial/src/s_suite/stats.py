"""Cluster-aware statistics for the S instrument (CPU only, numpy/scipy).

Everything here works on *independent units* (recordings, subjects, patients). Items (clips, windows, images) that belong to the same
unit are resampled together. Randomness comes from numpy's legacy RandomState, whose streams are frozen across numpy versions, so a
certificate computed with the same seed and inputs is reproducible bit-for-bit up to BLAS summation order (outputs are rounded).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, norm

Z_MDE = 2.8  # z_{0.975} + z_{0.80} = 2.80; programme 1's convention: MDE80 = 2.8 x SE of the (paired) difference


def unit_index(cluster):
    """cluster ids (any dtype) -> (inv, m): integer unit index per item and number of units."""
    cluster = np.asarray(cluster).astype(str)
    uniq, inv = np.unique(cluster, return_inverse=True)
    return inv.astype(np.int64), int(len(uniq))


def unit_sums(values, inv, m):
    """Sum item-level `values` (n,) or (n,K) into units -> (m,) or (m,K)."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.bincount(inv, weights=values, minlength=m)
    return np.stack([np.bincount(inv, weights=values[:, k], minlength=m) for k in range(values.shape[1])], axis=1)


def boot_weights(m, n_boot, seed):
    """(B, m) matrix of unit resampling counts (multinomial; each row sums to m). Deterministic in `seed`."""
    rs = np.random.RandomState(int(seed) % (2 ** 32))
    return rs.multinomial(m, np.full(m, 1.0 / m), size=int(n_boot)).astype(float)


def ratio_boot(S, n_c, W):
    """Bootstrap of the ratio estimator sum_c w_c S_c / sum_c w_c n_c for each column of S. -> (B, K)."""
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    return (W @ S) / (W @ n_c)[:, None]


def ratio_point(S, n_c):
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    return S.sum(0) / n_c.sum()


# ---------------------------------------------------------------------------------------------------------------------------
# AUC with unit-level resampling (binary labels; score = larger means more positive)

def _weighted_auc(y, score_rank_groups, n_groups, w):
    """AUC with sample weights w; groups index ties (score_rank_groups is the dense rank of the score, 0..n_groups-1)."""
    wp = np.bincount(score_rank_groups, weights=w * (y == 1), minlength=n_groups)
    wn = np.bincount(score_rank_groups, weights=w * (y == 0), minlength=n_groups)
    P, N = wp.sum(), wn.sum()
    if P <= 0 or N <= 0:
        return np.nan
    cum_n = np.cumsum(wn) - wn
    return float((wp * (cum_n + 0.5 * wn)).sum() / (P * N))


def auc_point(y, score, w=None):
    y = np.asarray(y)
    _, g = np.unique(score, return_inverse=True)
    w = np.ones(len(y)) if w is None else w
    return _weighted_auc(y, g, g.max() + 1, w)


def auc_boot(y, score, inv, W):
    """-> (B,) AUC values with units resampled by the rows of W (B, m)."""
    y = np.asarray(y)
    _, g = np.unique(score, return_inverse=True)
    ng = g.max() + 1
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        out[b] = _weighted_auc(y, g, ng, W[b][inv])
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# bounds

def cp_upper(k, n, alpha):
    """One-sided Clopper-Pearson upper bound at level 1-alpha for k successes of n (k and n may be non-integer)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1.0, n - k))


def eff_counts(p, se, n_items, n_units=None):
    """Effective binomial counts matching a cluster-bootstrap SE: n_eff = p(1-p)/se^2, clipped to [1, n_items]."""
    p = float(p)
    if not np.isfinite(se) or se <= 0 or p <= 0 or p >= 1:
        n_eff = float(n_units if n_units else n_items)
    else:
        n_eff = p * (1 - p) / se ** 2
    n_eff = float(min(max(n_eff, 1.0), n_items))
    return p * n_eff, n_eff


def ucb_effective(p, se, n_items, alpha, n_units=None):
    k, n = eff_counts(p, se, n_items, n_units)
    return cp_upper(k, n, alpha)


def zero_leak_floor(chance, n_units, alpha, K=1):
    """Smallest upper bound the design can deliver: observed rate == chance on `n_units` independent units, Bonferroni over K.
    (Programme 1 stress test: 150 units, chance 0.04 -> 0.077 at K=1, 0.111 at K=30.)"""
    k = int(round(chance * n_units))
    return cp_upper(k, n_units, alpha / max(K, 1))


def mde80(se):
    return Z_MDE * float(se)


def design_effect(se, p, n_items):
    base = p * (1 - p) / n_items
    return float(se ** 2 / base) if base > 0 else float("nan")


def r6(x, nd=6):
    """Round for stable serialisation (recursive for dict/list)."""
    if isinstance(x, dict):
        return {k: r6(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def percentile_ci(draws, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def onesided_z(alpha):
    return float(norm.ppf(1 - alpha))
