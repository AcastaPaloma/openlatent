"""Selection bias of the maximum over K attackers.

Two designs.

(1) `split_inflation` -- saved predictions only, no truth. Units are split at random into a SELECT half A and an EVAL half B (whole units on one
    side). For K attackers drawn from the pool:
        naive_B   = max_k acc_B(k)                      best-of-K reported on the evaluation units themselves
        valsel_B  = acc_B(argmax_k acc_A(k))            validation-selected attacker, scored on the units it was not selected on
        naive_A   = max_k acc_A(k)                      best-of-K on its own selection units
    inflation      = naive_B  - valsel_B   (what "report the best of K on the test set" adds over the validation-selected number)
    optimism       = naive_A  - valsel_B   (winner's curse of the maximum on the units that selected it)
    Both are averages over random splits (and random K-subsets when the pool is larger than K). Intervals: (a) the 95% range across splits
    and (b) a bootstrap over units (whole units resampled with multiplicities; all copies of a unit stay on one side of the split).

(2) `truth_inflation` -- planted-leak data, where the TRUE accuracy of each fitted attacker is known from a large fresh sample. Val and test
    are independent samples. Reports the winner's-curse bias of the best-of-K on the test, the bias of the validation-selected value
    (should be ~0), and the selection loss (true best minus true score of the selected attacker).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import stats as st


def _subset_cols(rs: np.random.RandomState, R: int, P: int, K: int):
    if K >= P:
        return np.tile(np.arange(P), (R, 1))
    return np.argsort(rs.rand(R, P), axis=1)[:, :K]


def _one_pass(S, n_c, w, K, R, rs):
    """One bootstrap-weight vector w (m,), R random splits. -> arrays (R,) of naive_B, valsel_B, naive_A."""
    m, P = S.shape
    present = np.where(w > 0)[0]
    nA = len(present) // 2
    # random half of the distinct present units
    order = np.argsort(rs.rand(R, len(present)), axis=1)
    mask = np.zeros((R, m))
    rows = np.arange(R)[:, None]
    mask[rows, present[order[:, :nA]]] = 1.0
    wa = mask * w
    wb = (1.0 - mask) * w
    wb[:, np.setdiff1d(np.arange(m), present)] = 0.0
    thA = (wa @ S) / np.maximum((wa @ n_c)[:, None], 1e-12)
    thB = (wb @ S) / np.maximum((wb @ n_c)[:, None], 1e-12)
    cols = _subset_cols(rs, R, P, K)
    a = np.take_along_axis(thA, cols, 1)
    b = np.take_along_axis(thB, cols, 1)
    kA = np.argmax(a, axis=1)
    valsel = b[np.arange(R), kA]
    return b.max(1), valsel, a.max(1)


def split_inflation(correct: np.ndarray, cluster, K_list: List[int], n_splits: int = 400, n_outer: int = 200, inner_per_outer: int = 20,
                    seed: int = 0) -> dict:
    """correct: (n, P) 0/1 correctness of P attackers on the same items. -> per K: mean inflation / optimism with intervals."""
    correct = np.asarray(correct, dtype=float)
    inv, m = st.unit_index(cluster)
    S = st.unit_sums(correct, inv, m)
    n_c = np.bincount(inv, minlength=m).astype(float)
    P = correct.shape[1]
    rs = np.random.RandomState(seed)
    out = {"P": P, "n_units": m, "K": {}}
    w1 = np.ones(m)
    for K in K_list:
        if K > P:
            continue
        nb, vs, na = _one_pass(S, n_c, w1, K, n_splits, np.random.RandomState(seed * 7919 + K))
        infl, opt = nb - vs, na - vs
        # outer bootstrap over units
        means_i, means_o = [], []
        rso = np.random.RandomState(seed * 104729 + K)
        for _ in range(n_outer):
            w = rso.multinomial(m, np.full(m, 1.0 / m)).astype(float)
            nb_, vs_, na_ = _one_pass(S, n_c, w, K, inner_per_outer, rso)
            means_i.append(float((nb_ - vs_).mean()))
            means_o.append(float((na_ - vs_).mean()))
        out["K"][K] = dict(inflation=float(infl.mean()), inflation_split_range95=[float(np.quantile(infl, 0.025)), float(np.quantile(infl, 0.975))],
                           inflation_boot_ci95=[float(np.quantile(means_i, 0.025)), float(np.quantile(means_i, 0.975))],
                           optimism=float(opt.mean()), optimism_split_range95=[float(np.quantile(opt, 0.025)), float(np.quantile(opt, 0.975))],
                           optimism_boot_ci95=[float(np.quantile(means_o, 0.025)), float(np.quantile(means_o, 0.975))],
                           valsel_mean=float(vs.mean()), naive_B_mean=float(nb.mean()))
    return out


def truth_inflation_rep(c_val: np.ndarray, u_val, c_test: np.ndarray, u_test, true_acc: np.ndarray, K_list: List[int], n_subsets: int,
                        rs: np.random.RandomState) -> dict:
    """One planted replicate. c_*: (n, P) correctness of the P fitted attackers; true_acc: (P,) population accuracy.
    For each K, `n_subsets` random K-subsets of the pool. -> per K means of the quantities in the module docstring."""
    c_val = np.asarray(c_val, dtype=float)
    c_test = np.asarray(c_test, dtype=float)
    P = c_val.shape[1]
    th_v = c_val.mean(0)
    th_t = c_test.mean(0)
    res = {}
    for K in K_list:
        if K > P:
            continue
        cols = _subset_cols(rs, n_subsets, P, K)
        tv, tt, ta = th_v[cols], th_t[cols], true_acc[cols]
        kV = tv.argmax(1)
        kT = tt.argmax(1)
        r = np.arange(n_subsets)
        naive = tt.max(1)
        valsel = tt[r, kV]
        res[K] = dict(
            naive_test=float(naive.mean()), valsel_test=float(valsel.mean()), inflation=float((naive - valsel).mean()),
            bias_naive=float((naive - ta[r, kT]).mean()),                  # winner's curse of best-of-K on the test
            bias_valsel=float((valsel - ta[r, kV]).mean()),               # honest estimate of the selected attacker (~0)
            true_best=float(ta.max(1).mean()), true_selected=float(ta[r, kV].mean()), selection_loss=float((ta.max(1) - ta[r, kV]).mean()),
            true_of_naive_pick=float(ta[r, kT].mean()))
    return res


def aggregate_truth(reps: List[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean over replicates (independent datasets) with percentile bootstrap CIs, per K and quantity."""
    rs = np.random.RandomState(seed)
    Ks = sorted(set.intersection(*[set(r) for r in reps]))
    out = {}
    for K in Ks:
        keys = reps[0][K].keys()
        M = {k: np.array([r[K][k] for r in reps]) for k in keys}
        idx = rs.randint(0, len(reps), (n_boot, len(reps)))
        out[K] = {k: dict(mean=float(v.mean()), ci95=[float(np.quantile(v[idx].mean(1), 0.025)), float(np.quantile(v[idx].mean(1), 0.975))])
                  for k, v in M.items()}
    return out
