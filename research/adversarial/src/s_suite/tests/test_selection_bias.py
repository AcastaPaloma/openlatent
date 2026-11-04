import numpy as np

from s_suite import selection_bias as sb


def equal_attackers(rs, n_units, P, acc, w=2):
    n = n_units * w
    cl = np.repeat(np.arange(n_units), w)
    C = (rs.rand(n, P) < acc).astype(float)
    return C, cl


def test_k1_has_no_inflation():
    C, cl = equal_attackers(np.random.RandomState(0), 150, 6, 0.1)
    r = sb.split_inflation(C, cl, [1], n_splits=200, n_outer=20, inner_per_outer=5)
    assert abs(r["K"][1]["inflation"]) < 1e-12 and abs(r["K"][1]["optimism"]) < 0.02 * 0 + 0.02


def test_inflation_grows_with_k_when_attackers_are_equal():
    C, cl = equal_attackers(np.random.RandomState(1), 150, 30, 0.1)
    r = sb.split_inflation(C, cl, [2, 5, 10, 30], n_splits=400, n_outer=30, inner_per_outer=10)["K"]
    infl = [r[k]["inflation"] for k in (2, 5, 10, 30)]
    opt = [r[k]["optimism"] for k in (2, 5, 10, 30)]
    assert infl[0] > 0 and infl[0] < infl[1] < infl[2] < infl[3]
    assert opt[0] > 0 and opt[3] > opt[0]
    # order of magnitude: half-split SE of accuracy at p=0.1, 150 units x 2 items on 75 units = 150 items -> 2.4 pt; E max of 10 normals ~1.54 sd
    assert 0.01 < infl[2] < 0.12
    lo, hi = r[10]["inflation_boot_ci95"]
    assert lo <= r[10]["inflation"] + 0.01 and hi >= r[10]["inflation"] - 0.01


def test_truth_rep_valsel_unbiased_naive_biased():
    rs = np.random.RandomState(2)
    P, true = 30, np.full(30, 0.1)
    reps = []
    for r in range(60):
        cv = (rs.rand(600, P) < true).astype(float)
        ct = (rs.rand(1500, P) < true).astype(float)
        reps.append(sb.truth_inflation_rep(cv, np.arange(600), ct, np.arange(1500), true, [1, 10, 30], 30, rs))
    agg = sb.aggregate_truth(reps, n_boot=200)
    assert abs(agg[10]["bias_valsel"]["mean"]) < 0.004
    assert agg[10]["bias_naive"]["mean"] > 0.01
    assert agg[30]["bias_naive"]["mean"] > agg[10]["bias_naive"]["mean"] > agg[1]["bias_naive"]["mean"] - 1e-9
    assert abs(agg[1]["bias_naive"]["mean"]) < 0.003
