import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from s_suite import stats as st


def test_cp_upper_known_values():
    assert st.cp_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-9)
    assert st.cp_upper(10, 10, 0.05) == 1.0
    assert st.cp_upper(5, 0, 0.05) == 1.0


def test_zero_leak_floor_matches_stress_test_table():
    # contrarian stress test, 25-way identity (chance 4%), 120-150 independent units: K=1 -> 7.7-8.6%, K=30 -> 11.1-12.5%
    f1 = st.zero_leak_floor(0.04, 150, 0.05, 1)
    f30 = st.zero_leak_floor(0.04, 150, 0.05, 30)
    assert 0.070 < f1 < 0.090
    assert 0.105 < f30 < 0.130
    assert st.zero_leak_floor(0.04, 4707, 0.05, 1) < 0.047  # CXR-sized test: floor is close to chance


def test_cluster_boot_se_exceeds_iid_se_with_shared_unit_effect():
    rs = np.random.RandomState(0)
    m, w = 100, 6
    u = rs.rand(m) < 0.3                       # unit-level success, all windows share it
    y = np.repeat(u.astype(float), w)
    cl = np.repeat(np.arange(m), w)
    inv, mm = st.unit_index(cl)
    S = st.unit_sums(y, inv, mm)
    n_c = np.bincount(inv).astype(float)
    W = st.boot_weights(mm, 1000, 1)
    se = st.ratio_boot(S, n_c, W)[:, 0].std(ddof=1)
    p = y.mean()
    iid = np.sqrt(p * (1 - p) / len(y))
    assert se > 1.8 * iid                      # design effect ~ w
    assert st.design_effect(se, p, len(y)) > 3


def test_boot_weights_deterministic_and_rows_sum_to_m():
    a = st.boot_weights(50, 10, 3)
    b = st.boot_weights(50, 10, 3)
    assert np.array_equal(a, b) and (a.sum(1) == 50).all()


def test_auc_matches_sklearn_with_ties():
    rs = np.random.RandomState(1)
    y = rs.randint(0, 2, 300)
    s = np.round(rs.randn(300) + y * 0.7, 1)   # ties
    assert st.auc_point(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    inv, m = st.unit_index(np.arange(300))
    W = np.ones((1, m))
    assert st.auc_boot(y, s, inv, W)[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_eff_counts_clipped():
    k, n = st.eff_counts(0.5, 0.001, 100)
    assert n == 100                             # SE below binomial -> cannot exceed item count
    k, n = st.eff_counts(0.2, 0.0, 100, n_units=40)
    assert n == 40
