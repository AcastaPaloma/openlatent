import numpy as np
import pytest

from s_suite import evidence_budget as eb


def curve(items, K=(1, 2, 5, 10), **kw):
    return eb.budget_curve(items, list(K), n_draws=60, n_boot=300, seed=1, **kw)


def test_independent_toy_schemes_agree_and_grow():
    it = eb.toy_items(rho=0.0, n_sessions=8, per_session=12, seed=0)
    r = curve(it)
    d = r["schemes"]["diverse"]["acc"]
    s = r["schemes"]["same_session"]["acc"]
    assert d[3] > d[0] + 0.15                      # evidence adds up (13% -> >30% at K=10)
    assert abs(d[3] - s[3]) < 0.2
    ind = r["independence"][-1]
    assert ind["available"]


def test_shared_session_nuisance_is_detected():
    it = eb.toy_items(rho=0.9, n_sessions=8, per_session=12, seed=1)
    r = curve(it)
    ind = r["independence"][-1]
    assert ind["diverse"] - ind["same_session"] > 0.08
    assert ind["dependence_detected"]
    assert r["schemes"]["same_session"]["acc"][3] < r["schemes"]["diverse"]["acc"][3]


def test_labels_shuffled_stays_at_chance_for_every_k():
    it = eb.toy_items(rho=0.0, n_sessions=6, per_session=6, seed=2)
    rs = np.random.RandomState(0)
    it2 = eb.Items(it.probs, rs.permutation(it.person), it.session)       # probabilities no longer match the person labels
    r = curve(it2, schemes=("random", "diverse"))
    for sc in ("random", "diverse"):
        acc = np.array(r["schemes"][sc]["acc"])
        assert acc.max() < 0.12                                           # chance 0.04


def test_n_star_interpolation():
    assert eb._n_star([1, 5, 10], [0.1, 0.4, 0.8], 0.6) == pytest.approx(7.5)
    assert eb._n_star([1, 5, 10], [0.1, 0.4, 0.8], 0.9) is None
    assert eb._n_star([1, 5, 10], [0.5, 0.6, 0.8], 0.3) == 1.0


def test_predicted_n_star_has_correct_order_of_magnitude_on_independent_toy():
    it = eb.toy_items(rho=0.0, n_sessions=8, per_session=12, seed=3)
    pred = eb.predicted_n_star(it.probs, it.person)
    assert pred["n_star_pred"] is not None and 10 < pred["n_star_pred"] < 80
    r = curve(it, K=(1, 2, 5, 10, 20, 40), targets=(0.9,))
    n_obs = r["schemes"]["diverse"]["n_star"]["0.9"]
    assert n_obs is not None and 0.2 < n_obs / pred["n_star_pred"] < 3


def test_infeasible_scheme_returns_nan_not_error():
    it = eb.toy_items(rho=0.0, n_sessions=2, per_session=3, seed=0)
    r = curve(it, K=(1, 5), schemes=("same_session", "diverse"))
    assert r["schemes"]["same_session"]["n_persons_eligible"][1] == 0
