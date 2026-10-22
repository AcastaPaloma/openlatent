import numpy as np
import pytest

from openlatent.scorers.dependence import distance_correlation, identity_alignment, linear_cka
from openlatent.scorers.probes import binary_probe, ridge_probe
from openlatent.scorers.reid import reid_scores

rng = np.random.default_rng(0)
N_ID, DIM = 100, 16
IDENTITIES = rng.normal(size=(N_ID, DIM))
GALLERY_IDS = np.arange(N_ID)
QUERY_IDS = np.repeat(GALLERY_IDS, 3)


def _reid(gallery, query, **kw):
    return reid_scores(gallery, GALLERY_IDS, query, QUERY_IDS, **kw)


def test_reid_identity_leak_is_found():
    noisy = IDENTITIES[QUERY_IDS] + 0.1 * rng.normal(size=(len(QUERY_IDS), DIM))
    r = _reid(IDENTITIES, noisy)
    assert r["top1"] > 0.95 and r["verification_auc"] > 0.99


def test_reid_random_features_near_chance():
    r = _reid(rng.normal(size=(N_ID, DIM)), rng.normal(size=(len(QUERY_IDS), DIM)))
    assert r["top1"] < 0.05
    assert abs(r["verification_auc"] - 0.5) < 0.05


def test_reid_constant_features_are_exactly_chance():
    r = _reid(np.ones((N_ID, DIM)), np.ones((len(QUERY_IDS), DIM)))
    assert r["top1"] == pytest.approx(r["chance_top1"])
    assert r["top5"] == pytest.approx(r["chance_top5"])
    assert r["mrr"] == pytest.approx(r["chance_mrr"])
    assert r["verification_auc"] == pytest.approx(0.5)


def test_reid_chunking_does_not_change_ranks():
    query = IDENTITIES[QUERY_IDS] + rng.normal(size=(len(QUERY_IDS), DIM))
    a, b = _reid(IDENTITIES, query, chunk=7), _reid(IDENTITIES, query, chunk=10_000)
    assert a["top1"] == b["top1"] and a["mrr"] == pytest.approx(b["mrr"])


def test_probes_detect_signal_and_are_repeatable():
    x = rng.normal(size=(400, 8))
    y = (x[:, 0] > 0).astype(int)
    groups = np.arange(200) // 2
    first = binary_probe(x[:200], y[:200], groups, x[200:], y[200:])
    assert first["auc"] > 0.95 and first["converged"]
    assert binary_probe(x[:200], y[:200], groups, x[200:], y[200:]) == first

    age = 50 + 10 * x[:, 1]
    assert ridge_probe(x[:200], age[:200], groups, x[200:], age[200:])["r2"] > 0.95


def test_ridge_cv_prevents_overfit_below_chance():
    x = rng.normal(size=(300, 250))  # nearly as many dims as rows, no signal
    age = rng.normal(50, 10, size=300)
    r = ridge_probe(x[:150], age[:150], np.arange(150), x[150:], age[150:])
    assert r["alpha"] >= 1e3 and r["r2"] > -0.05


def test_linear_cka_properties():
    x = rng.normal(size=(300, 10))
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)
    assert linear_cka(x, rng.normal(size=(300, 10))) < 0.1
    wide = rng.normal(size=(20, 500))  # exercises the Gram-matrix branch
    assert linear_cka(wide, wide) == pytest.approx(1.0)


def test_distance_correlation_catches_nonlinear_dependence():
    x = rng.normal(size=(400, 1))
    assert distance_correlation(x, x**2) > 0.3  # zero Pearson correlation, but dependent
    assert distance_correlation(x, rng.normal(size=(400, 1))) < 0.15
    assert distance_correlation(x, np.zeros((400, 3))) == 0.0


def test_identity_alignment():
    ids = np.repeat(np.arange(50), 4)
    clustered = IDENTITIES[ids % N_ID] + 0.05 * rng.normal(size=(len(ids), DIM))
    assert identity_alignment(clustered, ids) > 0.5
    assert abs(identity_alignment(rng.normal(size=(len(ids), DIM)), ids)) < 0.05
    assert abs(identity_alignment(rng.normal(size=(len(ids), DIM)), ids, kernel="linear")) < 0.05
    assert identity_alignment(np.ones((len(ids), DIM)), ids) == 0.0
