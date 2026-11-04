import numpy as np

from s_suite.attackers import POOL_NAMES, SUITE12, fit_predict_pool
from s_suite.planted import PlantedSpec, features, generate_replicate, raw_view


def run(spec, seed=0, names=("lr_raw_c0.1", "lr_feat_c1", "lda_feat_auto"), **kw):
    rep = generate_replicate(spec, seed, units_train=10, units_val=2, units_test=4, units_pop=8)
    vt = {"raw": raw_view(rep["train"]["X"]), "feat": features(rep["train"]["X"])}
    ve = {s: {"raw": raw_view(rep[s]["X"]), "feat": features(rep[s]["X"])} for s in ("test",)}
    out = fit_predict_pool(vt, rep["train"]["y"], ve, names=names, n_classes=spec.n_id)["test"]
    return {k: float((P.argmax(1) == rep["test"]["y"]).mean()) for k, P in out.items()}


def test_pool_declaration():
    assert len(POOL_NAMES) == 30 and len(set(POOL_NAMES)) == 30 and len(SUITE12) == 12


def test_null_is_chance_and_gain_leaks_through_features_not_raw():
    null = run(PlantedSpec(channel="null", dose=5.0))
    assert max(null.values()) < 0.09                                  # chance 0.04
    gain = run(PlantedSpec(channel="gain", dose=0.3))
    assert gain["lr_feat_c1"] > 0.3 and gain["lda_feat_auto"] > 0.3
    assert gain["lr_raw_c0.1"] < 0.12                                 # zero-mean gain is invisible to a linear model on raw windows


def test_bias_visible_to_raw_linear():
    b = run(PlantedSpec(channel="bias", dose=1.0))
    assert b["lr_raw_c0.1"] > 0.2


def test_determinism_and_unit_structure():
    spec = PlantedSpec(channel="spectral", dose=0.4)
    a = generate_replicate(spec, 3, 4, 2, 2, 2)
    b = generate_replicate(spec, 3, 4, 2, 2, 2)
    assert np.array_equal(a["train"]["X"], b["train"]["X"])
    c = generate_replicate(spec, 4, 4, 2, 2, 2)
    assert not np.array_equal(a["train"]["X"], c["train"]["X"])
    t = a["test"]
    # windows of one unit share the session offset: within-unit spread of window means is smaller than across-unit spread
    mu = t["X"].mean(1)[:, 0]
    within = np.mean([mu[t["unit"] == u].std() for u in np.unique(t["unit"])])
    assert within < mu.std()
    assert len(t["X"]) == 25 * 2 * spec.W
