import json

import numpy as np
import pytest

from s_suite import certificate as cf
from s_suite import stats as st


def make_split(rs, n_units, w, C, accs, y=None, shuffle=False):
    """Units of w items; attacker k correct with prob accs[k] per UNIT-level effect mixture. probs one-hot-ish."""
    n = n_units * w
    cl = np.repeat(np.arange(n_units), w)
    y = rs.randint(0, C, n) if y is None else y
    out = {}
    for k, a in enumerate(accs):
        # unit-level correlated success: p_unit ~ Beta around a (cluster effect), item success ~ Bernoulli(p_unit)
        pu = np.clip(rs.beta(a * 4 + 1e-3, (1 - a) * 4 + 1e-3, n_units), 0, 1)
        ok = rs.rand(n) < pu[cl]
        pred = np.where(ok, y, (y + 1 + rs.randint(0, C - 1, n)) % C)
        P = np.full((n, C), 0.01)
        P[np.arange(n), pred] = 1.0
        out[f"a{k}"] = P
    return y, cl, out


def decl(K, **kw):
    return cf.SuiteDecl("t", tuple(f"a{k}" for k in range(K)), n_boot=500, seed=3, n_classes=10, **kw)


def build(seed, accs, n_units=150, w=4, C=10):
    rs = np.random.RandomState(seed)
    yv, cv, sv = make_split(rs, 100, w, C, accs)
    yt, ct, stt = make_split(rs, n_units, w, C, accs)
    return cf.SplitPreds(yv, cv, sv), cf.SplitPreds(yt, ct, stt)


def test_deterministic_json_and_digest():
    val, test = build(0, [0.3, 0.2, 0.1])
    c1 = cf.certify(decl(3), val, test)
    c2 = cf.certify(decl(3), val, test)
    assert cf.to_json(c1) == cf.to_json(c2)
    assert c1["digest"] == c2["digest"]
    json.loads(cf.to_json(c1))                   # serialisable
    # different seed in the declaration changes the declaration digest
    d2 = cf.SuiteDecl("t", decl(3).attackers, n_boot=500, seed=4, n_classes=10)
    assert cf.certify(d2, val, test)["decl_digest"] != c1["decl_digest"]


def test_declaration_is_enforced():
    val, test = build(0, [0.3, 0.2, 0.1])
    with pytest.raises(ValueError):
        cf.certify(decl(4), val, test)           # a4 declared but absent
    del test.scores["a2"]
    with pytest.raises(ValueError):
        cf.certify(decl(3), val, test)
    val, test = build(0, [0.3, 0.2, 0.1, 0.1])
    with pytest.raises(ValueError):
        cf.certify(decl(3), val, test)           # undeclared extra attacker


def test_selected_is_chosen_on_val_not_test():
    rs = np.random.RandomState(5)
    val, test = build(1, [0.5, 0.1])
    # make attacker a1 best on test and a0 best on val
    c = cf.certify(decl(2), val, test)
    assert c["selected"]["name"] == "a0"
    test.scores["a0"], test.scores["a1"] = test.scores["a1"], test.scores["a0"]   # now a1 holds the strong test predictions
    c2 = cf.certify(decl(2), val, test)
    assert c2["selected"]["name"] == "a0"        # still val's pick although a1 is the better one on test
    assert c2["naive_max"]["name"] == "a1"


def test_leak_detected_and_null_not_flagged():
    val, test = build(2, [0.45, 0.3, 0.2])
    assert cf.certify(decl(3), val, test)["verdict"]["label"] == "LEAK"
    flagged = 0
    for s in range(40):
        val, test = build(100 + s, [0.1, 0.1, 0.1])        # chance = 0.1 for C=10
        c = cf.certify(decl(3), val, test)
        flagged += c["verdict"]["label"] == "LEAK"
    assert flagged <= 6                                    # nominal 5% of 40 = 2


def test_mde_floor_and_bound_ordering():
    val, test = build(3, [0.15, 0.12, 0.1, 0.1, 0.1])
    c = cf.certify(decl(5), val, test)
    sel = c["selected"]
    assert sel["mde80"] == pytest.approx(2.8 * sel["se"], abs=2e-6)
    for a in c["attackers"].values():
        assert a["mde80"] == pytest.approx(2.8 * a["se"], abs=2e-6)
        assert a["ucb_bonferroni"] >= a["ucb_onesided"] - 1e-9
    assert c["suite_bound"]["bonferroni"] >= max(a["ucb_onesided"] for a in c["attackers"].values()) - 1e-9
    assert c["floor"]["ucb_at_baseline_suite"] > c["floor"]["ucb_at_baseline_k1"] > c["baseline_used"]
    assert c["suite_bound"]["q_maxt"] <= c["suite_bound"]["z_bonferroni"] + 0.3   # max-t is never much worse than Bonferroni


def test_bonferroni_coverage_of_best_declared_attacker():
    # unit-level Bernoulli attackers with known true accuracies; the suite bound must cover the best true accuracy >= ~95%
    accs = [0.12, 0.11, 0.10, 0.10, 0.09]
    cover = 0
    R = 150
    for s in range(R):
        rs = np.random.RandomState(1000 + s)
        yv = rs.randint(0, 10, 400); cv = np.arange(400)
        yt = rs.randint(0, 10, 600); ct = np.arange(600)
        def mk(y):
            out = {}
            for k, a in enumerate(accs):
                ok = rs.rand(len(y)) < a
                pred = np.where(ok, y, (y + 1 + rs.randint(0, 9, len(y))) % 10)
                P = np.full((len(y), 10), 0.01); P[np.arange(len(y)), pred] = 1.0
                out[f"a{k}"] = P
            return out
        c = cf.certify(decl(5), cf.SplitPreds(yv, cv, mk(yv)), cf.SplitPreds(yt, ct, mk(yt)))
        cover += c["suite_bound"]["bonferroni"] >= max(accs)
    assert cover / R >= 0.93


def test_label_only_baseline_and_normalised_leakage():
    val, test = build(4, [0.4, 0.2])
    lab = np.full((len(test.y), 10), 0.1)
    lab[:, 3] = 0.9                                       # constant predictor of class 3
    c = cf.certify(decl(2), val, test, label_only=lab)
    assert c["label_only"] is not None
    assert c["baseline_used"] >= c["chance"]["uniform"]
    L = c["selected"]["normalised_leakage"]
    assert 0 < L < 1


def test_controls_invalidate_certificate():
    val, test = build(5, [0.4, 0.3])
    c = cf.certify(decl(2), val, test)
    good_val, good_test = build(6, [0.1, 0.1])
    ctrl_ok = cf.certify(decl(2), good_val, good_test)
    bad = cf.certify(decl(2), *build(7, [0.5, 0.4]))
    assert ctrl_ok["verdict"]["label"] != "LEAK"
    assert cf.attach_controls(c, {"shuffled": ctrl_ok})["valid"] is True
    out = cf.attach_controls(c, {"shuffled": bad})
    assert out["valid"] is False and out["controls"]["shuffled"]["passes"] is False


def test_paired_difference_requires_same_units_and_prints_mde():
    val, test = build(8, [0.3, 0.2])
    d = cf.paired_difference(decl(2), test, "a0", test, "a1")
    assert d["mde80"] == pytest.approx(2.8 * d["se"], abs=2e-6)
    other = cf.SplitPreds(test.y.copy(), np.roll(test.cluster, 1), dict(test.scores))
    with pytest.raises(ValueError):
        cf.paired_difference(decl(2), test, "a0", other, "a1")


def test_auc_metric_path():
    rs = np.random.RandomState(0)
    n = 600
    y = rs.randint(0, 2, n); cl = np.repeat(np.arange(150), 4)
    sc = {"s0": y * 1.0 + rs.randn(n), "s1": rs.randn(n)}
    sv = {"s0": np.concatenate([y * 1.0 + rs.randn(n)]), "s1": rs.randn(n)}
    d = cf.SuiteDecl("auc", ("s0", "s1"), metric="auc", n_boot=300, seed=1)
    c = cf.certify(d, cf.SplitPreds(y, cl, sv), cf.SplitPreds(y, cl, sc))
    assert c["chance"]["used"] == 0.5 and c["selected"]["name"] == "s0" and c["verdict"]["label"] == "LEAK"
