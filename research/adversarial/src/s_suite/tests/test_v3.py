import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from s_suite import canary as cy
from s_suite import certificate as cf
from s_suite import suite_v3 as sv
from s_suite import views as vw
from s_suite.planted import features as planted_features

from .test_certificate import build


def noise(n, T=64, C=2, seed=0, rho=0.5):
    rs = np.random.RandomState(seed)
    a = rs.randn(n, T, 1)
    X = np.concatenate([a] + [rho * a + np.sqrt(1 - rho ** 2) * rs.randn(n, T, 1) for _ in range(C - 1)], axis=2)
    return X.astype(np.float32)


def test_view_shapes_and_backward_compatibility():
    X = noise(10, T=32, C=4)
    v = vw.ViewBuilder().fit(X).transform(X)
    assert v["hb"].shape == (10, 2 * 4 * 16) and v["phase"].shape == (10, 6 * 3 * 16)
    assert v["lvlspec"].shape == (10, 2 * 4 + 4 * 16) and v["inter"].shape == (10, 4 * vw.NLEVEL * vw.NBANDS_INTER * 4)
    assert np.array_equal(v["feat"], planted_features(X))                      # suite v1/v2 view is unchanged
    X1 = noise(10, T=32, C=1)
    assert "phase" not in vw.ViewBuilder().fit(X1).transform(X1)


def test_phase_view_is_invariant_to_a_common_circular_shift():
    X = noise(6, T=64, C=3, seed=2)
    a = vw.phase_view(X)
    b = vw.phase_view(np.roll(X, 7, axis=1))
    assert np.allclose(a, b, atol=1e-4)


def test_suite_declaration_counts_and_digest():
    d2, d1 = sv.declaration(24, 2), sv.declaration(24, 1)
    assert len(d2.attackers) == 24 and len(d1.attackers) == 20 and len(set(d2.attackers)) == 24
    assert d2.attackers[:12] == sv.SUITE12 and d2.canary_groups == ("hb", "phase", "inter") and d1.canary_groups == ("hb", "inter")
    assert sv.declaration(24, 2).digest() == d2.digest() != d1.digest()


def _acc(model_view, X, y, tr, te):
    sc = StandardScaler().fit(model_view[tr])
    clf = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(model_view[tr]), y[tr])
    return (clf.predict(sc.transform(model_view[te])) == y[te]).mean()


@pytest.mark.parametrize("kind,good,blind", [("hb_tone", "hb", "feat"), ("phase_lag", "phase", "feat"), ("ampfreq", "inter", "feat")])
def test_each_canary_is_read_by_its_view_and_not_by_the_v1_features(kind, good, blind):
    N, n = 8, 1600
    y = np.random.RandomState(1).randint(0, N, n)
    X = noise(n, T=64, C=2, seed=3)
    tr, te = np.arange(n) < 1100, np.arange(n) >= 1100
    ref = cy.Reference(X[tr])
    Xc = cy.inject(X, y, kind, key=5, snr=3.0, ref=ref)
    v = vw.ViewBuilder().fit(Xc[tr]).transform(Xc)
    a_good, a_blind = _acc(v[good], Xc, y, tr, te), _acc(v[blind], Xc, y, tr, te)
    assert a_good > (2.0 if kind == "ampfreq" else 3.0) / N, (kind, a_good)
    assert a_blind < 1.8 / N + 0.04, (kind, a_blind)
    Xn = vw.ViewBuilder().fit(X[tr]).transform(X)
    assert _acc(Xn[good], X, y, tr, te) < 1.8 / N + 0.04                         # no canary, no signal


def _canary_cert(decl, build_seed=21, accs=(0.5, 0.4, 0.3)):
    val, test = build(build_seed, list(accs), n_units=3000, w=1)
    return cf.certify(decl, val, test, split_status="canary")


def test_below_margin_needs_both_bounds_and_canaries():
    names = tuple(f"a{k}" for k in range(3))
    decl = cf.SuiteDecl("t", names, n_boot=500, seed=3, n_classes=10, margin=0.04, views=("raw", "hb"), canary_groups=("hb",))
    val, test = build(0, [0.1, 0.1, 0.1], n_units=3000, w=1)
    c0 = cf.certify(decl, val, test)
    assert c0["verdict"]["label"] == "NOT_RESOLVED" and c0["verdict"]["assurance_withheld"] == ["hb"] and c0["verdict"]["below_margin_bonferroni"]
    ok = _canary_cert(decl)
    assert ok["verdict"]["label"] == "LEAK"
    c1 = cf.certify(decl, val, test, canaries={"hb": ok})
    assert c1["verdict"]["label"] == "BELOW_MARGIN" and c1["view_scope"]["canaries"]["hb"]["accepted"] is True
    assert c1["view_scope"]["not_covered"] and "hb" in c1["view_scope"]["declared_views"]
    # a canary of another declaration, a canary that read nothing, or one with too few units does not count
    other = cf.SuiteDecl("t2", names, n_boot=500, seed=3, n_classes=10, margin=0.04, views=("raw", "hb"), canary_groups=("hb",))
    assert cf.certify(decl, val, test, canaries={"hb": _canary_cert(other)})["verdict"]["label"] == "NOT_RESOLVED"
    weak = _canary_cert(decl, accs=(0.1, 0.1, 0.1))
    assert weak["verdict"]["label"] != "LEAK" and cf.certify(decl, val, test, canaries={"hb": weak})["verdict"]["label"] == "NOT_RESOLVED"
    small_val, small_test = build(22, [0.5, 0.4, 0.3], n_units=100, w=1)
    small = cf.certify(decl, small_val, small_test, split_status="canary")
    assert cf.certify(decl, val, test, canaries={"hb": small})["verdict"]["label"] == "NOT_RESOLVED"
    # decl without canary groups keeps the v2 behaviour except for the stricter both-bounds rule
    plain = cf.SuiteDecl("t", names, n_boot=500, seed=3, n_classes=10, margin=0.04)
    assert cf.certify(plain, val, test)["verdict"]["label"] == "BELOW_MARGIN"


def test_run_canaries_end_to_end_on_a_tiny_release():
    rs = np.random.RandomState(0)
    N = 6

    def split(per):
        y = np.repeat(np.arange(N), per * 4)
        unit = np.array([f"s{i}" for i in np.repeat(np.arange(N * per), 4)])
        return noise(len(y), T=32, C=2, seed=int(rs.randint(1e6))), y, unit
    splits = dict(train=split(6), val=split(2), test=split(4))
    decl = sv.declaration(N, 2, n_boot=200)
    certs = cy.run_canaries(splits, N, decl, groups=("hb",), snr={"hb": 2.5, "phase": 2.5, "inter": 2.5})
    assert set(certs) == {"hb"} and certs["hb"]["split_status"] == "canary" and certs["hb"]["decl_digest"] == decl.digest()
    assert certs["hb"]["data"]["n_test_units"] == N * 4
