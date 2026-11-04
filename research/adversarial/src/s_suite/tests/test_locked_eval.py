import numpy as np
import pytest

from s_suite import locked_eval as le


def onehot(pred, C):
    P = np.zeros((len(pred), C)); P[np.arange(len(pred)), pred] = 1.0
    return P


def test_only_scalars_and_ladder_semantics():
    rs = np.random.RandomState(0)
    yd, ys = rs.randint(0, 5, 200), rs.randint(0, 5, 300)
    ev = le.LockedEvaluator(yd, ys, eta=0.05, max_queries=10)
    s1 = ev.dev(onehot(rs.randint(0, 5, 200), 5))
    assert isinstance(s1, float)
    # perfect predictions beat the ladder; a second weaker query does not lower the shown score
    s2 = ev.dev(onehot(yd, 5))
    assert s2 >= 0.95
    s3 = ev.dev(onehot(rs.randint(0, 5, 200), 5))
    assert s3 == s2
    # small improvements below eta are not revealed
    ev2 = le.LockedEvaluator(yd, ys, eta=0.2, max_queries=10)
    a = ev2.dev(onehot(yd, 5))
    b = ev2.dev(onehot(yd, 5))
    assert a == b


def test_budget_and_single_final():
    yd, ys = np.arange(10) % 3, np.arange(12) % 3
    ev = le.LockedEvaluator(yd, ys, max_queries=2)
    ev.dev(onehot(yd, 3)); ev.dev(onehot(yd, 3))
    with pytest.raises(le.BudgetExceeded):
        ev.dev(onehot(yd, 3))
    f = ev.final(onehot(ys, 3), "abc")
    assert f == 1.0
    with pytest.raises(le.AlreadyFinal):
        ev.final(onehot(ys, 3), "abc")
    rep = ev.seal_report()
    assert rep["artifact_digest"] == "abc" and rep["n_queries"] == 2


def test_rejects_bad_shapes_and_nans():
    ev = le.LockedEvaluator(np.zeros(5, int), np.zeros(5, int))
    with pytest.raises(ValueError):
        ev.dev(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        ev.dev(np.full((5, 3), np.nan))


def test_subprocess_client_holds_no_labels(tmp_path):
    yd, ys = np.arange(40) % 4, np.arange(50) % 4
    path = str(tmp_path / "labels.npz")
    le.write_labels(path, yd, ys)
    cl = le.LockedClient(path, eta=0.01, max_queries=5)
    try:
        assert cl.status()["n_dev"] == 40
        assert cl.dev(onehot(yd, 4)) > 0.98
        assert cl.final(onehot(ys, 4), "deadbeef") == 1.0
        with pytest.raises(RuntimeError):
            cl.final(onehot(ys, 4), "x")
        assert not any(isinstance(v, np.ndarray) for v in vars(cl).values())
    finally:
        cl.close()


def test_shuffled_label_null_of_the_whole_search_loop():
    # 40 designs of pure noise against shuffled labels: the dev score the loop sees drifts above chance, the single sealed query does not
    finals, bests = [], []
    for seed in range(20):
        rs = np.random.RandomState(seed)
        C, nd, nsl = 10, 150, 400
        yd, ys = rs.permutation(rs.randint(0, C, nd)), rs.permutation(rs.randint(0, C, nsl))
        ev = le.LockedEvaluator(yd, ys, eta=0.005, max_queries=100)
        design = lambda i: (rs.rand(nd, C), rs.rand(nsl, C), f"d{i}")
        r = le.run_search_loop(ev, design, 40)
        finals.append(r["final"]); bests.append(r["best_dev_shown"])
    assert abs(np.mean(finals) - 0.1) < 0.02
    assert np.mean(bests) > 0.1 + 0.02            # winner's curse is visible on the dev side only
