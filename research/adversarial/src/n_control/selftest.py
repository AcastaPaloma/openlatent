"""Self-tests on a synthetic base (coloured noise with activity-dependent amplitude); run: python -m n_control.selftest   (CPU, ~1 min).

Checks: assignment balance and subject-disjointness, determinism of release and digest, null oracle at chance, oracle monotone in dose and near 1 at a large
dose for every family, shuffled twin oracle at chance, tone-invariance of the ampfreq level, zero-mean of the injected tones, canonical manifest bytes,
analyse() on fabricated certificates.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from . import analyse as an
from .assign import assign_sessions, check_assignment
from .base import Base
from .common import FAMILIES, N_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, canon
from .families import level_of
from .release import build_release, digest, oracle_report


def synth_base(seed=0, per_sess=20):
    rs = np.random.RandomState(seed)
    X, sub, act, trial, j = [], [], [], [], []
    codes = [(0, 1), (0, 2), (0, 11), (1, 3), (1, 4), (1, 12), (2, 7), (2, 8), (2, 15), (3, 9), (3, 16)]
    for s in range(24):
        off = rs.randn(2) * 0.3
        for a, t in codes:
            for w in range(per_sess):
                z = rs.randn(2, 160)
                k = np.ones(8) / 8
                zz = np.stack([np.convolve(z[c], k, mode="valid")[:128] for c in range(2)]) * (1 + a) * 2
                X.append(zz + off[:, None] + a * 0.3)
                sub.append(s); act.append(a); trial.append(t); j.append(w)
    return Base(np.array(X, dtype=np.float32), sub, act, trial, j)


def main():
    b = synth_base()
    assert b.n_sess == 264
    sp, pid = assign_sessions(b.sess_sub, b.sess_act, 5)
    cnt = check_assignment(sp, pid)
    for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        assert not (set(b.sess_sub[sp == s]) & set(b.sess_sub[sp != s])), "real subjects cross splits"
    e0 = dict(kind="null", family=None, amp=0.0, assign_seed=5, key="ab" * 16, shuffle_seed=0)
    r0 = build_release(b, e0)
    assert digest(r0) == digest(build_release(b, e0)), "digest not deterministic"
    res = {}
    for fam in FAMILIES:
        for amp in (0.0, 1.5 if fam != "stealth" else 0.3, 8.0 if fam == "bias" else (3.0 if fam != "stealth" else 0.6)):
            e = dict(kind="plant", family=fam, amp=amp, assign_seed=5, key="cd" * 16, shuffle_seed=0)
            rel = build_release(b, e)
            if amp == 0.0:
                assert np.allclose(rel["X"], b.X), "dose 0 must be the unplanted release"
            assert digest(rel) == digest(build_release(b, e))
            res[(fam, amp)] = oracle_report(b, e, rel)["planted"]
        lo, mid, hi = (res[(fam, a)]["win"] for a in sorted({k[1] for k in res if k[0] == fam}))
        assert lo < 0.09, (fam, "oracle at dose 0 must sit at chance", lo)
        assert hi > 0.9, (fam, "oracle at a large dose must be near 1", hi)
        assert lo <= mid + 0.02 and mid <= hi + 0.02, (fam, lo, mid, hi)
    # shuffled twin: the released-label oracle is at chance, the planted-label oracle is not
    e = dict(kind="shuf", family="specin", amp=3.0, assign_seed=5, key="cd" * 16, shuffle_seed=9)
    rep = oracle_report(b, e, build_release(b, e))
    assert rep["planted"]["win"] > 0.9 and rep["released"]["win"] < 0.1, rep
    # a zero-mean tone cannot change the ampfreq level
    from .families import inject
    X2 = inject(b.X, pid[b.sess], "ampfreq", 2.0, "cd" * 16, b.stats())
    assert (level_of(X2, b.stats()["qedges"]) == level_of(b.X, b.stats()["qedges"])).all(), "ampfreq level moved with the tone"
    # canonical bytes round trip
    obj = {"b": [1, 2.5, None], "a": {"z": 1, "y": "x"}}
    assert json.loads(canon(obj)) == obj and canon(json.loads(canon(obj))) == canon(obj)
    # analyse on fabricated certificates
    d = tempfile.mkdtemp()
    entries, rmap, orc, k = [], {}, {}, 0
    rs = np.random.RandomState(0)
    for kind, fam, lvl, n in [("null", None, 0, 30), ("shuf", "specin", 4, 10)] + [("plant", f, L, 6) for f in FAMILIES for L in (1, 2, 3, 4, 5)]:
        for _ in range(n):
            dg = "%064x" % k
            entries.append(dict(rid=k, kind=kind, family=fam, level=lvl, base="msda", amp=0.1, rep=0, assign_seed=1, key="00", shuffle_seed=0))
            rmap[dg] = k
            orc[str(k)] = dict(kind=kind) if kind == "null" else dict(kind=kind, planted=dict(win=0.1 * max(lvl, 1), sess=0.2, n_win=1, n_sess=1),
                                                                      released=dict(win=0.04, sess=0.04, n_win=1, n_sess=1))
            leak = (kind == "plant" and fam in ("bias", "specin") and lvl >= 3) or (kind == "plant" and fam == "xphase" and lvl == 5 and rs.rand() < 0.2)
            cert = dict(selected=dict(name="a", test=0.3 if leak else 0.045, se=0.01, excess_over_baseline=0.2, excess_lcb_onesided=0.1, excess_mde80=0.03),
                        suite_bound=dict(bonferroni=0.3, maxt=0.3), naive_max=dict(test=0.3), data=dict(n_test_units=88), baseline_used=0.05,
                        floor=dict(ucb_at_baseline_k1=0.1, ucb_at_baseline_suite=0.12), split_status="sealed", evaluator=dict(frozen_match=True),
                        verdict=dict(label="LEAK" if leak else "NOT_RESOLVED"), attackers={"a": dict(test=0.3)})
            json.dump(cert, open(os.path.join(d, dg + ".json"), "w"))
            k += 1
    man = dict(entries=entries)
    mp_ = os.path.join(d, "man.json")
    open(mp_, "wb").write(canon(man))
    json.dump(rmap, open(os.path.join(d, "release_map.json"), "w"))
    json.dump(orc, open(os.path.join(d, "oracle.json"), "w"))
    R, rows = an.analyse(mp_, d, d)
    assert R["G1"]["pass"] and R["G2"]["pass"], (R["G1"], R["G2"])
    assert R["primary"]["bias"]["status"] == "DETECTED" and R["primary"]["stealth"]["status"] == "MISSED", R["primary"]["stealth"]["status"]
    assert R["verdict"]["label"] == "NO", R["verdict"]
    print("n_control selftest OK", cnt)


if __name__ == "__main__":
    main()
