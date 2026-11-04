"""Self-tests for the N2 families on a synthetic base: python -m n_control.selftest2   (CPU, < 1 min).
Checks determinism, dose 0 at chance, a large dose near 1 for every family (session level for `order`), tone-invariance of the af2 level, shuffled twin at chance."""
from __future__ import annotations

import numpy as np

from .families2 import FAMILIES2, level1, inject2, stats2
from .n2 import build_release2, oracle_report2
from .release import digest
from .selftest import synth_base


def main():
    b = synth_base(1, per_sess=30)
    st = stats2(b)
    big = dict(bias=8.0, specin=3.0, hb2=1.5, phase2=2.0, af2=3.0, pac=8.0, order=3.0)
    for fam in FAMILIES2:
        res = {}
        for amp in (0.0, big[fam]):
            e = dict(kind="plant", family=fam, amp=amp, assign_seed=7, key="ef" * 16, shuffle_seed=0)
            rel = build_release2(b, e)
            assert digest(rel) == digest(build_release2(b, e)), fam
            if amp == 0.0:
                assert np.allclose(rel["X"], b.X), fam
            res[amp] = oracle_report2(b, e, rel)["planted"]
        lo, hi = res[0.0], res[big[fam]]
        key = "sess" if fam == "order" else "win"
        assert lo[key] < 0.12, (fam, lo)
        assert hi[key] > 0.85, (fam, hi)
        print(fam, "dose0 %.3f  large %.3f/%.3f" % (lo[key], hi["win"], hi["sess"]))
    e = dict(kind="shuf", family="pac", amp=8.0, assign_seed=7, key="ef" * 16, shuffle_seed=3)
    rep = oracle_report2(b, e, build_release2(b, e))
    assert rep["planted"]["win"] > 0.85 and rep["released"]["win"] < 0.12, rep
    X2 = inject2(b.X, np.zeros(len(b.X), dtype=int), b.sess, "af2", 3.0, "ef" * 16, st)
    assert (level1(X2, st["q1edges"]) == level1(b.X, st["q1edges"])).all()
    print("n_control selftest2 OK")


if __name__ == "__main__":
    main()
