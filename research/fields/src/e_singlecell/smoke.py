"""Synthetic dataset to exercise the SC-0 pipeline (no real data, numbers are NOT results).  python smoke.py  -> writes $E_CACHE/smoke"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_dir  # noqa: E402


def main(n=9000, G=400, seed=0):
    rng = np.random.RandomState(seed)
    nb, nt = 6, 9
    batch = rng.randint(nb, size=n)
    fine = rng.randint(nt, size=n)
    coarse = fine // 3
    donor = batch * 2 + rng.randint(2, size=n)
    tmean = rng.randn(nt, G) * 1.0 * (rng.rand(nt, G) < 0.25)
    bshift = rng.randn(nb, G) * 0.8
    dshift = rng.randn(nb * 2, G) * 0.3
    X = np.maximum(0, tmean[fine] + bshift[batch] + dshift[donor] + 1.0 + rng.randn(n, G) * 0.7).astype(np.float32)
    X[rng.rand(n, G) < 0.5] = 0
    Xs = sp.csr_matrix(X)
    d = cache_dir("smoke")
    names = [f"b{i}" for i in range(nb)]
    fn = [f"t{i}" for i in range(nt)]
    cn = [f"c{i}" for i in range(3)]
    sup = list(fn)
    held = [t for i, t in enumerate(sup) if i % 3 == 1]
    pd.DataFrame({"b": batch, "unit": donor, "fine": fine, "coarse": coarse, "study": [names[b] for b in batch],
                  "fine_name": [fn[i] for i in fine], "coarse_name": [cn[i] for i in coarse], "batch_name": [names[b] for b in batch],
                  "unit_name": donor.astype(str)}).to_pickle(d / "obs.pkl")
    sp.save_npz(d / "X.npz", Xs, compressed=False)
    sp.save_npz(d / "counts.npz", Xs, compressed=False)
    (d / "genes.txt").write_text("\n".join(f"g{i}" for i in range(G)))
    (d / "meta.json").write_text(json.dumps(dict(batches=names, fine_names=sup, coarse_names=cn, held_types=held, has_donor=True,
                                                 n_cells=n, n_genes=G, n_units=nb * 2)))
    print("smoke dataset written to", d)


if __name__ == "__main__":
    main()
