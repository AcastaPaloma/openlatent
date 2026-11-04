"""Cache encoder-0 embeddings (z0) for every fold of one dataset (CPU preprocessing; no metric).  python warm.py immune"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SC  # noqa: E402
from evaluate import get_z0  # noqa: E402

ds = sys.argv[1]
sc = SC(ds)
for f in range(len(sc.folds)):
    T, H = sc.fold_masks(f)
    t0 = time.time()
    Z = get_z0(sc, f, T)
    print(ds, "fold", f, "train", int(T.sum()), "held", int(H.sum()), "z0", Z.shape, "finite", bool(np.isfinite(Z).all()), f"{time.time() - t0:.0f}s", flush=True)
