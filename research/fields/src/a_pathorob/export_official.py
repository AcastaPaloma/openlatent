#!/usr/bin/env python
"""CPU only.  Convert cached embeddings to the official PathoROB feature store
(<repo>/data/features/<model>/<dataset>/<centre>.npz, one key '<slide_id>-<patch_id>' per patch) so that the unmodified
official Robustness Index / APD code can be run on our embeddings (replication anchor)."""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a_pathorob import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--datasets", nargs="+", default=C.DATASETS)
ap.add_argument("--features_dir", default=str(C.PATHOROB_REPO / "data" / "features"))
a = ap.parse_args()
for d in a.datasets:
    X = C.load_embeddings(a.model, d)
    idx = C.load_index(d)
    ids = (idx.slide_id.astype(str) + "-" + idx.patch_id.astype(str)).values
    out = Path(a.features_dir) / a.model / d
    out.mkdir(parents=True, exist_ok=True)
    for centre in idx.medical_center.unique():
        m = (idx.medical_center == centre).values
        f = out / f"{centre.replace(' ', '_')}.npz"
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **{i: x for i, x in zip(ids[m], X[m])})
        tmp.replace(f)
    print("exported", a.model, d, X.shape, flush=True)
