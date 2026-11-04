"""Validate the GPU AP (`ap.ap_queries`) against copairs 0.4.1 per query on S4 (raw encoder-0 profiles, and a random-projection embedding).
Writes validate_s4.json next to the cache.  Modes checked: xs (primary U) and nonrep (benchmark)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ap as AP  # noqa: E402
import common as C  # noqa: E402

d = C.cache_dir("s4")
X = np.load(d / "X.npy")
m = pd.read_parquet(d / "meta.parquet")
trt = (m["PertType"] == "trt").to_numpy() & (m["Metadata_JCP2022"] != "JCP2022_033954").to_numpy()
m, X = m[trt].reset_index(drop=True), X[trt]
comp = pd.factorize(m["Metadata_JCP2022"])[0]
src = pd.factorize(m["Metadata_Source"])[0]
plate, well, pert = m["Metadata_Plate"].to_numpy(), m["Metadata_Well"].to_numpy(), np.array(["trt"] * len(m))
rng = np.random.RandomState(0)
embs = {"raw": X, "rp32": (X - X.mean(0)) @ (rng.randn(X.shape[1], 32).astype(np.float32) / np.sqrt(X.shape[1]))}
out = {}
for name, E in embs.items():
    for mode in ("xs", "nonrep"):
        t0 = time.time()
        a = AP.ap_queries(E, plate, comp, src, well, pert, mode=mode)
        t1 = time.time() - t0
        import copairs.map as cp
        pd_diff = ["Metadata_Source"] if mode == "xs" else []
        res = cp.average_precision(m, E.astype(np.float32), pos_sameby=["Metadata_JCP2022"], pos_diffby=pd_diff, neg_sameby=["Metadata_Plate"],
                                   neg_diffby=["Metadata_JCP2022"], batch_size=20000)
        b = np.full(len(m), np.nan)
        b[res.index.to_numpy()] = res["average_precision"].to_numpy()
        both = ~np.isnan(a) & ~np.isnan(b)
        out[f"{name}|{mode}"] = dict(n_gpu=int((~np.isnan(a)).sum()), n_copairs=int((~np.isnan(b)).sum()), n_both=int(both.sum()),
                                      max_abs_diff=float(np.abs(a[both] - b[both]).max()), n_diff_gt_1e6=int((np.abs(a[both] - b[both]) > 1e-6).sum()), p9999_abs_diff=float(np.quantile(np.abs(a[both] - b[both]), 0.9999)), mean_abs_diff=float(np.abs(a[both] - b[both]).mean()), mean_gpu=float(np.nanmean(a)), mean_copairs=float(np.nanmean(b)), seconds_gpu=round(t1, 2))
        print(name, mode, out[f"{name}|{mode}"], flush=True)
(d / "validate_s4.json").write_text(json.dumps(out, indent=1))
