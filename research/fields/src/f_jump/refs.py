"""Transductive references (use the source id and all wells of the scenario, incl. held-out compounds): never counted as wins.

  python refs.py --scenario 4 [--only centre,swhiten64,combat,harmony]

T_centre: per-source mean-centring.  T_swhiten64: PCA-64 of all wells, then per-source full whitening.  T_combat: scanpy combat (key Metadata_Source).
T_harmony: harmonypy on the D-dim features with the benchmark's settings (max_iter_harmony=20, nclust=300, batch = Metadata_Source).
Written as float32 .npy under <cache>/<tag>/refs/.  (Sphering is fitted inside evaluate.py because its epsilon is chosen on the split's V.)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def whiten(Zb):
    mu = Zb.mean(0)
    Cv = np.cov((Zb - mu).T) + 1e-3 * np.eye(Zb.shape[1])
    w, V = np.linalg.eigh(Cv)
    return ((Zb - mu) @ ((V / np.sqrt(w)) @ V.T)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--only", default="centre,swhiten64,combat,harmony")
    a = ap.parse_args()
    d = C.cache_dir(f"s{a.scenario}")
    out = d / "refs"
    out.mkdir(exist_ok=True)
    X = np.load(d / "X.npy")
    src = pd.read_parquet(d / "meta.parquet")["Metadata_Source"].to_numpy()
    only = a.only.split(",")
    t0 = time.time()
    if "centre" in only:
        Y = X.copy()
        for s in np.unique(src):
            m = src == s
            Y[m] -= Y[m].mean(0)
        np.save(out / "T_centre.npy", Y)
        print(f"centre {time.time() - t0:.0f}s", flush=True)
    if "swhiten64" in only:
        mu = X.mean(0)
        _, S, Vt = np.linalg.svd(np.cov((X[:: max(1, len(X) // 100000)] - mu).T), full_matrices=False)
        V = Vt[:64].T
        Z = ((X - mu) @ V).astype(np.float32)
        Y = np.empty_like(Z)
        for s in np.unique(src):
            m = src == s
            Y[m] = whiten(Z[m])
        np.save(out / "T_swhiten64.npy", Y)
        print(f"swhiten64 {time.time() - t0:.0f}s", flush=True)
    if "combat" in only:
        import anndata as ad
        import scanpy as sc
        A = ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"Metadata_Source": pd.Categorical(src)}, index=np.arange(len(X)).astype(str)))
        Y = sc.pp.combat(A, key="Metadata_Source", inplace=False)
        np.save(out / "T_combat.npy", np.asarray(Y, np.float32))
        print(f"combat {time.time() - t0:.0f}s", flush=True)
    if "harmony" in only:
        import harmonypy
        ho = harmonypy.run_harmony(X.astype(np.float32), pd.DataFrame({"Metadata_Source": src}), "Metadata_Source", max_iter_harmony=20, nclust=300, verbose=False, random_state=0)
        Z = np.asarray(ho.Z_corr)
        Z = Z.T if Z.shape[0] == X.shape[1] and Z.shape[1] == X.shape[0] else Z
        np.save(out / "T_harmony.npy", Z.astype(np.float32))
        print(f"harmony {time.time() - t0:.0f}s shape {Z.shape}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
