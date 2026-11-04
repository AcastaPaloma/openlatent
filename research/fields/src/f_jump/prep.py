"""F / JS-1: scenario profiles -> encoder 0 (Arevalo `mad_int_featselect`), cached as X.npy (float32) + meta.parquet.

  python prep.py --scenario 4 [--plates s2]        (plates s2 = the S2 subset of the S4 download, for the R0 gate)

Re-implementation (same order as their Snakemake rules; BSD-3 recipe by Arevalo et al. 2024) of:
  write_parquet (merge well metadata, drop UNTREATED/UNKNOWN/BAD CONSTRUCT, drop wells with no metadata)
  compute_negcon_stats (per-plate DMSO median and MAD, scipy scale=1, columns without NaN/inf)
  select_variant_features (mad != 0 and |mad/median| > 1e-3 in EVERY plate with DMSO stats; plates without DMSO stats are dropped)
  mad_normalize (x - median_plate) / mad_plate
  INT (rank-based inverse normal, ordinal ties, seed 0, per feature over all wells of the scenario)
  featselect (pycytominer variance_threshold, then correlation_threshold, defaults)
No metric is computed here.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import plates as PL  # noqa: E402


def load_meta(sc):
    pl = PL.plate_table(C.META, sc)
    w = pd.read_csv(C.META / "well.csv.gz")
    w["Metadata_JCP2022"] = w["Metadata_JCP2022"].map(lambda x: C.MAPPER.get(x, x))
    w = w[~w["Metadata_JCP2022"].isin(C.DROP_IDS)]
    return pl, w


def load_raw(pl, workers=16):
    paths = [C.RAW / f"{p}.parquet" for p in pl.Metadata_Plate]
    cols0 = pq.ParquetFile(paths[0]).schema.names
    feat = [c for c in cols0 if not c.startswith("Meta")]
    counts = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    off = np.concatenate([[0], np.cumsum(counts)])
    X = np.empty((off[-1], len(feat)), np.float32)
    meta = np.empty((off[-1], 3), dtype=object)

    def rd(i):
        d = pd.read_parquet(paths[i])
        X[off[i]:off[i + 1]] = d[feat].to_numpy(np.float32)
        meta[off[i]:off[i + 1]] = d[["Metadata_Source", "Metadata_Plate", "Metadata_Well"]].to_numpy()

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(rd, range(len(paths))))
    return pd.DataFrame(meta, columns=["Metadata_Source", "Metadata_Plate", "Metadata_Well"]), X, feat


def rank_int_array(a, c=3.0 / 8, seed=0):
    rng = np.random.default_rng(seed=seed)
    ix = rng.permutation(len(a))
    rev = np.argsort(ix)
    rank = ss.rankdata(a[ix], method="ordinal")[rev]
    return ss.norm.ppf((rank - c) / (len(rank) - 2 * c + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--plates", default="", help="s2: restrict to the scenario-2 plates (R0 gate); tag becomes s2")
    ap.add_argument("--skip_featselect", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = dict(scenario=a.scenario)
    pl, wmeta = load_meta(a.scenario)
    tag = f"s{a.scenario}"
    if a.plates == "s2":
        pl = PL.plate_table(C.META, 2)
        tag = "s2"
    out = C.cache_dir(tag)
    log["n_plates"] = len(pl)
    meta, X, feat = load_raw(pl)
    print(f"loaded {X.shape} in {time.time() - t0:.0f}s", flush=True)
    # merge metadata, drop wells with no metadata
    m = meta.merge(pl[["Metadata_Source", "Metadata_Plate", "Metadata_Batch", "Metadata_PlateType"]], on=["Metadata_Source", "Metadata_Plate"], how="left")
    m = m.merge(wmeta, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"], how="left")
    keep = m["Metadata_JCP2022"].notna().to_numpy()
    m, X = m[keep].reset_index(drop=True), X[keep]
    log["n_wells_raw"], log["n_features_raw"] = int(keep.size), len(feat)
    log["n_wells_with_meta"] = int(X.shape[0])
    # negcon stats on columns without NaN / inf
    fin = np.isfinite(X).all(0)
    log["n_feat_finite"] = int(fin.sum())
    X, feat = X[:, fin], [f for f, k in zip(feat, fin) if k]
    plate = m["Metadata_Plate"].to_numpy()
    neg = (m["Metadata_JCP2022"] == "DMSO").to_numpy()
    plates_all = np.unique(plate)
    med, mad, havenegs = {}, {}, []
    for p in plates_all:
        i = np.where((plate == p) & neg)[0]
        if len(i) == 0:
            continue
        med[p] = np.median(X[i], 0)
        mad[p] = ss.median_abs_deviation(X[i], axis=0, nan_policy="omit")      # scale=1, as in the benchmark repo
        havenegs.append(p)
    log["n_plates_with_dmso"] = len(havenegs)
    okf = np.ones(len(feat), bool)
    for p in havenegs:
        cv = np.abs(np.nan_to_num(mad[p] / med[p], nan=0.0, posinf=0.0, neginf=0.0))
        okf &= (mad[p] != 0) & (cv > 1e-3)
    log["n_variant_features"] = int(okf.sum())
    inplate = np.isin(plate, havenegs)
    log["n_wells_dropped_no_dmso_plate"] = int((~inplate).sum())
    X, m, plate = X[inplate][:, okf], m[inplate].reset_index(drop=True), plate[inplate]
    feat = [f for f, k in zip(feat, okf) if k]
    for p in havenegs:
        i = plate == p
        X[i] = (X[i] - med[p][okf]) / mad[p][okf]
    print(f"mad done, {X.shape}, {time.time() - t0:.0f}s", flush=True)
    # INT
    def to_normal(j):
        X[:, j] = rank_int_array(X[:, j]).astype(np.float32)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(to_normal, range(X.shape[1])))
    print(f"int done, {time.time() - t0:.0f}s", flush=True)
    # feature selection (pycytominer)
    if not a.skip_featselect:
        from pycytominer.operations import correlation_threshold, variance_threshold
        df = pd.DataFrame(X, columns=feat)
        low = variance_threshold(df, feat)
        feat2 = [f for f in feat if f not in low]
        high = correlation_threshold(df, feat2)
        drop = set(low) | set(high)
        log["n_removed_variance"], log["n_removed_correlation"] = len(low), len(high)
        kept = np.array([f not in drop for f in feat])
        X, feat = X[:, kept], [f for f, k in zip(feat, kept) if k]
    log["n_features_final"], log["n_wells_final"] = len(feat), int(X.shape[0])
    m["PertType"] = C.pert_type(m["Metadata_JCP2022"].to_numpy())
    np.save(out / "X.npy", np.ascontiguousarray(X, np.float32))
    m.to_parquet(out / "meta.parquet")
    (out / "features.txt").write_text("\n".join(feat))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "prep_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1), flush=True)


if __name__ == "__main__":
    main()
