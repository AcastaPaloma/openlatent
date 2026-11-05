"""H3-001: the pre-assembled 10-source COMPOUND table (JUMP cpg0016 `profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect.parquet`, CC0, 2,810,741,997 B,
ETag 32fc704d018039d48503588aac65aa8c-336, 803,853 wells, 737 features) -> the cache format of prep.py (X.npy float32 + meta.parquet + features.txt) under the tag s10.
No recipe step is re-run: the table is pre-built (mad, INT and feature selection over all wells of all ten sources: transductive front end, stated in F.md section 23).
  python prep_assembled10.py --parquet ~/work/data/jump10/profiles_var_mad_int_featselect.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    a = ap.parse_args()
    t0 = time.time()
    f = pq.ParquetFile(a.parquet)
    names = f.schema_arrow.names
    mcols = [n for n in names if n.startswith("Metadata")]
    feat = [n for n in names if not n.startswith("Metadata")]
    tab = f.read()
    meta = tab.select(mcols).to_pandas()
    X = np.empty((tab.num_rows, len(feat)), np.float32)
    for j, c in enumerate(feat):
        X[:, j] = tab.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    print(f"read {X.shape} in {time.time() - t0:.0f}s", flush=True)
    log = dict(parquet=str(a.parquet), n_wells=int(X.shape[0]), n_features=len(feat), n_nonfinite=int((~np.isfinite(X)).sum()))
    pl = pd.read_csv(C.META / "plate.csv.gz")
    pl["Metadata_Plate"] = pl["Metadata_Plate"].astype(str)
    meta["Metadata_Plate"] = meta["Metadata_Plate"].astype(str)
    meta = meta.merge(pl[["Metadata_Source", "Metadata_Plate", "Metadata_Batch", "Metadata_PlateType"]].drop_duplicates(["Metadata_Source", "Metadata_Plate"]),
                      on=["Metadata_Source", "Metadata_Plate"], how="left")
    log["plate_type_missing"] = int(meta["Metadata_PlateType"].isna().sum())
    log["plate_types"] = meta["Metadata_PlateType"].value_counts(dropna=False).to_dict()
    meta["Metadata_JCP2022"] = meta["Metadata_JCP2022"].map(lambda x: C.MAPPER.get(x, x))
    keep = (~meta["Metadata_JCP2022"].isin(C.DROP_IDS)).to_numpy()
    log["n_dropped_control_ids"] = int((~keep).sum())
    meta, X = meta[keep].reset_index(drop=True), X[keep]
    meta["PertType"] = C.pert_type(meta["Metadata_JCP2022"].to_numpy())
    log["pert_types"] = meta["PertType"].value_counts().to_dict()
    log["wells_per_source"] = meta["Metadata_Source"].value_counts().sort_index().to_dict()
    out = C.cache_dir("s10")
    np.save(out / "X.npy", np.ascontiguousarray(X, np.float32))
    meta.to_parquet(out / "meta.parquet")
    (out / "features.txt").write_text("\n".join(feat))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "prep_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps(log, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
