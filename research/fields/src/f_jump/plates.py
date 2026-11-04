"""F / JUMP stage 1: plate lists per Arevalo scenario (their metadata rules) and the S3 download list.  Metadata only, no feature values.

  python plates.py --meta ~/work/data/jump_meta --scenario 4 --list        # prints plate keys
  python plates.py --meta ~/work/data/jump_meta --scenario 4 --keys > keys.txt

Rules copied from carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection preprocessing/metadata.py (BSD-3-Clause):
plates of the scenario's sources and plate types; source_3 batches in SOURCE3_BATCH_REDLIST are dropped except TARGET2 plates.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SOURCE3_BATCH_REDLIST = {"CP_32_all_Phenix1", "CP_33_all_Phenix1", "CP_34_mix_Phenix1", "CP_35_all_Phenix1", "CP_36_all_Phenix1", "CP59", "CP60"}
SCENARIOS = {
    1: dict(sources=["source_6"], types=["TARGET2"]),
    2: dict(sources=["source_2", "source_6", "source_10"], types=["TARGET2"]),
    3: dict(sources=["source_2", "source_6", "source_10"], types=["COMPOUND", "TARGET2"]),
    4: dict(sources=["source_2", "source_3", "source_6", "source_8", "source_10"], types=["TARGET2"]),
    5: dict(sources=["source_2", "source_3", "source_6", "source_8", "source_10"], types=["COMPOUND", "TARGET2"]),
}


def plate_table(meta_dir, sc):
    pl = pd.read_csv(Path(meta_dir) / "plate.csv.gz")
    pl = pl[(~pl.Metadata_Batch.isin(SOURCE3_BATCH_REDLIST)) | (pl.Metadata_PlateType == "TARGET2")]
    pl = pl[pl.Metadata_Source.isin(SCENARIOS[sc]["sources"]) & pl.Metadata_PlateType.isin(SCENARIOS[sc]["types"])]
    return pl.sort_values(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]).reset_index(drop=True)


def key_of(r):
    return f"cpg0016-jump/{r.Metadata_Source}/workspace/profiles/{r.Metadata_Batch}/{r.Metadata_Plate}/{r.Metadata_Plate}.parquet"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--keys", action="store_true")
    a = ap.parse_args()
    t = plate_table(a.meta, a.scenario)
    if a.keys:
        for r in t.itertuples():
            print(key_of(r))
    else:
        print(t.groupby(["Metadata_Source", "Metadata_PlateType"]).agg(plates=("Metadata_Plate", "nunique"), batches=("Metadata_Batch", "nunique")))
        print("plates", len(t), file=sys.stderr)
