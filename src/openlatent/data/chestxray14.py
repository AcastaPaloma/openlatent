"""NIH ChestX-ray14 metadata and patient-disjoint splits.

Every assignment here is a pure function of patient IDs, image IDs and a salt,
so the same metadata always yields the same partition, regardless of row order
or which subset of images is present.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

DEFAULT_SALT = "openlatent-v1"

# Data_Entry_2017.csv and Data_Entry_2017_v2020.csv use slightly different headers.
_COLUMNS = {
    "image_id": ("Image Index",),
    "findings": ("Finding Labels",),
    "followup": ("Follow-up #",),
    "patient_id": ("Patient ID",),
    "age": ("Patient Age",),
    "sex": ("Patient Sex", "Patient Gender"),
    "view": ("View Position",),
}


def _parse_age(value: object) -> float:
    """Parse ages given as ints or strings like '058Y', '010M', '020D'; >120 years becomes NaN."""
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d+)([YMD]?)", text)
    if match is None:
        return float("nan")
    years = int(match.group(1)) / {"": 1, "Y": 1, "M": 12, "D": 365}[match.group(2)]
    return years if years <= 120 else float("nan")


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load Data_Entry_2017*.csv into normalized columns:
    image_id, findings, followup, patient_id, age, sex, view."""
    raw = pd.read_csv(csv_path)
    out = {}
    for name, aliases in _COLUMNS.items():
        column = next((a for a in aliases if a in raw.columns), None)
        if column is None:
            raise ValueError(f"{csv_path}: none of {aliases} found in columns")
        out[name] = raw[column]
    meta = pd.DataFrame(out)
    meta["image_id"] = meta["image_id"].astype(str)
    meta["patient_id"] = meta["patient_id"].astype(int)
    meta["followup"] = meta["followup"].astype(int)
    meta["age"] = meta["age"].map(_parse_age).astype(float)
    meta["sex"] = meta["sex"].astype(str).str.strip().str.upper().str[0]
    if meta["image_id"].duplicated().any():
        raise ValueError(f"{csv_path}: duplicate image IDs")
    return meta.sort_values("image_id", ignore_index=True)


def findings_matrix(findings: pd.Series) -> np.ndarray:
    """Multi-hot (n, 14) matrix in FINDINGS order from pipe-separated label strings."""
    labels = findings.astype(str).str.split("|")
    return np.array([[name in row for name in FINDINGS] for row in labels], dtype=np.uint8)


def read_id_list(path: str | Path) -> list[str]:
    """Read an official split file such as test_list.txt (one image ID per line)."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def stable_unit(key: object, salt: str) -> float:
    """Map a key to [0, 1) with SHA-256, independent of PYTHONHASHSEED and platform."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def make_splits(
    meta: pd.DataFrame,
    test_ids: list[str] | None = None,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    probe_fit_frac: float = 0.5,
    salt: str = DEFAULT_SALT,
) -> pd.DataFrame:
    """Assign every image to a patient-disjoint split and evaluation roles.

    Columns of the result:
      split      encoder_train | encoder_val | eval. With ``test_ids`` (the official
                 test_list.txt, which NIH made patient-wise), eval is exactly those
                 images; otherwise ``test_frac`` of patients by hash.
      probe_role fit | test for eval images (patient-disjoint), else empty. Convex
                 probes are fit on one half of the eval patients and scored on the other.
      reid_role  gallery | query for eval patients with 2+ images, else empty. The
                 gallery holds each patient's earliest image; every later image is a query.
    """
    meta = meta.reset_index(drop=True)
    patients = pd.Series(meta["patient_id"].unique())

    if test_ids is not None:
        is_test_image = meta["image_id"].isin(set(test_ids))
        test_patients = set(meta.loc[is_test_image, "patient_id"])
        leaked = set(meta.loc[~is_test_image, "patient_id"]) & test_patients
        if leaked:
            raise ValueError(f"test list is not patient-disjoint: {len(leaked)} patients on both sides")
    else:
        test_patients = {p for p in patients if stable_unit(p, f"{salt}:test") < test_frac}
    val_patients = {
        p for p in patients if p not in test_patients and stable_unit(p, f"{salt}:val") < val_frac
    }

    pid = meta["patient_id"]
    split = np.where(
        pid.isin(test_patients), "eval", np.where(pid.isin(val_patients), "encoder_val", "encoder_train")
    )
    is_eval = split == "eval"

    probe_fit = pid.map(lambda p: stable_unit(p, f"{salt}:probe") < probe_fit_frac).to_numpy()
    probe_role = np.where(is_eval, np.where(probe_fit, "fit", "test"), "")

    reid_role = np.full(len(meta), "", dtype=object)
    eval_rows = meta[is_eval].sort_values(["patient_id", "followup", "image_id"])
    counts = eval_rows["patient_id"].map(eval_rows["patient_id"].value_counts())
    multi = eval_rows[counts >= 2]
    first = ~multi["patient_id"].duplicated()
    reid_role[multi.index[first.to_numpy()]] = "gallery"
    reid_role[multi.index[~first.to_numpy()]] = "query"

    return pd.DataFrame(
        {
            "image_id": meta["image_id"].to_numpy(),
            "patient_id": pid.to_numpy(),
            "split": split,
            "probe_role": probe_role,
            "reid_role": reid_role,
        }
    )


def check_splits(splits: pd.DataFrame) -> None:
    """Raise if any patient appears in more than one split or probe half."""
    for column in ("split", "probe_role"):
        rows = splits if column == "split" else splits[splits["split"] == "eval"]
        per_patient = rows.groupby("patient_id")[column].nunique()
        if (per_patient > 1).any():
            raise ValueError(f"{(per_patient > 1).sum()} patients span several values of {column!r}")
