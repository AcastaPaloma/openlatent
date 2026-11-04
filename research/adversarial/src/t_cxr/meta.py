"""ChestX-ray14 metadata for R-CXR (Builder-T).  Pure pandas/numpy; no images, no torch.

Splits are the OFFICIAL patient-disjoint lists (train_val_list.txt / test_list.txt, the same files PriCheXy-Net ships);
train_val patients are split further into attacker-train (`tr`) and attacker-validation (`va`) by SHA-256 of the patient id,
so every assignment is a pure function of ids and a salt (no RNG state, no row-order dependence).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "t_cxr-v1"
COLS = {"Image Index": "img", "Finding Labels": "findings", "Follow-up #": "fu", "Patient ID": "pid", "Patient Age": "age",
        "Patient Gender": "sex", "View Position": "view", "OriginalImage[Width": "W", "Height]": "H",
        "OriginalImagePixelSpacing[x": "sx", "y]": "sy"}
FINDINGS = ("Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia")


PRI = ("image_pairs_training_10000.txt", "image_pairs_validation_2000.txt")


def hid(pid, salt) -> str:
    """hashed patient id used in every uploaded prediction file."""
    return hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()[:12]


def unit(key, salt: str = SALT) -> float:
    """SHA-256 -> [0,1); independent of PYTHONHASHSEED and platform."""
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big") / 2**64


def load(csv: str | Path, test_list: str | Path, trval_list: str | Path, val_frac: float = 0.15) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns=COLS)[list(COLS.values())].copy()
    d["age"] = d["age"].astype(float)
    d = d[d["age"] <= 120].copy()                                   # v2020 csv has none; kept so the older csv behaves
    te = set(Path(test_list).read_text().split())
    tv = set(Path(trval_list).read_text().split())
    d["official"] = np.where(d["img"].isin(te), "test", np.where(d["img"].isin(tv), "trval", "none"))
    if (d["official"] == "none").any():
        raise ValueError("images in neither official list")
    pt = d.loc[d.official == "trval", "pid"].unique()
    va = {p for p in pt if unit(p, SALT + ":val") < val_frac}
    d["split"] = np.where(d.official == "test", "te", np.where(d["pid"].isin(va), "va", "tr"))
    if set(d.loc[d.split == "te", "pid"]) & set(d.loc[d.split != "te", "pid"]):
        raise ValueError("official test list is not patient-disjoint")
    d = d.sort_values(["pid", "fu", "img"], kind="stable").reset_index(drop=True)
    d["male"] = (d["sex"].str.strip().str.upper().str[0] == "M").astype(np.int8)
    d["pa"] = (d["view"].str.strip() == "PA").astype(np.int8)
    d["row"] = np.arange(len(d))
    return d


def seen_patients(pair_files, d: pd.DataFrame) -> set:
    """Patients that appear in PriCheXy-Net's training or validation pair lists = patients the released generator was fitted / early-stopped on."""
    row = dict(zip(d["img"], d["pid"]))
    seen = set()
    for f in pair_files:
        for line in Path(f).read_text().split("\n"):
            t = line.split()
            if len(t) >= 2:
                seen.add(row[t[0]]); seen.add(row[t[1]])
    return seen


def assign_roles(d: pd.DataFrame, seen: set, salt: str, rot: int = 0, val_share: float = 0.39) -> pd.DataFrame:
    """Three-way patient-disjoint roles (Auditor-2 F4 / FOCUS A2):
       S  defender-SEEN (released generator trained / early-stopped on them)   -> sensitivity arm and attacker-scaling rungs only
       among the generator-UNSEEN patients (multi- and single-visit alike), three hash buckets b = floor(3 * unit(pid, salt:rot)):
       T  sealed test = bucket `rot`;  V  validation = `val_share` of bucket (rot+1) % 3;  A  attacker-train = everything else unseen.
    `salt` is committed by the sealed-split custodian (main agent) AFTER the floor list is frozen; rot in {0,1,2} rotates the test units."""
    d = d.copy()
    bucket = d["pid"].map(lambda p: int(3 * unit(p, f"{salt}:rot")))
    u2 = d["pid"].map(lambda p: unit(p, f"{salt}:val"))
    d["role"] = np.where(d["pid"].isin(seen), "S", np.where(bucket == rot, "T", np.where((bucket == (rot + 1) % 3) & (u2 < val_share), "V", "A")))
    return d


def role_counts(d: pd.DataFrame) -> dict:
    n = d.groupby("pid").size()
    out = {}
    for r in ("S", "A", "V", "T"):
        p = d.loc[d.role == r, "pid"].unique()
        out[r] = dict(images=int((d.role == r).sum()), patients=int(len(p)), ge2=int((n[p] >= 2).sum()))
    return out


def counts(d: pd.DataFrame) -> dict:
    """Published-count audit: images, patients, multi-visit patients per split (feeds the power audit)."""
    out = {}
    n = d.groupby("pid").size()
    for s in ("tr", "va", "te"):
        p = d.loc[d.split == s, "pid"].unique()
        out[s] = dict(images=int((d.split == s).sum()), patients=int(len(p)), ge2=int((n[p] >= 2).sum()), ge3=int((n[p] >= 3).sum()))
    out["all"] = dict(images=len(d), patients=int(d.pid.nunique()), ge2=int((n >= 2).sum()), ge3=int((n >= 3).sum()))
    return out
