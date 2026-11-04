"""F / JS-1 (JUMP Cell Painting, floors only): paths, controls, compound universe, md5 splits, plate-disjoint halves.

Everything that must be fixed BEFORE any model number is fixed here and mirrored in results_notes/F.md (pre-registration).
Control definitions follow the Arevalo loader (carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection, BSD-3-Clause).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
META = Path(os.environ.get("F_META", HOME / "work/data/jump_meta"))
RAW = Path(os.environ.get("F_RAW", HOME / "work/data/f_jump/raw"))
CACHE = Path(os.environ.get("F_CACHE", HOME / "work/cache/f_jump"))

# Arevalo `MAPPER`: controls get readable names; ids that do not start with JCP are positive controls.
MAPPER = {
    "JCP2022_085227": "Aloxistatin", "JCP2022_037716": "AMG900", "JCP2022_025848": "Dexamethasone", "JCP2022_046054": "FK-866",
    "JCP2022_035095": "LY2109761", "JCP2022_064022": "NVS-PAK1-1", "JCP2022_050797": "Quinidine", "JCP2022_012818": "TC-S-7004",
    "JCP2022_033924": "DMSO", "JCP2022_999999": "UNTREATED", "JCP2022_UNKNOWN": "UNKNOWN", "JCP2022_900001": "BAD CONSTRUCT",
}
DROP_IDS = {"UNTREATED", "UNKNOWN", "BAD CONSTRUCT"}
EXCLUDE_FROM_MAP = {"JCP2022_033954"}      # Arevalo: "has many more replicates than any other"
SOURCES5 = ["source_2", "source_3", "source_6", "source_8", "source_10"]
SALTS = ["s0", "s1", "s2"]


def cache_dir(tag):
    d = CACHE / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def pert_type(jcp):
    s = pd.Series(jcp)
    out = pd.Series("trt", index=s.index)
    out[~s.str.startswith("JCP")] = "poscon"
    out[s == "DMSO"] = "negcon"
    return out.to_numpy()


def split_role(jcp_ids, salt):
    """'H' (held out), 'V' (validation), 'T' (fit) for each compound id.  h = int(md5(id + salt)); H if odd; else V if (h>>1)%5==0."""
    out = []
    for j in jcp_ids:
        h = int(hashlib.md5((str(j) + salt).encode()).hexdigest(), 16)
        out.append("H" if h & 1 else ("V" if (h >> 1) % 5 == 0 else "T"))
    return np.array(out)


def plate_halves(plate, source, salt, seed=0):
    """0/1 half label per well, plate-disjoint; plates alternate within each source after a seeded shuffle (every source in both halves)."""
    rng = np.random.RandomState(abs(hash(salt)) % (2 ** 31) if False else int(hashlib.md5(("half" + salt + str(seed)).encode()).hexdigest()[:8], 16))
    half = np.zeros(len(plate), np.int8)
    for s in np.unique(source):
        ps = np.unique(plate[source == s])
        ps = ps[rng.permutation(len(ps))]
        m = {p: i % 2 for i, p in enumerate(ps)}
        idx = np.where(source == s)[0]
        half[idx] = np.array([m[p] for p in plate[idx]], np.int8)
    return half
