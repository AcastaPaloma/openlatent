"""Loaders for programme 1's HAR harness predictions (HF KuanKuanKuan/openlatent-privacy, harness/<dataset>/eval/<release>/preds.npz).

preds.npz keys: 'secret|<secret>|<protocol>|<retrained|frozen>|<arch>' -> (n_test, n_classes) class probabilities, rows in the order of
result.json['secrets'][secret][protocol]['test_ids']. dataset.json maps item id -> labels and groups (recording = groups['source'],
scenario / scene / trial). Row i of every attacker of one (release, protocol) refers to the same item.

Independent unit used by S for the harness: the RECORDING (groups['source'], e.g. person01_boxing_d3 for KTH), not the clip.
Session used by the evidence-budget module: KTH person x scenario, Charades identity x scene; UTD-MHAD has no session variable (one sitting
per subject), so the session is the person and the independence test is unavailable there.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Cell:
    dataset: str
    release: str
    protocol: str
    ids: List[str]
    y: np.ndarray
    record: np.ndarray
    session: np.ndarray
    probs: Dict[str, np.ndarray]     # 'retrained|r3d' -> (n, C)

    @property
    def n_units(self) -> int:
        return len(np.unique(self.record))


_INDEX_CACHE: Dict[str, dict] = {}


def dataset_index(root: str, ds: str) -> dict:
    if (root, ds) not in _INDEX_CACHE:
        d = json.load(open(os.path.join(root, ds, "dataset.json")))
        _INDEX_CACHE[(root, ds)] = {it["id"]: it for it in d["items"]}
    return _INDEX_CACHE[(root, ds)]


def releases(root: str, ds: str) -> List[str]:
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, ds, "eval", "*", "preds.npz")))


def _session_of(ds: str, it: dict, secret_label: int) -> str:
    g = it.get("groups", {})
    if ds.startswith("kth"):
        return f"p{secret_label}_{g.get('scenario')}"
    if ds == "charades":
        return f"p{secret_label}_{g.get('scene')}"
    return f"p{secret_label}"


def load_cell(root: str, ds: str, release: str, protocol: str = "within", kinds=("retrained",), secret: str = "identity") -> Cell:
    d = os.path.join(root, ds, "eval", release)
    z = np.load(os.path.join(d, "preds.npz"))
    r = json.load(open(os.path.join(d, "result.json")))
    ids = r["secrets"][secret][protocol]["test_ids"]
    idx = dataset_index(root, ds)
    items = [idx[i] for i in ids]
    y = np.array([it["labels"][secret] for it in items], dtype=int)
    record = np.array([it["groups"]["source"] for it in items])
    session = np.array([_session_of(ds, it, int(lab)) for it, lab in zip(items, y)])
    probs = {}
    for k in z.files:
        p = k.split("|")
        if len(p) == 5 and p[0] == "secret" and p[1] == secret and p[2] == protocol and p[3] in kinds:
            probs[f"{p[3]}|{p[4]}"] = z[k]
    for k, v in probs.items():
        if v.shape[0] != len(ids):
            raise ValueError(f"{ds}/{release}/{k}: {v.shape[0]} rows for {len(ids)} test ids")
    return Cell(ds, release, protocol, list(ids), y, record, session, probs)


def correctness(cell: Cell) -> Dict[str, np.ndarray]:
    return {k: (v.argmax(1) == cell.y).astype(float) for k, v in cell.probs.items()}


def family_pool(root: str, ds: str, release_names: List[str], protocol: str = "within", kinds=("retrained",)):
    """Stack the correctness of every attacker of several releases scored on the SAME test items (e.g. a release and its seed / epoch /
    DINO variants). Releases whose test ids differ are rejected. -> (correct (n,P), names, Cell of the first release)."""
    cells = [load_cell(root, ds, r, protocol, kinds) for r in release_names]
    base = cells[0]
    cols, names = [], []
    for c in cells:
        if c.ids != base.ids:
            raise ValueError(f"{c.release} has different test items than {base.release}")
        for k, v in correctness(c).items():
            cols.append(v)
            names.append(f"{c.release}::{k}")
    return np.stack(cols, axis=1), names, base
