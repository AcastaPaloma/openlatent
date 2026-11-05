"""Near-duplicate pair list for H3-007 stage 1 (CPU): production compounds of the S5 universe with Morgan-2 (2048 bit) Tanimoto >= 0.8 or the same InChIKey connectivity block.
Writes nd_pairs.json = {'ids': [...], 'pairs': [[i, j], ...]} (indices into ids).  Same construction as ctrl_position.py (B0).
  CUDA_VISIBLE_DEVICES= python nd_pairs.py --out $HOME/work/f5_ctrl_position_meta/nd_pairs.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import evaluate as EV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import scipy.sparse as sps
import torch
RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(a.threads)
meta = pd.read_parquet(C.CACHE / "s5/meta.parquet")
U = EV.universe(meta, True)
ids = np.unique(U["jcp"][U["inu"]])
comp = pd.read_csv(C.META / "compound.csv.gz").drop_duplicates("Metadata_JCP2022").set_index("Metadata_JCP2022")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rows, cols, keys = [], [], []
for i, j in enumerate(ids):
    r = comp.loc[j]
    keys.append(r["Metadata_InChIKey"][:14] if isinstance(r["Metadata_InChIKey"], str) else None)
    mol = Chem.MolFromSmiles(r["Metadata_SMILES"]) if isinstance(r["Metadata_SMILES"], str) else None
    if mol is None and isinstance(r["Metadata_InChI"], str):
        mol = Chem.MolFromInchi(r["Metadata_InChI"])
    if mol is not None:
        on = list(gen.GetFingerprint(mol).GetOnBits())
        rows += [i] * len(on)
        cols += on
n = len(ids)
A = sps.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, 2048))
cnt = torch.as_tensor(np.asarray(A.sum(1)).ravel(), dtype=torch.float32)
AT = torch.as_tensor(A.T.toarray())
pairs = set()
for a0 in range(0, n, 400):
    blk = torch.as_tensor(A[a0:a0 + 400].toarray())
    inter = blk @ AT
    sim = inter / (cnt[a0:a0 + 400, None] + cnt[None, :] - inter).clamp(min=1)
    for r in range(sim.shape[0]):
        sim[r, : a0 + r + 1] = 0
    ii, jj = torch.where(sim >= 0.8)
    for x, y in zip(ii.tolist(), jj.tolist()):
        if cnt[a0 + x] > 0 and cnt[y] > 0:
            pairs.add((a0 + x, y))
kg = pd.Series(keys).dropna()
for _, g in kg.groupby(kg):
    g = list(g.index)
    for x in range(len(g)):
        for y in range(x + 1, len(g)):
            pairs.add((g[x], g[y]))
Path(a.out).write_text(json.dumps(dict(ids=[str(x) for x in ids], pairs=sorted(map(list, pairs)))))
print("pairs", len(pairs), flush=True)
