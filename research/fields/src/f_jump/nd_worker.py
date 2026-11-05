"""Tanimoto >= 0.8 pair search worker for ctrl_position.py (separate module without torch so that spawned processes stay light)."""
import numpy as np

_FPS = None


def init(fps):
    global _FPS
    _FPS = fps


def chunk(rng):
    from rdkit import DataStructs
    a, b = rng
    out = []
    for i in range(a, b):
        if _FPS[i] is None:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(_FPS[i], _FPS[i + 1:]))
        for k in np.where(sims >= 0.8)[0]:
            if _FPS[i + 1 + k] is not None:
                out.append((i, i + 1 + int(k)))
    return out
