"""Training-free re-identification by cosine nearest neighbour.

Ties are resolved by their expectation under a uniformly random tie-break, so a
constant encoder (every z identical) scores exactly chance rather than 100%.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma
from sklearn.metrics import roc_auc_score


def _harmonic(n: np.ndarray) -> np.ndarray:
    return digamma(np.asarray(n, dtype=np.float64) + 1.0) + np.euler_gamma


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def reid_scores(
    gallery: np.ndarray,
    gallery_ids: np.ndarray,
    query: np.ndarray,
    query_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    n_impostors: int = 20,
    seed: int = 0,
    chunk: int = 1024,
) -> dict:
    """Closed-set re-ID: rank the gallery for each query by cosine similarity.

    The gallery must hold exactly one item per identity, and every query identity
    must be in the gallery. Verification AUC uses each query's genuine pair plus
    ``n_impostors`` impostor pairs drawn with a fixed seed.
    """
    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)
    if len(np.unique(gallery_ids)) != len(gallery_ids):
        raise ValueError("gallery must contain one item per identity")
    position = {g: i for i, g in enumerate(gallery_ids)}
    missing = [q for q in query_ids if q not in position]
    if missing:
        raise ValueError(f"{len(missing)} queries have no gallery identity")
    true_idx = np.array([position[q] for q in query_ids])

    g = _l2_normalize(gallery)
    q = _l2_normalize(query)
    n_g = len(g)
    n_imp = min(n_impostors, n_g - 1)
    rng = np.random.default_rng(seed)

    topk = {k: [] for k in ks}
    mrr, genuine, impostor = [], [], []
    for start in range(0, len(q), chunk):
        rows = slice(start, start + chunk)
        sims = q[rows] @ g.T
        t = true_idx[rows]
        true_sim = sims[np.arange(len(t)), t]
        greater = (sims > true_sim[:, None]).sum(axis=1)
        equal = (sims == true_sim[:, None]).sum(axis=1)  # includes the true match
        for k in ks:
            topk[k].append(np.clip((k - greater) / equal, 0.0, 1.0))
        mrr.append((_harmonic(greater + equal) - _harmonic(greater)) / equal)

        # Impostors: uniform over the other n_g - 1 gallery items.
        draw = rng.integers(0, n_g - 1, size=(len(t), n_imp))
        draw += draw >= t[:, None]
        genuine.append(true_sim)
        impostor.append(np.take_along_axis(sims, draw, axis=1).ravel())

    genuine_all = np.concatenate(genuine)
    impostor_all = np.concatenate(impostor)
    labels = np.r_[np.ones_like(genuine_all), np.zeros_like(impostor_all)]
    result = {
        "n_gallery": n_g,
        "n_query": len(q),
        "mrr": float(np.concatenate(mrr).mean()),
        "chance_mrr": float(_harmonic(n_g) / n_g),
        "verification_auc": float(roc_auc_score(labels, np.r_[genuine_all, impostor_all])),
        "chance_verification_auc": 0.5,
    }
    for k in ks:
        result[f"top{k}"] = float(np.concatenate(topk[k]).mean())
        result[f"chance_top{k}"] = min(k, n_g) / n_g
    return result
