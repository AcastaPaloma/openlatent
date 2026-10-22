"""Fake ChestX-ray14-shaped metadata and features for tests and smoke runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openlatent.data.chestxray14 import FINDINGS


def fake_metadata(n_patients: int = 300, max_images: int = 6, seed: int = 0) -> pd.DataFrame:
    """Metadata with the normalized columns produced by ``load_metadata``."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(1, n_patients + 1):
        sex = "M" if rng.random() < 0.55 else "F"
        age = float(rng.integers(18, 90))
        for followup in range(int(rng.integers(1, max_images + 1))):
            present = [f for f in FINDINGS if rng.random() < 0.08]
            rows.append(
                {
                    "image_id": f"{patient:08d}_{followup:03d}.png",
                    "findings": "|".join(present) or "No Finding",
                    "followup": followup,
                    "patient_id": patient,
                    "age": age,
                    "sex": sex,
                    "view": "PA",
                }
            )
    return pd.DataFrame(rows).sort_values("image_id", ignore_index=True)


def fake_features(meta: pd.DataFrame, dim: int = 32, identity: float = 1.0, noise: float = 0.5, seed: int = 0) -> np.ndarray:
    """Features = identity * per-patient vector + sex/age signal + Gaussian noise.

    ``identity=0`` gives features that carry no patient identity beyond sex and age.
    """
    rng = np.random.default_rng(seed)
    patients = meta["patient_id"].to_numpy()
    patient_vecs = {p: rng.normal(size=dim) for p in np.unique(patients)}
    base = np.stack([patient_vecs[p] for p in patients]) * identity
    base[:, 0] += np.where(meta["sex"].to_numpy() == "M", 1.0, -1.0)
    base[:, 1] += (meta["age"].to_numpy() - 50.0) / 20.0
    return (base + noise * rng.normal(size=base.shape)).astype(np.float32)
