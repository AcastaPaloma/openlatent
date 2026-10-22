"""Feature files: an (n, d) .npy matrix plus a sibling .ids.txt with one image ID per row.

Raw inputs x and latents z = E(x) use the same format, so the harness scores both
the same way. Encode once and cache z to disk: that keeps GPU nondeterminism out
of the scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ids_path_for(npy_path: str | Path) -> Path:
    return Path(npy_path).with_suffix(".ids.txt")


def save_features(npy_path: str | Path, features: np.ndarray, image_ids: list[str]) -> None:
    if len(features) != len(image_ids):
        raise ValueError(f"{len(features)} feature rows but {len(image_ids)} IDs")
    np.save(npy_path, np.asarray(features))
    ids_path_for(npy_path).write_text("\n".join(image_ids) + "\n")


def load_features(npy_path: str | Path, ids_path: str | Path | None = None) -> tuple[np.ndarray, list[str]]:
    features = np.load(npy_path)
    ids = [line for line in Path(ids_path or ids_path_for(npy_path)).read_text().splitlines() if line]
    if len(features) != len(ids):
        raise ValueError(f"{npy_path}: {len(features)} rows but {len(ids)} IDs")
    if features.ndim != 2:
        features = features.reshape(len(features), -1)
    return features, ids


def pixel_features(image_dir: str | Path, image_ids: list[str], size: int = 64) -> np.ndarray:
    """Grayscale, bilinear-resized, flattened pixels in [0, 1].

    A deterministic raw-input baseline. It is a weak attacker on chest X-rays, so
    for the headline "before" number use embeddings from a frozen re-ID network
    (Packhauser et al., 2022) saved with ``save_features``.
    """
    from PIL import Image

    paths = {p.name: p for p in Path(image_dir).rglob("*.png")}
    missing = [i for i in image_ids if i not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not under {image_dir}, e.g. {missing[0]}")
    out = np.empty((len(image_ids), size * size), dtype=np.float32)
    for row, image_id in enumerate(image_ids):
        with Image.open(paths[image_id]) as image:
            small = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        out[row] = np.asarray(small, dtype=np.float32).ravel() / 255.0
    return out
