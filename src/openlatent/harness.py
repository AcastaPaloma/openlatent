"""Run every deterministic scorer on one feature set, and compare two reports.

Score raw x (or a frozen re-ID embedding of x) and the latent z = E(x) with the
same splits and config; ``compare_reports`` then gives leakage removed and
utility kept, both normalized against chance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from openlatent.data.chestxray14 import FINDINGS, findings_matrix, stable_unit
from openlatent.scorers.dependence import distance_correlation, identity_alignment, linear_cka
from openlatent.scorers.probes import ProbeConfig, binary_probe, multilabel_probe, ridge_probe
from openlatent.scorers.reid import reid_scores

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HarnessConfig:
    ks: tuple[int, ...] = (1, 5, 10)
    n_impostors: int = 20
    seed: int = 0
    dependence_n: int = 2000
    salt: str = "openlatent-v1"
    probe: ProbeConfig = field(default_factory=ProbeConfig)


# (report section, metric, chance key or constant). Higher means more leakage / more utility.
PRIVACY_METRICS = (
    ("reid", "top1", "chance_top1"),
    ("reid", "mrr", "chance_mrr"),
    ("reid", "verification_auc", "chance_verification_auc"),
    ("sex_probe", "auc", "chance"),
    ("age_probe", "r2", "chance"),
    ("identity_alignment", "rbf", 0.0),
)
UTILITY_METRICS = (("findings_probe", "mean_auc", "chance"),)


def _fingerprint(features: np.ndarray, image_ids: list[str]) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(features).tobytes())
    h.update("\n".join(image_ids).encode())
    return h.hexdigest()


def _align(features: np.ndarray, image_ids: list[str], meta: pd.DataFrame, splits: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Eval-split rows of meta+splits, in image_id order, with their feature rows."""
    table = splits[splits["split"] == "eval"].merge(
        meta.drop(columns="patient_id"), on="image_id", how="left", validate="one_to_one"
    )
    row_of = {image_id: i for i, image_id in enumerate(image_ids)}
    missing = [i for i in table["image_id"] if i not in row_of]
    if missing:
        raise ValueError(f"features are missing {len(missing)} eval images, e.g. {missing[0]}")
    table = table.sort_values("image_id", ignore_index=True)
    return np.asarray(features)[[row_of[i] for i in table["image_id"]]], table


def _dependence_rows(table: pd.DataFrame, n: int, salt: str) -> np.ndarray:
    """Up to n rows from whole re-ID patients, picked by hash so it is the same for x and z."""
    reid = table[table["reid_role"] != ""]
    order = sorted(reid["patient_id"].unique(), key=lambda p: stable_unit(p, f"{salt}:dependence"))
    chosen, total = [], 0
    counts = reid["patient_id"].value_counts()
    for patient in order:
        if total + counts[patient] > n:
            break
        chosen.append(patient)
        total += counts[patient]
    return reid.index[reid["patient_id"].isin(chosen)].to_numpy()


def run_report(
    features: np.ndarray,
    image_ids: list[str],
    meta: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: HarnessConfig = HarnessConfig(),
) -> dict:
    """Score one representation on the eval split. Same inputs always give the same report."""
    x, table = _align(features, image_ids, meta, splits)
    x = x.astype(np.float64)

    gallery = (table["reid_role"] == "gallery").to_numpy()
    query = (table["reid_role"] == "query").to_numpy()
    pid = table["patient_id"].to_numpy()
    reid = reid_scores(x[gallery], pid[gallery], x[query], pid[query], cfg.ks, cfg.n_impostors, cfg.seed)

    fit = (table["probe_role"] == "fit").to_numpy()
    test = (table["probe_role"] == "test").to_numpy()
    sex = (table["sex"] == "M").to_numpy().astype(int)
    sex_probe = binary_probe(x[fit], sex[fit], pid[fit], x[test], sex[test], cfg.probe)
    age = table["age"].to_numpy()
    a_fit, a_test = fit & ~np.isnan(age), test & ~np.isnan(age)
    age_probe = ridge_probe(x[a_fit], age[a_fit], pid[a_fit], x[a_test], age[a_test], cfg.probe)
    y = findings_matrix(table["findings"])
    findings_probe = multilabel_probe(x[fit], y[fit], pid[fit], x[test], y[test], FINDINGS, cfg.probe)

    dep = _dependence_rows(table, cfg.dependence_n, cfg.salt)
    alignment = {
        "rbf": identity_alignment(x[dep], pid[dep], "rbf"),
        "linear": identity_alignment(x[dep], pid[dep], "linear"),
        "n": int(len(dep)),
    }

    return {
        "schema": SCHEMA_VERSION,
        "fingerprint": _fingerprint(np.asarray(features), image_ids),
        "dim": int(x.shape[1]),
        "n_eval": int(len(table)),
        "config": json.loads(json.dumps(asdict(cfg))),  # as it reads back from disk
        "reid": reid,
        "sex_probe": sex_probe,
        "age_probe": age_probe,
        "identity_alignment": alignment,
        "findings_probe": findings_probe,
    }


def _chance(section: dict, chance: str | float) -> float:
    return float(section[chance]) if isinstance(chance, str) else float(chance)


def _relative(before: float | None, after: float | None, chance: float, min_margin: float) -> float | None:
    """(after - chance) / (before - chance), or None when before is not clearly above chance."""
    if before is None or after is None or before - chance < min_margin:
        return None
    return (after - chance) / (before - chance)


def compare_reports(before: dict, after: dict, min_margin: float = 0.01) -> dict:
    """Normalize each metric against chance.

    leakage_removed = (before - after) / (before - chance); 1.0 means the attacker
    is reduced to guessing, 0.0 means no change, negative means z leaks more.
    utility_kept    = (after - chance) / (before - chance) and the raw ratio after / before.
    Both are None when ``before`` is less than ``min_margin`` above chance: there
    is no measured signal on x to remove or keep.
    """
    if before["config"] != after["config"]:
        raise ValueError("reports were made with different harness configs")
    privacy, utility = {}, {}
    for section, metric, chance in PRIVACY_METRICS:
        b, a = before[section][metric], after[section][metric]
        c = _chance(before[section], chance)
        rel = _relative(b, a, c, min_margin)
        privacy[f"{section}.{metric}"] = {
            "before": b,
            "after": a,
            "chance": c,
            "leakage_removed": None if rel is None else 1.0 - rel,
        }
    for section, metric, chance in UTILITY_METRICS:
        b, a = before[section][metric], after[section][metric]
        c = _chance(before[section], chance)
        utility[f"{section}.{metric}"] = {
            "before": b,
            "after": a,
            "chance": c,
            "utility_kept": _relative(b, a, c, min_margin),
            "ratio": a / b if a is not None and b else None,
        }
    return {"privacy": privacy, "utility": utility}


def cross_dependence(
    x: np.ndarray,
    x_ids: list[str],
    z: np.ndarray,
    z_ids: list[str],
    meta: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: HarnessConfig = HarnessConfig(),
) -> dict:
    """Linear CKA and distance correlation between x and z on the eval subsample."""
    x_eval, table = _align(x, x_ids, meta, splits)
    z_eval, _ = _align(z, z_ids, meta, splits)
    rows = _dependence_rows(table, cfg.dependence_n, cfg.salt)
    xs, zs = x_eval[rows].astype(np.float64), z_eval[rows].astype(np.float64)
    return {"linear_cka": linear_cka(xs, zs), "distance_correlation": distance_correlation(xs, zs), "n": int(len(rows))}
