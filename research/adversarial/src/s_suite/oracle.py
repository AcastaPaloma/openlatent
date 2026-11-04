"""Oracle arm: best in-family attacker trained with ~20x the data on the same identities, scored on a large fresh sample."""
from __future__ import annotations

import numpy as np

from .attackers import fit_predict_pool
from .planted import PlantedSpec, features, generate_replicate, raw_view

ORACLE_NAMES = ["lr_raw_c0.1", "lda_raw_auto", "knn5_raw", "lr_feat_c1", "lda_feat_auto", "lgbm_feat", "mlp_feat_64"]


def oracle_accuracy(spec: PlantedSpec, seed: int = 0, units_train: int = 120, units_pop: int = 100):
    rep = generate_replicate(spec, seed, units_train=units_train, units_val=1, units_test=1, units_pop=units_pop)
    vt = {"raw": raw_view(rep["train"]["X"]), "feat": features(rep["train"]["X"])}
    ve = {"pop": {"raw": raw_view(rep["pop"]["X"]), "feat": features(rep["pop"]["X"])}}
    out = fit_predict_pool(vt, rep["train"]["y"], ve, names=ORACLE_NAMES, n_classes=spec.n_id)["pop"]
    accs = {k: float((P.argmax(1) == rep["pop"]["y"]).mean()) for k, P in out.items()}
    return max(accs.values()), accs
