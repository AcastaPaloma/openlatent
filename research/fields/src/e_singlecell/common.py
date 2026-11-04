"""Focus E / SC-0 (single-cell batch integration, floors only): dataset definitions, folds, cached loader.

Everything that must be fixed BEFORE any model number is fixed here and mirrored in results_notes/E.md (pre-registration):
folds (held-out batch groups), coarse-label maps, held-out-type rule, minimum class sizes.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
DATA = Path(os.environ.get("E_DATA", HOME / "work/data/e_singlecell"))
CACHE = Path(os.environ.get("E_CACHE", HOME / "work/cache/e_singlecell"))

MIN_GLOBAL = 60      # a fine type needs >= 60 cells in the dataset (after subsampling) to enter any label metric
MIN_TRAIN = 20       # ... and >= 20 cells in the fold's training batches
MIN_HELD = 10        # ... and >= 10 cells in the fold's held-out batches (otherwise it is skipped in that fold)

# ---------------------------------------------------------------------------------------------- coarse maps (hand-written, lineage level)
IMMUNE_COARSE = {
    "CD4+ T cells": "T", "CD8+ T cells": "T", "NKT cells": "T",
    "NK cells": "NK",
    "CD20+ B cells": "B", "CD10+ B cells": "B",
    "Plasma cells": "Plasma",
    "CD14+ Monocytes": "Myeloid", "CD16+ Monocytes": "Myeloid", "Monocyte-derived dendritic cells": "Myeloid", "Monocyte progenitors": "Myeloid",
    "Plasmacytoid dendritic cells": "pDC",
    "Erythrocytes": "Erythroid", "Erythroid progenitors": "Erythroid",
    "HSPCs": "Progenitor", "Megakaryocyte progenitors": "Progenitor",
}
PANCREAS_COARSE = {
    "alpha": "endocrine", "beta": "endocrine", "delta": "endocrine", "gamma": "endocrine", "epsilon": "endocrine",
    "acinar": "exocrine", "ductal": "exocrine",
    "activated_stellate": "stromal", "quiescent_stellate": "stromal", "schwann": "stromal",
    "endothelial": "endothelial",
    "macrophage": "immune", "mast": "immune", "t_cell": "immune",
}

# ---------------------------------------------------------------------------------------------- datasets
# folds: list of lists of BATCH values (the attacker's secret classes inside the fold); every batch is held out exactly once.
DATASETS = {
    "immune": dict(
        file="immune.h5ad", batch="batch", study="study", donor=None, fine="final_annotation", coarse_map=IMMUNE_COARSE, coarse_col=None,
        counts_layer="counts", subsample=None,
        folds=[["10X", "Villani"],
               ["Freytag", "Oetjen_A", "Oetjen_P", "Oetjen_U"],
               ["Sun_sample1_CS", "Sun_sample2_KC", "Sun_sample3_TB", "Sun_sample4_TC"]],
    ),
    "pancreas": dict(
        file="pancreas.h5ad", batch="tech", study=None, donor=None, fine="celltype", coarse_map=PANCREAS_COARSE, coarse_col=None,
        counts_layer="counts", subsample=None,
        folds=[["inDrop1", "inDrop2", "inDrop3", "inDrop4"],
               ["celseq", "celseq2", "smarter"],
               ["fluidigmc1", "smartseq2"]],
    ),
    "hlca": dict(
        file="hlca_core.h5ad", batch="dataset", study="study", donor="donor_id", fine="ann_finest_level", coarse_map=None, coarse_col="ann_level_2",
        counts_layer="raw", subsample=dict(max_cells=100_000, rare_below=2000, seed=20261104),
        folds=[["Banovich_Kropski_2020", "Misharin_2021"],
               ["Barbry_Leroy_2020", "Nawijn_2021", "Seibold_2020_10Xv2", "Seibold_2020_10Xv3", "Krasnow_2020"],
               ["Jain_Misharin_2021_10Xv1", "Jain_Misharin_2021_10Xv2", "Misharin_Budinger_2018", "Meyer_2019",
                "Lafyatis_Rojas_2019_10Xv1", "Lafyatis_Rojas_2019_10Xv2", "Teichmann_Meyer_2019"]],
    ),
    # synthetic dataset used only to exercise the pipeline (smoke.py); never reported as a result
    "smoke": dict(
        file=None, batch="batch", study=None, donor="donor", fine="fine", coarse_map=None, coarse_col="coarse", counts_layer="counts", subsample=None,
        folds=[["b0", "b1"], ["b2", "b3"], ["b4", "b5"]],
    ),
}


def cache_dir(ds):
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    return d


class SC:
    """prepared dataset: sparse log-normalised X (cells x genes), optional raw counts, integer codes for the columns that matter."""

    def __init__(self, ds):
        import scipy.sparse as sp
        self.ds = ds
        self.cfg = DATASETS[ds]
        d = cache_dir(ds)
        self.obs = pd.read_pickle(d / "obs.pkl")
        self.X = sp.load_npz(d / "X.npz").tocsr()
        self.counts = sp.load_npz(d / "counts.npz").tocsr() if (d / "counts.npz").exists() else None
        self.genes = np.array((d / "genes.txt").read_text().split("\n"))[: self.X.shape[1]]
        meta = json.loads((d / "meta.json").read_text())
        self.batches = meta["batches"]            # list of batch names, code = index
        self.fine_names = meta["fine_names"]      # supported fine types (>= MIN_GLOBAL), code = index; unsupported cells have code -1
        self.coarse_names = meta["coarse_names"]
        self.held_types = meta["held_types"]      # names of the held-out TYPES H (label-supervised arms never see them)
        self.n = self.X.shape[0]
        self.b = self.obs["b"].to_numpy().astype(np.int64)
        self.unit = self.obs["unit"].to_numpy().astype(np.int64)      # donor (HLCA) or batch (immune, pancreas): the bootstrap unit
        self.fine = self.obs["fine"].to_numpy().astype(np.int64)
        self.coarse = self.obs["coarse"].to_numpy().astype(np.int64)
        self.has_donor = bool(meta["has_donor"])
        name2code = {n: i for i, n in enumerate(self.batches)}
        self.folds = [np.array([name2code[x] for x in f]) for f in self.cfg["folds"]]
        self.held_codes = np.array([self.fine_names.index(t) for t in self.held_types], dtype=np.int64)

    def fold_masks(self, f):
        held = np.isin(self.b, self.folds[f])
        return ~held, held
