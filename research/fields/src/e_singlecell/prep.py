"""Prepare one dataset for SC-0: subsample (HLCA only), derive integer codes, supported fine types, held-out types, cache to disk.
CPU only, no model is fitted here.  Usage:  python prep.py immune|pancreas|hlca
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, DATASETS, MIN_GLOBAL, cache_dir  # noqa: E402


def main(ds):
    import anndata as ad
    cfg = DATASETS[ds]
    a = ad.read_h5ad(DATA / cfg["file"])
    obs = a.obs.copy()
    print(ds, "loaded", a.shape, flush=True)
    keep = obs[cfg["fine"]].notna().to_numpy()
    if ds == "hlca":
        keep &= (obs[cfg["fine"]].astype(str) != "None").to_numpy()
    rows = np.where(keep)[0]
    ss = cfg["subsample"]
    if ss is not None:
        rng = np.random.RandomState(ss["seed"])
        fine_all = obs[cfg["fine"]].astype(str).to_numpy()
        vc = pd.Series(fine_all[rows]).value_counts()
        rare = set(vc[vc < ss["rare_below"]].index)
        donor = obs[cfg["donor"]].astype(str).to_numpy()
        is_rare = np.array([f in rare for f in fine_all])
        sel_rare = rows[is_rare[rows]]
        common = rows[~is_rare[rows]]
        best = None
        for cap in range(2000, 100, -50):
            pick = [sel_rare]
            for dn in np.unique(donor[common]):
                idx = common[donor[common] == dn]
                if len(idx) > cap:
                    idx = rng.choice(idx, size=cap, replace=False)
                pick.append(idx)
            sel = np.sort(np.concatenate(pick))
            if len(sel) <= ss["max_cells"]:
                best = (cap, sel)
                break
            rng = np.random.RandomState(ss["seed"])           # same stream for every cap, so the result is a function of cap only
        cap, rows = best
        print(f"hlca subsample: cap per donor {cap} for non-rare types ({len(rare)} rare types kept whole): {len(rows)} cells", flush=True)
    a = a[rows].copy()
    obs = a.obs.copy()
    fine = obs[cfg["fine"]].astype(str).to_numpy()
    if cfg["coarse_map"] is not None:
        coarse = np.array([cfg["coarse_map"][f] for f in fine])
    else:
        coarse = obs[cfg["coarse_col"]].astype(str).to_numpy()
    batch = obs[cfg["batch"]].astype(str).to_numpy()
    batches = sorted(set(batch))
    donor_col = cfg["donor"]
    if donor_col is not None:
        unit_names = obs[donor_col].astype(str).to_numpy()
    else:
        unit_names = batch
    units = sorted(set(unit_names))
    # supported fine types (sorted by coarse then name); held-out types H = every third in that order (index % 3 == 1)
    vc = pd.Series(fine).value_counts()
    sup = sorted([f for f in vc.index if vc[f] >= MIN_GLOBAL], key=lambda f: (pd.Series(coarse[fine == f]).mode()[0], f))
    held = [t for i, t in enumerate(sup) if i % 3 == 1]
    coarse_names = sorted(set(coarse))
    f2c = {n: i for i, n in enumerate(sup)}
    o = pd.DataFrame({
        "b": [batches.index(x) for x in batch],
        "unit": [units.index(x) for x in unit_names],
        "fine": [f2c.get(f, -1) for f in fine],
        "coarse": [coarse_names.index(c) for c in coarse],
        "study": obs[cfg["study"]].astype(str).to_numpy() if cfg["study"] else batch,
        "fine_name": fine, "coarse_name": coarse, "batch_name": batch, "unit_name": unit_names,
    })
    d = cache_dir(ds)
    o.to_pickle(d / "obs.pkl")
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.tocsr().astype(np.float32)
    if cfg["counts_layer"] == "raw":
        C = a.raw.X
        var_names = a.raw.var_names
    else:
        C = a.layers[cfg["counts_layer"]]
        var_names = a.var_names
    C = (C if sp.issparse(C) else sp.csr_matrix(C)).tocsr()
    assert list(var_names) == list(a.var_names), "raw and X gene axes differ"
    nz = np.asarray((C > 0).sum(0)).ravel()
    gk = np.where(nz >= 10)[0]
    X, C = X[:, gk], C[:, gk].astype(np.float32)
    sp.save_npz(d / "X.npz", X, compressed=False)
    sp.save_npz(d / "counts.npz", C, compressed=False)
    (d / "genes.txt").write_text("\n".join(np.asarray(a.var_names)[gk]))
    meta = dict(batches=batches, fine_names=sup, coarse_names=coarse_names, held_types=held, has_donor=donor_col is not None,
                n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), n_units=len(units),
                x_max=float(X.max()), counts_is_int=bool(np.allclose(C.data[:1000], np.round(C.data[:1000]))))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("fine_names",)}, indent=1))
    # data statistics for the pre-registration (counts only; no model)
    t = pd.crosstab(o["batch_name"], o["fine_name"])
    print("cells per batch:\n", o["batch_name"].value_counts().sort_index().to_string())
    print("units per batch:\n", o.groupby("batch_name")["unit_name"].nunique().to_string())
    print("supported fine types:", len(sup), "held-out H:", held)
    print("coarse classes:", coarse_names)
    print("cells per supported type:\n", vc[sup].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
