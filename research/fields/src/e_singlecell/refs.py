"""T_scvi reference (transductive, label-free, batch covariate): scVI latent on all cells of one dataset.  GPU job.  python refs.py --dataset immune --out $OUT
Writes the 30-d latent to the cache (NOT to HF: derived embedding of a public dataset); $OUT gets scvi_info.json only.  ONE seed: 'not replicated'."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=None, help="default: scvi-tools heuristic min(400, 20000/n_cells*400)")
    a = ap.parse_args()
    import anndata as ad
    import scanpy as sc_
    import scvi
    scvi.settings.seed = a.seed
    S = SC(a.dataset)
    C = S.counts.copy()
    C.data = np.round(C.data)                       # Pancreas ships normalised (non-integer) counts: rounded, flagged in E.md
    ann = ad.AnnData(C)
    ann.layers["counts"] = ann.X.copy()
    ann.obs["batch"] = np.array(S.batches)[S.b]
    t0 = time.time()
    sc_.pp.highly_variable_genes(ann, n_top_genes=2000, flavor="seurat_v3", batch_key="batch", layer="counts", subset=True)
    scvi.model.SCVI.setup_anndata(ann, layer="counts", batch_key="batch")
    m = scvi.model.SCVI(ann, n_latent=30)
    m.train(max_epochs=a.max_epochs)
    Z = m.get_latent_representation().astype(np.float32)
    np.save(CACHE / a.dataset / "scvi_latent.npy", Z)
    info = dict(dataset=a.dataset, cells=int(Z.shape[0]), latent=int(Z.shape[1]), seconds=round(time.time() - t0, 1), scvi=scvi.__version__,
                epochs=int(len(m.history["elbo_train"])), final_elbo_train=float(m.history["elbo_train"].iloc[-1, 0]),
                counts_were_integer=bool(np.allclose(S.counts.data[:1000], np.round(S.counts.data[:1000]))))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scvi_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
