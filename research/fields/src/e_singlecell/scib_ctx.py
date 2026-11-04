"""scIB-style context scores (scib-metrics 0.5.6) on the HELD-OUT cells of each fold, for a subset of arms.  Context only; in no kill condition.
Not comparable with the original scIB code or with published totals (the package states this).  python scib_ctx.py --dataset immune --out $OUT"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A  # noqa: E402
from common import CACHE, SC  # noqa: E402
from evaluate import get_z0  # noqa: E402

ARMS = ["raw50", "pca16", "ldaC_pca16", "leace", "T_centre", "T_bwhiten16", "T_harmony", "T_scvi", "onehot_oracle", "ctrl_noise16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="", help="comma list; each fold also gets its own json (a crash in one fold must not lose the others)")
    a = ap.parse_args()
    import anndata as ad
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    sc = SC(a.dataset)
    scvi_p = CACHE / sc.ds / "scvi_latent.npy"
    scvi = np.load(scvi_p) if scvi_p.exists() else None
    arms_w = [x for x in ARMS if x != "T_scvi" or scvi is not None]
    res = {}
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(len(sc.folds)))
    for f in folds:
        T, H = sc.fold_masks(f)
        Z = get_z0(sc, f, T)
        arms = A.build_arms(Z, T, sc, a.seed, sc.folds[f], scvi=scvi, only=arms_w)
        held = np.where(H & (sc.fine >= 0))[0]
        rng = np.random.RandomState(100 + f)
        pick = np.sort(rng.choice(held, size=min(a.n, len(held)), replace=False))
        ann = ad.AnnData(np.zeros((len(pick), 1), np.float32))
        ann.obs["batch"] = np.array(sc.batches)[sc.b[pick]]
        ann.obs["label"] = np.array(sc.fine_names)[sc.fine[pick]]
        for k in arms_w:
            ann.obsm[k] = arms[k][pick]
        try:
            bm = Benchmarker(ann, "batch", "label", embedding_obsm_keys=arms_w, pre_integrated_embedding_obsm_key="raw50",
                             bio_conservation_metrics=BioConservation(), batch_correction_metrics=BatchCorrection(), n_jobs=8, progress_bar=False)
            bm.benchmark()
            df = bm.get_results(min_max_scale=False)
            res[f] = {k: {c: (float(v) if np.isfinite(v) else None) for c, v in df.loc[k].items() if isinstance(v, (int, float, np.floating))} for k in df.index if k in arms_w}
            print(f"fold {f}: ok", flush=True)
        except Exception as e:      # noqa: BLE001  (a metric can fail on a fold with 2 batches; report, do not hide)
            res[f] = {"error": repr(e)[:300]}
            print(f"fold {f}: FAILED {e!r}"[:400], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = "_f" + "".join(str(f) for f in folds) if a.folds else ""
    (Path(a.out) / f"scib_ctx_{a.dataset}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
