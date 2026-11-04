"""Frozen-embedder extraction for Focus D.  Writes float32 unit vectors to $D_CACHE/<embedder>/*.npy (never to $OUT, never uploaded:
the vectors are derived from third-party corpora).  Variants: raw docs/queries, and the oracle arm (docs with the secret-tier
words deleted from the text before embedding).

  python embed.py --embedder gtr_base [--masks rare,all] [--limit N]   (CPU or GPU; deterministic)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import CACHE, World

EMBEDDERS = {
    "gtr_base": dict(hf="sentence-transformers/gtr-t5-base", qprefix="", dprefix=""),
    "bge_small": dict(hf="BAAI/bge-small-en-v1.5", qprefix="Represent this sentence for searching relevant passages: ", dprefix=""),
    "e5_small": dict(hf="intfloat/e5-small-v2", qprefix="query: ", dprefix="passage: "),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="gtr_base")
    ap.add_argument("--masks", default="rare,all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip_raw", action="store_true")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    spec = EMBEDDERS[a.embedder]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(spec["hf"], device=dev)
    model.max_seq_length = 512
    commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    cache = Path(a.out) if a.out else CACHE / a.embedder
    cache.mkdir(parents=True, exist_ok=True)
    W = World()
    lim = a.limit or None

    def enc(texts, name, prefix=""):
        f = cache / f"{name}.npy"
        if f.exists() and not lim:
            print("have", name, flush=True)
            return
        t0 = time.time()
        x = model.encode([prefix + t for t in texts[:lim]], batch_size=a.bs, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        tmp = cache / f"{name}.tmp.npy"
        np.save(tmp, x)
        os.replace(tmp, f)
        print(f"{name} {x.shape} {time.time() - t0:.0f}s", flush=True)

    if not a.skip_raw:
        enc(W.nf_texts, "nf_docs", spec["dprefix"])
        enc(W.q_texts, "nf_queries", spec["qprefix"])
        enc(W.sf_texts, "sf_docs", spec["dprefix"])
    for m in [m for m in a.masks.split(",") if m]:
        enc([W.mask_text(t, m) for t in W.nf_texts], f"nf_docs_mask_{m}", spec["dprefix"])
        enc([W.mask_text(t, m) for t in W.sf_texts], f"sf_docs_mask_{m}", spec["dprefix"])
    (cache / "meta.json").write_text(json.dumps(dict(embedder=a.embedder, hf=spec["hf"], commit=commit, device=dev,
                                                     torch=torch.__version__, limit=a.limit,
                                                     n_vocab=len(W.vocab), n_rare=len(W.tier_cols["rare"]),
                                                     n_mid=len(W.tier_cols["mid"])), indent=1))


if __name__ == "__main__":
    main()
