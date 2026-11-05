"""Focus D3 stage 1a (GPU): frozen GTR-base embeddings of PII-Masking-300K documents (raw, PII-deleted oracle) and excerpts.

Reads texts from the scratch cache written by pii_prep.py and writes float32 unit vectors to the scratch cache only ($D3_CACHE/gtr_base/*.npy).
Nothing derived from the text is written to $OUT; $OUT gets one aggregate JSON (shapes, counts, a non-embedding utility reference).
Never prints text.  <= 8 CPU threads.
"""
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch

torch.set_num_threads(8)
CACHE3 = Path(os.environ.get("D3_CACHE", "~/work/cache/d3_pii")).expanduser()
OUT = Path(os.environ.get("OUT", "~/work/runs/d3_embed")).expanduser()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = 512
    commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    T = pickle.load(open(CACHE3 / "texts.pkl", "rb"))
    d = CACHE3 / "gtr_base"
    d.mkdir(exist_ok=True)
    shapes = {}
    for key, name in (("src", "docs"), ("oracle", "docs_oracle"), ("ex0", "ex0"), ("ex1", "ex1")):
        t0 = time.time()
        x = model.encode(T[key], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(d / f"{name}.npy", x)
        shapes[name] = list(x.shape)
        print(f"{name} {x.shape} {time.time() - t0:.0f}s", flush=True)
    # non-embedding utility reference on the KEPT (purged) V and C sets: TF-IDF cosine, excerpt vs source text, known-item nDCG@10
    from sklearn.feature_extraction.text import TfidfVectorizer
    M = np.load(CACHE3 / "meta.npz", allow_pickle=True)
    split, keep = M["split"], M["keep"]
    fit = np.where((split == 0) | (split == 1))[0]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32).fit([T["src"][i] for i in fit])
    ref = {}
    for g, nm in ((2, "V"), (3, "C")):
        idx = np.where((split == g) & keep)[0]
        D_ = tf.transform([T["src"][i] for i in idx])
        Q_ = tf.transform([T["ex0"][i] for i in idx])
        nd = []
        for s in range(0, len(idx), 500):
            S = (Q_[s: s + 500] @ D_.T).toarray()
            own = S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])]
            rank = (S > own[:, None]).sum(1) + 1
            nd.append(np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0))
        ref[nm] = dict(n=int(len(idx)), tfidf_ndcg10=float(np.concatenate(nd).mean()))
    (OUT / "summary_embed.json").write_text(json.dumps(dict(hf="sentence-transformers/gtr-t5-base", commit=commit, shapes=shapes, utility_reference_kept=ref,
                                                            note="aggregate only; vectors are in the scratch cache and are not uploaded"), indent=1))
    (d / "done.txt").write_text("ok")
    print("done", ref, flush=True)


if __name__ == "__main__":
    main()
