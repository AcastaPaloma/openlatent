"""Focus D (embedding privacy for retrieval): data, splits, token vocabulary, metrics.

Protocol is pre-registered in research/fields/results_notes/D.md.  Nothing here depends on a model.

Corpora (BEIR, cc-by-sa-4.0 per the HF cards): NFCorpus (target; 3,633 docs, train/dev/test qrels) and SciFact docs
(extra text for the defender and for the attacker, split into two disjoint halves).
Doc splits are md5-hash based so they are deterministic and independent of any seed:
  NFCorpus  D 30%  defender fit (adversary training, PCA, LEACE)        A 30%  attacker training
            V 15%  validation (config selection)                         C 25%  sealed test
  SciFact   ED 50% defender extra text                                  EA 50% attacker extra text
C docs, dev and test queries never enter any fit (not even as retrieval-loss negatives).
"""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA = Path(os.environ.get("D_DATA", "~/work/data/d_embedpriv")).expanduser()
CACHE = Path(os.environ.get("D_CACHE", "~/work/cache/d_embedpriv")).expanduser()
TOK = re.compile(r"[A-Za-z]{3,}")
RARE = (3, 30)   # document-frequency range (over the defender+attacker text pool) of the "rare" tier
MID = (31, 300)  # "mid" tier


def tokens(s):
    return [t for t in (m.lower() for m in TOK.findall(s)) if t not in ENGLISH_STOP_WORDS]


def _h(name, did):
    return int(hashlib.md5(f"d-split-v1|{name}|{did}".encode()).hexdigest()[:8], 16) % 100


def doc_split(name, did):
    h = _h(name, did)
    if name == "nfcorpus":
        return "D" if h < 30 else "A" if h < 60 else "V" if h < 75 else "C"
    return "ED" if h < 50 else "EA"


def load_beir(name):
    d = DATA / "beir" / name
    c = pd.read_parquet(d / "corpus.parquet")
    q = pd.read_parquet(d / "queries.parquet")
    qrels = {}
    for f in sorted(d.glob("qrels_*.tsv")):
        split = f.stem.split("_")[1]
        df = pd.read_csv(f, sep="\t")
        r = {}
        for a, b, s in zip(df["query-id"].astype(str), df["corpus-id"].astype(str), df["score"]):
            r.setdefault(a, {})[b] = int(s)
        qrels[split] = r
    ids = c["_id"].astype(str).tolist()
    texts = [((t or "").strip() + " " + (x or "").strip()).strip() for t, x in zip(c["title"], c["text"])]
    qids = q["_id"].astype(str).tolist()
    qtexts = [(t or "").strip() for t in q["text"]]
    return ids, texts, qids, qtexts, qrels


class World:
    """All text-side objects: ids, splits, token bags for the two secret tiers."""

    def __init__(self):
        self.nf_ids, self.nf_texts, self.q_ids, self.q_texts, self.qrels = load_beir("nfcorpus")
        self.sf_ids, self.sf_texts, _, _, _ = load_beir("scifact")
        self.nf_split = np.array([doc_split("nfcorpus", i) for i in self.nf_ids])
        self.sf_split = np.array([doc_split("scifact", i) for i in self.sf_ids])
        # vocabulary: document frequency over the training pools (NFCorpus D+A, SciFact all); V and C never contribute
        pool = [t for t, s in zip(self.nf_texts, self.nf_split) if s in ("D", "A")] + list(self.sf_texts)
        df = {}
        for t in pool:
            for w in set(tokens(t)):
                df[w] = df.get(w, 0) + 1
        rare = sorted(w for w, n in df.items() if RARE[0] <= n <= RARE[1])
        mid = sorted(w for w, n in df.items() if MID[0] <= n <= MID[1])
        self.vocab = rare + mid
        self.tier_cols = {"rare": np.arange(len(rare)), "mid": np.arange(len(rare), len(rare) + len(mid)),
                          "all": np.arange(len(rare) + len(mid))}
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.rare_set, self.mid_set = set(rare), set(mid)
        self.nf_bags = self.bags(self.nf_texts)
        self.sf_bags = self.bags(self.sf_texts)

    def bags(self, texts):
        rows, cols = [], []
        for i, t in enumerate(texts):
            for w in set(tokens(t)):
                j = self.w2i.get(w)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(self.vocab)))

    def mask_text(self, text, which):
        """Oracle arm: delete every secret-tier word from the text (whole-word, case-insensitive)."""
        kill = self.rare_set if which == "rare" else (self.rare_set | self.mid_set)
        return re.sub(r"\s+", " ", TOK.sub(lambda m: "" if m.group(0).lower() in kill else m.group(0), text)).strip()

    def idx(self, split, corpus="nf"):
        s = self.nf_split if corpus == "nf" else self.sf_split
        return np.where(s == split)[0]


# ------------------------------------------------------------------------------------------------ metrics
def ndcg10_per_query(Q, D, q_ids, d_ids, qrels_split):
    """BEIR-standard nDCG@10 (pytrec_eval ndcg_cut_10, graded linear gains) of cosine retrieval of Q over D.
    Returns (qid list, per-query array) for the queries in qrels_split."""
    import pytrec_eval
    qpos = {q: i for i, q in enumerate(q_ids)}
    qs = [q for q in qrels_split if q in qpos]
    Qs = np.asarray(Q)[[qpos[q] for q in qs]]
    S = Qs @ np.asarray(D).T
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    run = {}
    for r, q in enumerate(qs):
        run[q] = {d_ids[j]: float(S[r, j]) for j in top[r]}
    ev = pytrec_eval.RelevanceEvaluator({q: qrels_split[q] for q in qs}, {"ndcg_cut_10"})
    res = ev.evaluate(run)
    return qs, np.array([res[q]["ndcg_cut_10"] for q in qs])


def rprec_per_doc(scores, Y, cols):
    """R-precision per doc over a token tier: top-k of the attacker's scores vs the doc's true tier tokens, k = #true tokens.
    scores [n, |V|] (numpy), Y csr [n, |V|]. Docs with no tier token get NaN (excluded from means)."""
    S = np.asarray(scores)[:, cols]
    T = Y[:, cols].toarray() > 0
    k = T.sum(1)
    order = np.argsort(-S, axis=1)
    hit = np.take_along_axis(T, order, axis=1)
    ar = np.arange(S.shape[1])[None, :]
    hits = (hit & (ar < k[:, None])).sum(1)
    out = np.full(len(S), np.nan)
    ok = k > 0
    out[ok] = hits[ok] / k[ok]
    return out


def boot_idx(n, B, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))
