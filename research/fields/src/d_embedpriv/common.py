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
    load_beir.titles = [(t or "").strip() for t in c["title"]]  # side channel for the digits/entity scheme (title length = skip offset)
    return ids, texts, qids, qtexts, qrels


NUM = re.compile(r"\d+(?:[.,]\d+)*")          # numeric token: digit run with internal . or , (years, doses, sample sizes, p-values)
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")       # capitalised word (crude named-entity proxy, no NER model)


def ne_spans(text, skip=0):
    """Capitalised words that do not start a sentence/clause (previous non-space char not in .!?:;) and lie outside the title (first `skip` chars)."""
    out = []
    for m in CAP.finditer(text):
        i = m.start()
        if i < skip:
            continue
        j = i - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0 or text[j] in ".!?:;":
            continue
        if m.group(0).lower() in ENGLISH_STOP_WORDS:
            continue
        out.append((m.start(), m.end()))
    return out


class World:
    """All text-side objects: ids, splits, token bags for the secret tiers.
    scheme 'words' (default): rare/mid content-word tiers (D pre-registration section 2.1).
    scheme 'digits' (H2-009): n1 = numeric tokens with df >= 3 (primary), n2 = capitalised mid-sentence tokens with df >= 3 (secondary),
    n1s = numeric tokens with 3 <= df <= 30 (descriptive).  df is counted over NFCorpus D+A and all SciFact (never V, C)."""

    def __init__(self, scheme="words"):
        self.scheme = scheme
        self.nf_ids, self.nf_texts, self.q_ids, self.q_texts, self.qrels = load_beir("nfcorpus")
        self.nf_titles = list(load_beir.titles)
        self.sf_ids, self.sf_texts, _, _, _ = load_beir("scifact")
        self.sf_titles = list(load_beir.titles)
        self.nf_split = np.array([doc_split("nfcorpus", i) for i in self.nf_ids])
        self.sf_split = np.array([doc_split("scifact", i) for i in self.sf_ids])
        if scheme == "digits":
            return self._init_digits()
        self.tiers = ("rare", "mid")
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

    def _doc_sets(self, text, title_len):
        return set(NUM.findall(text)), set(text[a:b].lower() for a, b in ne_spans(text, title_len + 1 if title_len else 0))

    def _init_digits(self):
        self.tiers = ("n1", "n2", "n1s")
        pool = [(t, len(ti)) for t, ti, s in zip(self.nf_texts, self.nf_titles, self.nf_split) if s in ("D", "A")] + \
               list(zip(self.sf_texts, [len(x) for x in self.sf_titles]))
        d1, d2 = {}, {}
        for t, tl in pool:
            a, b = self._doc_sets(t, tl)
            for w in a:
                d1[w] = d1.get(w, 0) + 1
            for w in b:
                d2[w] = d2.get(w, 0) + 1
        n1 = sorted(w for w, n in d1.items() if n >= 3)
        n2 = sorted(w for w, n in d2.items() if n >= 3)
        self.vocab = n1 + n2
        self.df1, self.df2 = d1, d2
        n1s = np.array([i for i, w in enumerate(n1) if d1[w] <= 30])
        self.tier_cols = {"n1": np.arange(len(n1)), "n2": np.arange(len(n1), len(n1) + len(n2)), "n1s": n1s,
                          "all": np.arange(len(n1) + len(n2))}
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.nf_bags = self.bags(self.nf_texts, [len(x) for x in self.nf_titles])
        self.sf_bags = self.bags(self.sf_texts, [len(x) for x in self.sf_titles])

    def bags(self, texts, title_lens=None):
        rows, cols = [], []
        for i, t in enumerate(texts):
            if self.scheme == "digits":
                a, b = self._doc_sets(t, title_lens[i] if title_lens else 0)
                toks = a | b
            else:
                toks = set(tokens(t))
            for w in toks:
                j = self.w2i.get(w)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(self.vocab)))

    def mask_text(self, text, which, title_len=0):
        """Oracle arm: delete every secret-tier word from the text (whole-word, case-insensitive).
        digits scheme: which='digits' deletes ALL digit runs; which='n2' deletes the capitalised mid-sentence words outside the title."""
        if which == "digits":
            return re.sub(r"\s+", " ", NUM.sub("", text)).strip()
        if which == "n2":
            spans = ne_spans(text, title_len + 1 if title_len else 0)
            for a, b in reversed(spans):
                text = text[:a] + text[b:]
            return re.sub(r"\s+", " ", text).strip()
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
