"""Focus D3 stage 0 (CPU): PII-Masking-300K -> template clusters, splits, secrets, excerpts, reference features.

LICENCE / HANDLING (docs/data_licences/WHITELIST.md, owner approval 2026-11-05): the dataset is academic, non-commercial, AI4Privacy-acknowledged.
Everything derived from its text (texts, token vocabularies, bags, features, embeddings) is written ONLY to the scratch cache $D3_CACHE
(default ~/work/cache/d3_pii, outside the git checkout) and is never uploaded or committed.  This script prints counts and thresholds only:
never text, never entity values, never per-record data.  The aggregate summary goes to $D3_OUT/summary_prep.json.

  python pii_prep.py            (<= 8 threads; about 10 min)
"""
import hashlib
import json
import os
import pickle
import re
import time
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401

PII_DIR = Path(os.environ.get("D3_DATA", "~/work/data/d_embedpriv/pii")).expanduser()
CACHE3 = Path(os.environ.get("D3_CACHE", "~/work/cache/d3_pii")).expanduser()
OUT3 = Path(os.environ.get("D3_OUT", "~/work/runs/d3_prep")).expanduser()
FILES = ("1english_openpii_30k.jsonl", "1english_openpii_8k.jsonl")
NAME_L = {"GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3"}
PLACE_L = {"CITY", "STATE", "COUNTRY", "STREET", "BUILDING", "POSTCODE", "SECADDRESS", "GEOCOORD"}
PH = re.compile(r"\[[A-Z0-9_]+\]")
TOK = re.compile(r"[a-z0-9]+")
NB = 1 << 22


def md(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16) / float(1 << 48)


def load_rows():
    rows = []
    for fn in FILES:
        with open(PII_DIR / fn) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def delete_spans(text, spans):
    for s in sorted(spans, key=lambda e: -e["start"]):
        text = text[: s["start"]] + " " + text[s["end"]:]
    return re.sub(r"\s+", " ", text).strip()


def excerpts(ctx_words, key):
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    n = len(ctx_words)
    if n < 12:
        t = " ".join(ctx_words)
        return t, t
    h = n // 2
    out = []
    for lo, hi in ((0, h), (h, n)):
        seg = ctx_words[lo:hi]
        w = min(len(seg), int(rng.integers(12, 17)))
        s = int(rng.integers(0, len(seg) - w + 1))
        out.append(" ".join(seg[s: s + w]))
    return out[0], out[1]


def shingle_matrix(texts):
    rows, cols = [], []
    for i, t in enumerate(texts):
        w = t.lower().split()
        sh = {zlib.crc32(" ".join(w[k: k + 3]).encode()) % NB for k in range(max(1, len(w) - 2))}
        rows.extend([i] * len(sh))
        cols.extend(sh)
    return sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(texts), NB))


def edges(X, tau_min, chunk=400):
    """All pairs (i<j) with exact 3-shingle Jaccard >= tau_min."""
    a = np.asarray(X.sum(1)).ravel().astype(np.float64)
    XT = X.T.tocsr()
    I, J, V = [], [], []
    for s in range(0, X.shape[0], chunk):
        R = (X[s: s + chunk] @ XT).tocoo()
        i = R.row + s
        j = R.col
        keep = j > i
        i, j, c = i[keep], j[keep], R.data[keep].astype(np.float64)
        jac = c / (a[i] + a[j] - c)
        k = jac >= tau_min
        I.append(i[k]); J.append(j[k]); V.append(jac[k])
    return np.concatenate(I), np.concatenate(J), np.concatenate(V)


def components(n, I, J, V, tau):
    k = V >= tau
    g = sp.coo_matrix((np.ones(k.sum()), (I[k], J[k])), shape=(n, n))
    return sp.csgraph.connected_components(g, directed=False)[1]


def main():
    t0 = time.time()
    CACHE3.mkdir(parents=True, exist_ok=True)
    OUT3.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    N = len(rows)
    src = [r["source_text"] for r in rows]
    msk = [r["target_text"] for r in rows]  # PII-masked, typed placeholders kept
    gen = [PH.sub(" PIIX ", m) for m in msk]  # generic placeholder
    ctx = [re.sub(r"\s+", " ", PH.sub(" ", m)).strip() for m in msk]  # context only
    oracle = [delete_spans(r["source_text"], r["privacy_mask"]) for r in rows]
    print(f"docs {N}", flush=True)

    # ---------------- stage 0: template clusters, planted-duplicate calibration
    X = shingle_matrix(msk)
    rng = np.random.default_rng(0)
    plant_src = rng.choice(N, size=1000, replace=False)
    levels = [0, 10, 20, 30, 40, 50]
    ctx_vocab = [w for t in rng.choice(N, 2000, replace=False) for w in msk[t].lower().split() if not PH.fullmatch(w)]
    planted, plant_meta = [], []
    for p in levels:
        for i in plant_src:
            w = msk[i].split()
            pos = [k for k, x in enumerate(w) if not PH.fullmatch(x)]
            m = int(round(p / 100.0 * len(pos)))
            sel = set(rng.choice(pos, size=min(m, len(pos)), replace=False).tolist()) if m else set()
            w2 = [ctx_vocab[int(rng.integers(len(ctx_vocab)))] if k in sel else x for k, x in enumerate(w)]
            planted.append(" ".join(w2))
            plant_meta.append((i, p))
    XA = sp.vstack([X, shingle_matrix(planted)]).tocsr()
    TAU_MIN = 0.10
    I, J, V = edges(XA, TAU_MIN)
    print(f"candidate edges >= {TAU_MIN}: {len(V)}  ({time.time() - t0:.0f}s)", flush=True)
    real = (I < N) & (J < N)
    calib = []
    for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        lab = components(N, I[real], J[real], V[real], tau)
        sizes = np.bincount(lab)
        giant = float(sizes.max()) / N
        # planted recall: a copy has an edge >= tau to some doc in its source's cluster
        # (copies are rows N..; edges (src_real, copy) have j >= N)
        copy_edge = (J >= N) & (I < N) & (V >= tau)
        hit = {}
        for i_, j_ in zip(I[copy_edge], J[copy_edge]):
            hit.setdefault(j_ - N, set()).add(int(lab[i_]))
        rec = {}
        for p in levels:
            ok = [int(lab[plant_meta[q][0]]) in hit.get(q, set()) for q in range(len(plant_meta)) if plant_meta[q][1] == p]
            rec[p] = float(np.mean(ok))
        calib.append(dict(tau=tau, n_clusters=int(len(sizes)), giant_share=giant, singletons=float((sizes == 1).sum()) / len(sizes),
                          docs_in_multi=float((sizes[lab] >= 2).mean()), recall=rec))
        print(f"tau {tau:.2f} clusters {len(sizes)} giant {giant:.3f} recall@20% {rec[20]:.3f}", flush=True)
    # tau_dup := smallest threshold with giant component <= 5% of docs
    ok = [c for c in calib if c["giant_share"] <= 0.05]
    tau_dup = ok[0]["tau"] if ok else None
    cert = bool(ok and ok[0]["recall"][20] >= 0.95)
    summary = dict(n_docs=N, planted=dict(n=len(plant_src), levels=levels), calibration=calib, tau_dup=tau_dup, certified=cert)
    if tau_dup is None:
        (OUT3 / "summary_prep.json").write_text(json.dumps(summary, indent=1))
        print("STOP-0: no threshold with giant component <= 5%")
        return
    lab = components(N, I[real], J[real], V[real], tau_dup)
    sizes = np.bincount(lab)
    print(f"tau_dup {tau_dup} certified {cert} clusters {len(sizes)} giant {sizes.max() / N:.3f}", flush=True)

    # ---------------- splits by cluster (hash of the cluster's smallest doc index)
    first = np.full(len(sizes), N)
    np.minimum.at(first, lab, np.arange(N))
    h = np.array([md(f"d3split|{first[c]}") for c in range(len(sizes))])
    grp_c = np.where(h < 0.25, 0, np.where(h < 0.60, 1, np.where(h < 0.70, 2, 3)))  # D A V C
    hv = np.array([md(f"d3dval|{first[c]}") for c in range(len(sizes))])
    split = grp_c[lab]
    dval = (split == 0) & (hv[lab] < 0.10)
    names = ["D", "A", "V", "C"]
    summary["splits"] = {names[g]: dict(docs=int((split == g).sum()), clusters=int((grp_c == g).sum())) for g in range(4)}
    summary["splits"]["Dval"] = dict(docs=int(dval.sum()), clusters=int(len(set(lab[dval]))))
    print("splits", summary["splits"], flush=True)

    # ---------------- independent verifier: char 5-gram TF-IDF cosine on the masked text (generic placeholders)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    Xc = tv.fit_transform(gen + [PH.sub(" PIIX ", p) for p in planted])
    Xr, Xp = Xc[:N].tocsr(), Xc[N:].tocsr()
    src_cos = np.asarray(Xp.multiply(Xr[[m[0] for m in plant_meta]]).sum(1)).ravel()
    pl_levels = np.array([m[1] for m in plant_meta])
    thr = {p: float(np.percentile(src_cos[pl_levels == p], 5)) for p in levels}
    nonC = np.where(split != 3)[0]
    Cidx = np.where(split == 3)[0]
    mx = np.zeros(len(Cidx), np.float32)
    XnT = Xr[nonC].T.tocsr()
    for s in range(0, len(Cidx), 500):
        S = (Xr[Cidx[s: s + 500]] @ XnT).toarray()
        mx[s: s + 500] = S.max(1)
    summary["verifier"] = dict(planted_cos_p5=thr, frac_C_nearest_nonC_above_p5_at_30pct=float((mx >= thr[30]).mean()),
                               frac_C_nearest_nonC_above_p5_at_50pct=float((mx >= thr[50]).mean()),
                               C_nearest_nonC_cos_percentiles={str(q): float(np.percentile(mx, q)) for q in (50, 90, 99, 100)})
    print("verifier", summary["verifier"]["frac_C_nearest_nonC_above_p5_at_30pct"], flush=True)

    # ---------------- secrets: value tokens (df >= 10 over D and A), types
    poolmask = (split == 0) | (split == 1)
    doc_tok, tok_lab = [], {}
    for r in rows:
        d = set()
        for e in r["privacy_mask"]:
            for t in TOK.findall(e["value"].lower()):
                if len(t) >= 2:
                    d.add(t)
                    tok_lab.setdefault(t, {}).setdefault(e["label"], 0)
                    tok_lab[t][e["label"]] += 1
        doc_tok.append(d)
    df = {}
    for i in np.where(poolmask)[0]:
        for t in doc_tok[i]:
            df[t] = df.get(t, 0) + 1
    vocab = sorted(t for t, c in df.items() if c >= 10)
    w2i = {t: k for k, t in enumerate(vocab)}

    def tier(t):
        c = tok_lab[t]
        top = max(c, key=c.get)
        return "name" if top in NAME_L else ("place" if top in PLACE_L else "other")

    tiers = np.array([tier(t) for t in vocab])
    rr, cc = [], []
    for i, d in enumerate(doc_tok):
        for t in d:
            k = w2i.get(t)
            if k is not None:
                rr.append(i); cc.append(k)
    B = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(N, len(vocab)))
    labs = sorted({e["label"] for r in rows for e in r["privacy_mask"]} - {"CARDISSUER"})
    l2i = {l: k for k, l in enumerate(labs)}
    rr, cc = [], []
    for i, r in enumerate(rows):
        for l in {e["label"] for e in r["privacy_mask"]}:
            if l in l2i:
                rr.append(i); cc.append(l2i[l])
    Y = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(N, len(labs)))
    cnt = np.asarray(B.sum(1)).ravel()
    summary["secret"] = dict(vocab=len(vocab), tiers={t: int((tiers == t).sum()) for t in ("name", "place", "other")},
                             tokens_per_doc_by_split={names[g]: float(cnt[split == g].mean()) for g in range(4)},
                             docs_without_token_C=float((cnt[split == 3] == 0).mean()), n_types=len(labs),
                             value_coverage_C=float(np.mean([len(d & set(vocab)) / max(1, len(d)) for d in (doc_tok[i] for i in np.where(split == 3)[0][:3000])])))

    # ---------------- excerpts + reference features
    ex0, ex1 = [], []
    for i in range(N):
        a, b = excerpts(ctx[i].split(), f"d3ex|{i}")
        ex0.append(a); ex1.append(b)
    from sklearn.decomposition import TruncatedSVD
    feats = {}
    fitmask = np.where(poolmask)[0]
    for nm, tx in (("b1", msk), ("b1g", gen)):
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32).fit([tx[i] for i in fitmask])
        Xt = tf.transform(tx)
        feats[nm] = TruncatedSVD(256, random_state=0).fit(tf.transform([tx[i] for i in fitmask])).transform(Xt).astype(np.float32)
    # non-embedding utility reference: TF-IDF cosine, excerpt (first) vs source text, known-item nDCG@10 within V and C
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32).fit([src[i] for i in fitmask])
    util_ref = {}
    for g in (2, 3):
        idx = np.where(split == g)[0]
        D_ = tf.transform([src[i] for i in idx]); Q_ = tf.transform([ex0[i] for i in idx])
        nd = []
        for s in range(0, len(idx), 500):
            S = (Q_[s: s + 500] @ D_.T).toarray()
            own = S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])]
            rank = (S > own[:, None]).sum(1) + 1
            nd.append(np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0))
        nd = np.concatenate(nd)
        util_ref[names[g]] = dict(tfidf_ndcg10=float(nd.mean()), random_ndcg10=float(np.mean([1 / np.log2(r + 1) for r in range(1, 11)]) * 10 / len(idx)), n=int(len(idx)))
    summary["utility_reference"] = util_ref
    summary["excerpt_words_mean"] = float(np.mean([len(t.split()) for t in ex0]))

    # ---------------- write caches (scratch only)
    with open(CACHE3 / "texts.pkl", "wb") as f:
        pickle.dump(dict(src=src, oracle=oracle, ex0=ex0, ex1=ex1), f)
    sp.save_npz(CACHE3 / "bags.npz", B)
    sp.save_npz(CACHE3 / "types.npz", Y)
    np.savez(CACHE3 / "meta.npz", split=split, dval=dval, cluster=lab, tiers=tiers, vocab=np.array(vocab), labs=np.array(labs),
             b1=feats["b1"], b1g=feats["b1g"])
    summary["tau_dup_certified"] = cert
    summary["seconds"] = round(time.time() - t0)
    (OUT3 / "summary_prep.json").write_text(json.dumps(summary, indent=1))
    print("done", summary["seconds"], "s; summary at", OUT3 / "summary_prep.json", flush=True)


if __name__ == "__main__":
    main()
