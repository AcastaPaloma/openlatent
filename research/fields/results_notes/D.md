# Focus D (embedding privacy for retrieval): working log and two-hour kill test

Owner: Builder-D. Code: `research/fields/src/d_embedpriv/`. Job specs: `research/fields/jobs/specs/d_*.yaml`. Raw per-unit outputs: HF
`KuanKuanKuan/openlatent-fields` under `runs/d_*` (worker upload; metrics only, see licence note). Nothing below is a result unless it carries an
interval and a seed count; single-seed numbers are marked "not yet replicated".

## 0. Licences and data (checked 2026-11-04)

| asset | licence as stated | use here |
|---|---|---|
| BEIR NFCorpus (`BeIR/nfcorpus`, `BeIR/nfcorpus-qrels`) | HF card: cc-by-sa-4.0 (original NFCorpus: research use, Boteva et al. 2016) | read locally; never re-uploaded; metrics only leave the VM |
| BEIR SciFact (`BeIR/scifact`) | HF card: cc-by-sa-4.0 | extra text for defender / attacker; same handling |
| sentence-transformers/gtr-t5-base | apache-2.0 (not gated) | frozen embedder, revision 9801579c |
| BAAI/bge-small-en-v1.5, intfloat/e5-small-v2 | mit (not gated) | second embedder (sweep only) |
| ielabgroup/vec2text_gtr-base-st_{inversion,corrector} | apache-2.0 (not gated) | sweep only (Vec2Text-style attacker on GTR-base) |
| **ai4privacy/pii-masking-300k** | HF card `license: other`; `LICENSE.md` = **academic research and non-commercial use only, AI4Privacy must be acknowledged, and "redistribution, uploading to databases, sharing through any medium, or the creation and dissemination of derivative works" need a written licence; the data is watermarked.** The contrarian scout wrote "stated Apache 2.0" (copied from the SPARSE paper): that is wrong for the HF release. | **Not used in the kill test.** For the sweep: local use only (English jsonl downloaded to `~/work/data/d_embedpriv/pii` for academic evaluation), no text, no per-record embeddings and no per-record outputs are written to `$OUT`/HF/git; only aggregate metrics. Needs the programme owner's OK that an academic-only licence is acceptable; if not, swap in a permissively licensed synthetic PII set (to be checked) |
| MIMIC-III, any gated set | gated | not used |

## 1. Question and the claim being tested

Can a LEARNED map on frozen text embeddings keep retrieval utility (BEIR nDCG@10 computed on the released vectors) while defeating
recovery of the source text by attackers RETRAINED on the released vectors, beyond what the strongest trivial floor already does at equal
nDCG@10? (Focus A taught us the trap: against LEACE a learned adapter looked promising, against a label-supervised linear projection it had
no headroom. Here the analogous floors are PCA-d, random projection, Gaussian noise, LEACE of the secret-token subspace and a
retrieval-fine-tuned linear projection at matched d.)

Scope of the kill test: GTR-base (768-d, frozen) x NFCorpus. The secret is a **bag of content words** of the document (not PII: NFCorpus has none;
PII-Masking-300K is a sweep-stage secret). The result must therefore not be read as "PII is protected".

## 2. PRE-REGISTRATION v0 (committed 11:37Z, see git log; written before any attacker, floor or learned-arm number existed)

### 2.1 Data, splits, secret
* Corpus: NFCorpus 3,633 docs (text = title + " " + abstract). Queries: all 3,237; qrels train 2,590 / dev 324 / test 323 queries (graded 0-2).
  SciFact 5,183 docs as extra text.
* Doc splits by md5 hash (fixed; not seed dependent): NFCorpus **D** 30% (1,099; defender fit) / **A** 30% (1,080; attacker training) / **V** 15% (560;
  validation, config selection) / **C** 25% (894; sealed test). SciFact **ED** 50% (2,547; defender extra text) / **EA** 50% (2,636; attacker extra text).
  C docs, dev queries and test queries never enter any fit (not even as retrieval-loss negatives); V and dev queries are used for selection only.
* Secret = set of vocabulary words in the document. Words: lower-cased alphabetic tokens of length >= 3, English stop-words removed (sklearn list).
  Document frequency df is counted over NFCorpus D+A and all SciFact (never V, C). **Rare tier**: 3 <= df <= 30 (11,862 words, 13.5 per C doc);
  **mid tier**: 31 <= df <= 300 (3,002 words, 35.7 per C doc). The **rare tier is the primary secret** (closest to PII-like specific strings); mid tier is secondary
  and reported next to it. (Vocabulary sizes and words-per-doc were looked at before this text was written; they depend on no model.)
* Metric for the secret: **R-precision** per document = fraction of the doc's k true tier-words that are in the attacker's top-k scored words (k = number of
  true tier-words in the doc). Bootstrap unit = document (C: 894). Reference levels: **R_prior** = attacker that sees a constant vector (falls back to word
  frequency in its training pool); **R_raw** = attacker on raw GTR vectors. Normalised leakage L = (R - R_prior) / (R_raw - R_prior).

### 2.2 Released vector and utility
* Release = vector stored per document (unit norm after the map). The same deterministic map is applied to queries. Additive noise is applied to stored document
  vectors only and queries stay clean (this favours the noise floor; a both-sides variant is a sweep item).
* Utility = nDCG@10 (pytrec_eval `ndcg_cut_10`, BEIR convention, cosine, full 3,633-doc corpus) on the dev queries (V, selection) and the test queries
  (sealed). **Retention = nDCG@10(arm) / nDCG@10(raw)** on the same query set. Intervals: bootstrap over queries.

### 2.3 Attackers (always retrained from scratch on the released vectors of the attacker's pool = NFCorpus A + SciFact EA, 3,716 docs;
stochastic releases are sampled 4 times per pool doc so the attacker can denoise)
`ridge` (closed-form multi-output ridge on standardised z), `mlp` (2 x 1024 GELU, BCE over the 14,864 words, early stopping), `knn` (cosine, k and temperature chosen
on an inner split). Hyper-parameters/early stopping use an inner random 15% of the attacker pool only. **Worst case = max over the three attackers of the mean
R-precision** (also taken per bootstrap resample). Controls that must hold before any number is believed: constant vector -> prior ranking; attacker trained on
shuffled (vector, bag) pairs within 0.02 of R_prior; raw above R_prior (measurability, H1 below).
Not in the kill-test suite (stated so the claim is bounded): Vec2Text corrector, ALGEN-style few-shot alignment with an external decoder, LightGBM-type
attackers. The reported worst case is a lower bound on the true worst case.

### 2.4 Arms
Floors: `randproj` (Gaussian, d in 32..512), `pca` (PCA-d fitted on the defender pool D + ED, d in 16..512), `noise` (r = noise-to-signal norm ratio, 0.5..6),
`leace` (closed-form linear erasure of the top-r SVD directions of the defender-pool secret-token bag, r in 4..512, ridge 1e-2), `sup_lin` (linear map to d
trained with the retrieval loss only, PCA init; the task-supervised floor at matched d).
Controls: `const`, `shuffle`, `bott_mlp` (linear + MLP-residual map to d, same retrieval loss, **no adversary**: mandatory rate/architecture control).
Candidate: `adv` = `bott_mlp` + token-bag adversary re-initialised every 250 steps (loss: confusion towards the word-frequency prior, weight lambda); adversary data =
defender pool only. Same step budget (1,500 steps, 256 queries/step) for sup_lin, bott_mlp, adv.
Oracle (headroom only): `oracle_mask_rare` / `oracle_mask_all` = embeddings of documents with the rare (resp. rare+mid) words deleted from the text, attackers
retrained on those vectors and scored against the ORIGINAL bag. It is a reference for what removing the secret from the input costs, not a strict bound (a vector map
could hide topical context that deleting words leaves in).

### 2.5 Selection and kill rule (comparator named in advance)
* Operating points: tau = 0.97 (primary; "nDCG loss under 3%", from the scout's kill test) and tau = 0.90 (secondary).
* Per (family, seed): the config with the lowest **V** worst-case rare-tier R-precision among those with V retention >= tau (ties: smaller d, smaller lambda).
  A family with no config reaching tau is reported as "cannot reach tau". The number is that config on the sealed C docs / test queries.
* **Strongest floor F\*(tau)** = the floor family (randproj, pca, noise, leace, sup_lin; not the controls) with the lowest seed-mean V leakage at tau. Fixed from V, not from C.
* **Headroom check (floors + oracle only; learned arms are not run until this is written up and committed with the margin):**
  H1 measurability: R_raw - R_prior >= 0.05 (R-precision units) with a 95% bootstrap CI that excludes 0, else the benchmark cannot resolve differences: STOP.
  H2 room: headroom = L(F\*, 0.97) - L(oracle_mask_rare) >= 0.10, else NO-GO immediately (the floor is already within 0.10 of what deleting the secret from the input
  achieves). If the oracle itself has retention < 0.97 it is not a valid reference at tau = 0.97 and H2 is judged against the oracle at its own retention, flagged.
* **Margin m = min(0.10, 0.5 x headroom)** in normalised-leakage units, computed from the numbers above before any learned arm is run, then frozen in section 4.
* **Kill rule.** The candidate `adv` PASSES iff all hold at tau = 0.97 on the sealed C docs / test queries:
  K1 in >= 2 of 3 seeds, the paired bootstrap (B = 1000, over C docs) 95% lower bound of L(F\*) - L(adv) is >= m;
  K2 in those seeds adv's test retention >= F\*'s test retention - 0.01 (equal utility);
  K3 the same comparison adv vs `bott_mlp` has a lower bound >= m/2 (otherwise the gain is rate/architecture, not the adversary);
  K4 the mid-tier leakage of adv is not worse than F\*'s by more than 0.05 (no hiding behind the tier).
  Fail -> NO-GO for Focus D as specified. Seeds 0,1,2 for everything stochastic; doc splits are one fixed realisation (CIs cover test-document / test-query
  sampling and, separately, seed spread; not split-to-split variability). Single-seed statements are "not yet replicated".
* After a GO only (not part of the kill test): bge-small replication, SciFact as second utility set, STS Pearson, PII-Masking token recovery, VQ and contrastive adapters,
  Vec2Text/ALGEN attackers, both-sides noise.

## 3. Disclosure log (things seen before the numbers they could bias)
* 11:20Z vocabulary statistics (sizes, words per doc by split) and a 24-document CPU smoke embedding (no metric computed). Nothing else before this pre-registration.

## 4. Results
(filled in below as jobs finish)
