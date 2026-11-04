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
| **ai4privacy/pii-masking-300k** | HF card `license: other`; `LICENSE.md` = **academic research and non-commercial use only, AI4Privacy must be acknowledged, and "redistribution, uploading to databases, sharing through any medium, or the creation and dissemination of derivative works" need a written licence; the data is watermarked.** The contrarian scout wrote "stated Apache 2.0" (copied from the SPARSE paper): that is wrong for the HF release. | **NOT USED, including in the sweep, until the programme owner approves academic-only terms (coordinator decision 12:05Z).** The English jsonl was downloaded to the VM (`~/work/data/d_embedpriv/pii`, 130 MB, never read by any code here, never uploaded); it will be deleted if approval is not given. The secret is defined from public-licence text instead: the rare/mid content-word tier on BEIR documents (section 2.1). Consequence: all results are about recoverable text (words), not PII spans; a PII claim would need a permissively licensed PII set |
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
* After a GO only (not part of the kill test): bge-small replication, SciFact as second utility set, STS Pearson, VQ and contrastive adapters,
  Vec2Text/ALGEN attackers, both-sides noise.

## 3. Disclosure log (things seen before the numbers they could bias)
* 11:20Z vocabulary statistics (sizes, words per doc by split) and a 24-document CPU smoke embedding (no metric computed). Nothing else before this pre-registration.
* Pipeline mechanics were smoke-tested on SYNTHETIC vectors (`fake_embed.py`: random projection of the bag of words plus noise, CPU, `D_FAST=1` shrinks the MLP attacker). Those
  numbers (e.g. the control attackers returning the frequency prior, R_prior 0.004 vs raw 0.16 on synthetic vectors) only show that the code path runs and that the
  const/shuffle controls fall to the prior; they say nothing about GTR-base and are not reported as results. No learned-arm number on real embeddings exists as of this edit.

## 4. Status log
* 11:34Z `d_extract_gtr_base` (GTR-base embeddings of NFCorpus docs/queries, SciFact docs and the two masked-text oracle variants; about 3 GPU-minutes) queued; 11:57Z six guarded
  headroom jobs queued (`d_hr1_gtr_s{0,1,2}` = controls + oracle + LEACE + sup_lin; `d_hr2_gtr_s{0,1,2}` = random projection, PCA, noise). 12:04Z: **no d_ job has had a GPU yet**
  (29 min); builder C/A added 17 + 7 jobs at priority 6-10 ahead of them. CPU extraction is not an alternative (measured 0.8 docs/s with 24 threads on the shared machine: 3.6k docs = 75 min).
  Nothing has been run on real embeddings, so no headroom check, no margin, no result exists yet.
* Resume recipe once the extraction job is done: `d_hr*` jobs start by themselves (they wait up to 30 min for `~/work/cache/d_embedpriv/gtr_base/meta.json`); then
  `src/d_embedpriv/pull_runs.sh <dir>` and `python src/d_embedpriv/aggregate.py <dir>/d_hr*` give the controls/oracle/floors tables for the headroom check (section 2.5); the margin is then frozen here,
  and only then `python make_jobs.py learned 0,1,2` writes the learned-arm jobs (bott_mlp control + adv lambda 0.1/0.3/1/3, d 16..512).

* 12:05Z coordinator promoted `d_extract_gtr_base` (priority 3) and the six `d_hr*` jobs (priority 4); extraction started 12:05Z. Coordinator ruling: the 2-hour kill-test clock restarts when extraction starts (12:05Z, so the budget ends 14:05Z).

## 5. Results

### 5.1 Headroom stage (floors, controls, oracle; GTR-base x NFCorpus; seeds 0 and 1; no learned privacy arm yet)

Jobs: `d_extract_gtr_base` (embeddings, revision 9801579c, fp32, max 512 tokens), `d_hr1_gtr_s{0,1}` (controls, oracle, LEACE, sup_lin), `d_hr2_gtr_s{0,1}` (random projection, PCA, noise); code at
commit recorded in each `records.jsonl`. Seed 2 of `d_hr1` finished, `d_hr2_gtr_s2` is still queued, so the tables below use seeds 0 and 1 (2 seeds: "not yet replicated" in the 3-seed sense).
Raw GTR-base nDCG@10 on our embeddings: dev 0.3039, test 0.3022 (BEIR-published GTR-base NFCorpus values are 0.31-0.33, same range). R_C = worst-of-3 attacker R-precision on the 894 sealed C docs;
L = (R - R_prior)/(R_raw - R_prior); intervals are 95% bootstrap over C docs (B = 1000, resamples shared across arms, so differences are paired) averaged over the 2 seeds; they do not include seed spread
or split-to-split variability. Selection on V (560 docs, 324 dev queries), numbers on C (894 docs, 323 test queries).

**Verdict of the pre-registered headroom check: H1 PASS, H2 FAIL (headroom 0.041 [0.001, 0.081] < 0.10) => NO-GO for the learned kill test as pre-registered (section 2.5).**
The frozen margin from the pre-registered formula would be m = min(0.10, 0.5 x 0.041) = 0.020, i.e. below the noise of the comparison: no decidable margin exists.

(oracle_mask_rare reaches dev retention 0.700 < tau: added as an oracle-matched operating point for H2)

records 68, seeds [0, 1], C docs 894, test queries 323

#### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)

| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |
|---|---|---|---|---|---|---|
| const | rare | 0.004 [0.003, 0.005] | 0.004/0.004/0.004 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| const | mid | 0.039 [0.037, 0.041] | 0.038/0.038/0.038 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| raw | rare | 0.183 [0.174, 0.193] | 0.183/0.079/0.145 | 1.00 [1.00, 1.00] | 0.3022 | 1.000 / 1.000 |
| raw | mid | 0.253 [0.246, 0.260] | 0.253/0.186/0.181 | 1.00 [1.00, 1.00] | 0.3022 | 1.000 / 1.000 |
| oracle_mask_rare | rare | 0.110 [0.102, 0.120] | 0.110/0.063/0.099 | 0.59 [0.56, 0.63] | 0.2231 | 0.700 / 0.738 |
| oracle_mask_rare | mid | 0.276 [0.270, 0.283] | 0.276/0.188/0.199 | 1.11 [1.09, 1.13] | 0.2231 | 0.700 / 0.738 |
| oracle_mask_all | rare | 0.073 [0.065, 0.081] | 0.073/0.041/0.066 | 0.38 [0.35, 0.42] | 0.1433 | 0.397 / 0.474 |
| oracle_mask_all | mid | 0.155 [0.149, 0.160] | 0.155/0.145/0.140 | 0.54 [0.52, 0.56] | 0.1433 | 0.397 / 0.474 |

#### All configs (mean over seeds; V = selection split, C = sealed). R = worst-of-3 rare-tier R-precision

| arm | dim | seeds | retention dev | retention test | R_V rare | R_C rare | R_C mid |
|---|---|---|---|---|---|---|---|
| leace_r128 | 768 | 2 | 0.739 | 0.770 | 0.157 | 0.161 | 0.159 |
| leace_r16 | 768 | 2 | 0.938 | 0.949 | 0.174 | 0.179 | 0.208 |
| leace_r256 | 768 | 2 | 0.529 | 0.572 | 0.140 | 0.143 | 0.135 |
| leace_r4 | 768 | 2 | 0.980 | 0.991 | 0.180 | 0.186 | 0.232 |
| leace_r512 | 768 | 2 | 0.149 | 0.161 | 0.092 | 0.092 | 0.086 |
| leace_r64 | 768 | 2 | 0.845 | 0.881 | 0.166 | 0.170 | 0.180 |
| noise_r0.5 | 768 | 2 | 0.954 | 0.955 | 0.142 | 0.137 | 0.207 |
| noise_r1 | 768 | 2 | 0.804 | 0.805 | 0.114 | 0.118 | 0.165 |
| noise_r2 | 768 | 2 | 0.413 | 0.405 | 0.042 | 0.038 | 0.107 |
| noise_r3 | 768 | 2 | 0.209 | 0.197 | 0.018 | 0.018 | 0.080 |
| noise_r4 | 768 | 2 | 0.135 | 0.128 | 0.011 | 0.011 | 0.066 |
| noise_r6 | 768 | 2 | 0.076 | 0.075 | 0.006 | 0.006 | 0.052 |
| pca_d512 | 512 | 2 | 0.977 | 0.975 | 0.180 | 0.182 | 0.248 |
| pca_d256 | 256 | 2 | 0.968 | 0.966 | 0.169 | 0.162 | 0.232 |
| pca_d128 | 128 | 2 | 0.884 | 0.913 | 0.140 | 0.140 | 0.211 |
| pca_d64 | 64 | 2 | 0.759 | 0.752 | 0.124 | 0.128 | 0.184 |
| pca_d32 | 32 | 2 | 0.519 | 0.527 | 0.106 | 0.105 | 0.166 |
| pca_d16 | 16 | 2 | 0.312 | 0.275 | 0.077 | 0.077 | 0.147 |
| randproj_d512 | 512 | 2 | 0.927 | 0.942 | 0.178 | 0.181 | 0.249 |
| randproj_d256 | 256 | 2 | 0.888 | 0.868 | 0.163 | 0.158 | 0.227 |
| randproj_d128 | 128 | 2 | 0.727 | 0.753 | 0.132 | 0.128 | 0.193 |
| randproj_d64 | 64 | 2 | 0.534 | 0.547 | 0.104 | 0.110 | 0.156 |
| randproj_d32 | 32 | 2 | 0.293 | 0.333 | 0.079 | 0.082 | 0.122 |
| sup_lin_d512 | 512 | 2 | 0.957 | 1.071 | 0.174 | 0.175 | 0.236 |
| sup_lin_d256 | 256 | 2 | 0.938 | 1.047 | 0.156 | 0.156 | 0.217 |
| sup_lin_d128 | 128 | 2 | 0.901 | 1.012 | 0.139 | 0.138 | 0.186 |
| sup_lin_d64 | 64 | 2 | 0.800 | 0.876 | 0.131 | 0.128 | 0.162 |
| sup_lin_d32 | 32 | 2 | 0.625 | 0.667 | 0.116 | 0.113 | 0.151 |
| sup_lin_d16 | 16 | 2 | 0.419 | 0.427 | 0.083 | 0.081 | 0.121 |

##### Headroom check (pre-registered H1, H2; floors + oracle only)

H1 measurability: R_raw - R_prior (rare tier, worst of 3 attackers) = 0.179 [0.170, 0.189] (rule: >= 0.05 and CI excludes 0): PASS

oracle_mask_rare: L_rare = 0.592 [0.558, 0.631], test retention 0.738 (< 0.97: not a valid reference at tau = 0.97, so H2 is judged at the oracle's own dev retention tau_O = 0.700)

| tau | strongest floor F* (by V) | F* configs | F* test retention | L_rare(F*) [95% CI] | L_rare(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2 (>= 0.10) |
|---|---|---|---|---|---|---|---|
| 0.970 (oracle not valid at this tau; reference only) | pca | s0:pca_d512, s1:pca_d512 | 0.975 | 0.993 [0.981, 1.006] | 0.592 | 0.401 [0.362, 0.435] | n/a |
| 0.900 (oracle not valid at this tau; reference only) | sup_lin | s0:sup_lin_d128, s1:sup_lin_d128 | 1.012 | 0.747 [0.712, 0.783] | 0.592 | 0.155 [0.108, 0.197] | n/a |
| 0.700 | noise | s0:noise_r1, s1:noise_r1 | 0.805 | 0.633 [0.599, 0.665] | 0.592 | 0.041 [0.001, 0.081] | FAIL |

#### Selected on V at retention >= 0.97 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 0/2 | cannot reach tau | | | | | |
| pca | 2/2 | s0:d512, s1:d512 | 0.976 [0.949, 1.001] | 0.182 [0.173, 0.191] | 0.993 [0.981, 1.006] | 0.976 [0.968, 0.985] | 0.992 |
| noise | 0/2 | cannot reach tau | | | | | |
| leace | 2/2 | s0:r4, s1:r4 | 0.991 [0.969, 1.014] | 0.186 [0.176, 0.196] | 1.014 [0.997, 1.030] | 0.901 [0.887, 0.915] | 0.996 |
| sup_lin | 0/2 | cannot reach tau | | | | | |

Strongest floor F*(0.97) (lowest seed-mean V leakage): **pca** (V L_rare 0.992; C L_rare 0.993)

#### Selected on V at retention >= 0.9 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 2/2 | s0:d512, s1:d512 | 0.943 [0.920, 0.966] | 0.181 [0.171, 0.190] | 0.987 [0.975, 0.998] | 0.981 [0.974, 0.987] | 0.980 |
| pca | 2/2 | s0:d256, s1:d256 | 0.966 [0.937, 0.993] | 0.162 [0.153, 0.171] | 0.883 [0.860, 0.905] | 0.902 [0.888, 0.915] | 0.932 |
| noise | 2/2 | s0:r0.5, s1:r0.5 | 0.955 [0.935, 0.974] | 0.137 [0.129, 0.145] | 0.744 [0.720, 0.766] | 0.785 [0.770, 0.799] | 0.781 |
| leace | 2/2 | s0:r16, s1:r16 | 0.950 [0.920, 0.982] | 0.179 [0.170, 0.189] | 0.979 [0.962, 0.995] | 0.792 [0.776, 0.807] | 0.962 |
| sup_lin | 2/2 | s0:lin_d128, s1:lin_d128 | 1.013 [0.934, 1.097] | 0.138 [0.128, 0.148] | 0.747 [0.712, 0.783] | 0.690 [0.673, 0.708] | 0.762 |

Strongest floor F*(0.9) (lowest seed-mean V leakage): **sup_lin** (V L_rare 0.762; C L_rare 0.747)

#### Selected on V at retention >= 0.7 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 2/2 | s0:d128, s1:d128 | 0.755 [0.712, 0.793] | 0.128 [0.120, 0.137] | 0.693 [0.669, 0.722] | 0.719 [0.703, 0.735] | 0.723 |
| pca | 2/2 | s0:d64, s1:d64 | 0.753 [0.705, 0.800] | 0.127 [0.118, 0.137] | 0.689 [0.654, 0.723] | 0.680 [0.662, 0.699] | 0.680 |
| noise | 2/2 | s0:r1, s1:r1 | 0.806 [0.776, 0.839] | 0.118 [0.108, 0.127] | 0.633 [0.599, 0.665] | 0.592 [0.575, 0.608] | 0.621 |
| leace | 2/2 | s0:r128, s1:r128 | 0.770 [0.721, 0.814] | 0.161 [0.152, 0.172] | 0.878 [0.854, 0.902] | 0.561 [0.542, 0.581] | 0.862 |
| sup_lin | 2/2 | s0:lin_d64, s1:lin_d64 | 0.878 [0.806, 0.954] | 0.128 [0.118, 0.138] | 0.692 [0.656, 0.727] | 0.581 [0.561, 0.606] | 0.720 |

Strongest floor F*(0.7) (lowest seed-mean V leakage): **noise** (V L_rare 0.621; C L_rare 0.633)



### 5.2 Reading of the headroom stage (what the numbers say; hypotheses are marked as such)
1. **The attacker is measurable and the controls hold (H1).** On raw GTR-base vectors the worst retrained attacker has R-precision 0.183 on rare words versus 0.004 for the frequency prior
   (constant vector) and 0.003-0.004 for shuffled pairs; L is defined on that 0.179 gap. The ridge (linear) attacker is the strongest on raw vectors (0.183), then kNN (0.145), then the MLP (0.079): the cheap linear attacker is
   the worst case, as in Focus C.
2. **Deleting the secret words from the input does not buy more than the floors do (H2).** The oracle (rare words removed from the document text before embedding) keeps only 0.738 of test nDCG@10 and still leaves L_rare = 0.592 [0.558, 0.631]
   (the attacker re-infers deleted words from context; mid-tier recovery even rises, L_mid 1.11). Rare words (df 3-30, 13.5 per document) are retrieval-critical in NFCorpus. At the oracle's own operating point (dev retention 0.70) the strongest
   floor, Gaussian noise on stored vectors (r = 1; retention 0.805 on test), reaches L_rare 0.633 [0.599, 0.665]: higher retention than the oracle at a leakage 0.041 [0.001, 0.081] above it. Removing the secret from the input is therefore not a
   better utility/privacy trade than adding noise. This is the B' lesson again (oracle headroom below the margin => no mask-type arm can pass), but note the limit: the oracle is a *reference*, not a bound on what a vector map could do.
3. **At the pre-registered 3% operating point the floors give essentially no privacy.** Selected on V at retention >= 0.97, the only feasible floors are PCA-512 (L 0.993) and LEACE r = 4 (L 1.014). Random projection, noise and the
   retrieval-trained linear map cannot reach 0.97 dev retention. At 0.90: noise r = 0.5 (L 0.744) and sup_lin d = 128 (L 0.747) lead; at 0.70: noise r = 1 (0.633), PCA-64 (0.689), sup_lin d = 64 (0.692), random projection d = 128 (0.693).
   So the privacy gain of any floor is bought at 5-20% nDCG; nothing is free at 3%.
4. **LEACE of the secret-token subspace is the weakest floor** (L 0.88 even at 0.77 retention; r = 512 destroys retrieval, 0.16): the token bag is not linearly separable from the retrieval signal, and the retrained ridge attacker
   recovers words through the directions LEACE leaves. (Linear erasure removes linear predictability on the fit pool, the attacker retrains on new pool documents, as in Focus A.)
5. **Dev/test discordance of the retrieval-trained linear map.** sup_lin has *higher* test than raw nDCG (retention 1.07 at d = 512, 1.01 at d = 128) but *lower* dev nDCG (0.957 and 0.901). The pre-registered rule selects on dev, so sup_lin is
   excluded at 0.97 although on the sealed test it would qualify (d = 128: retention 1.013 [0.934, 1.097], L 0.747). Reading C instead of V would make sup_lin d = 128 the strongest floor at 0.97 (L 0.75 against PCA-512's 0.99): the
   conclusion "floors give nothing at 3%" depends on the dev split. With 324/323 queries the retention CIs are wide (+-0.05 to 0.08), which is the same size as the 3% target.

### 5.3 Post-hoc learned arms (NOT part of the kill decision)
The coordinator asked for the learned kill test to be run after the headroom tables. Because H2 had already failed, anything below is **post-hoc** and cannot flip the verdict; it only tests whether the pre-registered reading
(no room) is too pessimistic, since the oracle is not a bound for a vector map. Jobs `d_ph_lrn_gtr_s{0,1}` (queued 12:30Z): bott_mlp (control) d in {16..512}, adv lambda in {0.3, 3} x d in {16..512}, seeds 0 and 1,
selection on V as in 2.5, comparison margin fixed post hoc at 0.10 (the cap in the pre-registered formula) for K1-K4. Results are added below when the jobs finish.

(pending)
