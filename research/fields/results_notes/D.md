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

### 5.1 Headroom stage (floors, controls, oracle; GTR-base x NFCorpus; seeds 0, 1, 2; no learned privacy arm)

Jobs: `d_extract_gtr_base` (embeddings, revision 9801579c, fp32, max 512 tokens), `d_hr1_gtr_s{0,1,2}` (controls, oracle, LEACE, sup_lin), `d_hr2_gtr_s{0,1,2}` (random projection, PCA, noise); the code commit is recorded in
each `records.jsonl`. Raw GTR-base nDCG@10 on our embeddings: dev 0.3039, test 0.3022 (BEIR-published GTR-base NFCorpus values are 0.31-0.33, same range).
R_C = worst-of-3 attacker R-precision on the 894 sealed C docs; L = (R - R_prior)/(R_raw - R_prior); intervals are 95% bootstrap over C docs (B = 1000, resamples shared across arms, so differences are paired)
averaged over the 3 seeds; they do not include seed spread or split-to-split variability (one fixed document split). Selection on V (560 docs, 324 dev queries), numbers on C (894 docs, 323 test queries).
Aggregation clarification made after seeing the 3-seed table: a family is a candidate for the strongest floor F\* only if it reaches tau in every seed (noise r = 0.5 reaches 0.97 dev retention in 1 of 3 seeds, which
would otherwise have made it F\*(0.97) on one seed). It does not touch H2 (all families are feasible at the oracle-matched tau).

**Verdict of the pre-registered headroom check: H1 PASS, H2 FAIL (headroom 0.039 [-0.003, 0.077] < 0.10) => NO-GO for the learned kill test as pre-registered (section 2.5).**
The frozen margin from the pre-registered formula would be m = min(0.10, 0.5 x 0.039) = 0.020, below the resolution of the comparison: no decidable margin exists.

(oracle_mask_rare reaches dev retention 0.700 < tau: added as an oracle-matched operating point for H2)

records 102, seeds [0, 1, 2], C docs 894, test queries 323

#### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)

| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |
|---|---|---|---|---|---|---|
| const | rare | 0.004 [0.003, 0.005] | 0.004/0.003/0.004 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| const | mid | 0.039 [0.037, 0.041] | 0.038/0.038/0.038 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| raw | rare | 0.183 [0.174, 0.193] | 0.183/0.079/0.145 | 1.00 [1.00, 1.00] | 0.3022 | 1.000 / 1.000 |
| raw | mid | 0.253 [0.246, 0.260] | 0.253/0.186/0.181 | 1.00 [1.00, 1.00] | 0.3022 | 1.000 / 1.000 |
| oracle_mask_rare | rare | 0.110 [0.102, 0.120] | 0.110/0.062/0.099 | 0.59 [0.56, 0.63] | 0.2231 | 0.700 / 0.738 |
| oracle_mask_rare | mid | 0.276 [0.270, 0.283] | 0.276/0.189/0.199 | 1.11 [1.09, 1.13] | 0.2231 | 0.700 / 0.738 |
| oracle_mask_all | rare | 0.073 [0.065, 0.081] | 0.073/0.040/0.066 | 0.38 [0.35, 0.42] | 0.1433 | 0.397 / 0.474 |
| oracle_mask_all | mid | 0.155 [0.149, 0.160] | 0.155/0.145/0.140 | 0.54 [0.52, 0.56] | 0.1433 | 0.397 / 0.474 |

#### All configs (mean over seeds; V = selection split, C = sealed). R = worst-of-3 rare-tier R-precision

| arm | dim | seeds | retention dev | retention test | R_V rare | R_C rare | R_C mid |
|---|---|---|---|---|---|---|---|
| leace_r128 | 768 | 3 | 0.732 | 0.774 | 0.156 | 0.161 | 0.160 |
| leace_r16 | 768 | 3 | 0.938 | 0.950 | 0.175 | 0.180 | 0.209 |
| leace_r256 | 768 | 3 | 0.524 | 0.568 | 0.140 | 0.143 | 0.135 |
| leace_r4 | 768 | 3 | 0.980 | 0.992 | 0.180 | 0.185 | 0.231 |
| leace_r512 | 768 | 3 | 0.148 | 0.167 | 0.093 | 0.092 | 0.086 |
| leace_r64 | 768 | 3 | 0.846 | 0.879 | 0.166 | 0.171 | 0.180 |
| noise_r0.5 | 768 | 3 | 0.960 | 0.958 | 0.143 | 0.137 | 0.207 |
| noise_r1 | 768 | 3 | 0.809 | 0.806 | 0.115 | 0.117 | 0.165 |
| noise_r2 | 768 | 3 | 0.413 | 0.393 | 0.041 | 0.038 | 0.108 |
| noise_r3 | 768 | 3 | 0.212 | 0.192 | 0.018 | 0.017 | 0.080 |
| noise_r4 | 768 | 3 | 0.135 | 0.124 | 0.011 | 0.010 | 0.066 |
| noise_r6 | 768 | 3 | 0.079 | 0.072 | 0.006 | 0.006 | 0.052 |
| pca_d512 | 512 | 3 | 0.977 | 0.975 | 0.180 | 0.182 | 0.247 |
| pca_d256 | 256 | 3 | 0.968 | 0.966 | 0.169 | 0.162 | 0.232 |
| pca_d128 | 128 | 3 | 0.884 | 0.913 | 0.140 | 0.140 | 0.211 |
| pca_d64 | 64 | 3 | 0.759 | 0.752 | 0.124 | 0.128 | 0.184 |
| pca_d32 | 32 | 3 | 0.519 | 0.527 | 0.106 | 0.105 | 0.166 |
| pca_d16 | 16 | 3 | 0.312 | 0.275 | 0.077 | 0.077 | 0.147 |
| randproj_d512 | 512 | 3 | 0.925 | 0.944 | 0.178 | 0.181 | 0.248 |
| randproj_d256 | 256 | 3 | 0.886 | 0.865 | 0.164 | 0.158 | 0.227 |
| randproj_d128 | 128 | 3 | 0.748 | 0.755 | 0.132 | 0.128 | 0.194 |
| randproj_d64 | 64 | 3 | 0.544 | 0.544 | 0.106 | 0.110 | 0.157 |
| randproj_d32 | 32 | 3 | 0.288 | 0.328 | 0.078 | 0.081 | 0.122 |
| sup_lin_d512 | 512 | 3 | 0.957 | 1.068 | 0.174 | 0.175 | 0.235 |
| sup_lin_d256 | 256 | 3 | 0.940 | 1.044 | 0.156 | 0.156 | 0.217 |
| sup_lin_d128 | 128 | 3 | 0.904 | 1.012 | 0.140 | 0.138 | 0.187 |
| sup_lin_d64 | 64 | 3 | 0.799 | 0.875 | 0.131 | 0.129 | 0.162 |
| sup_lin_d32 | 32 | 3 | 0.630 | 0.666 | 0.117 | 0.112 | 0.151 |
| sup_lin_d16 | 16 | 3 | 0.419 | 0.429 | 0.082 | 0.082 | 0.121 |

#### Headroom check (pre-registered H1, H2; floors + oracle only)

H1 measurability: R_raw - R_prior (rare tier, worst of 3 attackers) = 0.179 [0.170, 0.189] (rule: >= 0.05 and CI excludes 0): PASS

oracle_mask_rare: L_rare = 0.593 [0.559, 0.631], test retention 0.738 (< 0.97: not a valid reference at tau = 0.97, so H2 is judged at the oracle's own dev retention tau_O = 0.700)

| tau | strongest floor F* (by V) | F* configs | F* test retention | L_rare(F*) [95% CI] | L_rare(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2 (>= 0.10) |
|---|---|---|---|---|---|---|---|
| 0.970 (oracle not valid at this tau; reference only) | leace | s0:leace_r4, s1:leace_r4, s2:leace_r4 | 0.992 | 1.008 [0.996, 1.021] | 0.593 | 0.416 [0.376, 0.451] | n/a |
| 0.900 (oracle not valid at this tau; reference only) | sup_lin | s0:sup_lin_d128, s1:sup_lin_d128, s2:sup_lin_d128 | 1.012 | 0.748 [0.712, 0.783] | 0.593 | 0.155 [0.109, 0.198] | n/a |
| 0.700 | noise | s0:noise_r1, s1:noise_r1, s2:noise_r1 | 0.806 | 0.632 [0.599, 0.663] | 0.593 | 0.039 [-0.003, 0.077] | FAIL |

#### Selected on V at retention >= 0.97 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 0/3 | cannot reach tau | | | | | |
| pca | 3/3 | s0:d512, s1:d512, s2:d512 | 0.976 [0.949, 1.001] | 0.182 [0.172, 0.191] | 0.991 [0.980, 1.003] | 0.974 [0.966, 0.983] | 0.993 |
| noise | 1/3 | s2:r0.5 | 0.966 [0.942, 0.988] | 0.137 [0.129, 0.146] | 0.743 [0.717, 0.769] | 0.781 [0.765, 0.798] | 0.792 |
| leace | 3/3 | s0:r4, s1:r4, s2:r4 | 0.992 [0.969, 1.016] | 0.185 [0.175, 0.194] | 1.008 [0.996, 1.021] | 0.898 [0.887, 0.911] | 0.992 |
| sup_lin | 0/3 | cannot reach tau | | | | | |

Strongest floor F*(0.97) (lowest seed-mean V leakage): **leace** (V L_rare 0.992; C L_rare 1.008)

#### Selected on V at retention >= 0.9 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 3/3 | s0:d512, s1:d512, s2:d512 | 0.944 [0.921, 0.967] | 0.180 [0.171, 0.190] | 0.985 [0.975, 0.995] | 0.979 [0.973, 0.985] | 0.984 |
| pca | 3/3 | s0:d256, s1:d256, s2:d256 | 0.966 [0.937, 0.993] | 0.162 [0.153, 0.171] | 0.883 [0.861, 0.904] | 0.902 [0.889, 0.916] | 0.932 |
| noise | 3/3 | s0:r0.5, s1:r0.5, s2:r0.5 | 0.959 [0.941, 0.976] | 0.137 [0.129, 0.145] | 0.744 [0.721, 0.764] | 0.784 [0.770, 0.797] | 0.785 |
| leace | 3/3 | s0:r16, s1:r16, s2:r16 | 0.951 [0.922, 0.982] | 0.180 [0.170, 0.189] | 0.981 [0.966, 0.997] | 0.793 [0.779, 0.808] | 0.967 |
| sup_lin | 3/3 | s0:lin_d128, s1:lin_d128, s2:lin_d128 | 1.013 [0.935, 1.096] | 0.138 [0.128, 0.149] | 0.748 [0.712, 0.783] | 0.693 [0.675, 0.710] | 0.767 |

Strongest floor F*(0.9) (lowest seed-mean V leakage): **sup_lin** (V L_rare 0.767; C L_rare 0.748)

#### Selected on V at retention >= 0.7 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 3/3 | s0:d128, s1:d128, s2:d128 | 0.756 [0.715, 0.793] | 0.128 [0.121, 0.137] | 0.694 [0.671, 0.720] | 0.723 [0.708, 0.738] | 0.722 |
| pca | 3/3 | s0:d64, s1:d64, s2:d64 | 0.753 [0.705, 0.800] | 0.127 [0.118, 0.137] | 0.689 [0.654, 0.723] | 0.679 [0.662, 0.699] | 0.680 |
| noise | 3/3 | s0:r1, s1:r1, s2:r1 | 0.806 [0.779, 0.836] | 0.117 [0.108, 0.127] | 0.632 [0.599, 0.663] | 0.590 [0.575, 0.605] | 0.625 |
| leace | 3/3 | s0:r128, s1:r128, s2:r128 | 0.774 [0.729, 0.817] | 0.161 [0.151, 0.171] | 0.875 [0.852, 0.899] | 0.564 [0.545, 0.583] | 0.860 |
| sup_lin | 3/3 | s0:lin_d64, s1:lin_d64, s2:lin_d64 | 0.876 [0.806, 0.953] | 0.129 [0.120, 0.139] | 0.699 [0.664, 0.733] | 0.582 [0.563, 0.605] | 0.720 |

Strongest floor F*(0.7) (lowest seed-mean V leakage): **noise** (V L_rare 0.625; C L_rare 0.632)



### 5.2 Reading of the headroom stage (what the numbers say; hypotheses are marked as such)
1. **The attacker is measurable and the controls hold (H1).** On raw GTR-base vectors the worst retrained attacker has R-precision 0.183 on rare words versus 0.004 for the frequency prior
   (constant vector) and 0.003-0.004 for shuffled pairs; L is defined on that 0.179 gap. The ridge (linear) attacker is the strongest on raw vectors (0.183), then kNN (0.145), then the MLP (0.079): the cheap linear attacker is
   the worst case, as in Focus C.
2. **Deleting the secret words from the input does not buy more than the floors do (H2).** The oracle (rare words removed from the document text before embedding) keeps only 0.738 of test nDCG@10 and still leaves L_rare = 0.593 [0.559, 0.631]
   (the attacker re-infers deleted words from context; mid-tier recovery even rises, L_mid 1.11). Rare words (df 3-30, 13.5 per document) are retrieval-critical in NFCorpus. At the oracle's own operating point (dev retention 0.70) the strongest
   floor, Gaussian noise on stored vectors (r = 1; retention 0.806 on test), reaches L_rare 0.632 [0.599, 0.663]: higher retention than the oracle at a leakage 0.039 [-0.003, 0.077] above it. Removing the secret from the input is therefore not a
   better utility/privacy trade than adding noise. This is the B' lesson again (oracle headroom below the margin => no mask-type arm can pass), but note the limit: the oracle is a *reference*, not a bound on what a vector map could do.
3. **Floors reaching 3% nDCG loss in every seed give essentially no privacy; the first real gain costs about 4%.** Selected on V at retention >= 0.97, the floors feasible in all 3 seeds are PCA-512 (L 0.991) and LEACE r = 4 (L 1.008).
   Gaussian noise r = 0.5 sits on the boundary: dev retention 0.960 on average (>= 0.97 in only 1 of 3 seeds), test retention 0.958 and L_rare 0.743 [0.717, 0.769] (the qualifying seed: test retention 0.966). Random projection and the retrieval-trained linear map
   cannot reach 0.97 on dev. The noise grid has no level between 0 and 0.5, so a floor at exactly 3% with L around 0.8-0.9 may exist; not run. At 0.90: noise r = 0.5 (L 0.744) and sup_lin d = 128 (L 0.748) lead; at 0.70: noise r = 1 (0.632), PCA-64 (0.689),
   sup_lin d = 64 (0.699), random projection d = 128 (0.694). So a noise floor already moves L from 1.0 to 0.74 for about 4% nDCG, and L 0.63 costs about 19%.
4. **LEACE of the secret-token subspace is the weakest floor** (L 0.875 even at 0.774 retention; r = 512 destroys retrieval, 0.167): the token bag is not linearly separable from the retrieval signal, and the retrained ridge attacker
   recovers words through the directions LEACE leaves. (Linear erasure removes linear predictability on the fit pool, the attacker retrains on new pool documents, as in Focus A.)
5. **Dev/test discordance of the retrieval-trained linear map.** sup_lin has *higher* test than raw nDCG (retention 1.07 at d = 512, 1.01 at d = 128) but *lower* dev nDCG (0.957 and 0.904). The pre-registered rule selects on dev, so sup_lin is
   excluded at 0.97 although on the sealed test it would qualify (d = 128: retention 1.013 [0.935, 1.096], L 0.748). Reading C instead of V would make sup_lin d = 128 the strongest floor at 0.97 (L 0.75). With 324/323 queries the retention CIs
   are wide (+-0.05 to 0.08), the same size as the 3% target, so operating points near 0.97 are not resolvable on this benchmark.

### 5.3 Post-hoc learned arms (NOT part of the kill decision)
The coordinator asked for the learned kill test to be run after the headroom tables. Because H2 had already failed, anything below is **post-hoc** and cannot flip the verdict; it only tests whether the pre-registered reading
(no room) is too pessimistic, since the oracle is not a bound for a vector map. Jobs `d_ph_lrn_gtr_s{0,1}` (queued 12:30Z): bott_mlp (control) d in {16..512}, adv lambda in {0.3, 3} x d in {16..512}, seeds 0 and 1,
selection on V as in 2.5, comparison margin fixed post hoc at 0.10 (the cap in the pre-registered formula) for K1-K4. Results are added below when the jobs finish.

(pending)

## 6. H2-009 check: digit/number secret (pre-registered 15:10Z VM clock, before any attacker, floor or oracle number on this secret)

Requested by the coordinator as the cheap check in the programme's stop/pivot rule (hypotheses/RANKING_R2.md section 7, H2-009). Budget ~0.5 GPU-h, one builder-hour, **one attempt**.
Queue priority 4 (reserved for the main agent) is used **once**, for the `d_dig_*` jobs, at the coordinator's explicit instruction. PII-Masking-300K is not used.

### 6.1 What changes relative to section 2 (everything else - splits D/A/V/C and ED/EA, release definition, utility, attackers, floors, controls, selection on V / numbers on C, bootstrap over C docs, seeds 0,1,2 - is unchanged)
* **Secret tiers** (`World("digits")`): **n1 (primary)** = numeric tokens = maximal digit runs with internal "." or "," (`\d+([.,]\d+)*`: years, doses, sample sizes, p-values; "95%" gives "95") with df >= 3 over NFCorpus D+A and all SciFact (no upper df cap, as in H2-009); 1,745 tokens, 8.8 per C doc, 11.2% of C docs have none (excluded from means).
  **n2 (secondary, descriptive, never used for pass/fail)** = capitalised mid-sentence words (not after . ! ? : ; and not in the title or the first body word; crude named-entity proxy, **no NER model**) with df >= 3: 1,285 tokens, 2.7 per C doc, 34% of C docs have none.
  **n1s (descriptive)** = n1 tokens with 3 <= df <= 30: 1,432 tokens, 1.9 per C doc, 47% of C docs have none. Metric: R-precision per document, as in 2.1. The frequency prior is the strongest trivial reference here (the top n1 tokens are "1", "2", "3", "4", "95" with df 880-2,660), so L is normalised exactly as before: L = (R - R_prior)/(R_raw - R_prior).
* **Oracle** `oracle_mask_digits`: every digit run is deleted from the document text before embedding (queries untouched); attackers are retrained on those vectors and scored against the ORIGINAL n1 bag. `oracle_mask_n2` (descriptive) deletes the n2 words outside the title.
* Floors: same families; the LEACE concept is the SVD-r span of the defender-pool n1+n2 bag; the noise grid is {0.25, 0.5, 1, 2, 3, 4} (adds 0.25, drops 6, because section 5 showed the 3% boundary lies between r = 0 and r = 0.5). Controls: const, shuffled pairs.

### 6.2 Pass conditions (fixed now)
* **H1 (measurability) PASS** iff, on the sealed C docs, mean over 3 seeds, R_raw - R_prior on n1 (worst of ridge/mlp/knn) >= 0.05 with the 95% bootstrap CI (over C docs) excluding 0, and the const / shuffled-pairs controls are within 0.02 of R_prior. (H2-009's own wording; "GTR-base recovers the number tier above its frequency prior".)
* **H2 (room) PASS** iff both: (a) `oracle_mask_digits` retention >= 0.97 on the test queries AND >= 0.97 on the dev queries (seed-independent, the oracle embeddings are fixed; retention = nDCG@10 / raw nDCG@10 on the same queries);
  (b) headroom = L_n1(F\*, tau = 0.97) - L_n1(oracle_mask_digits) >= 0.10 with a 95% paired bootstrap lower bound > 0, where F\* is the floor family (randproj, pca, noise, leace, sup_lin) that reaches tau on V in **every** seed, with the lowest seed-mean V leakage, selected exactly as in 2.5.
  If no floor family reaches 0.97 in every seed, (b) is judged at tau = 0.90 and flagged. Headroom >= 0.30 is the H2-009 prediction and is reported as "strong", it is not the gate.
* **Margin** (computed from this floors + oracle run, then frozen in this file before any learned job is created): m = min(0.10, 0.5 x headroom).
* **Both H1 and H2 must pass before any learned arm.** If both pass, exactly one learned kill test runs (jobs `d_dig_lrn_gtr_s{0,1,2}`: `bott_mlp` control and `adv` lambda in {0.3, 3}, d in {16..512}, 3 seeds), judged by K1-K4 of 2.5 with tier n1 as the primary and n2 as the second tier in K4, F\* named from the same run. Fail on any of H1, H2, K1-K4 = this check fails; no second attempt.

### 6.3 Disclosure (before the numbers)
* Seen before this pre-registration: tier sizes, tokens per doc, df quantiles and example tokens from the text only (stage 0); the section 5 results for the *word* secret, including the retention curves of the secret-agnostic floors (random projection, PCA, noise use the same embeddings and seeds, so their retention numbers on this run are already known; what is new is the attacker side and the oracle).
* `d_ph_lrn_gtr_s{0,1}` (post-hoc learned arms for the word secret) have finished but were not aggregated at the time of writing; they are unrelated to this check.
* An N2 bug (words after a title were counted as mid-sentence) was fixed in stage 0 before any model number; the numbers above are after the fix.

### 6.4 Results (GTR-base x NFCorpus, secret = numeric tokens n1; seeds 0, 1, 2; jobs `d_dig_check_gtr_s{0,1,2}`, priority 4, one run each)
Run notes: the extraction job `d_dig_extract_gtr` wrote all four masked-embedding files (digits and n2, NFCorpus and SciFact) and then crashed at the very last line (`KeyError: 'rare'` while writing its meta file,
a scheme bug fixed in the next commit); the worker marked it failed. I verified the files (3,633 / 5,183 rows, unit norm, 87% of NFCorpus rows differ from the raw embedding) and wrote the `meta_digits.json` marker by hand
(CPU, no GPU process) so the guarded check jobs could start. Raw GTR-base nDCG@10 dev 0.3039 / test 0.3022 (identical to section 5). Intervals: 95% bootstrap over the 894 sealed C docs, shared resamples, mean of 3 seeds.

**Verdict: H1 FAIL. The check fails; the learned arms (bott_mlp, adv, K1-K4) were NOT run, as pre-registered in 6.2. No margin was needed.**
* **H1: R_raw - R_prior = 0.010 [0.002, 0.018] (rule: >= 0.05).** The worst retrained attacker on raw GTR-base vectors recovers number tokens at R-precision 0.214 versus 0.204 for the frequency prior (ridge 0.204, mlp 0.215, knn 0.187; the controls hold: shuffled pairs 0.205 [0.191, 0.218], const 0.204 [0.190, 0.218]).
  Raw GTR vectors carry almost no information about which numbers a document contains beyond how common each number is.
* **H2(a) passes, H2(b) is undecidable.** The oracle that deletes every digit run keeps dev 1.005 / test 0.991 of nDCG@10 (so the premise "retrieval does not use numbers" is true), but its attacker recovery is 0.215, the same as raw (0.214):
  there is nothing for it to remove. L = (R - R_prior)/(R_raw - R_prior) divides by 0.010 and is noise (values such as +14, -69, +21 in the tables below); L on n1 must be ignored. In absolute terms every arm sits at 0.204-0.216 R-precision,
  i.e. floors, oracle and raw are indistinguishable on this secret. H2(b) is reported FAIL (headroom lower bound is not positive), but the correct reading is "no measurable leakage to defend".
* Attacker limits that bound the claim: three attackers trained on 3,716 documents (no Vec2Text corrector, no external decoder). A frequency prior is a strong reference for numbers (top tokens "1", "2", "3", "95"), and a corrector trained on millions of passages might recover more.
  The result is "not recoverable by this retrained suite", not "not recoverable".
* **Descriptive tier n2 (capitalised mid-sentence words, a crude entity proxy; not part of the pass/fail).** Raw excess is large: 0.228 vs prior 0.070 (+0.158). The oracle that deletes n2 words keeps dev 0.987 / test 0.978 and still leaves L_n2 = 0.80 [0.73, 0.86]
  (re-inference from context, as with rare words). The strongest floors at 0.97 reach L_n2 0.81 [0.75, 0.88] (noise r = 0.25-0.5, test retention 0.988) - equal to the oracle within 0.01, so a headroom of about 0 on this tier as well; at 0.90 noise r = 0.5 gets 0.68, below the oracle.
  The pattern is the same as for rare words: what is leaked is recoverable from context, deleting the secret from the input does not remove it, and noise already matches the oracle. I did not run a pre-registered K-test on n2 and did not choose it after seeing the numbers as a rescue.

##### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)

| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |
|---|---|---|---|---|---|---|
| const | n1 | 0.204 [0.190, 0.218] | 0.205/0.204/0.205 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| const | n2 | 0.070 [0.060, 0.082] | 0.070/0.067/0.070 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| raw | n1 | 0.214 [0.201, 0.228] | 0.204/0.215/0.187 | -27.62 [-368.09, 1.00] | 0.3022 | 1.000 / 1.000 |
| raw | n2 | 0.228 [0.205, 0.251] | 0.228/0.143/0.167 | 1.00 [1.00, 1.00] | 0.3022 | 1.000 / 1.000 |
| oracle_mask_digits | n1 | 0.215 [0.201, 0.228] | 0.202/0.215/0.183 | 14.11 [-1.55, 6.33] | 0.2994 | 1.005 / 0.991 |
| oracle_mask_digits | n2 | 0.220 [0.198, 0.245] | 0.220/0.145/0.170 | 0.95 [0.91, 0.99] | 0.2994 | 1.005 / 0.991 |
| oracle_mask_n2 | n1 | 0.214 [0.200, 0.228] | 0.206/0.214/0.185 | -10.04 [-93.79, 2.17] | 0.2956 | 0.987 / 0.978 |
| oracle_mask_n2 | n2 | 0.195 [0.174, 0.219] | 0.196/0.138/0.156 | 0.80 [0.73, 0.86] | 0.2956 | 0.987 / 0.978 |


##### All configs (mean over seeds; V = selection split, C = sealed). R = worst-of-3 primary-tier R-precision

| arm | dim | seeds | retention dev | retention test | R_V n1 | R_C n1 | R_C n2 |
|---|---|---|---|---|---|---|---|
| leace_r128 | 768 | 3 | 0.792 | 0.765 | 0.215 | 0.206 | 0.174 |
| leace_r16 | 768 | 3 | 0.963 | 0.931 | 0.212 | 0.206 | 0.205 |
| leace_r256 | 768 | 3 | 0.523 | 0.508 | 0.214 | 0.204 | 0.141 |
| leace_r4 | 768 | 3 | 0.996 | 0.964 | 0.216 | 0.208 | 0.220 |
| leace_r64 | 768 | 3 | 0.904 | 0.884 | 0.216 | 0.206 | 0.185 |
| noise_r0.25 | 768 | 3 | 0.992 | 1.000 | 0.229 | 0.211 | 0.206 |
| noise_r0.5 | 768 | 3 | 0.960 | 0.958 | 0.224 | 0.212 | 0.177 |
| noise_r1 | 768 | 3 | 0.809 | 0.806 | 0.225 | 0.212 | 0.145 |
| noise_r2 | 768 | 3 | 0.413 | 0.393 | 0.216 | 0.208 | 0.109 |
| noise_r3 | 768 | 3 | 0.212 | 0.192 | 0.211 | 0.205 | 0.095 |
| noise_r4 | 768 | 3 | 0.135 | 0.124 | 0.210 | 0.204 | 0.091 |
| pca_d512 | 512 | 3 | 0.977 | 0.975 | 0.224 | 0.212 | 0.222 |
| pca_d256 | 256 | 3 | 0.968 | 0.966 | 0.223 | 0.212 | 0.209 |
| pca_d128 | 128 | 3 | 0.884 | 0.913 | 0.229 | 0.213 | 0.185 |
| pca_d64 | 64 | 3 | 0.759 | 0.752 | 0.226 | 0.213 | 0.154 |
| pca_d32 | 32 | 3 | 0.519 | 0.527 | 0.221 | 0.210 | 0.147 |
| randproj_d512 | 512 | 3 | 0.925 | 0.944 | 0.227 | 0.216 | 0.214 |
| randproj_d256 | 256 | 3 | 0.886 | 0.865 | 0.227 | 0.216 | 0.193 |
| randproj_d128 | 128 | 3 | 0.748 | 0.755 | 0.224 | 0.213 | 0.165 |
| randproj_d64 | 64 | 3 | 0.544 | 0.544 | 0.223 | 0.213 | 0.144 |
| randproj_d32 | 32 | 3 | 0.288 | 0.328 | 0.222 | 0.209 | 0.125 |
| sup_lin_d512 | 512 | 3 | 0.957 | 1.068 | 0.222 | 0.213 | 0.205 |
| sup_lin_d256 | 256 | 3 | 0.940 | 1.044 | 0.224 | 0.213 | 0.196 |
| sup_lin_d128 | 128 | 3 | 0.904 | 1.012 | 0.223 | 0.210 | 0.179 |
| sup_lin_d64 | 64 | 3 | 0.799 | 0.875 | 0.222 | 0.214 | 0.154 |
| sup_lin_d32 | 32 | 3 | 0.630 | 0.666 | 0.224 | 0.211 | 0.146 |


##### Headroom check (pre-registered H1, H2; floors + oracle only)

H1 measurability: R_raw - R_prior (n1 tier, worst of 3 attackers) = 0.010 [0.002, 0.018] (rule: >= 0.05 and CI excludes 0): FAIL

oracle_mask_digits: L_n1 = 14.112 [-1.548, 6.325], retention dev 1.005 / test 0.991; H2(a) (both >= 0.97): PASS

| tau | strongest floor F* (by V) | F* configs | F* test retention | L_n1(F*) [95% CI] | L_n1(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2(b) (>= 0.10, lower bound > 0) |
|---|---|---|---|---|---|---|---|
| 0.970 | leace | s0:leace_r4, s1:leace_r4, s2:leace_r4 | 0.964 | -69.354 [-1483.489, 0.796] | 14.112 | -83.466 [-1602.887, -0.098] | FAIL |
| 0.900 (fallback if no floor reaches 0.97 in every seed) | leace | s0:leace_r16, s1:leace_r16, s2:leace_r16 | 0.931 | -82.981 [-1692.116, 0.554] | 14.112 | -97.093 [-1874.463, -0.280] | n/a |


### 6.5 Post-hoc learned arms for the WORD secret (jobs `d_ph_lrn_gtr_s{0,1}`, finished; seeds 0 and 1; not part of any decision)
Selection on V as in 2.5; numbers on C; margin m = 0.10 fixed post hoc. The adversary is not better than its no-adversary control and costs utility: dev retention never reaches 0.90 for any `adv` config (best 0.838 at d = 256, lambda 3, versus `bott_mlp` 0.935 at d = 512)
at the same attacker recovery (adv d256: R_C 0.159 vs bott d256 0.154). Selected at dev retention >= 0.70: adv (d = 128, lambda 0.3) L_rare 0.731 [0.705, 0.756] at test retention 0.803; bott_mlp (d = 64) 0.683 [0.647, 0.719] at 0.896; noise r = 1 0.633 [0.599, 0.665] at 0.806; sup_lin d = 64 0.692 at 0.878.
At 0.97 and at 0.90 `adv` has no feasible config, so K1-K4 are not satisfiable (K1+K2 in 0 of 2 seeds). The post-hoc evidence therefore agrees with the H2 verdict: the adversarial arm adds nothing over a rate/architecture bottleneck or noise.
