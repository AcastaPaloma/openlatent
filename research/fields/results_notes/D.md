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

### 5.3 Post-hoc learned arms (superseded: the adversarial arm was a FAILED IMPLEMENTATION, see 7.1)
The coordinator asked for the learned kill test to be run after the headroom tables. Jobs `d_ph_lrn_gtr_s{0,1}` (queued 12:30Z): bott_mlp (control) d in {16..512}, adv lambda in {0.3, 3} x d in {16..512}, seeds 0 and 1.
**The `adv` arm of these jobs collapsed in all 24 fits** (retrieval loss at ln 2,739 = 7.915 from step 250; audit F-01, section 7.1): it is not evidence about adversarial training. The `bott_mlp` control in the same jobs trained normally (retrieval loss 4.5-4.9)
and is used as the control in section 7. Results of these `adv` fits are kept in the raw records only.

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


### 6.5 Post-hoc learned arms for the WORD secret (jobs `d_ph_lrn_gtr_s{0,1}`): `adv` = FAILED IMPLEMENTATION (collapsed), no conclusion
~~The adversary is not better than its no-adversary control and costs utility ... the adversarial arm adds nothing over a rate/architecture bottleneck or noise.~~ **Struck (2026-11-04, audit F-01):** all 24 `adv` fits sat at the uniform-softmax retrieval loss ln 2,739 = 7.915 from step 250
(the unnormalised confusion term, 101-154 nats at step 0, dominated the retrieval loss; the map collapsed to a constant), so their low retention and "no feasible config at 0.90/0.97" reflect a dead map, not the adversary. Only the `bott_mlp` rows of those jobs are valid
(selected at dev retention >= 0.90: d 256/128, L_rare 0.788 [0.764, 0.811] at test retention 1.010; at >= 0.70: d = 64, 0.683 [0.647, 0.719] at 0.896). The repaired arm is run in section 7.

## 7. D2: repaired adversarial arm and the pre-registered learned kill test at tau 0.90 / 0.97 (reopened by the coordinator after audit F-01, F-10; pre-registered 19:10Z VM clock, before any repaired-arm number exists)

### 7.0 Two verdicts, kept side by side
* **verdict_original** (section 5, unchanged, 3 seeds): the pre-registered H2 failed at the oracle's own retention (tau = 0.70): headroom 0.039 [-0.003, 0.077] < 0.10. Original learned kill test: not run as pre-registered; the post-hoc `adv` fits were collapsed (7.1).
* **verdict_D2** (this section): at tau = 0.90 and 0.97 the same H2 passes against the oracle taken as a reference (headroom 0.155 [0.109, 0.198] and 0.416 [0.376, 0.451], 3 seeds; the oracle itself only reaches retention 0.738, so it is a reference there, not a valid bound). D2 runs the learned kill test K1-K4 at those two points with a repaired adversarial arm. A D2 PASS or FAIL does not overwrite verdict_original; it is reported next to it.

### 7.1 What failed and the repair
`train_log` of all 24 post-hoc `adv` fits (d_ph_lrn_gtr_s{0,1}): the retrieval loss is 7.911-7.915 from step 250 to the end (ln of the 2,739 docs outside C = 7.9153, the uniform-softmax value), the confusion term goes 90-154 nats at step 0 to about 0, i.e. the map collapsed to a constant that the adversary cannot decode. With lambda 0.3-3 on an unnormalised sum over 14,864 word logits the confusion term dominated the retrieval loss (about 8-12 nats). Dev nDCG of those fits was at or near chance. B' flagged its own collapsed masks as a failed implementation; this arm should have been flagged the same way.
Repair (`train.py`, spec key `conf_norm`): the confusion term is divided by its value at step 0 (detached; adversary already warmed up on the initial PCA-d map), so lambda is a relative weight: loss = L_retrieval + lambda x conf / conf_0. Every fit (all arms) gets a collapse check: final retrieval loss (mean of the last two logs) > ln N - 0.5 = 7.415 => the fit is flagged `collapsed`, stays in the records, and is excluded from selection.

### 7.2 D2 arm and grid (fixed now)
`adv` with `conf_norm = true`, lambda in {0.001, 0.01, 0.1} (coordinator) plus {1, 3} (added by me: after normalisation the first three add at most about 0.1 nats against a retrieval loss of 5-8, so they may be too weak to move leakage; all five lambdas are members of the family and all enter the V-selection pool), d in {32, 64, 128, 256, 512}, seeds 0 and 1, same data, 1,500 steps, 256 queries/step, adversary re-initialised every 250 steps (warm-up 150, 2 adversary steps per map step) as before. 50 fits, about 35 s each. Words secret (rare tier), GTR-base x NFCorpus, splits and attackers as in section 2. Controls and floors: the existing `d_hr1/d_hr2` (floors, seeds 0, 1) and the valid `bott_mlp` rows of `d_ph_lrn` (seeds 0, 1; the retrieval-loss code path is unchanged by the repair).
"The repaired arm trains" := in each seed at least one non-collapsed config exists. If not, D2 stops with "still collapsed" and K1-K4 are not evaluated.

### 7.3 Kill test and margin (fixed now, from floors + control only)
* Operating points tau in {0.90, 0.97} (selection on V dev retention, numbers on C), 2 seeds. F\*(tau) = floor family (randproj, pca, noise, leace, sup_lin) that reaches tau on V in **every** seed with the lowest seed-mean V leakage, as in 2.5 (sup_lin at 0.90; at 0.97 PCA-512 / LEACE r = 4 with L about 1.0). The candidate is the V-selected non-collapsed `adv` config per seed.
* K1-K4 as in 2.5: K1 paired-bootstrap (over C docs) 95% lower bound of L(F\*) - L(adv) >= m; K2 adv test retention >= F\* test retention - 0.01; K3 the same comparison adv vs `bott_mlp` has lower bound >= m/2; K4 second-tier (mid-word) leakage of adv not worse than F\* by more than 0.05.
  With only 2 seeds the rule is **K1-K4 in both seeds** (stricter than the original 2 of 3). If adv or `bott_mlp` cannot reach tau in a seed, K3 (or the whole test) fails at that tau. Fail at both taus => the repaired learned arm does not beat the floors; pass at either => learned arm GO for a 3-seed replication, not yet a sweep.
* **Margin**: m(tau) = max(min(0.10, 0.5 x headroom(tau)), MDE80(tau)), i.e. the original formula, never lower than the printed minimum detectable effect. Headroom: 0.155 (tau 0.90) and 0.416 (tau 0.97) from the 3-seed headroom table. MDE80 = 2.8 x the paired bootstrap SD of L(F\*) - L(bott_mlp) printed by `aggregate.py --mde` on floors and the valid control only, seeds 0 and 1: **0.031** at tau 0.90 (F\* = sup_lin, SD 0.0111); at tau 0.97 `bott_mlp` cannot reach the threshold on dev so no pair exists, and I use the largest MDE printed anywhere (0.046 at tau 0.70, F\* = noise, SD 0.0164).
  Result: **m(0.90) = max(min(0.10, 0.0775), 0.031) = 0.0775; m(0.97) = max(min(0.10, 0.208), 0.046) = 0.10.** Both margins exceed their MDE, so a pass or a fail is decidable.
* Budget: <= 2 GPU-h; queue priority 4 (reserved for the main agent) used once for these jobs, at the coordinator's instruction. No PII-Masking-300K.

### 7.4 Digits (n1s), post-hoc re-analysis of section 6 (audit F-10)
Disclosure first: the n1s numbers (raw R-precision 0.0408 vs frequency prior 0.0163) were told to me by the audit before this paragraph was written, so the threshold below is **post hoc**: it was chosen knowing the point ratio is about 2.5, and is set well below it. The absolute 0.05 threshold of 6.2 was scale-blind: the n1 prior is 0.204 because the top "numbers" are "1", "2", "95".
Scale-aware H1s (n1s = numeric tokens with 3 <= df <= 30, the rare-number tier; 1,432 tokens, 1.9 per C doc, 47% of C docs have none and are excluded): **PASS iff the ratio R_raw / R_prior >= 1.5 with the 95% bootstrap lower bound >= 1.25 and the absolute excess R_raw - R_prior has a lower bound > 0** (same attackers, 3 seeds, bootstrap over C docs). H2s as in 6.2 with the primary tier n1s. The attacker suite is weak for this question: three attackers trained on 3,716 documents, no Vec2Text or external-decoder attacker, so a pass or a fail is about this suite only. No learned digit arm is run in this budget; a pass would only reopen the digit tier for a separate pre-registration.

### 7.5 Disclosure (before the D2 numbers)
Seen before this pre-registration: all of sections 5 and 6, the collapsed `adv` train logs, the valid `bott_mlp` and floor tables for seeds 0 and 1, the audit text. Mechanics of the repaired code were smoke-tested on synthetic vectors (CPU) after this section was committed; no real-embedding number for a repaired arm exists at this point.

### 7.6 D2 results (words secret, GTR-base x NFCorpus; seeds 0 and 1; jobs `d_d2_lam{0p001,0p01,0p1,1,3}_gtr_s{0,1}`, priority 4 once)
All 10 jobs finished (50 fits; VM clock 19:40Z). The K-test selections and numbers below are from the complete grid (the last job, lambda 0.1 seed 1, did not change any selected config).

**verdict_D2: FAIL at tau = 0.90 and at tau = 0.97 (the repaired learned arm does not beat the strongest floor). verdict_original (NO-GO, headroom 0.039 [-0.003, 0.077] at tau 0.70) stands unchanged next to it.**
* **The repaired arm trains.** 48 of 50 fits are healthy (final retrieval loss 4.6-7.4, not the uniform-softmax 7.915); only the two lambda = 3, d = 512 fits collapsed (7.91) and are excluded from selection by the pre-registered check. The sentence "the adversary adds nothing" is therefore now *testable*, and for this setting it is supported (below), but it was not supported by the original run.
* **lambda <= 0.1 is indistinguishable from the no-adversary bottleneck**: at every d the adv rows equal `bott_mlp` to the third decimal (d = 256: dev retention 0.929 / 0.922 / 0.927 for lambda 0.001 / 0.01 / 0.1 vs 0.922 for bott; R_C 0.154 / 0.154 / 0.161 vs 0.154). The in-training adversary stays strong (confusion/initial 5-15 at the end): the normalised weight is too small to move the map.
* **lambda = 1 and 3 reduce the in-training confusion (ratio 10 -> 2.2 -> 0.7) but cost retention and do not reduce retrained-attacker recovery**: d = 256: dev retention 0.830 (lambda 1) and 0.644 (lambda 3) vs 0.922 for bott, while R_C is 0.169 and 0.167 vs 0.154, i.e. the retrained ridge/MLP/kNN suite recovers *more* words. The map hides information from the co-trained MLP adversary, not from freshly trained attackers (the generic "protection holds only against the adversary family it optimised against" failure).
* **K-test at tau = 0.90** (F\* = sup_lin d = 128, L_rare 0.747 [0.712, 0.783]; margin m = 0.0775; adv selected on V: d = 128, lambda 0.01, both seeds): L(F\*) - L(adv) = 0.015 [-0.003, 0.034] (seed 0) and 0.015 [-0.004, 0.033] (seed 1), needed lower bound >= 0.0775: **K1 fails in both seeds**; K2 fails marginally (adv test retention 0.988 / 0.999 vs F\* 1.011 / 1.013 minus 0.01); K3 passes in seed 0 (0.100 [0.061, 0.140] over the control), but that is a V-selection artefact: in seed 0 the control's d = 128 fit has dev retention 0.894 (just under the 0.90 threshold) so the control is selected at d = 256 (L higher), while the adv d = 128 fit is at 0.90; the matched-d rows in the table below are equal and fails in seed 1 (0.011); K4 passes. PASS needs all four in both seeds: FAIL.
* **K-test at tau = 0.97**: no adv config reaches 0.97 dev retention in either seed (best 0.940 at d = 512, lambda 0.001-0.1; the same ceiling as `bott_mlp` 0.935 and sup_lin 0.957), F\* = PCA-512 (L 0.993): **not satisfiable, FAIL**. The margin was decidable (MDE80 0.031 at tau 0.90 from floors + control, below m = 0.0775).
* Reading: with a valid adversary (MLP token-bag, re-initialised every 250 steps, confusion-to-prior loss), lambda in {0.001..3}, d in {32..512}, 1,500 steps and 2 seeds, the learned arm is not better than the floors or than the rate/architecture control at equal retention, and larger lambda is strictly worse. This does **not** test other adversary designs (gradient reversal, population of adversaries, larger steps) or Vec2Text-class attackers.

##### Adversarial arm vs the no-adversary control, by lambda and d (mean of the seeds that exist; retention = nDCG@10 / raw; R = worst-of-3 rare-tier R-precision)
| arm | lambda | d | fits | collapsed | dev ret | test ret | R_V | R_C | confusion / initial at the end |
|---|---|---|---|---|---|---|---|---|---|
| bott | 0 | 128 | 2 | 0 | 0.902 | 0.998 | 0.135 | 0.137 | n/a |
| adv | 0.001 | 128 | 2 | 0 | 0.900 | 1.001 | 0.135 | 0.138 | 10.59 |
| adv | 0.01 | 128 | 2 | 0 | 0.903 | 0.994 | 0.134 | 0.135 | 9.88 |
| adv | 0.1 | 128 | 2 | 0 | 0.902 | 0.987 | 0.137 | 0.134 | 7.07 |
| adv | 1 | 128 | 2 | 0 | 0.782 | 0.831 | 0.147 | 0.140 | 2.13 |
| adv | 3 | 128 | 2 | 0 | 0.599 | 0.608 | 0.147 | 0.142 | 0.66 |
| bott | 0 | 256 | 2 | 0 | 0.922 | 1.019 | 0.154 | 0.154 | n/a |
| adv | 0.001 | 256 | 2 | 0 | 0.929 | 1.013 | 0.154 | 0.154 | 13.03 |
| adv | 0.01 | 256 | 2 | 0 | 0.922 | 1.008 | 0.155 | 0.154 | 11.27 |
| adv | 0.1 | 256 | 2 | 0 | 0.927 | 1.023 | 0.162 | 0.161 | 7.72 |
| adv | 1 | 256 | 2 | 0 | 0.830 | 0.872 | 0.169 | 0.169 | 2.20 |
| adv | 3 | 256 | 2 | 0 | 0.644 | 0.630 | 0.167 | 0.167 | 0.71 |
(All 25 adv (lambda, d) cells and the 6 bott cells were computed; d = 32, 64, 512 show the same pattern; the per-fit collapse table is the aggregate output of `aggregate.py` on the D2 directories: 2 of 50 fits collapsed.)

### Selected on V at retention >= 0.9 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 2/2 | s0:d512, s1:d512 | 0.943 [0.920, 0.966] | 0.181 [0.171, 0.190] | 0.987 [0.975, 0.998] | 0.981 [0.974, 0.987] | 0.980 |
| pca | 2/2 | s0:d256, s1:d256 | 0.966 [0.937, 0.993] | 0.162 [0.153, 0.171] | 0.883 [0.860, 0.905] | 0.902 [0.888, 0.915] | 0.932 |
| noise | 2/2 | s0:r0.5, s1:r0.5 | 0.955 [0.935, 0.974] | 0.137 [0.129, 0.145] | 0.744 [0.720, 0.766] | 0.785 [0.770, 0.799] | 0.781 |
| leace | 2/2 | s0:r16, s1:r16 | 0.950 [0.920, 0.982] | 0.179 [0.170, 0.189] | 0.979 [0.962, 0.995] | 0.792 [0.776, 0.807] | 0.962 |
| sup_lin | 2/2 | s0:lin_d128, s1:lin_d128 | 1.013 [0.934, 1.097] | 0.138 [0.128, 0.148] | 0.747 [0.712, 0.783] | 0.690 [0.673, 0.708] | 0.762 |
| bott_mlp | 2/2 | s0:mlp_d256, s1:mlp_d128 | 1.010 [0.933, 1.094] | 0.145 [0.137, 0.154] | 0.788 [0.764, 0.811] | 0.742 [0.727, 0.756] | 0.790 |
| adv | 2/2 | s0:d128_n0.01, s1:d128_n0.01 | 0.995 [0.917, 1.078] | 0.135 [0.125, 0.146] | 0.732 [0.700, 0.768] | 0.660 [0.644, 0.676] | 0.736 |

Strongest floor F*(0.9) (lowest seed-mean V leakage): **sup_lin** (V L_rare 0.762; C L_rare 0.747)


##### Kill rule at tau=0.97, margin m=0.1 (pre-registered K1-K4), F*=pca

| seed | adv config | F* config | adv test ret. | F* test ret. | L(F*)-L(adv) [95% CI] | L(bott)-L(adv) [95% CI] | mid: L(adv)-L(F*) |
|---|---|---|---|---|---|---|---|
| 0 | n/a | | | | | | |
| 1 | n/a | | | | | | |

K1+K2 (>= m with equal utility) in 0/2 seeds; K1+K2+K3 in 0/2; K4 in 0/2. PASS requires K1+K2+K3+K4 in >= 2 of 2 seeds (all of them when only 2 seeds exist): **FAIL** (0 seeds pass all four).

##### Selected on V at retention >= 0.9 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 2/2 | s0:d512, s1:d512 | 0.943 [0.920, 0.966] | 0.181 [0.171, 0.190] | 0.987 [0.975, 0.998] | 0.981 [0.974, 0.987] | 0.980 |
| pca | 2/2 | s0:d256, s1:d256 | 0.966 [0.937, 0.993] | 0.162 [0.153, 0.171] | 0.883 [0.860, 0.905] | 0.902 [0.888, 0.915] | 0.932 |
| noise | 2/2 | s0:r0.5, s1:r0.5 | 0.955 [0.935, 0.974] | 0.137 [0.129, 0.145] | 0.744 [0.720, 0.766] | 0.785 [0.770, 0.799] | 0.781 |
| leace | 2/2 | s0:r16, s1:r16 | 0.950 [0.920, 0.982] | 0.179 [0.170, 0.189] | 0.979 [0.962, 0.995] | 0.792 [0.776, 0.807] | 0.962 |
| sup_lin | 2/2 | s0:lin_d128, s1:lin_d128 | 1.013 [0.934, 1.097] | 0.138 [0.128, 0.148] | 0.747 [0.712, 0.783] | 0.690 [0.673, 0.708] | 0.762 |
| bott_mlp | 2/2 | s0:mlp_d256, s1:mlp_d128 | 1.010 [0.933, 1.094] | 0.145 [0.137, 0.154] | 0.788 [0.764, 0.811] | 0.742 [0.727, 0.756] | 0.790 |
| adv | 2/2 | s0:d128_n0.01, s1:d128_n0.01 | 0.995 [0.917, 1.078] | 0.135 [0.125, 0.146] | 0.732 [0.700, 0.768] | 0.660 [0.644, 0.676] | 0.736 |

Strongest floor F*(0.9) (lowest seed-mean V leakage): **sup_lin** (V L_rare 0.762; C L_rare 0.747)

##### Kill rule at tau=0.9, margin m=0.0775 (pre-registered K1-K4), F*=sup_lin

| seed | adv config | F* config | adv test ret. | F* test ret. | L(F*)-L(adv) [95% CI] | L(bott)-L(adv) [95% CI] | mid: L(adv)-L(F*) |
|---|---|---|---|---|---|---|---|
| 0 | adv_d128_n0.01 | sup_lin_d128 | 0.988 | 1.011 | 0.015 [-0.003, 0.034] (fail; ret fail) | 0.100 [0.061, 0.140] (ok) | -0.031 (ok) |
| 1 | adv_d128_n0.01 | sup_lin_d128 | 0.999 | 1.013 | 0.015 [-0.004, 0.033] (fail; ret fail) | 0.011 [-0.004, 0.027] (fail) | -0.030 (ok) |

K1+K2 (>= m with equal utility) in 0/2 seeds; K1+K2+K3 in 0/2; K4 in 2/2. PASS requires K1+K2+K3+K4 in >= 2 of 2 seeds (all of them when only 2 seeds exist): **FAIL** (0 seeds pass all four).


### Selected on V at retention >= 0.97 (config per family and seed; numbers on sealed C)

| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |
|---|---|---|---|---|---|---|---|
| randproj | 0/2 | cannot reach tau | | | | | |
| pca | 2/2 | s0:d512, s1:d512 | 0.976 [0.949, 1.001] | 0.182 [0.173, 0.191] | 0.993 [0.981, 1.006] | 0.976 [0.968, 0.985] | 0.992 |
| noise | 0/2 | cannot reach tau | | | | | |
| leace | 2/2 | s0:r4, s1:r4 | 0.991 [0.969, 1.014] | 0.186 [0.176, 0.196] | 1.014 [0.997, 1.030] | 0.901 [0.887, 0.915] | 0.996 |
| sup_lin | 0/2 | cannot reach tau | | | | | |
| bott_mlp | 0/2 | cannot reach tau | | | | | |
| adv | 0/2 | cannot reach tau | | | | | |

Strongest floor F*(0.97) (lowest seed-mean V leakage): **pca** (V L_rare 0.992; C L_rare 0.993)


##### Kill rule at tau=0.97, margin m=0.1 (pre-registered K1-K4), F*=pca

| seed | adv config | F* config | adv test ret. | F* test ret. | L(F*)-L(adv) [95% CI] | L(bott)-L(adv) [95% CI] | mid: L(adv)-L(F*) |
|---|---|---|---|---|---|---|---|
| 0 | n/a | | | | | | |
| 1 | n/a | | | | | | |

K1+K2 (>= m with equal utility) in 0/2 seeds; K1+K2+K3 in 0/2; K4 in 0/2. PASS requires K1+K2+K3+K4 in >= 2 of 2 seeds (all of them when only 2 seeds exist): **FAIL** (0 seeds pass all four).


### 7.7 Digits n1s (rare-number tier), scale-aware re-analysis (post hoc thresholds, section 7.4; jobs `d_dig_check_gtr_s{0,1,2}`, 3 seeds, no new compute)
* **H1s PASSES: R_raw = 0.0408, R_prior = 0.0163, ratio 2.58 [1.53, 4.12]; absolute excess 0.025 [0.011, 0.040].** On the rare-number tier GTR-base vectors leak above the frequency prior, as the audit said. The absolute-0.05 H1 of section 6 was scale-blind; the primary-tier result "H1 fails" (n1: excess 0.010) stays on record but is a measurement failure of that tier (prior 0.204), not a statement that numbers are unrecoverable.
* **H2s fails / is undecidable.** The oracle that deletes every digit run keeps retention dev 1.005 / test 0.991 (H2(a) passes) but its attacker recovery is 0.036 [0.025, 0.049], i.e. it removes only about 20% of the excess (L = 0.79 [0.31, 1.35], denominator held at its bootstrap mean because the raw excess is tiny): the rest is inferred from context. The strongest floor at tau = 0.97 (noise r = 0.25-0.5, test retention 0.988) has L = 0.85 [0.33, 1.45]; headroom = 0.064 [-0.199, 0.329] < 0.10, lower bound below 0: **the comparison has a resolution of about +-0.26 in L at this attacker strength and cannot show a gap in either direction.**
* **Attacker suite is weak for this question**: three attackers trained on 3,716 documents (ridge 0.033, mlp 0.024, knn 0.041 on raw for n1s), no Vec2Text or external-decoder attacker; the n1s excess is 0.025 of R-precision with 47% of C docs having no n1s token. A stronger suite would raise R_raw and could change both H1s and H2s. Status: **undecidable at this power, not a resolved null.** No learned digit arm was run (not requested, outside the 2 GPU-h).

### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)

| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |
|---|---|---|---|---|---|---|
| const | n1s | 0.017 [0.011, 0.024] | 0.016/0.013/0.016 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| const | n1 | 0.204 [0.190, 0.218] | 0.205/0.204/0.205 | 0.00 [0.00, 0.00] | 0.0122 | 0.020 / 0.040 |
| raw | n1s | 0.041 [0.029, 0.055] | 0.033/0.024/0.041 | 1.00 [0.46, 1.63] | 0.3022 | 1.000 / 1.000 |
| raw | n1 | 0.214 [0.201, 0.228] | 0.204/0.215/0.187 | 1.00 [0.19, 1.79] | 0.3022 | 1.000 / 1.000 |
| raw|shuffled pairs | n1s | 0.013 [0.010, 0.017] | 0.009/0.012/0.006 | -0.13 [-0.36, 0.08] | 0.3022 | 1.000 / 1.000 |
| raw|shuffled pairs | n1 | 0.205 [0.191, 0.218] | 0.181/0.206/0.149 | 0.06 [-0.17, 0.32] | 0.3022 | 1.000 / 1.000 |
| oracle_mask_digits | n1s | 0.036 [0.025, 0.049] | 0.031/0.023/0.035 | 0.79 [0.31, 1.35] | 0.2994 | 1.005 / 0.991 |
| oracle_mask_digits | n1 | 0.215 [0.201, 0.228] | 0.202/0.215/0.183 | 0.99 [0.26, 1.68] | 0.2994 | 1.005 / 0.991 |
| oracle_mask_n2 | n1s | 0.035 [0.025, 0.048] | 0.031/0.026/0.035 | 0.77 [0.30, 1.30] | 0.2956 | 0.987 / 0.978 |
| oracle_mask_n2 | n1 | 0.214 [0.200, 0.228] | 0.206/0.214/0.185 | 0.94 [0.14, 1.72] | 0.2956 | 0.987 / 0.978 |


### Headroom check (pre-registered H1, H2; floors + oracle only)

Scale-aware H1s (post hoc): R_raw = 0.0408, R_prior = 0.0163, ratio = 2.58 [1.53, 4.12] (rule: >= 1.5, lower bound >= 1.25, absolute excess lower bound > 0): PASS

H1 measurability: R_raw - R_prior (n1s tier, worst of 3 attackers) = 0.025 [0.011, 0.040] (rule: >= 0.05 and CI excludes 0): FAIL

oracle_mask_digits: L_n1s = 0.786 [0.309, 1.348], retention dev 1.005 / test 0.991; H2(a) (both >= 0.97): PASS

| tau | strongest floor F* (by V) | F* configs | F* test retention | L_n1s(F*) [95% CI] | L_n1s(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2(b) (>= 0.10, lower bound > 0) |
|---|---|---|---|---|---|---|---|
| 0.970 | noise | s0:noise_r0.25, s1:noise_r0.25, s2:noise_r0.5 | 0.988 | 0.849 [0.332, 1.447] | 0.786 | 0.064 [-0.199, 0.329] | FAIL |
| 0.900 (fallback if no floor reaches 0.97 in every seed) | noise | s0:noise_r0.5, s1:noise_r0.5, s2:noise_r0.5 | 0.958 | 0.814 [0.310, 1.370] | 0.786 | 0.028 [-0.240, 0.287] | n/a |


## 8. D3: real PII entity spans as the secret (PII-Masking-300K); PRE-REGISTRATION v0 (2026-11-05, committed before any number on this dataset)

**Acknowledgement.** Numbers in this section are computed on the AI4Privacy `pii-masking-300k` dataset (HF `ai4privacy/pii-masking-300k`, © AI4Privacy 2024; synthetic text with annotated PII spans of fictitious people). Use is academic and non-commercial under the dataset's `LICENSE.md`; owner approval 2026-11-05 with the conditions of `docs/data_licences/WHITELIST.md`.
**Handling (conditions a-e).** The English jsonl files sit on the VM scratch disk (`~/work/data/d_embedpriv/pii`, outside the git checkout). Nothing derived from the text is committed or uploaded: no text samples, no embeddings, caches, trained adapters/attackers, or per-record/per-document outputs. Every D3 job sets `hf_allow: ["*.json","*.log","*.md"]`, its uploaded JSON holds aggregates only, logs print no text or values, per-document arrays and caches stay in scratch, and cluster-bootstrap aggregation is run on the VM (the arrays are not copied to the laptop). Non-commercial use only; if that changes the approval is void. Budget <= 4.0 GPU-h, queue priority 5, <= 8 CPU threads.
verdict_original of Focus D (rare-word NO-GO at tau 0.70, D2 FAIL) is untouched; D3 is a new, separately registered attempt on a different secret. Nothing here loosens an earlier kill.

### 8.1 Question and design
1. **Question.** With real PII entity spans as the secret and a retrieval-style utility, does a learned adapter on a frozen GTR-base embedding beat the cheap floors (additive noise/dither, PCA, random projection, rate bottleneck, closed-form linear erasure) at matched utility?
2. **Corpus.** English files of PII-Masking-300K (train 29,908 + validation docs; 28 entity labels, about 6.5 spans and 50 words per doc). Utility and secret are both defined on the same documents, so the "task implies the secret" trap (rare words in D) is avoided by construction: queries are built from the NON-PII text.
3. **Independent unit = template cluster**, not record. The data are synthetic and templated, so near-duplicates (same prose, different PII values) would leak between train and test.
4. **Secrets (attacker targets).** Primary **S2 = PII value tokens**: lower-cased alphanumeric tokens (length >= 2) of the annotated spans with document frequency >= 10 over the defender+attacker pools (D, A; never V, C). Metric: R-precision per document (k = number of the doc's vocabulary PII tokens; top-k of the attacker's scores; documents with none excluded). Descriptive sub-tiers: name tokens (GIVENNAME*, LASTNAME*), place tokens (CITY, STATE, COUNTRY, STREET, BUILDING, POSTCODE, SECADDRESS, GEOCOORD), others. Secondary **S1 = entity-type presence** (27 labels; CARDISSUER dropped, 5 occurrences): R-precision of the top-k types, k = number of types present.
5. **Attackers, always retrained on the released vectors** of the attacker pool: ridge, MLP (2 x 1024), cosine kNN (exactly as in section 2.3), worst case = max over the three of the mean R-precision (also per resample). Stochastic releases are sampled 4 times per pool document.
6. **Reference attackers (non-embedding and trivial).** R_prior = attacker given a constant vector (frequency prior). **B1** = the same three attackers on 256-d LSA features of TF-IDF of the PII-MASKED text with typed placeholders kept ("[EMAIL]" etc.), i.e. a bag-of-words baseline that is given the entity TYPES for free (the strongest non-embedding feature); **B1g** = LSA features of the masked text with every placeholder replaced by one generic token (context only, types unknown). Shuffled-pairs control: attackers trained with the bags permuted within the pool; must be within 0.02 of R_prior.
7. **Utility.** Known-item retrieval with a non-PII query: for each document two disjoint excerpts (12-16 contiguous words, one from each half) are cut from its PII-masked text with every placeholder deleted; the first excerpt of V / C documents is the evaluation query, the second excerpt of defender documents is the training query for learned maps. Corpus = the documents of the evaluated split, relevant document = the one the excerpt came from, nDCG@10 (single relevant item). Retention = nDCG@10(arm) / nDCG@10(raw) on the same queries. Content-free reference = random ranking; non-embedding reference = TF-IDF cosine on the source text. There is no task label to build a "label-only" arm from; the closest analogues are the TF-IDF reference and `sup_lin` (retrieval-trained linear map, the supervised floor). Queries are never noised; additive noise is applied to stored document vectors only (as in D).
8. **Splits and template certification (stage 0, CPU, before any model).** Near-duplicate detector on the PII-masked text: 3-word-shingle Jaccard, edges at Jaccard >= tau_dup, clusters = connected components (union-find). **Calibration on planted duplicates**: 1,000 random documents are copied with p in {0, 10, 20, 30, 40, 50}% of their context words replaced by random vocabulary words (PII values also re-drawn from other documents of the same type); the planted copies enter the pool but are discarded afterwards. tau_dup := the smallest threshold (most recall) such that the giant component holds <= 5% of the documents; it is **certified** iff planted recall (copy lands in its source's cluster) >= 0.95 at p = 20%, else STOP-0. Independent verification with a different detector (character 5-gram TF-IDF cosine, calibrated on the same planted copies): after splitting, the fraction of C documents whose nearest D/A/V neighbour exceeds the planted-copy 5th percentile at p = 30% must be <= 1%. A demonstration control (no gate): the kNN attacker on raw vectors under a record-level random split vs the cluster split, to show how much template leakage the record split would have added.
   Cluster-hash splits (fixed, seed-independent): **D** 25% of clusters (defender fit; 10% of D clusters held out as **Dval** for the in-loop adversary check), **A** 35% (attacker pool), **V** 10% (selection), **C** 30% (sealed). C documents, V documents and their queries never enter any fit. Value overlap across splits is legitimate (the attacker's vocabulary comes from D, A) and is reported as coverage.
9. **Bootstrap** over clusters (B = 1000, shared resamples across arms so differences are paired), 3 seeds for floors/controls, 2 seeds (0, 1) for learned arms. MDE80 = 2.8 x paired-bootstrap SD of L(F\*) - L(bott_mlp) at the operating point, printed from floors and the control only, before any adversarial number. Normalised leakage L = (R - R_prior)/(R_raw - R_prior).

### 8.2 Floors-first ordering and gates (written now)
* **Stage 0 (CPU):** tier sizes, cluster statistics, planted-duplicate calibration, independent verification. **STOP-0** if tau_dup is not certified, or if fewer than 800 clusters fall in C.
* **Stage 1 (GPU ~0.3 h):** embeddings (raw, PII-deleted oracle, excerpts), controls, reference attackers, raw attackers, record-vs-cluster demonstration. **H1 (measurability) PASS iff** on C, 3 seeds, for S2: gap-closing share G = (R_raw - R_ref)/(1 - R_ref) >= 0.10 with cluster-bootstrap 95% lower bound >= 0.05, where R_ref = max(R_prior, R_B1) (the raw embedding must beat both the frequency prior and the type-aware bag-of-words baseline); controls hold. (Scale-aware: G is the share of the remaining distance to perfect recovery that the raw embedding covers; the absolute threshold that failed in section 6 is not used.) **STOP-1** if H1 fails: the benchmark cannot measure embedding leakage at this attacker strength; report that and stop.
* **Stage 2 (floors + oracle, ~0.7 h):** floors random projection (d 32..512), PCA (16..512), noise (r 0.25..4), LEACE of the defender-pool PII-token subspace (r 4..256), `sup_lin` (d 32..512) and `bott_mlp` (the rate bottleneck, no adversary), plus the oracle (PII spans deleted from the text before embedding, queries untouched). Selection on V at tau in {0.97, 0.90}: lowest V worst-case S2 R-precision among configs with V retention >= tau, per family and seed; **F\*(tau)** = the floor family reaching tau in every seed with the lowest seed-mean V leakage. **H2 PASS iff** (a) oracle retention >= 0.97 on V and on C queries; (b) headroom = L(F\*, 0.97) - L(oracle) >= 0.10 with paired-bootstrap lower bound > 0 (fallback tau 0.90, flagged, if no floor reaches 0.97 in every seed). **STOP-2** if H2 fails: no headroom, no learned arm.
* **Stage 3 (learned adversarial arm, only after H1 and H2 pass; <= 1.5 GPU-h):** `adv` = bott_mlp + token-bag adversary with the D2-repaired loss (confusion term divided by its step-0 value, lambda relative; collapse check: reject a fit whose final retrieval loss exceeds ln(N_D) - 0.5). **Engagement check first, on lambda in {1, 3, 10, 30}, d = 256, seed 0:** a configuration is *engaged* iff both (E1) the in-loop head, scored on held-out Dval documents, has L_head = (R_head - R_prior)/(R_MLP,raw - R_prior) <= 0.5, where R_MLP,raw is a fresh MLP trained on raw D-fit vectors and scored on Dval (so head and yardstick are measured the same way); and (E2) the worst fresh attacker of the suite, retrained from scratch on the released vectors of the attacker pool and scored on V, has L_fresh <= 0.5. Memorisation floor (E3, calibration, not a gate): the same MLP trained on Gaussian-noise latents of the pool's shape paired with the true bags; in-sample scores are reported next to the held-out scores so that no in-sample number is read as engagement (the lesson of audit F-02). **If no configuration is engaged: STOP-3, the adversarial arm is closed with "engagement never achieved" and no K-test is run for it** (the floors, `bott_mlp` and `sup_lin` are then the complete learned-versus-floor evidence). If engaged: the engaged lambdas and their neighbours x d in {64, 128, 256, 512}, seeds 0, 1, K1-K4 of 2.5 at tau 0.90 and 0.97 with m(tau) = max(min(0.10, 0.5 x headroom(tau)), MDE80(tau)), K1-K4 required in both seeds.
* Verdict form: **verdict_D3_gate** (H1/H2 outcome) and **verdict_D3_learned** (engagement and K-test), both reported, with the earlier Focus D verdicts beside them.

### 8.3 Disclosure (before any D3 number)
Seen: the dataset schema and counts (29,908 English train documents, 28 labels, 6.5 spans and 50 words per document, 127,576 distinct values, no repeated masked text), the licence text; the D / D2 / digits results of sections 5-7 (including that deleting digits did not remove the digit-tier leakage because it is inferred from context, and that D2's adversarial arm did not engage fresh attackers at lambda 0.001-3). No text sample and no attacker/floor/oracle number on this dataset has been looked at. Design choices made with that background: the use of non-PII excerpts as queries, the type-aware B1 baseline, the held-out engagement test, G as the scale-aware gate.

### 8.4 Stage-0 outcome and Amendment A1 (made before any embedding, attacker or utility number; stage-0 statistics only)
Stage 0 ran (56 s, CPU, 8 threads; 37,854 English documents). Word-3-shingle detector on the masked text: planted-copy recall at 20% word replacement is 0.993 / 0.976 / 0.850 at Jaccard 0.20 / 0.25 / 0.30; the smallest threshold whose giant component is <= 5% of documents is **tau_dup = 0.25** (30,541 clusters, giant component 4.5%, 24% of documents in multi-document clusters, recall 0.976 at 20%, 0.28 at 30%): **certified by the registered criterion.** Splits by cluster hash: D 11,310 docs / 7,602 clusters (Dval 816 / 745), A 12,139 / 10,705, V 3,680 / 2,915, C 10,725 / 9,319 (>= 800 clusters in C).
**The independent verifier did NOT pass its registered criterion.** Character-5-gram TF-IDF cosine on the masked text: planted-copy 5th percentile 0.80 / 0.66 / 0.54 / 0.43 / 0.34 at 10 / 20 / 30 / 40 / 50% replacement; 26.1% of C documents have a D/A/V neighbour at or above the 30% level (0.54) and 72.9% at the 50% level; the natural nearest-neighbour cosine of C to non-C has median 0.42, 90th percentile 0.67, 99th 0.90. So shared boilerplate and word choice, not only word-order-preserving template copies, are common across the splits. The word-shingle clusters cannot remove that (they would need a giant component).
**Amendment A1 (tightening only; nothing is loosened).** Evaluation sets are purged: a C document is dropped if its nearest D/A/V document under the independent detector has cosine >= 0.54 (the planted-copy 5th percentile at 30% replacement), a V document likewise against D/A. D and A are not purged (the attacker legitimately has same-distribution data). All V/C numbers, queries and corpora below use the purged sets; the kept counts are reported next to every table. The purge makes the verifier criterion true by construction, so it is not a proof of independence by itself; the proof of effect is the record-split vs cluster-split demonstration (8.1 item 8) and the kept-set nearest-neighbour percentiles. Consequence stated up front: the kept C is the less templated part of C, so utility and leakage numbers are on a harder, more diverse test set than a random split would give.
Other stage-0 facts (no model involved): secret vocabulary = 2,194 PII value tokens with df >= 10 over D+A (name tier 46 tokens, place tier 682, other 1,466; the names are mostly unique per document, so the token-bag secret covers recurring categorical values, not unique strings such as e-mails or IPs: a limitation); 6.0 vocabulary tokens per C document, 22% of C documents have none (excluded from the means); 47% of a document's PII tokens are in the vocabulary. Queries: first excerpts of 13.6 words on average; non-embedding utility reference (TF-IDF cosine on the source text) known-item nDCG@10 = 0.895 on V and on C (random ranking 0.0004-0.0012).

### 8.5 D3 stage 1 result (3 seeds; jobs `d_pii_embed`, `d_pii_s1_s{0,1,2}`, priority 5, about 0.2 GPU-h; AI4Privacy pii-masking-300k, aggregates only)
Sealed C after the A1 purge: 7,927 documents in 7,365 clusters (V: 2,738 / 2,344). Intervals are cluster-bootstrap 95%. R = worst-of-3 retrained attackers (ridge / MLP / kNN), S2 value tokens (2,194), mean of 3 seeds.
| arm | R_val on C | name tier | place tier | type presence (S1) | known-item nDCG@10 on C |
|---|---|---|---|---|---|
| constant vector (frequency prior) | 0.103 [0.100, 0.107] | 0.112 | 0.167 | 0.268 | 0.0006 (random) |
| shuffled pairs (control) | 0.104 [0.101, 0.108] | 0.100 | 0.157 | 0.265 | 0.6821 |
| B1g: LSA of masked text, one generic placeholder (context only) | 0.140 [0.136, 0.145] | 0.187 | 0.173 | 0.579 | n/a |
| B1: LSA of masked text WITH typed placeholders (types free) | 0.183 [0.178, 0.188] | 0.249 | 0.208 | 0.974 | n/a |
| **raw GTR-base** | **0.241 [0.235, 0.248]** | 0.443 | 0.313 | 0.702 | 0.6821 (V 0.7188) |
| oracle: PII spans deleted before embedding | 0.132 [0.128, 0.136] | 0.191 | 0.159 | 0.510 | **0.8228 (retention 1.206 [1.188, 1.225])** |
Non-embedding utility reference (TF-IDF cosine on the source text, kept sets): known-item nDCG@10 0.938 (V) / 0.945 (C). Controls hold (shuffled pairs 0.104 vs prior 0.103).
Template-leakage demonstration (raw vectors, worst-of-3 R_val): record-level random split 0.256 vs cluster split with purge 0.241 (ridge 0.256 vs 0.241, MLP 0.215 vs 0.193, kNN 0.175 vs 0.145): the record split would have inflated the attackers by about 0.015 absolute (6% relative; kNN +21% relative), in the direction expected.
**H1 (registered): gap-closing share G_raw = 0.072 [0.066, 0.078] over max(prior, B1); rule >= 0.10 with lower bound >= 0.05: FAIL. STOP-1 as registered: floors, learned arms and the engagement check are NOT run under the D3 pre-registration.**
verdict_D3_gate = H1 FAIL (STOP-1). What is and is not measured: the raw embedding does leak PII values above the frequency prior (0.241 vs 0.103, 2.3x) and above the context-only baseline (0.241 vs 0.140), but not enough above a bag-of-words attacker that is handed the entity types (0.183), so the pooled tier cannot register a 10% gap-closing share at this attacker strength. Descriptive tiers (not registered as the gate): the name tier (46 tokens) has R_raw 0.443 vs B1 0.249 vs prior 0.112, G = 0.259 [0.198, 0.318]; the place tier 0.313 vs 0.208 vs 0.167, G = 0.132 [0.116, 0.148]; type presence is better predicted by B1 (0.974, types given) than by the raw embedding (0.702) and the context-only B1g (0.579). The oracle (PII deleted) removes 76% of the name-tier excess and 82% of the pooled excess and RAISES the utility by 21%, because the non-PII queries match PII-free documents better.

### 8.6 D3b (post hoc, clearly outside the registered gate): floors and oracle on the tiers where leakage is measurable
The registered D3 plan stops at STOP-1 and that verdict stands. Because the failure is dilution (the pooled tier is mostly values that the raw embedding does not carry beyond the entity type) and the VM and budget are free, the floors are run anyway, **post hoc**, on the **name tier** as the primary (chosen after seeing 8.5; the place tier and the pooled tier stay descriptive). This can inform a future separately registered attempt; it cannot reverse verdict_D3_gate, and no margin or pass is claimed from it. Rule fixed before the floor numbers: G_name >= 0.10 with lower bound >= 0.05 (already seen: 0.259); H2' = (a) oracle retention >= 0.97 on V and C (seen: 1.19 / 1.21) and (b) L_name(F\*, tau 0.97) - L_name(oracle) >= 0.10 with lower bound > 0; if (b) fails: no learned arm; if it passes: the engagement check of 8.2 (E1 AND E2, memorisation floor) on the name tier, and only if engaged the K1-K4 with margin max(min(0.10, 0.5 x headroom), MDE80). Same families, grids, selection and seeds as 8.2 (floors 3 seeds). Same handling conditions.

### 8.7 D3b floors result (post hoc, name tier primary; 3 seeds; jobs `d_pii_fa_s*`, `d_pii_fb_s*`; AI4Privacy pii-masking-300k, aggregates only)
Sealed C (7,927 documents, 7,365 clusters), cluster-bootstrap 95% intervals, selection on V per family and seed. L_name = (R_name - R_prior)/(R_raw - R_prior) (denominator held at its bootstrap mean). Oracle (PII deleted): L_name = 0.239 [0.120, 0.361], retention V 1.190 / C 1.206 (H2'(a) PASS).
| family at tau = 0.97 (V retention) | configs | retention C | R_name | L_name [CI] |
|---|---|---|---|---|
| random projection | feasible in 1 of 3 seeds | 0.964 | 0.423 | 0.94 [0.79, 1.08] |
| PCA | cannot reach tau | | | |
| additive noise (docs only) | r = 0.5 (3 seeds) | 0.971 [0.968, 0.974] | 0.350 | 0.72 [0.59, 0.83] |
| LEACE of the PII-token subspace | r = 128 | 0.984 [0.976, 0.992] | 0.408 | 0.89 [0.75, 1.04] |
| **sup_lin (retrieval-trained linear map)** | **d = 64** | **1.016 [1.004, 1.028]** | 0.249 | **0.41 [0.30, 0.54]** |
| bott_mlp (rate bottleneck, no adversary) | d = 64 | 1.017 [1.005, 1.029] | 0.247 | 0.41 [0.29, 0.53] |
**F\*(0.97) = sup_lin d = 64** (V leakage 0.433; C L_name 0.413). Headroom L(F\*) - L(oracle) = **0.174 [0.090, 0.255] at tau 0.97: H2'(b) PASS**; at tau 0.90 (F\* = sup_lin d = 32, C retention 0.870, L_name 0.32) headroom 0.085 [0.012, 0.155]: FAIL. MDE80 of F\* vs the bott_mlp control: 0.058 (tau 0.97), 0.057 (tau 0.90); the control and the supervised floor are indistinguishable (paired difference 0.006).
Reading: on the name tier a 64-d retrieval-trained map (linear or MLP, no adversary) already removes about 60% of the name leakage (L 0.41) while RAISING known-item nDCG@10 by 1.6% (retention above 1.1 for d >= 128: the supervised maps overfit to the PII-free excerpt task, which is also what the oracle exploits). Noise needs retention 0.97 to leave L 0.72; PCA/random projection/LEACE leave 0.9. Descriptive pooled tier (all 2,194 value tokens): the closed-form floor LEACE r = 128 is the strongest at tau 0.97 (retention 0.984, L = 0.32 [0.29, 0.35]) with headroom to the oracle (L 0.207) of 0.115 [0.084, 0.150]; at tau 0.90 headroom -0.016 [-0.044, 0.013] (LEACE L 0.19 is below the oracle).

### 8.8 D3b engagement check and conditional K-test (post hoc; registered here before any adversarial number)
H2' passes at tau 0.97 only, so the adversarial arm is tested at tau 0.97 only. Engagement check as in 8.2 with the primary tier = name: configs lambda in {1, 3, 10, 30}, d = 256, seed 0, two adversary designs: `adv` (the registered full-bag adversary over all 2,194 tokens) and `adv_tname` (confusion and adversary restricted to the 46 name tokens, a tighter target for a name-tier question). Engaged iff E1 (in-loop head on held-out Dval, L_head_name <= 0.5 against the fresh-MLP-on-raw yardstick) AND E2 (worst fresh attacker retrained on the released vectors, scored on V, L_fresh_name <= 0.5) AND the fit is not collapsed; memorisation floor on Gaussian latents reported (E3). If nothing is engaged: STOP-3 "engagement never achieved" for the adversarial arm. If engaged: full grid for the engaged design (lambda neighbours x d in {64, 128, 256, 512}, seeds 0 and 1), K1-K4 at tau 0.97 against F\*(0.97) = sup_lin d = 64 with margin m = max(min(0.10, 0.5 x 0.174), MDE80 0.058) = **0.087**, K1-K4 required in both seeds (K4 second tier = type presence). Bar: L_name(adv) <= about 0.33 at retention >= 1.006, i.e. most of the way to the oracle (0.24).

### 8.9 D3b engagement result and verdicts (post hoc, seed 0; job `d_pii_eng_s0`; AI4Privacy pii-masking-300k, aggregates only)
Eight adversarial fits (lambda 1, 3, 10, 30 x {full-bag `adv`, name-only `adv_tname`}, d = 256, D2-repaired loss with collapse check). **Engaged (E1 AND E2): 0 of 8.** Collapse check: 3 of 8 collapsed (`adv` lambda 10, 30; `adv_tname` lambda 30; final retrieval loss 9.2 vs ln 10,494 = 9.26).
| arm | retention V | final retrieval loss | pooled tier: E1 L_head / E2 L_fresh (ridge/MLP/kNN) | name tier: E1 L_head / E2 L_fresh (ridge/MLP/kNN) |
|---|---|---|---|---|
| adv lambda 1 | 1.035 | 3.43 | 0.21 / 0.68 (0.68/0.30/-0.07) | 2.00* / 0.91 (0.91/0.37/0.48) |
| adv lambda 3 | 0.841 | 5.50 | 0.24 / 0.68 (0.68/0.27/-0.13) | 0.00* / 0.97 (0.97/0.29/0.26) |
| adv lambda 10 / 30 | 0.69 / 0.81 | collapsed | -0.04 / -0.10 ; E2 0.72 / 0.74 | collapsed |
| adv_tname lambda 1 | 1.091 | 2.52 | n/a | -3.00* / 0.75 (0.75/0.14/0.25) |
| adv_tname lambda 3 | 1.066 | 3.12 | n/a | 1.50* / 0.88 (0.88/0.16/0.23) |
| adv_tname lambda 10 | 0.547 | 7.11 | n/a | 1.50* / 0.77 (0.77/0.21/0.13) |
| adv_tname lambda 30 | 0.678 | collapsed | n/a | collapsed |
\* E1 on the name tier is not informative: the yardstick gap on the 816 Dval documents is tiny (fresh MLP on raw defender vectors R = 0.194 vs frequency prior 0.161), so L_head swings between -3 and +2. On the pooled tier E1 is meaningful and PASSES at lambda 1 and 3 (the co-trained MLP head, scored on held-out Dval, falls to about the prior: L_head 0.21 and 0.24; its in-sample R stays at 0.35 / 0.20 vs the 0.12 memorisation floor, so in-sample numbers would have misled). **E2 fails everywhere: the freshly retrained RIDGE attacker keeps L_fresh 0.68-1.00 (the MLP and kNN attackers are partly fooled, 0.3-0.5, some negative).** The adversary hides what its own MLP head reads and leaves linearly decodable information in place; this is the same signature as D2, now with a held-out in-loop check and a memorisation-floor calibration. Memorisation floor on Gaussian latents: in-sample R 0.126 (name tier) / 0.119 (pooled), held-out 0.161 / 0.094.
**STOP-3: engagement never achieved. The adversarial arm is closed for this benchmark; no K-test was run.** (Not evidence that an adversary cannot work: only that these two designs with this loss, lambda range and step budget never made a fresh linear attacker lose.)

**Verdicts (this section's final form).**
* **verdict_D3_gate (registered): H1 FAIL, G = 0.072 [0.066, 0.078] < 0.10: STOP-1.** Stands.
* **D3b (post hoc, name tier, not a registered pass):** the raw embedding leaks names measurably (G = 0.259 [0.198, 0.318] over max(prior, B1)); oracle retention 1.21 (H2'(a) PASS); headroom of the strongest floor over the oracle 0.174 [0.090, 0.255] at tau 0.97 (PASS) and 0.085 [0.012, 0.155] at tau 0.90 (FAIL).
* **verdict_D3_learned:** the cheap non-adversarial 64-d retrieval-trained maps (`sup_lin`, `bott_mlp`) ARE the strongest floors (name tier L 0.41 at retention 1.016; they are statistically indistinguishable from each other, difference 0.006, MDE80 0.058); the adversarial arm never engaged. So the answer to the registered question is: **no learned adapter beat the cheap floors at matched retention, because the only learned adapters that trained are floors and the adversarial candidate failed the engagement check.** Pooled tier, descriptive: closed-form LEACE (r = 128) is the strongest floor at tau 0.97 (L 0.32 at retention 0.984; oracle 0.21).
* Limits: GTR-base only; token-bag attackers on 2,194 recurring PII value tokens (unique strings such as e-mails and IPs are not covered; no Vec2Text or LLM inverter); purged evaluation sets (7,927 C documents); the name tier has 46 tokens (about one per document); one adversary family; seed-0 engagement only; the benchmark is synthetic and templated (template leakage inflates attackers by about 0.015 absolute under a record split).
* Compute and handling: 11 jobs, about 0.94 worker-hours including guard waits (about 0.8 GPU-h), priority 5, <= 8 threads. Every job spec set `hf_allow` json/log/md; the HF repo holds 43 files under `runs/d_pii_*` (32 json, 11 log, nothing else), logs print arm names and numbers only, caches/embeddings/per-document arrays/vocabulary stay in `~/work/cache/d3_pii` and `~/work/runs/d_pii_*` on the VM, aggregation ran on the VM. Acknowledgement: AI4Privacy `pii-masking-300k`, © AI4Privacy 2024, academic non-commercial use.
