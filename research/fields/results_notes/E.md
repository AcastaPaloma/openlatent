# Focus E / SC-0 (single-cell batch integration, FLOORS ONLY): working log

Owner: Builder-E. Code: `research/fields/src/e_singlecell/`. Job specs: `research/fields/jobs/specs/e_*.yaml`. Raw per-cell predictions (no expression data, no embeddings):
HF `KuanKuanKuan/openlatent-fields` under `runs/e_*` (worker upload). Status: CHECK-ONLY focus (SC-0 of `hypotheses/RANKING_R2.md` section 6); NOT locked into `FOCUS.md`
unless SC-K0..K3 all clear (charter rule 6). Nothing below is a result unless it carries an interval and a seed count; single-seed numbers are marked "not yet replicated".

## 0. Licences (verified 2026-11-04 before any download)

| asset | licence | how verified | use here |
|---|---|---|---|
| scIB integration datasets (Luecken et al. 2022, figshare 10.6084/m9.figshare.12420968.v8): `Immune_ALL_human.h5ad` (33,506 cells, 10 batches, 16 types), `human_pancreas_norm_complexBatch.h5ad` (16,382 cells, 9 batches, 14 types) | **CC BY 4.0** | figshare API `articles/12420968` field `license` = {value 1, name "CC BY 4.0"} | read locally on the VM; never re-uploaded; only per-cell predictions leave the VM |
| HLCA core (Sikkema et al. 2023, CELLxGENE collection 6f6d381a-..., dataset 066943a2-fdac-4b29-b348-40cede398e4e, 584,944 cells, 107 donors, 14 datasets/11 studies) | **CC-BY 4.0** (CELLxGENE Discover data submission policy: published datasets "subject to a CC-BY 4.0 license") | CELLxGENE docs page "Contribute and Publish Data" (fetched); the Curation API v1 collection/dataset records carry NO per-dataset `license` key (key list checked), so the licence is the platform-wide policy, not a per-dataset field. The original studies' own terms were not individually checked. | same handling. Subsample of <= 100k cells only |
| harmonypy 2.1.0 | GPL-3.0 (as I recall; not re-verified) | pip metadata not read | run as a reference only, nothing copied or redistributed |
| scvi-tools 1.3.3, scib-metrics 0.5.6, scanpy 1.11.5, anndata 0.11.4, lightgbm 4.7.0 | BSD-3 / MIT class | from the package pages, not re-opened | used as libraries; scIB-E (AGPL-3.0) is NOT used or copied |
| scFM weights (Geneformer, UCE, scGPT) | not touched in SC-0 | | stage 2 only, licences to be verified then |

No gated or non-commercial asset is used. Nothing needs the programme owner's approval for SC-0.

## 1. Disclosure log (things seen before the numbers they could bias)

* 13:10-13:30Z VM clock: downloaded the three AnnData files; read the `obs` metadata tables (column names, per-batch/per-type/per-donor CELL COUNTS, donor-to-dataset nesting) in order to define the folds
  and the coarse-label maps below. No expression value was inspected beyond `X.max()` and whether the `counts` layer is integer. No embedding, floor, attacker or probe has been computed on real data
  as of this commit; the pipeline code is exercised on SYNTHETIC data only (section 6) and those numbers are not reported as results.
* Observed while preparing: the Pancreas `counts` layer is NOT integer (values 1.0019583, 2.0078535, ...: size-factor-normalised, not recoverable by multiplying with `obs.size_factors`);
  the scVI reference on Pancreas therefore uses `round(counts)` and is approximate. Immune `counts` is integer.
* The ranker looked at A/C outputs on the VM; I did not read A/C/D numbers beyond the notes and journal.

## 2. PRE-REGISTRATION v0 (committed before any model number on real data)

### 2.1 Data, units, folds
* Datasets (3): **Immune** (scIB, 33,506 cells, 10 batches in 5 studies, 16 fine types), **Pancreas** (scIB, 16,382 cells, 9 batches, 14 fine types, 10 supported), **HLCA core subsample**
  (<= 100,000 cells, 107 donors, 14 `dataset` batches in 11 studies, 61 `ann_finest_level` types).
* Secret (what the attackers recover) = the dataset's own batch label: Immune `batch`, Pancreas `tech`, HLCA `dataset` (the object's `batch_condition`). Donor-ID attackers (HLCA secondary secret) are NOT part of SC-0.
* Utility labels: fine = Immune `final_annotation`, Pancreas `celltype`, HLCA `ann_finest_level`; a fine type is SUPPORTED if it has >= 60 cells in the dataset (Immune 16 of 16; Pancreas 10 of 14: acinar, activated_stellate,
  alpha, beta, delta, ductal, endothelial, gamma, macrophage, quiescent_stellate; HLCA: decided on the realised subsample, recorded in the status log). Cells of unsupported types stay in every unsupervised fit
  and every attacker evaluation but are not used as probe labels. Coarse = hand-written lineage map for Immune (8 classes: T, NK, B, Plasma, Myeloid, pDC, Erythroid, Progenitor) and Pancreas (5 classes:
  endocrine, exocrine, stromal, endothelial, immune) (`common.py`), HLCA `ann_level_2` (11 classes; the spec's "<= 10" is exceeded by one, kept as is).
* Independent unit for intervals: **donor** in HLCA (107 donors, nested in datasets, none spans two datasets); in Immune and Pancreas there is no donor field, the unit is the **batch** (10 / 9), so those
  intervals are DESCRIPTIVE (the spec's fallback) and cannot carry a verdict alone.
* HLCA subsample (seeded 20261104, done before any model): cells with a finest label; finest types with < 2,000 cells in the core are kept whole; for the other cells each donor contributes at most `cap`
  cells chosen at random, `cap` = the largest multiple of 50 giving <= 100,000 cells in total (realised numbers in the status log).
* **Folds = leave-batch-group-out; every batch is held out exactly once; no study is split across train and held-out** (the held-out groups follow studies, so same-study batches never sit on both sides):
  Immune F1 {10X, Villani}, F2 {Freytag, Oetjen_A, Oetjen_P, Oetjen_U}, F3 {Sun_sample1_CS, Sun_sample2_KC, Sun_sample3_TB, Sun_sample4_TC};
  Pancreas F1 {inDrop1-4}, F2 {celseq, celseq2, smarter}, F3 {fluidigmc1, smartseq2};
  HLCA F1 {Banovich_Kropski_2020, Misharin_2021} (40 donors), F2 {Barbry_Leroy_2020, Nawijn_2021, Seibold_2020_10Xv2, Seibold_2020_10Xv3, Krasnow_2020} (37 donors),
  F3 {Jain_Misharin_2021_10Xv1/v2, Misharin_Budinger_2018, Meyer_2019, Lafyatis_Rojas_2019_10Xv1/v2, Teichmann_Meyer_2019} (30 donors).
  The folds are fixed (not seed dependent): this is the cross-fit by construction, and **seeds replicate noise realisations and attacker training only, not the held-out units** (the lesson of A's
  RandomState(1234)). Everything that is fitted (HVG list, scaling, PCA, LDA, LEACE, random projections, noise scale) is fitted on the fold's TRAINING batches only; held-out cells are only projected.
* Held-out cell types **H**: the supported types sorted by (coarse class, name); H = the types at positions 1, 4, 7, ... (0-based, i % 3 == 1) of that list (Immune: CD20+ B cells, CD14+ Monocytes,
  Monocyte-derived dendritic cells, HSPCs, CD8+ T cells; Pancreas: beta, endothelial, macrophage; HLCA: from the realised list). Label-supervised floors (`*_H` versions) are fitted without the cells of H-types;
  the probes are always trained on all supported types, so H measures "classes the code was not fitted on".
* Per-fold probe/evaluation type set: supported types with >= 20 training cells and >= 10 held-out cells in that fold (others skipped in that fold).

### 2.2 Encoder 0 (the "frozen released embedding" of SC-0)
Per fold: 2,000 highly variable genes (`scanpy.pp.highly_variable_genes`, flavor `seurat`, no batch key) on the fold's training cells of the provided log-normalised `X` (scIB: scran-pooled log1p;
HLCA: ln(1+CP10K), CELLxGENE `X`), per-gene standardisation with training mean/sd, clip +-10, PCA-50 (`sklearn` randomised SVD, random_state 0) on training cells; **z0** = the 50 PC scores of every cell.
All arms below act on z0 (except scVI, which acts on counts), so every inductive arm is a deterministic or noise-added function of one cell's z0 and never uses a batch id at inference.

### 2.3 Arms (floors and references only; no learned arm is run in SC-0)
Inductive floors (batch id not needed at inference): `raw50` (z0); `pca{8,16,32}`; `rp{16,32}` (Gaussian random projection of per-dimension-standardised z0); `noise_r{0.5,1,2}` (z0 + N(0,(r*sd_j)^2),
sd from training cells); `dith16_r{0.5,1}` (PCA-16 + the same noise: the H2-014 dithered label-free code); `ldaC` (LDA on coarse labels, C-1 dims), `ldaC_pca16`, `ldaC_pca32` (LDA dims padded with PCA of the
LDA-orthogonal residual of z0 up to 16 / 32 dims); `ldaF16` (first 16 discriminant directions of fine-label LDA; label ceiling for supervised codes); each of these four in a full-fit version and an `_H`
version (H-types hidden from the fit); `leace` (closed-form LEACE of the TRAINING batch id from z0, ridge 1e-3, 50 dims) and `leace_pca16`.
Transductive references (use batch ids and the held-out cells of the same batch; never counted as wins and never in the inductive-floor selection): `T_centre` (per-batch mean-centring of z0),
`T_bwhiten16` (PCA-16 of z0 then per-batch full whitening), `T_harmony` (harmonypy on the fold's z0 with the batch key, all cells), `T_scvi` (scVI latent, 30-d, batch covariate, 2,000 HVG seurat_v3 on
counts, all cells, default scvi-tools epochs heuristic, ONE seed, trained once per dataset: "not replicated").
Oracle label control (the Open Problems `embed_cell_types`): `onehot_oracle` = one-hot of the TRUE fine type of every cell including held-out cells; it is a ceiling for label metrics, not a method.
Controls: `ctrl_const` (constant vector), `ctrl_noise16` (iid N(0,1), 16-d), `ctrl_shuffle_batch` (`raw50` with the attacker's training batch labels permuted over cells), `ctrl_shuffle_labels` (`raw50`, probe
trained with permuted fine labels).

### 2.4 Metrics (all computed on the HELD-OUT batches of each fold, models retrained on the released embedding)
* **Leakage L.** Per fold, held-out cells are split into two halves (HLCA: donor-disjoint within each held-out dataset, halves alternate over donors sorted by a seeded hash; Immune/Pancreas: random halves of cells
  within each batch, because no finer unit exists). Four attackers are retrained from scratch on the released embedding of one half and score the other half, then swap (cross-fit; every held-out cell gets
  one test prediction per attacker): `linear` (class-balanced softmax, L-BFGS, L2 in {1e-3,1e-1} chosen on an inner split), `mlp` (2 x 256, early-stopping epoch from an inner split), `knn` (cosine, k in {10,30}),
  `lgbm` (200 trees, lr 0.1, 15 leaves, early stopping on an inner split). Features standardised with the attacker's own training statistics. Score = balanced accuracy over the fold's held-out batches
  (chance_f = 1/B_f, B_f = 2/4/4 Immune, 4/3/2 Pancreas, 2/5/7 HLCA). Pooled accuracy A_k = mean over folds; chance = mean over folds. **Worst case = max_k A_k** (taken inside every bootstrap resample).
  **L = (worst - chance) / (worst_raw50 - chance)** (raw50 -> 1, chance -> 0; spec: "normalised to raw and chance"). The label-only floor (predict the batch from the true fine type with training-half
  frequencies) is computed and reported next to chance as context.
* **Fine-label utility U_fine.** Probes trained on the fold's training-batch cells (supported types), scored on held-out-batch cells: class-balanced `linear` softmax (L2 1e-2) and cosine `knn` (k=15);
  macro-F1 pooled over folds over the types present; **U = max(linear, knn)** per resample. `U_hold` = same, macro over H-types only, using the `_H` versions of label-supervised arms.
  Chance reference u0 = macro-F1 of `ctrl_const`. "Above-chance" = U - u0.
* **Retention R** (context, in no kill condition): fraction of a held-out cell's 10 nearest z0 neighbours among the TRAINING-batch cells that are among its 30 nearest in the released space (<= 3,000 held-out
  cells and <= 30,000 training cells per fold, fixed subsample).
* **scIB-style scores** (context, in no kill condition): scib-metrics 0.5.6 bio-conservation and batch-correction aggregates on held-out cells (<= 8,000 per fold) for a subset of arms, labelled "scib-metrics, not
  comparable to the original scIB or to published totals". Reported if time allows; its absence does not affect any verdict.

### 2.5 Intervals, seeds, selection
* Seeds 0, 1, 2 for every stochastic ingredient (random projections, noise draws, cross-fit half assignment, attacker initialisation). The folds and the held-out units are identical across seeds, so the
  seed spread is NOT a replication of the test units (cross-fit across folds is). Point estimate = mean over seeds; resamples are shared across arms and seeds (paired).
* Bootstrap: B = 1000 resamples of UNITS (HLCA donors; Immune/Pancreas batches, descriptive), metrics recomputed on the resampled cells' predictions; 95% percentile intervals. HLCA also gets a
  study-cluster bootstrap (resample the 11 studies, then donors inside) as a conservative second interval for the headline quantities.
* "Inductive floor selection" (needed for K3): for each outer fold f, among the inductive-floor configurations (every arm in 2.3 except `T_*`, `onehot_oracle`, controls, `_H` versions, `ldaF16`) pick the one
  with the highest U_fine pooled over the OTHER folds subject to its pooled L over the other folds <= L_ref(other folds) + 0.05; its held-out predictions on fold f are used for fold f.
  (V = the other folds; no held-out information of fold f enters its choice.) The test-selected best configuration is also reported as an optimistic bound.
* The "best transductive reference" of a dataset = the `T_*` arm with the highest pooled U_fine (a reference, so choosing it on the same data is not a selection effect on any claim).

### 2.6 Power audit (SC-K0)
Floor-vs-floor, on HLCA donors (units): for each of the 6 pairs among {`pca16`, `rp16`, `dith16_r0.5`, `ldaC_pca16`} compute dU = U_fine(a) - U_fine(b) (seed 0 predictions), its donor-bootstrap SE, the
null distribution of dU by a paired within-donor permutation (for every donor independently swap the two arms' predictions with probability 1/2, 2,000 permutations), the two-sided 5% critical value
c = 97.5th percentile of |dU_null|, and **MDE80 = c + 0.8416 x SE** (the true difference detectable with 80% power). The reported MDE80 is the MEDIAN over the 6 pairs. The same table is printed for Immune
and Pancreas (batch units, descriptive).

### 2.7 Kill conditions (any one stops SC-0 and Focus E; text from RANKING_R2 section 6, interpretations marked [I])
* **SC-K0 instrument.** HLCA subsample has < 8 studies or < 40 donors in held-out folds, or the floor-vs-floor MDE80 on macro-F1 (2.6, HLCA) is > 0.05. [I] "held-out folds" = all folds jointly, since every
  donor is held out once.
* **SC-K1 measurability.** Raw batch attacker on held-out batches < 3x chance or < 0.15 above chance, or any control > chance + 0.03. [I] (a) The literal 3x-chance arm is arithmetically impossible whenever
  mean chance > 1/3 (Immune mean chance 0.333, Pancreas 0.361), so it is applied as acc >= min(3 x chance, 0.85); the 0.15-above-chance arm is literal. (b) controls: worst-of-4 attacker accuracy of
  `ctrl_const`, `ctrl_noise16`, `ctrl_shuffle_batch` <= chance + 0.03, and `ctrl_shuffle_labels` U <= u0 + 0.03. (c) Evaluated per dataset; a dataset that fails K1 is excluded from K2/K3; SC-0 is killed
  by K1 if HLCA fails or fewer than 2 datasets remain.
* **SC-K2 under-determination.** `ldaC_pca16` keeps > 70% of raw50's above-chance fine-label / held-type macro-F1: r_fine = (U_fine(ldaC_pca16) - u0)/(U_fine(raw50) - u0) and r_hold = same with U_hold and
  `ldaC_pca16_H`. [I] The label-only floor "holds" for a dataset iff **both** r_fine > 0.70 and r_hold > 0.70 (the reading "either" is printed beside it); SC-0 is killed by K2 if it holds on >= 2 of the 3
  datasets including HLCA (same aggregation as K3).
* **SC-K3 headroom.** The selected best inductive floor (2.5) at batch leakage <= L(best transductive reference) + 0.05 is within 0.05 macro-F1 (U_fine) of that reference on >= 2 of the 3 datasets, one of them
  HLCA. [I] "within" = U_ref - U_floor <= 0.05; if no inductive configuration reaches L <= L_ref + 0.05 in the selection folds, that dataset counts as "headroom exists" (K3 does not fire there).
* **GO for stage 2** = none of K0..K3 fires. Otherwise NO-GO, with the first-failing condition named. A K0 firing is reported as "NO-GO (instrument)": the answer is then not "no headroom" but "cannot be decided with these units".
* Sensitivity prints (they do not change the verdict): K2 under "either"; K3 with the test-selected floor; K3 with `U_hold` and `_H` versions; HLCA study-cluster intervals.

### 2.8 Stage-2 claim (only after GO; NOT run in SC-0)
Label-free AE+adversary / cVAE / dither exceed the best inductive floor by >= +0.05 macro-F1 at matched L, 95% CI above 0 (bootstrap over donors, folds as repeats), >= 2 of 3 seeds, V-selected,
power audit first. Budget <= 5 GPU-h and one builder-day, one attempt.

### 2.9 Interpretations that go beyond the spec text (listed so they cannot be moved later)
1. 3x chance capped at 0.85 (K1). 2. K2 "both"; K2 and K3 share the "2 of 3 incl. HLCA" aggregation. 3. K3's V-selection is leave-one-fold-out over the other folds. 4. Immune/Pancreas intervals are descriptive
(units = batches). 5. HLCA coarse = `ann_level_2` with 11 classes. 6. scVI is one seed and the Pancreas counts are rounded normalised counts. 7. `U = max(linear, knn)` (best-case probe) and L uses the worst-case attacker.
8. Any change after the first real number is appended as a dated amendment below and may not relax a kill condition.

## 3. Amendments (each dated; none may relax a kill condition)

* **A1, 13:50Z VM clock, BEFORE any real-data model number (found while exercising the pipeline on the synthetic set; no real number existed).**
  (a) Macro-F1 chance reference. Section 2.4 set u0 = macro-F1 of `ctrl_const`; the synthetic run showed that a class-balanced probe fed noise or permuted labels predicts all classes at random and scores about 1/K
  (0.11 with 9 types) while a constant predictor scores about 1/K^2 (0.05), so `ctrl_const` is the wrong zero and would have made the K1 control check fail by construction. **u0 is now the analytic macro-F1 of a classifier that
  predicts uniformly at random over the K evaluated types**, F1_c = 2 pi_c (1/K) / (pi_c + 1/K) averaged over the classes present (pi_c = class share of the evaluated held-out cells; for U_hold averaged over the H classes, K still all
  evaluated types). K1 controls: U(`ctrl_const`), U(`ctrl_noise16`), U(`ctrl_shuffle_labels`) each <= u0 + 0.03. K2's "above chance" uses this u0 (and the H-class version for r_hold). This cannot move K2/K3 in a
  favourable direction by itself: it changes the zero of both numerator and denominator of r by the same amount.
  (b) Bootstrap chance. In a resample a held-out batch whose donors are all absent has no cells; balanced accuracy and chance (1/B_f) are now computed over the held-out batches PRESENT in the resample (a constant predictor
  otherwise scored 0.83 in such resamples). Point estimates are unchanged.

## 4. Status log

* 13:22Z `prep.py` done for Immune and Pancreas (cached under `~/work/cache/e_singlecell`, not in git). HLCA prep done 13:28Z: cap 750 cells per donor for the non-rare types, 28 rare finest types kept whole,
  **97,232 cells, 107 donors, 14 datasets, 11 studies, 60 supported fine types** (61 minus nothing: `Lymphatic EC proliferating`, 28 cells, is the only one below 60), 24,869 genes expressed in >= 10 cells.
  Held-out types H for HLCA (20): Basal resting, Deuterosomal, Goblet (subsegmental), Multiciliated (nasal), Suprabasal, AT1, EC aerocyte capillary, EC venous pulmonary, Alveolar fibroblasts, Pericytes,
  Lymphatic EC differentiating, CD4 T cells, Plasma cells, Alveolar Mph CCL3+, Alveolar macrophages, DC2, Migratory DCs, Plasmacytoid DCs, Smooth muscle FAM83D+, SMG serous (bronchial).
  Cells per held-out fold: F1 32,326 (40 donors), F2 38,897 (37 donors), F3 26,009 (30 donors). Two HLCA coarse classes (`Hematopoietic stem cells`, `Mesothelium`) contain a single finest type each.
  Held-out types: Pancreas {beta, endothelial, macrophage} (endothelial and macrophage are the only members of their coarse class, so `ldaC_*_H` never sees those lineages).
* 13:34Z pipeline exercised on a SYNTHETIC dataset (CPU, 9,000 cells, 12 donors, 6 batches, 9 types): evaluation job, aggregator, power audit and verdict all run end to end (4.4 min on CPU). Numbers from that run are not reported.
