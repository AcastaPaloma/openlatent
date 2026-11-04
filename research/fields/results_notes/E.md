# Focus E / SC-0 (single-cell batch integration, FLOORS ONLY): working log

Owner: Builder-E. Code: `research/fields/src/e_singlecell/`. Job specs: `research/fields/jobs/specs/e_*.yaml`. Raw per-cell predictions (no expression data, no embeddings):
HF `KuanKuanKuan/openlatent-fields` under `runs/e_*` (worker upload). Status: CHECK-ONLY focus (SC-0 of `hypotheses/RANKING_R2.md` section 6); NOT locked into `FOCUS.md`
unless SC-K0..K3 all clear (charter rule 6). **VERDICT 2026-11-04 14:58Z: NO-GO for stage 2 (section 5.1); Focus E is not locked.** Nothing below is a result unless it carries an interval and a seed count; single-seed numbers are marked "not yet replicated".

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

* 13:59Z (AFTER the pre-registration, after the first real numbers): I ran the aggregator on seed 0 of each dataset with B=100 as a pipeline check and read the tables. Consequences, all logged: (i) I noticed that the registered K3 reference rule
  ("the T_* arm with the highest U_fine") picked per-batch centring, a reference that leaks more than raw, on HLCA seed 0 (scVI was not yet available); I did NOT change the rule, I added per-reference and Pareto printouts (`verdict.py`) as
  sensitivities; (ii) bootstrap L is set to NaN when the resampled raw attacker excess is < 0.10 (the cluster bootstrap produced a ratio of -99,000); point estimates are unaffected; (iii) `scib_ctx.py` was made per-fold after an XLA crash;
  (iv) the dense floor frontier (A3) was specified before the first number but run afterwards. No kill condition, threshold or selection rule was altered after any real number.

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

* **A1, 13:37Z VM clock, BEFORE any real-data model number (found while exercising the pipeline on the synthetic set; no real number existed).**
  (a) Macro-F1 chance reference. Section 2.4 set u0 = macro-F1 of `ctrl_const`; the synthetic run showed that a class-balanced probe fed noise or permuted labels predicts all classes at random and scores about 1/K
  (0.11 with 9 types) while a constant predictor scores about 1/K^2 (0.05), so `ctrl_const` is the wrong zero and would have made the K1 control check fail by construction. **u0 is now the analytic macro-F1 of a classifier that
  predicts uniformly at random over the K evaluated types**, F1_c = 2 pi_c (1/K) / (pi_c + 1/K) averaged over the classes present (pi_c = class share of the evaluated held-out cells; for U_hold averaged over the H classes, K still all
  evaluated types). K1 controls: U(`ctrl_const`), U(`ctrl_noise16`), U(`ctrl_shuffle_labels`) each <= u0 + 0.03. K2's "above chance" uses this u0 (and the H-class version for r_hold). This cannot move K2/K3 in a
  favourable direction by itself: it changes the zero of both numerator and denominator of r by the same amount.
  (b) Bootstrap chance. In a resample a held-out batch whose donors are all absent has no cells; balanced accuracy and chance (1/B_f) are now computed over the held-out batches PRESENT in the resample (a constant predictor
  otherwise scored 0.83 in such resamples). Point estimates are unchanged.

* **A2, 13:41Z, before any real-data model number.** scVI training length: the scvi-tools heuristic would give 400 epochs on Pancreas and 241 on Immune (about 15 GPU-min each, over the GPU budget); capped at `--max_epochs 100`
  for Immune and Pancreas; HLCA keeps the default heuristic (82 epochs). The scVI reference is a transductive context arm and is excluded from every selection, so this cannot decide K2/K3 except through
  the choice of the "best transductive reference", which is the arm with the highest U_fine among the four. A shorter-trained scVI can only LOWER the reference, i.e. make K3 easier to fire.
* **A3, 13:49Z, before any real-data model number (additive code only).** A dense floor frontier (`dithN_rR` for N in {8,16,32}, R in {0.25,0.5,1,2,4}; `pca{4,6,12,24}`; `ldaC_pca{8,24}`) can be run as separate jobs
  (`evaluate.py --arms dense`). It is NOT part of the pre-registered K3 grid; it exists to test whether a "no inductive floor reaches L_ref + 0.05, so headroom exists" reading is an artefact of holes in the floor grid
  (the ranker's note on A). It will be reported as a clearly labelled post-hoc sensitivity and cannot flip a pre-registered verdict by itself.

## 4. Status log

* 13:22Z `prep.py` done for Immune and Pancreas (cached under `~/work/cache/e_singlecell`, not in git). HLCA prep done 13:28Z: cap 750 cells per donor for the non-rare types, 28 rare finest types kept whole,
  **97,232 cells, 107 donors, 14 datasets, 11 studies, 60 supported fine types** (of 61; `Lymphatic EC proliferating`, 28 cells, is the only one below 60), 24,869 genes expressed in >= 10 cells.
  Held-out types H for HLCA (20): Basal resting, Deuterosomal, Goblet (subsegmental), Multiciliated (nasal), Suprabasal, AT1, EC aerocyte capillary, EC venous pulmonary, Alveolar fibroblasts, Pericytes,
  Lymphatic EC differentiating, CD4 T cells, Plasma cells, Alveolar Mph CCL3+, Alveolar macrophages, DC2, Migratory DCs, Plasmacytoid DCs, Smooth muscle FAM83D+, SMG serous (bronchial).
  Cells per held-out fold: F1 32,326 (40 donors), F2 38,887 (37 donors), F3 26,019 (30 donors). Two HLCA coarse classes (`Hematopoietic stem cells`, `Mesothelium`) contain a single finest type each.
  Held-out types: Pancreas {beta, endothelial, macrophage} (endothelial and macrophage are the only members of their coarse class, so `ldaC_*_H` never sees those lineages).
* 13:34Z pipeline exercised on a SYNTHETIC dataset (CPU, 9,000 cells, 12 donors, 6 batches, 9 types): evaluation job, aggregator, power audit and verdict all run end to end (4.4 min on CPU). Numbers from that run are not reported.
* 13:40Z queued (all validated with `yaml.safe_load`, specs in `jobs/specs/e_*.yaml`): `e_scvi_{immune,pancreas,hlca}` (scVI reference, GPU), `e_floors_{immune,pancreas,hlca}_s{0,1,2}` (everything in 2.3 except
  `T_scvi`; the controls `ctrl_shuffle_*` included), `e_tscvi_*` (the `T_scvi` arm for seeds 0-2, waits for the latent). **Priority 4 (reserved band) was used ONCE for the first wave: the three `e_scvi_*` jobs and the three
  seed-0 floors jobs, as the new-focus rule allows; seeds 1-2 are at 10-11 and `e_tscvi` at 12.** Budget: scVI ~15 GPU-min in total; the floors jobs are CPU-bound attacker evaluation (<= 8 threads) that hold a GPU slot.
  Encoder-0 warm-up for all folds took 2-9 s per fold on CPU (`warm.py`; finiteness check only, no metric).
* 13:56Z-14:05Z `e_floors_hlca_s0` (949 s), `e_floors_immune_s0`, `e_floors_pancreas_s0` finished; seed-0 pipeline check (B=100) read at 13:59Z (see disclosure). 14:05-14:31Z seeds 1-2, `e_tscvi_*`, `e_dense_*`, `e_scib_*` finished
  (`e_scib_hlca` crashed twice in fold 2 with an XLA CPU JIT allocation failure; retried per fold as `e_scib_hlca_c`). 14:31-14:50Z final aggregation (B=1000; HLCA also B=300 study-cluster) and post-hoc dense-frontier aggregation.
  Independent random-forest re-check of the per-batch-centring leakage finding run on CPU at ~14:10Z.
* 14:58Z verdict recorded in section 5.1: NO-GO (SC-K2 3/3 incl. HLCA; SC-K1 on HLCA and Immune; SC-K0 and SC-K3 clear). Final reply to the main agent follows.

## 5. Results (run 2026-11-04 13:40-14:50Z VM clock; 3 datasets x seeds 0,1,2 x 3 folds; every number below comes from `src/e_singlecell/aggregate.py` on the stored per-cell predictions)

Raw outputs: HF `KuanKuanKuan/openlatent-fields` `runs/e_floors_*`, `runs/e_tscvi_*`, `runs/e_dense_*`, `runs/e_scvi_*`, `runs/e_scib_*` (predictions and logs only: no expression values, no embeddings). Aggregates in git:
`research/fields/results/e_sc0/{hlca,immune,pancreas}_final.{md,json}` (the pre-registered grid, B=1000), `*_dense.{md,json}` (post-hoc sensitivity, seed 0, B=200), `scib_ctx_*.json`, `scvi_info_*.json`.
Intervals: 95% bootstrap over donors (HLCA, 107 units) or over batches (Immune 10, Pancreas 9: DESCRIPTIVE), resamples shared across arms and seeds. All three seeds share the same held-out units, so the seeds
do not replicate the units (section 2.5). The run commits differ between jobs because workers pull before each job; the only run-time code change between the first and the last floors job is the additive dense-arm block
(commit afa1624), inactive unless `--arms dense` is passed.

### 5.1 Verdict
**NO-GO for stage 2, as pre-registered. SC-K2 fires on 3 of 3 datasets (HLCA included); SC-K1 fires on HLCA and Immune (reading 2.9-1); SC-K0 and SC-K3 clear. Focus E is NOT written into FOCUS.md. No learned arm was run.**
SC-K2 alone decides it under every reading I can construct (both/either, donor or study-cluster intervals, seed-averaged). The verdict does not rest on the capped 3x-chance reading of K1, and it does not rest on the K3 reference rule.

| quantity | hlca | immune | pancreas |
|---|---|---|---|
| cells / units / studies | 97232 / 107 / 11 | 33506 / 10 / 5 | 16382 / 9 / 9 |
| chance (mean 1/B_f) | 0.281 | 0.333 | 0.361 |
| K0 median MDE80 (floor vs floor, macro-F1) | 0.035 | 0.116 | 0.098 |
| K1 raw50 worst attacker acc [CI] | 0.670 [0.644, 0.802] | 0.752 [0.619, 0.827] | 0.937 [0.902, 0.950] |
| K1 threshold min(3 x chance, 0.85) | 0.843 | 0.850 | 0.850 |
| K1 controls ok / pass | True / False | True / False | True / True |
| K2 r_fine [CI] | 0.77 [0.75, 0.79] | 0.97 [0.93, 1.03] | 0.91 [0.86, 0.95] |
| K2 r_hold [CI] | 0.82 [0.80, 0.84] | 0.92 [0.91, 0.94] | 0.90 [0.87, 0.93] |
| K2 label-only floor holds (both / either) | True / True | True / True | True / True |
| K3 best reference by U (U, L) | T_scvi (0.816, 0.80) | T_scvi (0.787, 1.16) | T_centre (0.941, 1.03) |
| K3 selected floor (U, L) / gap U_ref - U_floor [CI] | (0.700, 0.78) / 0.116 [0.099, 0.136] | (0.731, 0.99) / 0.056 [0.020, 0.115] | (0.947, 0.99) / -0.006 [-0.013, 0.000] |
| K3 fires (registered / test-selected floor) | False / False | False / False | True / True |
| per-reference: T_harmony U, L; best floor U; gap [CI] | 0.668, 0.37; no floor reaches L | 0.702, 0.76; noise_r0.5 0.643; 0.059 [0.001, 0.190] | 0.911, 0.54; dith16_r1 0.662; 0.249 [0.202, 0.312] |
| per-reference: T_scvi U, L; best floor U; gap [CI] | 0.816, 0.80; noise_r0.5 0.754; 0.062 [0.050, 0.073] | 0.787, 1.16; ldaC_pca32 0.734; 0.053 [0.017, 0.113] | 0.941, 1.00; leace 0.947; -0.006 [-0.020, 0.003] |

(Per-reference rows: each transductive reference taken alone as the comparator, test-selected best inductive floor with L <= L_ref + 0.05; this is a sensitivity, the registered K3 uses the single reference with the highest U_fine and V-selection.)

### 5.2 Reading (observations first, hypotheses marked)
1. **The instrument is adequate on HLCA and inadequate on the two scIB sets (K0).** 11 studies, 107 donors; floor-vs-floor MDE80 on macro-F1 is 0.028-0.057 over the six pairs, median **0.035** (<= 0.05), so a +0.05 difference between two arms
   is resolvable there (A's lesson: 12-48 units could not do this). With batches as units, Immune and Pancreas give 0.116 and 0.098: they cannot adjudicate +0.05 and serve as point-estimate corroboration only.
2. **K2: a 16-d code keeps most of the fine-label information, with or without labels.** `ldaC_pca16` retains r_fine = 0.77 [0.75, 0.79] (HLCA; study-cluster interval [0.72, 0.80]), 0.97 [0.93, 1.03] (Immune), 0.91 [0.86, 0.95] (Pancreas) of raw's
   above-chance fine-label macro-F1, and r_hold = 0.82, 0.92, 0.90 of the held-type macro-F1. The LABEL-FREE `pca16` retains the same (0.78 / 0.98 / 0.93): the registered statistic fires because 16 dimensions suffice, not because the
   coarse label carries the structure. The pure label-only code without PCA padding does collapse on two datasets (`ldaC`: r_fine 0.52 HLCA, 0.84 Immune, 0.51 Pancreas; r_hold 0.69, 0.86, 0.80), which is the A/C regime the spec had in
   mind, but K2 was registered on the padded code. HLCA is the closest call (0.77 against 0.70).
3. **What the label-only/PCA codes do not do is hide the batch.** L (normalised worst-case retrained attacker, raw = 1) is 0.84 (`ldaC_pca16`), 0.93 (`pca16`), 0.95 (`leace`) on HLCA; the floors that do lower L pay for it in utility:
   HLCA `noise_r2` L 0.47 at U 0.43, `noise_r1` L 0.70 at U 0.68, `noise_r0.5` L 0.83 at U 0.75. LEACE of the training batch ids removes almost nothing out of sample (L 0.95 / 0.90 / 0.98), the A pattern again.
4. **K3 clears: label-free batch-aware references beat the inductive floors at matched leakage** (this is the headroom premise of H2-010, "Hh"). HLCA, best reference by U = scVI (U 0.816, L 0.80; one seed, 82 epochs): the V-selected inductive floor at
   L <= 0.85 reaches U 0.700 (gap **+0.116 [0.099, 0.136]**; test-selected best floor `noise_r0.5` 0.754, gap +0.062 [0.050, 0.073]). Harmony (U 0.668, L 0.37) has no inductive floor within its leakage in the registered grid
   (lowest `noise_r2`, L 0.47); in the post-hoc dense grid the best floor at L 0.45 is `dith32_r2` with U 0.312: gap **+0.354 [0.334, 0.371]**. Immune is borderline (+0.056 [0.020, 0.115], descriptive), Pancreas fires (the best reference by U is
   per-batch centring, U 0.941 L 1.03, which a LEACE floor matches; its scVI is approximate: normalised counts rounded, ties raw). Registered K3 needs 2 of 3 including HLCA, so it does not fire.
   This is the known gap between batch-aware and batch-blind representations; SC-0 says nothing about whether a learned INDUCTIVE map can close it (that is the stage-2 claim, not run).
5. **K1: the raw batch signal is only moderately readable from a 50-d PCA code when the attacker must generalise across donors/samples.** Raw worst-case attacker 0.670 [0.644, 0.802] on HLCA (2.4x its mean chance of 0.281;
   fold chances 0.50 / 0.20 / 0.14 give 0.61 / 0.87 / 0.55), 0.752 on Immune (the Sun samples fold, chance 0.25, gives 0.50), 0.937 on Pancreas. Cell-type composition alone predicts the batch at 0.452 (HLCA), 0.583 (Immune), 0.536 (Pancreas)
   (label-only floor), and the `onehot_oracle` control reproduces it (0.456 / 0.583 / 0.536 attackers), a consistency check of the pipeline. All controls pass (constant, noise, batch-label shuffle <= chance + 0.013; probe controls <= u0 + 0.002).
   K1 fires on HLCA and Immune only because the literal "3x chance" is unattainable at these fold chances and I capped it at 0.85 (2.9-1); with the +0.15-above-chance arm alone all three datasets pass.
6. **By-product, not tested further: per-batch normalisation INCREASES the leakage seen by retrained nonlinear attackers.** `T_centre` L 1.54 [1.40, 1.80] (HLCA), 1.23 (Immune), 1.03 (Pancreas); `T_bwhiten16` 1.23 / 1.03 / 0.85, while the linear
   attacker falls to chance (0.32 / 0.32 / 0.33) and the kNN-based scIB-style batch score rises (Pancreas 0.41 to 0.61, Immune 0.40 to 0.61). An independent scikit-learn random-forest re-implementation of the attack gives the same direction
   (cross-fit balanced accuracy raw to centred: Pancreas fold 0 0.789 to 0.868; Immune fold 2 0.491 to 0.657; Immune fold 0 1.00 to 0.995), so it is not a pipeline artefact. Hypothesis: centring removes the batch mean, which the donor-disjoint
   attacker could not use anyway, but leaves batch-specific covariance and composition-shaped structure that nonlinear attackers read. This contradicts the programme's use of "oracle per-source normalisation" as a strong floor for
   nonlinear attackers, in this setting.
7. **scVI helps utility, not always leakage.** `T_scvi` fine-label U 0.816 (HLCA, raw 0.787), 0.787 (Immune, raw 0.734), 0.941 (Pancreas, raw 0.947); L 0.80 / 1.16 / 1.00. Harmony lowers L most (0.37 / 0.76 / 0.54) and costs U (-0.12 / -0.03 / -0.04 vs raw).

### 5.3 Floors tables with intervals (the pre-registered grid; U = fine-label macro-F1, U_hold = macro-F1 on held-out types, L = normalised worst-case retrained batch attacker)
Columns: arm, released dimension d, U_fine and U_hold with 95% bootstrap intervals, worst-case attacker accuracy with interval, L with interval, accuracy of each attacker family, retention (raw-neighbourhood kNN overlap, context). `_H` = label-supervised fit without the held-out
types (their U_hold is the held-class number; their U_fine is the fine-label number with a third of the types never seen by the code). `T_*` and `onehot_oracle` are references/ceilings, never wins.

#### HLCA
chance (mean over folds 1/B_f) = 0.281; label-only floor (batch from true fine type) = 0.452

| arm | d | U_fine [95% CI] | U_hold [95% CI] | worst attacker acc [95% CI] | L [95% CI] | lin / mlp / knn / lgbm | retention |
|---|---|---|---|---|---|---|---|
| raw50 | 50 | 0.787 [0.744, 0.797] | 0.792 [0.761, 0.809] | 0.670 [0.644, 0.802] | 1.00 [1.00, 1.00] | 0.649 / 0.670 / 0.606 / 0.647 | 0.90 |
| pca8 | 8 | 0.463 [0.427, 0.474] | 0.474 [0.436, 0.497] | 0.578 [0.557, 0.700] | 0.76 [0.68, 0.81] | 0.506 / 0.578 / 0.556 / 0.571 | 0.23 |
| pca16 | 16 | 0.619 [0.578, 0.632] | 0.607 [0.569, 0.630] | 0.643 [0.622, 0.754] | 0.93 [0.85, 0.99] | 0.576 / 0.643 / 0.602 / 0.624 | 0.49 |
| pca32 | 32 | 0.735 [0.695, 0.744] | 0.741 [0.710, 0.759] | 0.656 [0.631, 0.785] | 0.96 [0.94, 0.99] | 0.634 / 0.656 / 0.605 / 0.640 | 0.77 |
| rp16 | 16 | 0.670 [0.635, 0.682] | 0.711 [0.681, 0.727] | 0.598 [0.574, 0.719] | 0.81 [0.75, 0.84] | 0.528 / 0.598 / 0.556 / 0.558 | 0.52 |
| rp32 | 32 | 0.745 [0.703, 0.755] | 0.760 [0.730, 0.778] | 0.642 [0.611, 0.765] | 0.93 [0.88, 0.95] | 0.619 / 0.642 / 0.586 / 0.596 | 0.66 |
| noise_r0.5 | 50 | 0.754 [0.716, 0.764] | 0.771 [0.740, 0.787] | 0.605 [0.574, 0.719] | 0.83 [0.75, 0.88] | 0.605 / 0.602 / 0.555 / 0.576 | 0.47 |
| noise_r1 | 50 | 0.675 [0.637, 0.683] | 0.712 [0.683, 0.728] | 0.552 [0.520, 0.651] | 0.70 [0.54, 0.75] | 0.552 / 0.548 / 0.524 / 0.519 | 0.18 |
| noise_r2 | 50 | 0.430 [0.400, 0.442] | 0.503 [0.467, 0.519] | 0.464 [0.446, 0.558] | 0.47 [0.27, 0.51] | 0.464 / 0.461 / 0.451 / 0.438 | 0.03 |
| dith16_r0.5 | 16 | 0.474 [0.443, 0.485] | 0.483 [0.450, 0.506] | 0.524 [0.507, 0.633] | 0.63 [0.49, 0.67] | 0.488 / 0.524 / 0.503 / 0.507 | 0.13 |
| dith16_r1 | 16 | 0.323 [0.302, 0.334] | 0.337 [0.311, 0.355] | 0.469 [0.446, 0.565] | 0.48 [0.29, 0.52] | 0.446 / 0.469 / 0.455 / 0.448 | 0.04 |
| ldaC | 10 | 0.413 [0.388, 0.422] | 0.552 [0.527, 0.567] | 0.558 [0.531, 0.676] | 0.71 [0.62, 0.77] | 0.525 / 0.558 / 0.524 / 0.546 | 0.19 |
| ldaC_pca16 | 16 | 0.607 [0.578, 0.616] | 0.648 [0.624, 0.665] | 0.608 [0.584, 0.731] | 0.84 [0.78, 0.87] | 0.565 / 0.608 / 0.561 / 0.596 | 0.41 |
| ldaC_pca32 | 32 | 0.738 [0.700, 0.748] | 0.737 [0.706, 0.755] | 0.668 [0.644, 0.795] | 0.99 [0.96, 1.02] | 0.636 / 0.668 / 0.611 / 0.648 | 0.81 |
| leace | 50 | 0.775 [0.732, 0.784] | 0.783 [0.752, 0.800] | 0.651 [0.617, 0.778] | 0.95 [0.90, 0.98] | 0.625 / 0.651 / 0.579 / 0.624 | 0.80 |
| leace_pca16 | 16 | 0.654 [0.613, 0.665] | 0.663 [0.631, 0.685] | 0.629 [0.605, 0.748] | 0.89 [0.83, 0.92] | 0.585 / 0.629 / 0.590 / 0.598 | 0.51 |
| ldaF16 | 16 | 0.586 [0.554, 0.594] | 0.591 [0.563, 0.609] | 0.593 [0.569, 0.715] | 0.80 [0.73, 0.84] | 0.506 / 0.593 / 0.546 / 0.575 | 0.28 |
| ldaC_H | 10 | 0.449 [0.427, 0.457] | 0.552 [0.527, 0.567] | 0.550 [0.527, 0.671] | 0.69 [0.60, 0.73] | 0.512 / 0.550 / 0.513 / 0.540 | 0.20 |
| ldaC_pca16_H | 16 | 0.629 [0.600, 0.637] | 0.648 [0.624, 0.665] | 0.609 [0.585, 0.729] | 0.84 [0.78, 0.88] | 0.554 / 0.609 / 0.556 / 0.591 | 0.41 |
| ldaC_pca32_H | 32 | 0.740 [0.701, 0.749] | 0.737 [0.706, 0.755] | 0.657 [0.630, 0.787] | 0.97 [0.94, 0.99] | 0.626 / 0.657 / 0.603 / 0.633 | 0.79 |
| ldaF16_H | 16 | 0.616 [0.582, 0.624] | 0.591 [0.563, 0.609] | 0.590 [0.567, 0.720] | 0.79 [0.73, 0.83] | 0.520 / 0.590 / 0.547 / 0.574 | 0.34 |
| T_centre | 50 | 0.744 [0.707, 0.753] | 0.752 [0.727, 0.770] | 0.882 [0.857, 0.951] | 1.54 [1.40, 1.80] | 0.323 / 0.873 / 0.668 / 0.882 | 0.50 |
| T_bwhiten16 | 16 | 0.383 [0.353, 0.401] | 0.381 [0.347, 0.405] | 0.759 [0.728, 0.842] | 1.23 [1.08, 1.47] | 0.222 / 0.759 / 0.541 / 0.654 | 0.05 |
| T_harmony | 50 | 0.668 [0.645, 0.680] | 0.669 [0.645, 0.692] | 0.425 [0.410, 0.575] | 0.37 [0.31, 0.42] | 0.331 / 0.425 / 0.420 / 0.407 | 0.34 |
| T_scvi | 30 | 0.816 [0.778, 0.824] | 0.795 [0.768, 0.810] | 0.592 [0.556, 0.753] | 0.80 [0.74, 0.88] | 0.534 / 0.592 / 0.575 / 0.571 | 0.21 |
| onehot_oracle | 61 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.456 [0.424, 0.578] | 0.45 [0.31, 0.51] | 0.453 / 0.448 / 0.456 / 0.447 |  |
| ctrl_const | 1 | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | 0.281 [0.280, 0.459] | 0.00 [-0.05, 0.15] | 0.281 / 0.274 / 0.277 / 0.279 |  |
| ctrl_noise16 | 16 | 0.014 [0.012, 0.015] | 0.023 [0.020, 0.026] | 0.282 [0.275, 0.375] | 0.00 [-0.24, 0.05] | 0.282 / 0.276 / 0.279 / 0.275 |  |
| ctrl_shuffle_batch |  | nan [nan, nan] | nan [nan, nan] | 0.278 [0.267, 0.372] | -0.01 [-0.25, 0.07] | 0.273 / 0.268 / 0.278 / 0.264 |  |
| ctrl_shuffle_labels |  | 0.015 [0.012, 0.018] | 0.022 [0.018, 0.025] | nan [nan, nan] | nan [nan, nan] | nan / nan / nan / nan |  |

**K1 measurability**: raw50 worst-attacker accuracy 0.670 (0.6438592270417881, 0.8024736717538461) vs chance 0.281 (excess 0.389, needs >= 0.15 and >= min(3 x chance, 0.85) = 0.843): FAIL. Controls (attacker excess over chance; probe U minus the analytic random-guess macro-F1 u0=0.013): {'ctrl_const': 0.0, 'ctrl_noise16': 0.0015155246996620964, 'ctrl_shuffle_batch': -0.0031479064670335566} / {'ctrl_const': -0.012471371823909304, 'ctrl_noise16': 0.0005782790369044527, 'ctrl_shuffle_labels': 0.0017493127255173522}; allowed <= 0.03: ok. => K1 FAIL
per-fold raw50 attackers: fold 0 (2 batches, chance 0.50): worst 0.610; fold 1 (5 batches, chance 0.20): worst 0.867; fold 2 (7 batches, chance 0.14): worst 0.550

**K2 under-determination**: ldaC_pca16 retains r_fine = 0.77 (0.7503076512234675, 0.7890749165638539) of raw50's above-chance fine-label macro-F1 and r_hold = 0.82 (0.7981909554523184, 0.8377901162607595) of its held-type macro-F1 (threshold 0.70). Label-only floor holds: both = True, either = True (pre-registered reading: both).

**K3 headroom**: best transductive reference = T_scvi (U 0.816, L 0.80); references: T_centre U 0.744 L 1.54, T_bwhiten16 U 0.383 L 1.23, T_harmony U 0.668 L 0.37, T_scvi U 0.816 L 0.80
V-selected inductive floor per fold (selection on the other folds, L <= L_ref + 0.05): {0: 'noise_r0.5', 1: 'rp16', 2: 'noise_r1'}
selected floor pooled: U 0.700, L 0.78; U_ref - U_floor = 0.116 (0.09891316109659787, 0.13632853115390745); L_floor - L_ref = -0.025 (-0.18169528440502408, 0.04609138418254842); K3 fires on this dataset: False
sensitivity: test-selected floor {'arm': 'noise_r0.5', 'U': 0.7543460155714801, 'L': 0.8316512650717662, 'gap': 0.06182307635393691, 'fires': False}; hold-type version {'ref': 'T_scvi', 'arm': 'noise_r0.5', 'Uh_ref': 0.7953035515487895, 'Uh': 0.7705440020138088, 'gap': 0.024759549534980718, 'fires': True}
per-reference reading (each reference as the comparator; test-selected best inductive floor with L <= L_ref + 0.05; negative gap = floor better):
  - T_centre: U 0.744 L 1.54; best floor raw50 U 0.787 L 1.00; U_ref - U_floor = -0.043 (-0.053, -0.029)
  - T_bwhiten16: U 0.383 L 1.23; best floor raw50 U 0.787 L 1.00; U_ref - U_floor = -0.404 (-0.423, -0.366)
  - T_harmony: U 0.668 L 0.37; NO inductive floor reaches L <= 0.42 (lowest noise_r2 L 0.47)
  - T_scvi: U 0.816 L 0.80; best floor noise_r0.5 U 0.754 L 0.83; U_ref - U_floor = 0.062 (0.05, 0.073)
inductive Pareto set (arm, L, U): [('noise_r2', 0.469, 0.43), ('dith16_r0.5', 0.625, 0.474), ('noise_r1', 0.696, 0.675), ('noise_r0.5', 0.832, 0.754), ('leace', 0.951, 0.775), ('raw50', 1.0, 0.787)]

**Study-cluster bootstrap** (resample the 11 studies, then donors inside; B=300); conservative interval for the cross-study claim:
| arm | U_fine | U_hold | L |
|---|---|---|---|
| raw50 | 0.787 [0.722, 0.798] | 0.792 [0.725, 0.833] | 1.00 [1.00, 1.00] |
| pca16 | 0.619 [0.553, 0.640] | 0.607 [0.521, 0.664] | 0.93 [0.72, 1.02] |
| pca32 | 0.735 [0.673, 0.747] | 0.741 [0.679, 0.783] | 0.96 [0.86, 0.98] |
| ldaC_pca16 | 0.607 [0.542, 0.617] | 0.648 [0.572, 0.690] | 0.84 [0.49, 0.92] |
| ldaC_pca16_H | 0.629 [0.562, 0.639] | 0.648 [0.572, 0.690] | 0.84 [0.48, 0.93] |
| leace | 0.775 [0.707, 0.785] | 0.783 [0.713, 0.824] | 0.95 [0.80, 0.99] |
| T_centre | 0.744 [0.675, 0.756] | 0.752 [0.700, 0.802] | 1.54 [1.39, 3.01] |
| T_harmony | 0.668 [0.604, 0.687] | 0.669 [0.610, 0.723] | 0.37 [-0.95, 0.37] |
| T_scvi | 0.816 [0.759, 0.823] | 0.795 [0.742, 0.836] | 0.80 [0.40, 0.98] |
| dith16_r0.5 | 0.474 [0.425, 0.498] | 0.483 [0.417, 0.547] | 0.63 [-0.25, 0.63] |
r_fine cluster CI (0.7190302649322291, 0.8027436646765279), r_hold cluster CI (0.7580220843850475, 0.8504210595735697)

**K0 power audit** (seed 0 predictions; units = donors): n_units 107, n_studies 11
| pair | dU | SE (boot) | crit (perm 5%) | MDE80 |
|---|---|---|---|---|
| pca16 vs rp16 | -0.049 | 0.009 | 0.022 | 0.029 |
| pca16 vs dith16_r0.5 | 0.147 | 0.007 | 0.035 | 0.040 |
| pca16 vs ldaC_pca16 | 0.012 | 0.010 | 0.021 | 0.029 |
| rp16 vs dith16_r0.5 | 0.197 | 0.007 | 0.051 | 0.057 |
| rp16 vs ldaC_pca16 | 0.062 | 0.007 | 0.022 | 0.028 |
| dith16_r0.5 vs ldaC_pca16 | -0.135 | 0.007 | 0.034 | 0.040 |
median MDE80 over pairs = 0.035 (needs <= 0.05 on HLCA)

#### Immune
chance (mean over folds 1/B_f) = 0.333; label-only floor (batch from true fine type) = 0.583

| arm | d | U_fine [95% CI] | U_hold [95% CI] | worst attacker acc [95% CI] | L [95% CI] | lin / mlp / knn / lgbm | retention |
|---|---|---|---|---|---|---|---|
| raw50 | 50 | 0.734 [0.655, 0.789] | 0.756 [0.710, 0.802] | 0.752 [0.619, 0.827] | 1.00 [1.00, 1.00] | 0.733 / 0.750 / 0.702 / 0.752 | 0.65 |
| pca8 | 8 | 0.595 [0.550, 0.692] | 0.702 [0.648, 0.737] | 0.712 [0.556, 0.794] | 0.91 [0.49, 0.93] | 0.671 / 0.712 / 0.686 / 0.708 | 0.19 |
| pca16 | 16 | 0.722 [0.644, 0.772] | 0.732 [0.691, 0.780] | 0.748 [0.614, 0.827] | 0.99 [0.92, 1.05] | 0.721 / 0.748 / 0.717 / 0.744 | 0.37 |
| pca32 | 32 | 0.730 [0.652, 0.783] | 0.748 [0.702, 0.793] | 0.748 [0.621, 0.827] | 0.99 [0.95, 1.02] | 0.727 / 0.748 / 0.700 / 0.748 | 0.53 |
| rp16 | 16 | 0.541 [0.473, 0.583] | 0.582 [0.527, 0.610] | 0.681 [0.515, 0.769] | 0.83 [0.19, 0.83] | 0.655 / 0.681 / 0.651 / 0.670 | 0.18 |
| rp32 | 32 | 0.660 [0.583, 0.700] | 0.686 [0.634, 0.713] | 0.716 [0.558, 0.802] | 0.91 [0.54, 0.92] | 0.700 / 0.716 / 0.674 / 0.700 | 0.28 |
| noise_r0.5 | 50 | 0.643 [0.499, 0.739] | 0.655 [0.526, 0.737] | 0.642 [0.456, 0.722] | 0.74 [-0.15, 0.72] | 0.640 / 0.642 / 0.623 / 0.638 | 0.09 |
| noise_r1 | 50 | 0.506 [0.291, 0.633] | 0.517 [0.315, 0.636] | 0.589 [0.370, 0.652] | 0.61 [-0.73, 0.58] | 0.589 / 0.587 / 0.567 / 0.580 | 0.03 |
| noise_r2 | 50 | 0.333 [0.160, 0.435] | 0.357 [0.178, 0.462] | 0.549 [0.304, 0.594] | 0.51 [-1.20, 0.46] | 0.549 / 0.541 / 0.521 / 0.536 | 0.01 |
| dith16_r0.5 | 16 | 0.619 [0.475, 0.707] | 0.623 [0.496, 0.702] | 0.619 [0.417, 0.690] | 0.68 [-0.47, 0.66] | 0.613 / 0.619 / 0.605 / 0.616 | 0.09 |
| dith16_r1 | 16 | 0.474 [0.275, 0.577] | 0.481 [0.293, 0.591] | 0.570 [0.343, 0.637] | 0.57 [-0.89, 0.52] | 0.570 / 0.567 / 0.553 / 0.565 | 0.03 |
| ldaC | 7 | 0.629 [0.552, 0.669] | 0.659 [0.576, 0.715] | 0.666 [0.488, 0.752] | 0.79 [-0.03, 0.79] | 0.613 / 0.666 / 0.630 / 0.662 | 0.17 |
| ldaC_pca16 | 16 | 0.713 [0.658, 0.751] | 0.700 [0.662, 0.744] | 0.742 [0.605, 0.822] | 0.98 [0.85, 1.01] | 0.713 / 0.742 / 0.708 / 0.738 | 0.37 |
| ldaC_pca32 | 32 | 0.734 [0.652, 0.790] | 0.721 [0.679, 0.768] | 0.747 [0.614, 0.826] | 0.99 [0.91, 1.01] | 0.725 / 0.747 / 0.699 / 0.747 | 0.56 |
| leace | 50 | 0.709 [0.611, 0.776] | 0.745 [0.691, 0.791] | 0.712 [0.566, 0.804] | 0.90 [0.54, 0.91] | 0.668 / 0.712 / 0.662 / 0.711 | 0.52 |
| leace_pca16 | 16 | 0.698 [0.596, 0.756] | 0.724 [0.656, 0.773] | 0.699 [0.546, 0.791] | 0.87 [0.41, 0.87] | 0.633 / 0.699 / 0.654 / 0.695 | 0.28 |
| ldaF16 | 15 | 0.738 [0.664, 0.794] | 0.686 [0.596, 0.731] | 0.705 [0.549, 0.787] | 0.89 [0.45, 0.89] | 0.666 / 0.705 / 0.670 / 0.699 | 0.27 |
| ldaC_H | 7 | 0.667 [0.575, 0.713] | 0.659 [0.576, 0.715] | 0.666 [0.491, 0.749] | 0.79 [0.04, 0.79] | 0.615 / 0.666 / 0.630 / 0.661 | 0.16 |
| ldaC_pca16_H | 16 | 0.692 [0.623, 0.738] | 0.700 [0.662, 0.744] | 0.732 [0.592, 0.821] | 0.95 [0.73, 0.98] | 0.701 / 0.732 / 0.704 / 0.730 | 0.31 |
| ldaC_pca32_H | 32 | 0.716 [0.635, 0.769] | 0.721 [0.679, 0.768] | 0.741 [0.605, 0.826] | 0.97 [0.83, 1.02] | 0.716 / 0.741 / 0.699 / 0.740 | 0.54 |
| ldaF16_H | 10 | 0.682 [0.585, 0.749] | 0.686 [0.596, 0.731] | 0.683 [0.515, 0.771] | 0.83 [0.22, 0.83] | 0.634 / 0.683 / 0.647 / 0.675 | 0.20 |
| T_centre | 50 | 0.739 [0.662, 0.794] | 0.749 [0.697, 0.800] | 0.847 [0.751, 0.895] | 1.23 [1.20, 2.09] | 0.315 / 0.840 / 0.677 / 0.847 | 0.43 |
| T_bwhiten16 | 16 | 0.586 [0.459, 0.666] | 0.563 [0.467, 0.626] | 0.766 [0.657, 0.826] | 1.03 [0.72, 1.53] | 0.312 / 0.766 / 0.559 / 0.667 | 0.04 |
| T_harmony | 50 | 0.702 [0.644, 0.769] | 0.725 [0.676, 0.770] | 0.653 [0.467, 0.737] | 0.76 [-0.11, 0.76] | 0.534 / 0.624 / 0.549 / 0.653 | 0.48 |
| T_scvi | 30 | 0.787 [0.743, 0.821] | 0.820 [0.775, 0.854] | 0.817 [0.719, 0.846] | 1.16 [1.12, 1.84] | 0.745 / 0.817 / 0.759 / 0.800 | 0.04 |
| onehot_oracle | 17 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.583 [0.391, 0.695] | 0.60 [-0.90, 0.69] | 0.583 / 0.583 / 0.338 / 0.583 |  |
| ctrl_const | 1 | 0.016 [0.011, 0.043] | 0.034 [0.016, 0.044] | 0.333 [0.292, 0.667] | 0.00 [-2.04, 0.00] | 0.333 / 0.333 / 0.333 / 0.333 |  |
| ctrl_noise16 | 16 | 0.050 [0.044, 0.064] | 0.061 [0.052, 0.075] | 0.337 [0.254, 0.480] | 0.01 [-2.59, -0.12] | 0.334 / 0.331 / 0.337 / 0.336 |  |
| ctrl_shuffle_batch |  | nan [nan, nan] | nan [nan, nan] | 0.344 [0.257, 0.444] | 0.03 [-2.56, -0.11] | 0.338 / 0.314 / 0.344 / 0.333 |  |
| ctrl_shuffle_labels |  | 0.049 [0.044, 0.064] | 0.080 [0.056, 0.100] | nan [nan, nan] | nan [nan, nan] | nan / nan / nan / nan |  |

**K1 measurability**: raw50 worst-attacker accuracy 0.752 (0.6193220818656813, 0.8267617735924638) vs chance 0.333 (excess 0.419, needs >= 0.15 and >= min(3 x chance, 0.85) = 0.850): FAIL. Controls (attacker excess over chance; probe U minus the analytic random-guess macro-F1 u0=0.062): {'ctrl_const': 0.0, 'ctrl_noise16': 0.0032518301014092255, 'ctrl_shuffle_batch': 0.010713769248697491} / {'ctrl_const': -0.04650905612861566, 'ctrl_noise16': -0.012174146924099967, 'ctrl_shuffle_labels': -0.012722645020850416}; allowed <= 0.03: ok. => K1 FAIL
per-fold raw50 attackers: fold 0 (2 batches, chance 0.50): worst 1.000; fold 1 (4 batches, chance 0.25): worst 0.753; fold 2 (4 batches, chance 0.25): worst 0.503

**K2 under-determination**: ldaC_pca16 retains r_fine = 0.97 (0.926397854380289, 1.029974568533551) of raw50's above-chance fine-label macro-F1 and r_hold = 0.92 (0.905576666358355, 0.9408617633594354) of its held-type macro-F1 (threshold 0.70). Label-only floor holds: both = True, either = True (pre-registered reading: both).

**K3 headroom**: best transductive reference = T_scvi (U 0.787, L 1.16); references: T_centre U 0.739 L 1.23, T_bwhiten16 U 0.586 L 1.03, T_harmony U 0.702 L 0.76, T_scvi U 0.787 L 1.16
V-selected inductive floor per fold (selection on the other folds, L <= L_ref + 0.05): {0: 'raw50', 1: 'ldaC_pca32', 2: 'raw50'}
selected floor pooled: U 0.731, L 0.99; U_ref - U_floor = 0.056 (0.01965340261906642, 0.11525895537967211); L_floor - L_ref = -0.168 (-0.8851899287668907, -0.12343523946505552); K3 fires on this dataset: False
sensitivity: test-selected floor {'arm': 'ldaC_pca32', 'U': 0.7344908898067755, 'L': 0.9872122494312162, 'gap': 0.052589420717416346, 'fires': False}; hold-type version {'ref': 'T_scvi', 'arm': 'raw50', 'Uh_ref': 0.8198603620746278, 'Uh': 0.7556973128010047, 'gap': 0.06416304927362304, 'fires': False}
per-reference reading (each reference as the comparator; test-selected best inductive floor with L <= L_ref + 0.05; negative gap = floor better):
  - T_centre: U 0.739 L 1.23; best floor ldaC_pca32 U 0.734 L 0.99; U_ref - U_floor = 0.004 (-0.011, 0.024)
  - T_bwhiten16: U 0.586 L 1.03; best floor ldaC_pca32 U 0.734 L 0.99; U_ref - U_floor = -0.149 (-0.216, -0.113)
  - T_harmony: U 0.702 L 0.76; best floor noise_r0.5 U 0.643 L 0.74; U_ref - U_floor = 0.059 (0.001, 0.19)
  - T_scvi: U 0.787 L 1.16; best floor ldaC_pca32 U 0.734 L 0.99; U_ref - U_floor = 0.053 (0.017, 0.113)
inductive Pareto set (arm, L, U): [('noise_r2', 0.514, 0.333), ('dith16_r1', 0.565, 0.474), ('noise_r1', 0.611, 0.506), ('dith16_r0.5', 0.681, 0.619), ('noise_r0.5', 0.738, 0.643), ('leace_pca16', 0.872, 0.698), ('leace', 0.904, 0.709), ('ldaC_pca16', 0.976, 0.713), ('ldaC_pca32', 0.987, 0.734)]

**K0 power audit** (seed 0 predictions; units = batches): n_units 10, n_studies 5
| pair | dU | SE (boot) | crit (perm 5%) | MDE80 |
|---|---|---|---|---|
| pca16 vs rp16 | 0.143 | 0.013 | 0.111 | 0.121 |
| pca16 vs dith16_r0.5 | 0.108 | 0.040 | 0.071 | 0.105 |
| pca16 vs ldaC_pca16 | 0.008 | 0.015 | 0.024 | 0.036 |
| rp16 vs dith16_r0.5 | -0.034 | 0.040 | 0.094 | 0.127 |
| rp16 vs ldaC_pca16 | -0.134 | 0.018 | 0.102 | 0.118 |
| dith16_r0.5 vs ldaC_pca16 | -0.100 | 0.050 | 0.072 | 0.115 |
median MDE80 over pairs = 0.116 (needs <= 0.05 on HLCA)

#### Pancreas
chance (mean over folds 1/B_f) = 0.361; label-only floor (batch from true fine type) = 0.536

| arm | d | U_fine [95% CI] | U_hold [95% CI] | worst attacker acc [95% CI] | L [95% CI] | lin / mlp / knn / lgbm | retention |
|---|---|---|---|---|---|---|---|
| raw50 | 50 | 0.947 [0.920, 0.960] | 0.972 [0.930, 0.989] | 0.937 [0.902, 0.950] | 1.00 [1.00, 1.00] | 0.931 / 0.937 / 0.881 / 0.928 | 0.61 |
| pca8 | 8 | 0.735 [0.665, 0.805] | 0.832 [0.746, 0.911] | 0.798 [0.702, 0.834] | 0.76 [-0.19, 0.75] | 0.705 / 0.798 / 0.730 / 0.791 | 0.35 |
| pca16 | 16 | 0.890 [0.842, 0.924] | 0.974 [0.941, 0.988] | 0.870 [0.797, 0.891] | 0.88 [0.38, 0.88] | 0.813 / 0.870 / 0.809 / 0.859 | 0.53 |
| pca32 | 32 | 0.936 [0.908, 0.951] | 0.973 [0.938, 0.987] | 0.927 [0.888, 0.941] | 0.98 [0.91, 0.99] | 0.914 / 0.927 / 0.875 / 0.916 | 0.59 |
| rp16 | 16 | 0.660 [0.597, 0.699] | 0.813 [0.742, 0.851] | 0.849 [0.784, 0.869] | 0.85 [0.23, 0.84] | 0.801 / 0.849 / 0.811 / 0.825 | 0.23 |
| rp32 | 32 | 0.854 [0.804, 0.880] | 0.911 [0.873, 0.937] | 0.903 [0.855, 0.920] | 0.94 [0.70, 0.94] | 0.878 / 0.903 / 0.840 / 0.873 | 0.34 |
| noise_r0.5 | 50 | 0.902 [0.861, 0.929] | 0.945 [0.914, 0.963] | 0.873 [0.811, 0.895] | 0.89 [0.46, 0.88] | 0.869 / 0.873 / 0.837 / 0.857 | 0.30 |
| noise_r1 | 50 | 0.758 [0.692, 0.805] | 0.859 [0.802, 0.885] | 0.786 [0.693, 0.825] | 0.74 [-0.24, 0.72] | 0.786 / 0.786 / 0.761 / 0.766 | 0.10 |
| noise_r2 | 50 | 0.463 [0.396, 0.504] | 0.624 [0.485, 0.712] | 0.662 [0.542, 0.707] | 0.52 [-1.29, 0.47] | 0.662 / 0.657 / 0.624 / 0.624 | 0.02 |
| dith16_r0.5 | 16 | 0.815 [0.755, 0.859] | 0.942 [0.902, 0.963] | 0.717 [0.588, 0.748] | 0.62 [-0.93, 0.59] | 0.696 / 0.717 / 0.687 / 0.703 | 0.18 |
| dith16_r1 | 16 | 0.662 [0.599, 0.712] | 0.859 [0.810, 0.882] | 0.617 [0.480, 0.659] | 0.44 [-1.75, 0.39] | 0.613 / 0.617 / 0.595 / 0.602 | 0.06 |
| ldaC | 4 | 0.516 [0.464, 0.558] | 0.787 [0.721, 0.838] | 0.652 [0.506, 0.744] | 0.50 [-1.36, 0.54] | 0.494 / 0.652 / 0.593 / 0.651 | 0.04 |
| ldaC_pca16 | 16 | 0.868 [0.807, 0.910] | 0.882 [0.821, 0.916] | 0.871 [0.796, 0.901] | 0.89 [0.46, 0.88] | 0.813 / 0.871 / 0.820 / 0.863 | 0.53 |
| ldaC_pca32 | 32 | 0.940 [0.914, 0.953] | 0.864 [0.792, 0.907] | 0.927 [0.888, 0.942] | 0.98 [0.92, 0.99] | 0.914 / 0.927 / 0.875 / 0.916 | 0.60 |
| leace | 50 | 0.947 [0.922, 0.960] | 0.972 [0.936, 0.989] | 0.924 [0.884, 0.941] | 0.98 [0.88, 0.98] | 0.914 / 0.924 / 0.873 / 0.915 | 0.53 |
| leace_pca16 | 16 | 0.931 [0.898, 0.952] | 0.975 [0.942, 0.988] | 0.851 [0.773, 0.875] | 0.85 [0.27, 0.84] | 0.777 / 0.851 / 0.780 / 0.841 | 0.34 |
| ldaF16 | 9 | 0.955 [0.925, 0.965] | 0.819 [0.724, 0.884] | 0.733 [0.618, 0.778] | 0.65 [-0.69, 0.63] | 0.551 / 0.733 / 0.636 / 0.728 | 0.10 |
| ldaC_H | 3 | 0.550 [0.491, 0.589] | 0.787 [0.721, 0.838] | 0.626 [0.464, 0.748] | 0.46 [-1.55, 0.55] | 0.526 / 0.626 / 0.572 / 0.617 | 0.04 |
| ldaC_pca16_H | 16 | 0.879 [0.819, 0.910] | 0.882 [0.821, 0.916] | 0.870 [0.796, 0.895] | 0.88 [0.39, 0.88] | 0.810 / 0.870 / 0.798 / 0.855 | 0.50 |
| ldaC_pca32_H | 32 | 0.879 [0.827, 0.916] | 0.864 [0.792, 0.907] | 0.925 [0.882, 0.939] | 0.98 [0.88, 0.98] | 0.914 / 0.925 / 0.874 / 0.911 | 0.55 |
| ldaF16_H | 6 | 0.866 [0.823, 0.898] | 0.819 [0.724, 0.884] | 0.706 [0.582, 0.764] | 0.60 [-0.91, 0.59] | 0.529 / 0.705 / 0.630 / 0.706 | 0.09 |
| T_centre | 50 | 0.941 [0.916, 0.955] | 0.963 [0.927, 0.978] | 0.952 [0.931, 0.970] | 1.03 [1.02, 1.18] | 0.333 / 0.950 / 0.812 / 0.952 | 0.30 |
| T_bwhiten16 | 16 | 0.840 [0.781, 0.876] | 0.930 [0.868, 0.965] | 0.850 [0.802, 0.892] | 0.85 [0.23, 0.85] | 0.337 / 0.850 / 0.599 / 0.791 | 0.07 |
| T_harmony | 50 | 0.911 [0.879, 0.948] | 0.965 [0.910, 0.985] | 0.670 [0.579, 0.738] | 0.54 [-1.18, 0.50] | 0.561 / 0.670 / 0.617 / 0.644 | 0.25 |
| T_scvi | 30 | 0.941 [0.908, 0.956] | 0.950 [0.884, 0.982] | 0.937 [0.907, 0.955] | 1.00 [0.97, 1.05] | 0.883 / 0.937 / 0.893 / 0.917 | 0.17 |
| onehot_oracle | 11 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.536 [0.386, 0.731] | 0.30 [-2.10, 0.31] | 0.536 / 0.535 / 0.366 / 0.536 |  |
| ctrl_const | 1 | 0.051 [0.040, 0.059] | 0.000 [0.000, 0.000] | 0.361 [0.333, 0.667] | 0.00 [-2.35, 0.00] | 0.361 / 0.361 / 0.361 / 0.361 |  |
| ctrl_noise16 | 16 | 0.063 [0.053, 0.071] | 0.047 [0.033, 0.060] | 0.363 [0.296, 0.465] | 0.00 [-3.39, -0.10] | 0.358 / 0.359 / 0.363 / 0.360 |  |
| ctrl_shuffle_batch |  | nan [nan, nan] | nan [nan, nan] | 0.373 [0.296, 0.487] | 0.02 [-3.27, -0.09] | 0.373 / 0.362 / 0.370 / 0.367 |  |
| ctrl_shuffle_labels |  | 0.064 [0.053, 0.073] | 0.059 [0.035, 0.082] | nan [nan, nan] | nan [nan, nan] | nan / nan / nan / nan |  |

**K1 measurability**: raw50 worst-attacker accuracy 0.937 (0.9019829386592478, 0.9498562855682748) vs chance 0.361 (excess 0.576, needs >= 0.15 and >= min(3 x chance, 0.85) = 0.850): ok. Controls (attacker excess over chance; probe U minus the analytic random-guess macro-F1 u0=0.076): {'ctrl_const': 1.065916261622224e-05, 'ctrl_noise16': 0.0023713528264169836, 'ctrl_shuffle_batch': 0.012250847608970938} / {'ctrl_const': -0.025258422251920702, 'ctrl_noise16': -0.012599305006553596, 'ctrl_shuffle_labels': -0.012100048300877325}; allowed <= 0.03: ok. => K1 PASS
per-fold raw50 attackers: fold 0 (4 batches, chance 0.25): worst 0.852; fold 1 (3 batches, chance 0.33): worst 0.965; fold 2 (2 batches, chance 0.50): worst 0.996

**K2 under-determination**: ldaC_pca16 retains r_fine = 0.91 (0.8550629935623161, 0.9488220834542505) of raw50's above-chance fine-label macro-F1 and r_hold = 0.90 (0.8659965956149474, 0.928145380257601) of its held-type macro-F1 (threshold 0.70). Label-only floor holds: both = True, either = True (pre-registered reading: both).

**K3 headroom**: best transductive reference = T_centre (U 0.941, L 1.03); references: T_centre U 0.941 L 1.03, T_bwhiten16 U 0.840 L 0.85, T_harmony U 0.911 L 0.54, T_scvi U 0.941 L 1.00
V-selected inductive floor per fold (selection on the other folds, L <= L_ref + 0.05): {0: 'leace', 1: 'raw50', 2: 'leace'}
selected floor pooled: U 0.947, L 0.99; U_ref - U_floor = -0.006 (-0.01284023810644038, 2.430537693510944e-06); L_floor - L_ref = -0.038 (-0.24415407833112746, -0.033740712520559546); K3 fires on this dataset: True
sensitivity: test-selected floor {'arm': 'leace', 'U': 0.947158491439635, 'L': 0.9772882568016616, 'gap': -0.006482220931400162, 'fires': True}; hold-type version {'ref': 'T_harmony', 'arm': 'dith16_r1', 'Uh_ref': 0.9652457598976995, 'Uh': 0.859059382410828, 'gap': 0.10618637748687143, 'fires': False}
per-reference reading (each reference as the comparator; test-selected best inductive floor with L <= L_ref + 0.05; negative gap = floor better):
  - T_centre: U 0.941 L 1.03; best floor leace U 0.947 L 0.98; U_ref - U_floor = -0.006 (-0.014, -0.001)
  - T_bwhiten16: U 0.840 L 0.85; best floor leace_pca16 U 0.931 L 0.85; U_ref - U_floor = -0.090 (-0.146, -0.049)
  - T_harmony: U 0.911 L 0.54; best floor dith16_r1 U 0.662 L 0.44; U_ref - U_floor = 0.249 (0.202, 0.312)
  - T_scvi: U 0.941 L 1.00; best floor leace U 0.947 L 0.98; U_ref - U_floor = -0.006 (-0.02, 0.003)
inductive Pareto set (arm, L, U): [('dith16_r1', 0.445, 0.662), ('dith16_r0.5', 0.618, 0.815), ('leace_pca16', 0.85, 0.931), ('leace', 0.977, 0.947)]

**K0 power audit** (seed 0 predictions; units = batches): n_units 9, n_studies 9
| pair | dU | SE (boot) | crit (perm 5%) | MDE80 |
|---|---|---|---|---|
| pca16 vs rp16 | 0.224 | 0.020 | 0.155 | 0.172 |
| pca16 vs dith16_r0.5 | 0.073 | 0.009 | 0.055 | 0.063 |
| pca16 vs ldaC_pca16 | 0.022 | 0.012 | 0.025 | 0.035 |
| rp16 vs dith16_r0.5 | -0.152 | 0.027 | 0.111 | 0.134 |
| rp16 vs ldaC_pca16 | -0.202 | 0.027 | 0.154 | 0.176 |
| dith16_r0.5 vs ldaC_pca16 | -0.051 | 0.009 | 0.040 | 0.047 |
median MDE80 over pairs = 0.098 (needs <= 0.05 on HLCA)

### 5.4 Sensitivities (none changes the verdict)
* **Dense floor frontier (post hoc, A3; seed 0, B=200, `*_dense.md`).** Adds `dithN_rR` (N in 8/16/32, R in 0.25-4), `pca{4,6,12,24}`, `ldaC_pca{8,24}` to the inductive pool. It fills the low-leakage end that the registered grid left empty (HLCA Pareto set now runs
  from L 0.10 at U 0.04 to raw). Per reference: HLCA Harmony (U 0.666, L 0.41) vs best floor `dith32_r2` (L 0.45, U 0.312): gap +0.354 [0.334, 0.371]; HLCA scVI (U 0.816, L 0.83) vs `noise_r0.5` (L 0.85, U 0.757): +0.059 [0.048, 0.072];
  Pancreas Harmony +0.208 [0.140, 0.285] (best floor `dith8_r0.25`, U 0.701 at L 0.54), scVI -0.006; Immune Harmony +0.064 [0.014, 0.201], scVI +0.053 [0.017, 0.111] (descriptive). The registered K3 outcome is unchanged
  (Immune/HLCA do not fire, Pancreas fires). The denser grid matters only where the registered grid had no feasible floor at all (Harmony, HLCA), where it turns "no floor reaches this leakage" into a measured gap of 0.35.
* **K3 variants (test-selected floors).** Instead of the V-selected floor: HLCA gap 0.062 (fires: no), Immune 0.053 (no), Pancreas -0.006 (yes). Hold-type version (U_hold, `_H` fits, its own best reference): HLCA scVI gap 0.025 (fires), Immune 0.064 (no),
  Pancreas Harmony 0.106 (no). Both variants give 1 of 3 datasets firing, so K3 stays clear under them. (Registered K3 is the V-selected U_fine version; the variants are listed so that nobody has to wonder.)
* **HLCA study-cluster bootstrap** (resample 11 studies, then donors; B=300): U(raw50) 0.787 [0.722, 0.798], U(ldaC_pca16) 0.607 [0.542, 0.617], U(T_scvi) 0.816 [0.759, 0.823]; r_fine [0.719, 0.803], r_hold [0.758, 0.850]: K2 still holds on HLCA at
  the study level (lower bounds above 0.70), narrowly for r_fine.
* **Attackers.** The linear attacker is not the strongest on raw HLCA (lin 0.649, mlp 0.670, lgbm 0.647, knn 0.606), unlike C; the MLP is the worst case in most cells and LightGBM on the per-batch-normalised arms.

### 5.5 scIB-style context (scib-metrics 0.5.6, held-out cells, <= 8,000 per fold (HLCA: see status log), mean over folds; NOT comparable with published scIB totals)
### scIB-style context, immune (scib-metrics 0.5.6, held-out cells, mean over folds ['0', '1', '2']; NOT comparable with published scIB totals)
| arm | bio conservation | batch correction | total (0.6 bio + 0.4 batch) | iLISI | KBET | graph conn. | cLISI | isolated labels | KMeans NMI |
|---|---|---|---|---|---|---|---|---|---|
| raw50 | 0.677 | 0.396 | 0.565 | 0.280 | 0.319 | 0.751 | 0.985 | 0.610 | 0.685 |
| pca16 | 0.675 | 0.391 | 0.561 | 0.249 | 0.327 | 0.840 | 0.985 | 0.624 | 0.676 |
| ldaC_pca16 | 0.672 | 0.386 | 0.558 | 0.258 | 0.293 | 0.836 | 0.984 | 0.617 | 0.671 |
| leace | 0.655 | 0.470 | 0.581 | 0.260 | 0.399 | 0.734 | 0.980 | 0.603 | 0.659 |
| T_centre | 0.666 | 0.608 | 0.643 | 0.225 | 0.383 | 0.720 | 0.981 | 0.603 | 0.672 |
| T_bwhiten16 | 0.586 | 0.702 | 0.632 | 0.424 | 0.487 | 0.801 | 0.970 | 0.577 | 0.525 |
| T_harmony | 0.686 | 0.576 | 0.642 | 0.251 | 0.610 | 0.675 | 0.984 | 0.584 | 0.704 |
| T_scvi | 0.670 | 0.560 | 0.626 | 0.248 | 0.365 | 0.899 | 0.997 | 0.620 | 0.706 |
| onehot_oracle | 1.000 | 0.770 | 0.908 | 0.505 | 0.820 | 0.756 | 1.000 | 1.000 | 1.000 |
| ctrl_noise16 | 0.344 | 0.662 | 0.471 | 0.546 | 0.675 | 0.129 | 0.739 | 0.488 | 0.004 |

### scIB-style context, pancreas (scib-metrics 0.5.6, held-out cells, mean over folds ['0', '1', '2']; NOT comparable with published scIB totals)
| arm | bio conservation | batch correction | total (0.6 bio + 0.4 batch) | iLISI | KBET | graph conn. | cLISI | isolated labels | KMeans NMI |
|---|---|---|---|---|---|---|---|---|---|
| raw50 | 0.761 | 0.412 | 0.621 | 0.065 | 0.386 | 0.932 | 1.000 | 0.665 | 0.797 |
| pca16 | 0.776 | 0.461 | 0.650 | 0.136 | 0.445 | 0.956 | 1.000 | 0.699 | 0.788 |
| ldaC_pca16 | 0.774 | 0.455 | 0.647 | 0.130 | 0.440 | 0.956 | 1.000 | 0.690 | 0.779 |
| leace | 0.762 | 0.420 | 0.625 | 0.076 | 0.384 | 0.936 | 1.000 | 0.667 | 0.798 |
| T_centre | 0.731 | 0.605 | 0.680 | 0.022 | 0.375 | 0.928 | 1.000 | 0.657 | 0.750 |
| T_bwhiten16 | 0.667 | 0.739 | 0.696 | 0.302 | 0.672 | 0.909 | 0.997 | 0.599 | 0.650 |
| T_harmony | 0.780 | 0.701 | 0.748 | 0.377 | 0.778 | 0.929 | 1.000 | 0.671 | 0.829 |
| T_scvi | 0.695 | 0.467 | 0.604 | 0.088 | 0.391 | 0.927 | 1.000 | 0.630 | 0.732 |
| onehot_oracle | 1.000 | 0.765 | 0.906 | 0.612 | 0.776 | 0.673 | 1.000 | 1.000 | 1.000 |
| ctrl_noise16 | 0.335 | 0.686 | 0.475 | 0.606 | 0.693 | 0.184 | 0.710 | 0.482 | 0.005 |

### scIB-style context, hlca (scib-metrics 0.5.6, held-out cells, fold 1 ONLY (4,000 held-out cells); NOT comparable with published scIB totals)
| arm | bio conservation | batch correction | total (0.6 bio + 0.4 batch) | iLISI | KBET | graph conn. | cLISI | isolated labels | KMeans NMI |
|---|---|---|---|---|---|---|---|---|---|
| raw50 | 0.623 | 0.481 | 0.566 | 0.099 | 0.764 | 0.877 | 0.993 | 0.666 | 0.676 |
| pca16 | 0.586 | 0.449 | 0.531 | 0.132 | 0.720 | 0.797 | 0.989 | 0.619 | 0.614 |
| ldaC_pca16 | 0.597 | 0.537 | 0.573 | 0.160 | 0.760 | 0.821 | 0.989 | 0.617 | 0.636 |
| leace | 0.621 | 0.513 | 0.578 | 0.123 | 0.781 | 0.863 | 0.992 | 0.665 | 0.668 |
| T_centre | 0.617 | 0.633 | 0.623 | 0.055 | 0.674 | 0.837 | 0.993 | 0.671 | 0.658 |
| T_bwhiten16 | 0.546 | 0.675 | 0.598 | 0.228 | 0.708 | 0.746 | 0.980 | 0.628 | 0.504 |
| T_harmony | 0.614 | 0.646 | 0.627 | 0.267 | 0.832 | 0.858 | 0.991 | 0.644 | 0.656 |
| T_scvi | 0.631 | 0.595 | 0.617 | 0.107 | 0.752 | 0.899 | 0.996 | 0.604 | 0.712 |
| onehot_oracle | 0.983 | 0.747 | 0.889 | 0.446 | 0.913 | 0.901 | 1.000 | 0.917 | 1.000 |
| ctrl_noise16 | 0.360 | 0.593 | 0.453 | 0.604 | 0.303 | 0.154 | 0.902 | 0.414 | 0.095 |

HLCA: only fold 1 produced a result (4,000 cells). `e_scib_hlca` and `e_scib_hlca_b` crashed in fold 2 with an XLA CPU JIT allocation failure; in the per-fold retry `e_scib_hlca_c` folds 0 and 2 failed with an ARPACK error inside scib-metrics ("No shifts could be applied"). Not pursued: this table is context, in no kill condition. Reading: the literature ordering appears (Harmony and per-batch centring/whitening raise the batch score, the one-hot oracle saturates bio, noise scores 0.45 total), and the kNN-based batch scores rise for `T_centre` in all three datasets where the retrained attackers say leakage rises (section 5.2 point 6): the two families of metric disagree in direction for that arm.

### 5.6 What this does and does not establish
* It establishes, for one frozen encoder (2,000-HVG PCA-50), three datasets and the registered folds: (i) a +0.05 macro-F1 difference between arms is resolvable on HLCA (MDE80 0.035) but not on the scIB sets; (ii) a 16-d code, label-free or coarse-label-supervised,
  keeps 0.77-0.98 of the above-chance fine-label F1 and 0.76-1.00 of the held-type F1, so the "coarse label under-determines the representation" premise (H2-010 Hc; falsifier "coarse LDA keeps > 85% of raw" met on Immune and Pancreas, not on HLCA) is not what limits a learned
  adapter here; (iii) label-free batch-aware references (scVI, Harmony) sit above the inductive floor frontier in leakage-utility space on HLCA, so the open question moves from "utility under-determination" to "can an inductive adapter reach scVI's point"; (iv) per-batch centring/whitening is not a safe floor against
  retrained nonlinear attackers.
* It does NOT establish anything about a learned arm (none was run), about scFM encoders (Geneformer, UCE, scGPT untested), or about donor-ID attackers. Seeds replicate noise only. Immune and Pancreas intervals are over 10 / 9 batches. The K3 reference rule
  ("highest U_fine") can select a reference that leaks MORE than raw (it did on Pancreas, per-batch centring), which makes the registered test lenient toward firing; the per-reference table in 5.1 and 5.4 shows the scVI/Harmony reading. scVI is one seed and short
  (HLCA 82 epochs, Immune/Pancreas capped at 100), and the Pancreas scVI ran on rounded normalised counts. Attacker hyperparameters are fixed small grids. Harmony and the other `T_*` references are fitted per fold on that fold's z0 and use the held-out cells' batch ids by construction.
* Decision value for the programme: SC-0 closes Focus E as a check (NO-GO), consistent with the pattern in A, C and D that a label or low-dimensional projection already keeps the utility; it differs in that a measurable leakage-utility gap to batch-aware references
  does exist. If the programme wants one more single-cell attempt, the registered stage-2 claim needs rewriting around that gap (an inductive map matching scVI/Harmony-level L at their U, tested on HLCA donors where MDE80 is 0.035) rather than around utility retention; that would be a new pre-registration, not a continuation of this one.

### 5.7 Compute (worker time from `job_result.json`; the attacker evaluation is CPU-bound and holds a GPU slot)
scVI: 1,096 s GPU in total (HLCA 676, Immune 282, Pancreas 138) = 0.30 GPU-h; floors: HLCA 949/911/887 s, Immune 321/269/297 s, Pancreas 206/253/208 s per seed (4,300 s); `T_scvi` arm 261 s; dense frontier 841 s; scIB context (Immune, Pancreas) 762 s;
two failed HLCA scIB attempts (XLA CPU JIT allocation crash, 1,478 s) and a per-fold retry; aggregation on CPU (HLCA B=1000 plus cluster B=300 about 14 min). Total about 2.4 worker-hours; GPU-kernel time is a small fraction of it (scVI 0.30 GPU-h dominates), inside the 0.5 GPU-h budget.
Priority 4 was used once (first wave), as declared in section 4.

## 6. E2 (REOPENED 2026-11-04 by the coordinator after Auditor-1 F-06): inductive map to the scVI / Harmony point on HLCA. PRE-REGISTRATION E2 v0

**verdict_original (kept, unchanged): SC-0 = NO-GO (SC-K2 3/3 datasets, SC-K1 HLCA + Immune). E2 is a NEW attempt, reported alongside it; it neither continues nor overrides SC-0, and it does not edit FOCUS.md.** Written 2026-11-04 ~19:00Z VM clock,
BEFORE any E2 learned-arm number exists (no E2 job has been queued or run when this block is committed).

### 6.1 Why a new pre-registration, and what is rewritten
SC-K2 compared `ldaC_pca16` with raw (retention of above-chance F1) although that arm has L 0.84, i.e. it hides little batch, and label-free `pca16` retains the same share. It is replaced by a **leakage-matched** gate (E2-K2'). SC-K3 found a real transductive gap on HLCA (scVI
U 0.816 at L 0.80 vs the V-selected inductive floor 0.700, +0.116 [0.099, 0.136]; Harmony U 0.668 at L 0.37 vs the dense-grid floor +0.354 [0.334, 0.371]). Neither reference is inductive (batch covariate / batch ids, all cells). E2 asks the question SC-0 stopped short of:
**can an INDUCTIVE encoder (expression to latent; no batch covariate at inference; no held-out cell at fit) reach those points, measured against the V-selected inductive floor envelope?**

### 6.2 Disclosure (what is already known when this is written)
Known from SC-0 and not hidden: the HLCA floor and reference numbers of section 5 (all 3 seeds, B=1000), the dense-grid frontier for seed 0, and therefore the outcome of E2-K2' at the scVI target (the best label-only/PCA arm at L <= 0.85, `ldaC_pca16`, has U 0.607, gap 0.209 to the target).
NOT known: any number from a learned inductive arm. The grid below was chosen by me after seeing SC-0, from the SC-0 leakage range (floors cover L 0.10-1.0; targets L 0.80 and 0.37). A TRAINING-ONLY diagnostic (6.5) may rescale the beta grid before any held-out metric of a learned arm is read; such a rescale is logged as an amendment.

### 6.3 Data, folds, units, instrument (unchanged from SC-0 unless stated)
HLCA core subsample only (97,232 cells, 107 donors, 11 studies), the same three study-grouped folds, same encoder-0 HVG list per fold (2,000 HVG on training cells), same attackers (linear, MLP, kNN, LightGBM, donor-disjoint cross-fit halves, seeds 0-2), same probes (U = max(linear, kNN) macro-F1),
same L (worst-case retrained attacker, normalised to `raw50` and chance, present-class chance), same analytic u0. Immune and Pancreas are NOT run (batch units: MDE80 0.10-0.12, cannot adjudicate 0.05). Intervals: donor bootstrap (107 donors, B=1000) primary; study-cluster bootstrap (11 studies, B=300) reported next to it.
Folds are fixed, so the three seeds replicate training noise and the cross-fit over the three held-out study groups is the replication of units.

### 6.4 Learned arms (all INDUCTIVE: input = the fold's standardised 2,000-HVG expression of ONE cell, output = a 16-d code; HVG list, scaling, encoder, decoder and adversary are fitted on the fold's TRAINING cells only; the training batch id is used only as a training-time covariate/adversary label)
Encoder MLP 2000-512-256-(mu, logvar) of dimension d = 16, GELU, decoder 16(+B_train one-hot for cVAE)-256-512-2000, loss = 0.5 * sum_j (x_j - xhat_j)^2 + beta * KL(q(z|x) || N(0,I)); the RELEASED code is a sample z = mu + sigma * eps (fixed seeded eps), so beta sets a rate. Families (13 configs, each x 3 folds x 3 seeds):
* `vae_b{2,10,50}` — no batch information anywhere: the mandatory NO-ADVERSARY rate/dither control.
* `cvae_b{2,10,50}` — decoder conditioned on the training batch id (encoder never sees it): the inductive analogue of scVI.
* `vaeadv_b10_l{1,4,16}` — `vae` plus a batch adversary on the released code: gradient reversal with weight lambda, adversary = MLP 16-128-B_train, class-balanced CE over training batches, RE-INITIALISED every 250 steps (a retrained adversary, not a co-adapted one).
* `cvaeadv_b10_l{1,4,16}` — both.
* `vaeplc_b10_l4` — placebo: the adversary is trained on cell-level PERMUTED batch labels (any gain over `vae_b10` that survives here is not adversarial).
Training: AdamW lr 1e-3, weight decay 1e-4, batch 1024, 30 epochs, KL annealed linearly over the first 5 epochs, seed s in {0,1,2}. No selection on held-out batches inside training.
**Checks, computed on TRAINING cells only and printed for every arm:** (i) *adapter trained*: split the fold's training donors into two halves; a freshly retrained linear and MLP attacker (class-balanced) on one half's codes must score the other half at balanced accuracy <= chance_train + 0.05 (chance_train = 1/B_train, worst of the two); required for the four adversarial families and the placebo; (ii) *collapse*: mean KL < 1 nat or training reconstruction R^2 < 50% of PCA-16's R^2 on the same cells.
An arm that fails (i) or (ii) in ANY fold (seed-mean) is INELIGIBLE for the claim (still reported). This answers the audit: an adversary that never reached its own objective cannot be called a null.

### 6.5 Training-only diagnostic and the beta scale
Before the full run, one job trains `vae_b{2,10,50}` and `vaeadv_b10_l{1,4,16}` on fold 0, seed 0, and prints training KL, reconstruction R^2 and the in-sample adversary accuracy (no held-out cell is scored). If the median-beta `vae` has KL < 1 nat (collapse) or > 150 nats (no compression), the whole beta grid is multiplied by one factor from {0.1, 0.3, 3, 10} chosen to bring the median KL to 5-50 nats; the factor is written as amendment E2-A1 BEFORE the full jobs are queued. lambda is never rescaled.

### 6.6 Floors, targets, matched-leakage selection
* **Inductive floor pool** = the SC-0 registered grid + the dense frontier, 3 seeds each (the dense arms existed for seed 0 only; seeds 1-2 are run as part of E2 with identical code). `T_*` references and `onehot_oracle` are not floors.
* **Targets** (fixed from SC-0, HLCA, 3-seed means): T_scvi = (U 0.816, L 0.80), primary; T_harmony = (U 0.668, L 0.37), secondary. A configuration is feasible for target T if its pooled L <= L_T + 0.05 (0.85 and 0.42).
* **Cross-fold selection** (as SC-0 K3): for each outer fold f, among the feasible configurations of the pool (learned: eligible ones only; floors: pool above) choose the one with the highest U_fine on the OTHER two folds, subject to feasibility on those folds; its predictions on fold f are used. Result: `sel_learned(T)` and `sel_floor(T)`, pooled over folds.
  If no floor is feasible at T, the floor envelope is U = 0 and this is reported; if no learned config is feasible, the arm fails.
* **Gain** Delta(T) = U(sel_learned) - U(sel_floor), paired donor bootstrap. Also reported: fraction of the floor-to-target gap closed, Delta / (U_T - U(sel_floor)); and the same table with the test-selected best configuration per family as an optimistic bound.
* **Attribution** (reported, not part of GO): `vaeadv` vs `vae` and `cvaeadv` vs `cvae` at matched L; `vaeplc` vs `vae`.

### 6.7 Margin and power (decided before the run)
HLCA floor-vs-floor MDE80 from SC-0 (six pairs): 0.028 / 0.029 / 0.029 / 0.040 / 0.040 / 0.057, median **0.035**. The margin is **m = 0.05** (1.4x the median, within the pair range). After the run the MDE80 of the primary comparison (`sel_learned` vs `sel_floor`, donor-level swap permutation + bootstrap SE, as in 2.6)
is printed; if it exceeds m the verdict is UNDERPOWERED, not NO-GO. With SE ~ 0.007-0.010 for a single comparison a Delta of 0.05 is 5-7 SE, so the design is decidable; if the printed MDE80 of the primary comparison were > 0.05 I would stop and say so.

### 6.8 Decision rules (any E2-K failing stops the claim; verdict reported next to verdict_original)
* **E2-K0 (power):** MDE80 (primary comparison) <= m. Else UNDERPOWERED.
* **E2-K1' (measurability):** raw50 worst attacker >= chance + 0.15 and the SC-0 controls <= chance + 0.03 (the unattainable 3x arm is dropped for HLCA fold chances 0.14-0.50, as stated in SC-0 2.9-1; SC-0 itself stays NO-GO on its capped reading). Satisfied by the SC-0 raw50 predictions (excess 0.389); recomputed in the E2 aggregation.
* **E2-K2' (leakage-matched under-determination, replaces SC-K2):** the best LABEL-ONLY / PCA family arm (`pca{4,6,8,12,16,24,32}`, `rp{16,32}`, `ldaC`, `ldaC_pca{8,16,24,32}`, `ldaF16`, `leace_pca16`; noise/dither arms excluded) feasible at the PRIMARY target (L <= 0.85, cross-fold selected) has U_T - U_family <= m = 0.05, i.e. it already
  fills the target at the target's leakage. Printed sensitivity: retained fraction (U_family - u0) / (U_T - u0) against the old 0.70. If K2' fires, E2 stops (NO-GO, label-only floor holds at matched leakage).
* **E2-K3' (the claim):** at the primary target, (a) sel_learned pooled L <= 0.85; (b) Delta(T_scvi) >= m with donor-bootstrap 95% lower bound > 0; (c) seed-level Delta >= m in >= 2 of 3 seeds; (d) the selected configuration(s) eligible (6.4 checks pass).
  **E2-GO** = K0, K1', not K2', and K3' (a)-(d). **E2-NO-GO** = K1' or K3' fails with the upper 95% bound of Delta < m (resolved null), or K2' fires. **E2-INCONCLUSIVE** = point Delta < m <= upper bound, or K0 fails (UNDERPOWERED). The Harmony target is reported with the same rule and does not enter GO.
* An E2-GO would only nominate Focus E for locking; the lock decision stays with the main agent (charter rule 6).

### 6.9 Budget, queue, code
<= 3 GPU-h. Expected: 13 configs x 3 folds x 3 seeds trainings of ~10-40 s plus attacker evaluation (~12 s per arm-fold on HLCA) = about 0.5 h per seed on one GPU slot, three seed jobs in parallel, plus a 2-minute diagnostic and two CPU-bound dense-floor jobs (seeds 1, 2).
Priority 4 is used ONCE for the first wave (diagnostic, three seed jobs, two dense-floor jobs). Code: `src/e_singlecell/{inductive,evaluate_e2,aggregate_e2}.py`. Specs `jobs/specs/e_e2_*.yaml`, names prefixed `e_`.

### 6.10 Amendments and status log (E2)
* **E2-A0 (18:30Z VM clock, no number involved):** the header of section 6 says "~19:00Z"; the pre-registration was committed (ce7611d) at 18:13Z VM clock, before any E2 code or job existed. Code committed afterwards (`inductive.py`, `evaluate_e2.py`, `aggregate_e2.py`).
* 18:29Z queued at priority 4 (the one allowed use for E2): `e_e2_smoke` (synthetic dataset, wiring check, numbers not reported) and `e_e2_diag` (training-only diagnostic, 6.5). The three seed jobs and the two dense-floor jobs are queued only after the diagnostic has been read.
* **E2-A2 (2026-11-04T18:49Z, local UTC; VM clock not queried, VM saturated), TIGHTENING ONLY, before any adversarial number exists (no E2 job has produced output).** Prompted by Builder A's calibration (A.md AUDIT FOLLOW-UP (4)): an in-sample adversary criterion "within 0.05 of chance" is unattainable at small n because a fresh adversary memorises noise.
  The 6.4 trained-check (i) is replaced by BOTH of: (a) the adversary is scored on donors it was NOT fit on (the existing donor-disjoint halves: train on one half of the training donors, score the other half; excess over chance_train <= 0.05, worst of linear/MLP); and (b) a MEMORISATION FLOOR is measured, not assumed:
  the same attacker code (linear, MLP), trained and scored in-sample on a 15,000-cell subsample of the fold's training cells, applied to (1) iid N(0,1) 16-d noise latents of the same size and (2) dithered PCA-16 latents (PCA-16 + 0.5 sd Gaussian noise, the SC-0 `dith16_r0.5`); both are also scored donor-disjoint. The arm's own in-sample adversary accuracy must be <= the noise-latent in-sample floor + 0.05
  (the dithered-PCA numbers are printed as the leaky reference, not used as a bar). An adversarial arm must pass (a) AND (b) in every fold (seed-mean) to be eligible; previously only the weaker original criterion applied. No-adversary rate/dither arms and the placebo adversary stay as planned. Implemented in `evaluate_e2.py` (meta `floors`, check fields `in_linear`, `in_mlp`, `floor_in_noise`) and `aggregate_e2.py` (eligibility).
* **E2 diagnostic read (21:27Z VM clock; training-only, fold 0, seed 0, `e_e2_diag`, run with the A2 floor code).** Training KL / reconstruction R^2 (PCA-16 = 0.264): `vae_b2` 14.8 nats / 0.405; `vae_b10` (the median beta) 4.5 / 0.326; `vae_b50` 0.6 / 0.108 (collapsed, ineligible). The median-beta KL is inside the 1-150 nat window, so NO beta rescale (E2-A1 not needed).
  Memorisation floor, HLCA fold 0 (12 training batches, chance 0.083): noise latents in-sample linear 0.102 / MLP 0.124, donor-disjoint 0.075 / 0.078; dithered PCA-16 in-sample 0.448 / 0.544, donor-disjoint 0.365 / 0.399. The adversaries do NOT meet the A2 check at lambda in {1, 4, 16}: donor-disjoint MLP excess over chance 0.22-0.24 and in-sample MLP 0.49-0.56 against a floor of 0.124 + 0.05;
  `vaeadv_b10_l16` is no more invariant than `vae_b10` (in-sample MLP 0.489 vs 0.468). Under the pre-registered rule (lambda is never rescaled) the adversarial families are expected to be INELIGIBLE and E2 then tests the non-adversarial inductive arms (`vae`, `cvae`) against the floor envelope; I report that the adversary did not converge instead of reading any adversarial null.
* 21:30Z queued at priority 5: `e_e2_s{0,1,2}` (13 configs x 3 folds) and `e_e2_dense_hlca_s{1,2}` (dense floors for the two seeds that lack them).

### 6.11 E2 RESULT, pre-registered grid (3 seeds x 3 folds x 13 configs; HLCA; run 21:30-22:00Z VM clock; aggregates `results/e_e2/e2_final.{md,json}`; jobs `e_e2_s{0,1,2}` 1,203-1,237 s each, `e_e2_dense_hlca_s{1,2}` 574 s, diagnostic `e_e2_diag` 175 s: about 1.8 GPU-slot-hours)
**verdict_original: SC-0 = NO-GO (unchanged). verdict_tightened (E2, pre-registered rule incl. amendment A2): E2-NO-GO, resolved. No inductive encoder in the registered grid reaches the scVI point; the best eligible one is BELOW the V-selected inductive floor envelope.**

* **Instrument.** E2-K1' PASS (raw50 worst attacker 0.670 vs chance 0.281, excess 0.389; controls: attacker excess const 0.000, noise 0.002, batch-shuffle -0.003; probe U minus u0: const -0.012, noise 0.001, label-shuffle 0.002). E2-K0 PASS: realised MDE80 of the primary comparison 0.020 (SE 0.005) against the margin m = 0.05 (SC-0 floor-pair median 0.035), so the design was decidable, not underpowered.
* **Trained-check A2 (memorisation floor, unseen donors).** Floors measured on HLCA training cells, seed-mean (fold 0 / 1 / 2; chance_train 0.083 / 0.111 / 0.143): noise latents in-sample linear 0.104 / 0.136 / 0.162, MLP 0.288 / 0.398 / 0.257; donor-disjoint 0.080 / 0.116 / 0.144 (= chance); dithered PCA-16 in-sample MLP 0.538 / 0.525 / 0.630, donor-disjoint MLP 0.393 / 0.353 / 0.459.
  So the in-sample floor is NOT chance (MLP 0.26-0.40 on pure noise), as Builder A found, and unseen-donor scoring is what stays at chance. Result: **0 of 8 adversarial/placebo configs are eligible.** `vaeadv_*`: donor-disjoint MLP accuracy 0.30-0.40 (excess over chance 0.22-0.25) at every lambda in {1, 4, 16}, the same as the no-adversary `vae_b10` (0.31-0.40);
  `vaeadv_b10_l16` lowers the linear attacker (0.15-0.16 vs 0.26-0.31) and nothing else. `cvaeadv_*`: donor-disjoint excess small (MLP 0.128 / 0.135 / 0.178 vs chance) but in-sample MLP 0.48-0.53 against floor + 0.05 of 0.34 / 0.45 / 0.31: memorisation above the floor. The placebo adversary is indistinguishable from `vae_b10` (U 0.421 vs 0.421, L 0.50 vs 0.49)
  and so is the real one (U 0.413-0.418, L 0.49-0.50): **at lambda <= 16, 30 epochs, the adversary did not converge and had no measurable effect**; this is not an adversarial null. `vae_b50` and `cvae_b50` collapsed (KL 0.6-0.7 nats, U 0.05) and are ineligible. Eligible: `vae_b2`, `vae_b10`, `cvae_b2`, `cvae_b10`.
* **Learned configs (U_fine [donor CI], L [CI]; 3-seed means).** `vae_b2` 0.701 [0.670, 0.709], L 0.68 [0.58, 0.72]; `cvae_b2` 0.673 [0.646, 0.683], L 0.68 [0.52, 0.72]; `vae_b10` 0.421, L 0.49; `cvae_b10` 0.363, L 0.48 (cVAE lowers L ~0.1 below the matching VAE at equal rate and costs ~0.03-0.06 of U). References: scVI 0.816 [0.780, 0.826] / L 0.80; Harmony 0.668 [0.647, 0.678] / L 0.37; raw50 0.787.
* **E2-K2' (leakage-matched under-determination), replaces SC-K2: does NOT fire.** At the scVI leakage (L <= 0.85) the best label-only/PCA arm (cross-fold: `rp16` in all folds) has U 0.670 at L 0.81: gap to the target 0.146 (> 0.05). Sensitivity: retained fraction vs target 0.82, which WOULD exceed the old 0.70 threshold; the absolute gap is the registered quantity.
* **E2-K3' at the scVI point (L <= 0.85):** selected learned = `vae_b2` in all folds (U 0.701, L 0.68 [0.58, 0.72]); selected inductive floor = `noise_r0.5` / `dith32_r0.5` / `noise_r0.5` (U 0.738, L 0.81). **Delta = U_learned - U_floor = -0.037 [-0.045, -0.026]** (donor bootstrap, B=1000); study-cluster [-0.051, -0.016]; per seed -0.032 / -0.050 / -0.030; fraction of the floor-to-scVI gap closed -0.48.
  Condition (a) leakage ok, (b) Delta >= 0.05 fails, (c) 0 of 3 seeds, (d) eligible; upper bound -0.026 < m: **resolved null, E2-NO-GO.**
* **Harmony point (L <= 0.42), secondary:** no eligible learned config is feasible (the only configs at L <= 0.42 are the collapsed `b50` arms, U 0.05); label-only/PCA family also infeasible. Does not enter GO.
* **Attribution.** vae vs vaeadv vs vaeplc at b10: no difference (above). cVAE (decoder conditioned on the training batch) is the only mechanism that lowered leakage at fixed rate. Nothing here supports "the adversary adds nothing" (it did not train, A2); it supports "a rate/dither VAE and a batch-conditioned decoder are the only effective inductive arms in this grid".
* **Honest statement of the registered outcome.** In the registered grid an inductive map does NOT close the gap to the scVI/Harmony point. Two caveats that the registered rule hides, stated before any post hoc run: (1) the learned grid has a HOLE between `b2` (L 0.68) and raw (L 1.0), so the rule "best configuration with L <= 0.85" compares `vae_b2` at L 0.68 with floors at L 0.81;
  at matched leakage ~0.70 the floor envelope is `noise_r1` (U 0.675, L 0.70) and `vae_b2` (U 0.701, L 0.68) is about +0.03 above it (point estimate, unverified), so the registered NO-GO may partly be a grid-coverage artefact; (2) the adversaries never trained, so the "learned invariance" half of the hypothesis is untested, not refuted.
  Both are addressed by a clearly labelled POST HOC fill (6.12), which cannot change the registered verdict.
