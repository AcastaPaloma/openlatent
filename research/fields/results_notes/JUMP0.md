# JUMP0: 20-minute metadata check for H2-011 (stage 0), 2026-11-04

Scope: metadata only. No image-derived feature values were downloaded. What was read: GitHub metadata CSVs (about 15 MB), two PDFs (7.5 MB), parquet footers and the four index columns
(Source, Plate, Well, JCP2022) of one assembled profile table (6 MB, no feature column), S3 key listings, READMEs. Total about 45 MB local plus 6 MB metadata copied to the VM
(`~/work/data/jump_meta`, 6 files, sha256 below). No GPU, `~/queue` untouched, no git commands.
Tags: [O] opened and read by me; [C] computed by me from the opened metadata; [I] my interpretation; [E] my estimate, not measured.

## 0. Verdict

**Gate (replicate structure): PASS.** **Worth a floors-only stage 1 (CPU, <= 0.5 GPU-h, 1 builder-day): PASS, marginal, with the margins recast (section 7) and a kill-first ordering.**
The decisive unknown in RANKING_R2 section 6 ("do non-Target2 compounds have wells in >= 2 sources? from memory mostly single-source") is refuted: 81,735 of 82,276 production compounds (99.3%) have wells in >= 2 of the 5 Arevalo sources [C].
My own estimate that all four kill conditions clear is about 0.10 [E], because published numbers already show a very small utility spread (K3 pre-pressure, section 5). The check is cheap and decisive, which is the only reason it passes.

Three facts that decided it:
1. Cross-source replicates exist by design: 99.3% of the 82k production compounds appear in >= 2 of the 5 benchmark sources (67,818 in >= 3), 96.6% of compound-source pairs are a single well, and 97.2% of positive pairs are cross-source [C].
2. The benchmark's own mAP spread across ten methods is tiny: in scenario 5 Harmony is +0.05 (negcon) / +0.02 (nonrep) over the no-correction baseline, the best method +0.06 / +0.04 [O, SI]. H2-011's "+0.05 mAP over floors" is outside the whole published range; the room is on the leakage axis (baseline iLISI 0.01 / kBET 0.01, Harmony 0.25 / 0.03).
3. K2 (does a 301-dim compound-LDA leak source?) is NOT establishable from any published number I could open; the within-source oracle Hj is undefined for 96.6% of Arevalo-5 compound-source pairs. Both must be measured in stage 1.

## 1. Facts found (URL, what I opened)

| # | Fact | Source (what I opened) |
|---|---|---|
| F1 | Scenarios: S1 302 Target2 compounds, 1 source (source_6), 13 batches; S2 302, 3 sources (2, 6, 10), 43 batches; S3 82,278 compounds, same 3 sources; S4 302, 5 sources (2, 3, 6, 8, 10), 46 batches, 3 microscope types; S5 82,412 compounds, same 5 sources, 60 batches. Batch variable = Metadata_Batch (S1) or Metadata_Source (S2-S5). **So H2-011's "scenarios 4 and 5 = 302 and 82k" is right; the scout's "302 is smallest" refers to S1.** | https://pmc.ncbi.nlm.nih.gov/articles/PMC11297288/ (WebFetch x2); configs `inputs/conf/scenario_1..5.json` at https://github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection (raw files read) |
| F2 | "Target2" = JUMP-Target-2-Compound plate, run in every batch. Its 302 ids = 293 test compounds + 8 positive controls + DMSO [C: 302 distinct JCP2022 on TARGET2 plates, 9 of them in perturbation_control.csv]. | `plate.csv.gz`, `well.csv.gz`, `perturbation_control.csv` from https://github.com/jump-cellpainting/datasets/tree/main/metadata (commit 799fe5a, 2025-01-14) |
| F3 | The 5 benchmark sources: source_2 CV8000, source_3 Opera Phenix, source_6 CV8000, source_8 ImageXpress Micro Confocal, source_10 CV8000. `download_data.sh` sets `sources="source_2 source_3 source_6 source_8 source_10"`. | `microscope_config.csv`; `download_data.sh` (raw) |
| F4 | Library design: partners exchange nominated compounds; wave 1 compounds "typically screened in five replicates across five sites", wave 2 "five replicates across three sites"; 12 production sources; compound screened at 10 uM except source_7 (0.625 uM); Target-2 run in every batch of every source. | JUMP preprint https://www.biorxiv.org/content/10.1101/2023.03.23.534023v1.full (WebFetch, first 100k of 126k characters; the last 26k characters were not read) |
| F5 | Licence of the data: CC0 1.0 (three independent statements, see section 4). | registry page, gallery README, Arevalo Data availability |
| F6 | 5-source per-plate profile tables: 1,216 parquet files, 16.17 GB (exact sum of S3 key sizes); layout `cpg0016-jump/{source}/workspace/profiles/{batch}/{plate}/{plate}.parquet`. All 13 sources have well-level parquet profiles (source_15 also csv). | `https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&prefix=cpg0016-jump/source_X/workspace/profiles/` (ListObjectsV2, keys and sizes only) |
| F7 | A pre-assembled well-level table exists: `cpg0016-jump/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect.parquet` (the Arevalo "mad_int_featselect" recipe), 2,810,741,997 B, 803,853 wells, 4 metadata columns + 737 features, 10 sources (1, 2, 3, 5, 6, 7, 8, 9, 10, 11; no 4, 13, 15), one row group. A `..._harmony.parquet` twin (2.84 GB, same shape) exists. Pre-featselect `profiles_var_mad_int.parquet` = 12.1 GB, 3,180 features. | footer range-reads with pyarrow; 4 index columns read (6 MB); HEAD for ETag `32fc704d018039d48503588aac65aa8c-336` and Last-Modified **2026-11-03 14:38:58 GMT** (the assembled objects were rewritten yesterday; per-plate files are dated 2024-03-06, so pin by ETag or use per-plate files) |
| F8 | Published SI per-scenario scores for 11 methods (S1, S2, S3, S5 tables as text; S4 only inside Fig. 5, not extractable). Section 5. | SI PDF https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-50613-5/MediaObjects/41467_2024_50613_MOESM1_ESM.pdf (text extracted with pypdf); main PDF https://carpenter-singh-lab.broadinstitute.org/files/anne/files/191_Arevalo_NatureComm_2024.pdf |
| F9 | Near-duplicate split risk is negligible: of 115,795 assembled compound ids, 115,790 distinct InChIKey first blocks; only 5 skeletons carry two ids (10 ids). | `compound.csv.gz` (8.6 MB, sha256 8885960e...) |
| F10 | Newer related release: JUMP-Lite v1.0 (6 sources: 2, 4, 6, 7, 8, 13; 551 plates; 3,776 compound identities; per-site DINOv2/OpenPhenom/MorphEm/SubCell embeddings) under `source_all`; image-derived and not a well-level cross-source compound set. Not used. | `.../publication_data/2026_jump_lite/metadata/v1.0/README.md` and `.../target_2/v1.0/README.md` (read) |

## 2. Replicate-structure tables (all [C], from `well.csv.gz` + `plate.csv.gz` + `perturbation_control.csv`; code in the appendix)

Exclusions copied from the Arevalo loader: controls (perturbation_control.csv), `JCP2022_UNKNOWN`, `JCP2022_033954`, and source_3 batches in SOURCE3_BATCH_REDLIST (except TARGET2).
Reproduction check against the paper: S3 production total wells per compound: 14,185 with 1 well (paper "around 15,000 with one replicate"), 79,066 with <= 3 (paper "~79,000"); S5 plates 1,029 / batches 61 vs paper 60; S3 batches 44 vs 43 (metadata HEAD drift of about 1).

### 2a. Scenario view (non-control compounds)

| Scenario | Plates (batches) | Compounds | Distribution over #sources | Cross-source share of positive pairs |
|---|---|---|---|---|
| S1 (source_6, Target2) | 21 (13) | 293 | 1 source: 293 | 0 |
| S2 (3 src, Target2) | 37 (29) | 293 | 3: 292, 2: 1 | 0.59 |
| S3 (3 src, production + Target2) | 689 (44) | 82,266 | 1: 14,190; 2: 43,961; 3: 24,115 | 0.715 (0.936 production only) |
| S4 (5 src, Target2) | 66 (46) | 293 | 5: 292, 4: 1 | 0.73 |
| S5 (5 src, production + Target2) | 1,029 (61) | 82,402 | 1: 541; 2: 13,912; 3: 33,950; 4: 33,326; 5: 673 | 0.810 (**0.972 production only**) |

S5 production only (COMPOUND plates): 82,276 compounds; >= 2 sources 81,735 (99.3%); >= 3 sources 67,818; >= 4 sources 33,742; singletons in one source only 541 (0.7%).
Compounds per source: source_10 57,804; source_2 57,485; source_3 34,811; source_6 58,655; source_8 57,217. Largest membership patterns: 10+2+6+8 (11,873), 10+6+8 (11,547), 10+2+3+6 (11,260).

### 2b. Wells per compound per source

| Set | Wells per compound-source pair (S5 production) | Meaning |
|---|---|---|
| Arevalo-5 production | 1 well: 257,711 (96.6%); 2: 7,041; 3: 521; 4: 241; 5: 133; >= 6: 279 | replicates are almost purely across sources; 4,745 compounds (5.8%) have any within-source duplicate |
| Arevalo-5 Target2 (293 compounds) | median wells per compound per source: source_2 10 (min 4, max 20), source_3 25 (24, 50), source_6 21 (9, 42), source_8 4 (4, 8), source_10 6 (5, 12) | within-source and cross-source replicates in all 5 sources; 21, 25, 10, 4, 6 plates |
| Total wells per production compound, all sources | median 5; counts at 3: 6,319; 4: 22,147; 5: 45,742; 7: 18,451 | matches "five replicates" |

### 2c. Per-source table, 10-source assembled table (production = COMPOUND plates; assembled index columns joined to plate.csv.gz)

| source | microscope | production plates (batches) | compounds | share also in >= 1 other source | median wells / compound | singleton share | Target2 plates | Target2 wells / compound (median) |
|---|---|---|---|---|---|---|---|---|
| source_1 | Opera Phenix | 51 (6) | 59,737 | 1.000 | 1 | 0.956 | 0 | none |
| source_2 | CV8000 | 219 (13) | 57,485 | 1.000 | 1 | 0.958 | 10 | 10 |
| source_3 | Opera Phenix | 112 (13) | 34,811 | 1.000 | 1 | 0.998 | 25 | 25 |
| source_5 | CV8000 | 197 (33) | 32,289 | 0.999 | 2 | 0.310 | 24 | 24 |
| source_6 | CV8000 | 225 (13) | 58,655 | 1.000 | 1 | 0.958 | 21 | 21 |
| source_7 | CV7000 | 121 (8) | 6,053 | 0.463 | 5 | 0.000 | 7 | 7 |
| source_8 | ImageXpress | 199 (4) | 57,217 | 1.000 | 1 | 0.984 | 4 | 4 |
| source_9 | Opera Phenix (1536-well) | 99 (11) | 31,845 | 1.000 | 4 | 0.000 | 9 | 36 |
| source_10 | CV8000 | 205 (17) | 56,917 | 1.000 | 1 | 0.963 | 6 | 6 |
| source_11 | Operetta | 172 (5) | 32,245 | 1.000 | 2 | 0.399 | 7 | 7 |

Not in the assembled table: source_15 (Opera Phenix, 147 compound plates, 41,125 compounds in metadata; per-plate parquet 10.5 GB, added 2025-01), source_4 (ORF + Target1/2), source_13 (CRISPR + Target2).
Production compounds in the 10-source table: 115,720; sources per compound: 1: 3,294; 2: 1,260; 3: 50,204; 4: 35,971; 5: 23,347; >= 6: 1,644; at least 2 sources 112,426 (97.2%).
The assembled table has about 90% of the COMPOUND-plate wells listed in the metadata (750,114 production rows vs about 832k listed for the same 10 sources [E]); the exclusion rule for the rest was not documented in anything I opened.

Within-source replicates of production compounds exist only in sources 5, 7, 9, 11 (and are on different plates): >= 2 wells for 22,287 / 6,053 / 31,845 / 19,365 compounds, 100% / 100% / 100% / 99.6% on >= 2 plates (source_9: 4 plates per batch, one batch). So the within-source oracle Hj is definable on production compounds only outside the Arevalo-5 set.

### 2d. Unit counts relevant to power (K0)

| Unit | Arevalo-5 | 10-source assembled |
|---|---|---|
| sources with >= 30k production compounds that have a positive elsewhere | 5 | 9 (1, 2, 3, 5, 6, 8, 9, 10, 11) |
| microscope models | 3 (CV8000 x3, Opera x1, ImageXpress x1) | 5 (CV8000 x4, Opera x3, CV7000, ImageXpress, Operetta); single-source models: CV7000, ImageXpress, Operetta |
| eligible production compounds (>= 2 sources) | 81,735 from metadata, 81,100 inside the assembled table (md5 split held-out half 40,266; per source-fold, held-out compounds with a query well in s and a positive elsewhere: 17,228 (source_3) to 28,618 (source_6)) | 112,426 (held-out half about 56k [E]) |
| production plates | about 930 (1,029 incl. Target2) | 1,713 incl. all plate types |
| Target2 held-out half (oracle arm) | 146 compounds | 146 compounds |

## 3. Files, locations, sizes

- Data root: `s3://cellpainting-gallery/cpg0016-jump/` (region us-east-1, public, `aws s3 cp --no-sign-request`, or `https://cellpainting-gallery.s3.amazonaws.com/<key>`).
- Per-source per-plate parquet (4,762 features, MAD etc. not applied): source_2 3.08 GB (229 files), source_3 4.04 (303), source_6 3.24 (246), source_8 2.85 (216), source_10 2.95 (222); other sources: 1 2.08 (55), 5 3.43 (260), 7 1.74 (128), 9 4.25 (108), 11 2.34 (180), 15 10.54 (316 incl. csv), 4 3.66 (277), 13 2.09 (154).
- No per-scenario aggregated tables exist on S3: the Snakemake rule `write_parquet` concatenates plates into `outputs/scenario_k/raw.parquet`. Exact download sizes by scenario (sum of plate files in the scenario plate list): **S1 0.28 GB, S2 0.49 GB, S3 9.17 GB, S4 0.88 GB, S5 13.70 GB** (all 5 sources: 16.17 GB).
- Nearest pre-aggregated alternative: assembled COMPOUND v1.0: `profiles_var_mad_int_featselect.parquet` 2.81 GB (737 features, ready for the stage-1 floors), `..._harmony.parquet` 2.84 GB (JUMP's own Harmony output, batch key not verified), `profiles_var_mad_int.parquet` 12.1 GB. Also `v1.0a` (same sizes) and `COMPOUND_no_source7`.
- Metadata: `jump-cellpainting/datasets/metadata` well.csv.gz 5.98 MB, plate.csv.gz 9.7 KB, compound.csv.gz 8.6 MB, compound_source.csv.gz 0.38 MB, microscope_config.csv, perturbation_control.csv. sha256 (pinned at commit 799fe5a): well fde9fe0678b8d9973f17e106e4cf7e62017110ee3b700f10e27122e1d5f291a1; plate 541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319; compound 8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a; perturbation_control d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed; microscope_config e10a15cc...; compound_source 229b7adb.... Copied to the VM (`~/work/data/jump_meta`, 6 MB; compound.csv.gz not copied).

## 4. Licences

| Item | Licence | Evidence |
|---|---|---|
| cpg0016-jump images and profiles (Cell Painting Gallery) | **CC0 1.0 Universal**; citation requested | registry page https://registry.opendata.aws/cellpainting-gallery/ ("CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"); https://github.com/broadinstitute/cellpainting-gallery/blob/main/README.md ("All the data is released with CC0 1.0 Universal (CC0 1.0)"); Arevalo Data availability ("released under CC0 1.0 Universal license"). The gallery asks to acknowledge the AWS Open Data sponsorship and cite Chandrasekaran 2023. |
| Derived embeddings from CC0 profiles | no restriction from the data side; HF upload allowed by charter rule 2 | [I] |
| `jump-cellpainting/datasets` metadata repo | BSD-3-Clause (GitHub API `license.spdx_id`); the raw `LICENSE` path I tried returned 404; no separate data licence stated in the metadata README | https://api.github.com/repos/jump-cellpainting/datasets/license |
| Arevalo benchmark code | BSD-3-Clause (GitHub API); Zenodo record 11644046 (68 KB code zip only, no data or scores) is tagged cc-by-4.0 in its Zenodo metadata | https://api.github.com/repositories/661860261; https://zenodo.org/api/records/11644046 |
| Dependencies | copairs BSD-3 (env pins v0.4.1; latest 0.6.0 released 2026-11-02); scib MIT (the env uses the fork `johnarevalo/scib@frozen_paper`, licence not checked); scvi-tools BSD-3; scanorama MIT; pycytominer BSD-3; **harmonypy GPL-3.0 (run as a reference, do not redistribute)** | PyPI JSON and GitHub API |
| JUMP-Lite | derived from cpg0016; cite JUMP and the model papers | its README |

## 5. Is the benchmark's published evaluation deterministic and runnable on the profile tables with fixed code?

Yes for the metric, with caveats [O: repo files]:
- Environment is pinned (`environment.yaml`: scanpy 1.10.0, scvi-tools 1.1.2, harmonypy 0.0.9, copairs @v0.4.1, scib fork `frozen_paper`). Run: `snakemake -c3 --configfile inputs/conf/scenario_k.json` after `download_data.sh`. The Snakefile defaults to scenario_1 and the workflow `mad_int_featselect`.
- mAP is copairs `average_precision` (cosine similarity, `batch_size=20000`) then `mean_average_precision(threshold=0.05, null_size=10000, seed=0)`: deterministic given the pinned library. "negcon" = negatives are same-plate DMSO wells; "nonrep" = negatives are same-plate wells of other compounds; positives are all same-compound wells (`pos_diffby=[]`, so within-plate, within-source and cross-source pairs all count).
- Explicit seeds in the repo: only the copairs null (seed 0) and the Sphering epsilon grid (`seed=[6,12,2022]`). scVI, DESC, kBET (R) and Leiden use library defaults; the paper reports no repeated runs ("Variability in LISI label scores is <1e-3 for most of the scenarios" is the only variability statement). Not verified: seed behaviour of scVI/DESC/kBET.
- It is **not a held-out-compound protocol**. All compounds are evaluated with all their wells; Sphering's epsilon is chosen among 25 values by the mean of the two mAPs on the same labels (`correct/sphering.py: select_best`); Harmony is run with `nclust=300` ("Number of compounds"); DESC needs labels. H2-011's train/held-out compound split needs a new wrapper around `copairs.average_precision` and new fold code. This is the main builder cost.
- Metadata is fetched at HEAD (`plate.csv.gz`, `well.csv.gz`), only `microscope_config.csv` is pinned to a commit, hence the 1-batch drift above. Pin the sha256 in section 3.
- Which of `mean_map`, `fraction_below_p`, `fraction_below_corrected_p` is plotted in the SI "mAP (controls)/(nonrep)" columns is not stated in the text I read (code has all three per metric). S1 shows 0.94-0.95 for every method, which looks like a fraction rather than a mean [I]. **Unit ambiguity is open; check the Source Data before using levels.**

### Published numbers I can use (SI Sup. Figs 2-5, 2-decimal rounding, columns: kBET, iLISI(batch), mAP controls, mAP nonrep)

| Method | S5 kBET | S5 iLISI | S5 mAP ctrl | S5 mAP nonrep | S3 kBET | S3 iLISI | S3 mAP ctrl | S3 mAP nonrep |
|---|---|---|---|---|---|---|---|---|
| Baseline (MAD, INT, featselect) | 0.01 | 0.01 | 0.21 | 0.03 | 0.03 | 0.00 | 0.19 | 0.03 |
| Sphering (DMSO, epsilon tuned on mAP) | 0.01 | 0.00 | 0.22 | 0.03 | 0.02 | 0.00 | 0.21 | 0.03 |
| Combat | 0.01 | 0.02 | 0.24 | 0.04 | 0.03 | 0.02 | 0.22 | 0.04 |
| Harmony | 0.03 | 0.25 | 0.26 | 0.05 | 0.06 | 0.25 | 0.24 | 0.05 |
| scVI | 0.04 | 0.23 | 0.24 | 0.05 | 0.07 | 0.27 | 0.24 | 0.05 |
| DESC (uses labels) | 0.01 | 0.27 | 0.22 | 0.04 | 0.00 | 0.15 | 0.14 | 0.02 |
| Seurat RPCA | 0.10 | 0.31 | 0.27 | 0.06 | 0.15 | 0.28 | 0.25 | 0.05 |
| Seurat CCA (R, 17 days for all JUMP) | 0.13 | 0.49 | 0.27 | 0.07 | 0.17 | 0.51 | 0.28 | 0.06 |

What follows from these, and only these:
- K1-type measurability: with no correction, source is fully separable by kNN metrics (iLISI 0.01 / kBET 0.01 in S5). Linear mean/scale or negcon-whitening methods (Combat, Sphering) leave it unchanged (iLISI 0.00-0.02). A retrained attacker's accuracy is not published.
- K3-type pressure: the full spread of ten methods on mAP in S5 is +0.06 / +0.04 over baseline; the Python-implementable transductive references (Harmony, scVI, Combat) are +0.05 / +0.02 at most. Leakage-side spread is large (iLISI 0.01 to 0.49).
- K2 (compound-supervised LDA): **no published number exists** for any compound-supervised linear projection on JUMP. DESC is the only label-using method and is not a linear projection. What I can say structurally [I]: the assembled table has 737 features, so a 301-dim LDA keeps 41% of the dimensions and cannot be assumed too small to carry source; but LDA's within-class scatter pooled over 3-5 sources contains exactly the source differences of the same compound, so it is expected to suppress them. Both directions are arguments, not evidence.
- Sup. Fig. 6 (Seurat CCA, S3): compounds present in only 2 of 3 sources retrieve better than compounds in all 3 (n = 10,136 vs 23,782), i.e. retrieval gets harder as more sources must be aligned (authors' reading).

## 6. Corrections to H2-011 text

1. Scenario labels are right (S4 = 302 Target2 compounds, 5 sources; S5 = 82,412, 5 sources); "~82k" for S5 and the "five laboratories, 3 imaging systems" statement are consistent with the metadata (source_2, 6, 10 CV8000; source_3 Opera; source_8 ImageXpress).
2. Hj ("within-source mAP minus cross-source mAP of the best transductive method >= 0.10") is not defined for 96.6% of Arevalo-5 production compound-source pairs. It is defined on Target2 (held-out half = 146 compounds; MDE80 of the oracle gap about 0.07 assuming a per-compound AP SD of 0.3 [E, not measured], so descriptive only) and on production compounds of sources 5, 7, 9, 11 (outside the Arevalo-5 set).
3. The "+0.05 mAP over floors" stage-2 margin and the Hk margin of 0.05 exceed the whole published range of ten methods on the benchmark mAP (section 5). Recast (section 7).
4. "Falsified if Harmony/Sphering are within 0.05 of the within-source oracle" cannot be judged on production compounds; replace it by JK3.
5. The assembled `profiles_var_mad_int_featselect` applies rank-INT and correlation feature selection jointly over all sources (and all held-out compounds): it is a transductive "encoder 0". Treat it as a shared fixed front end for all arms (a limitation, stated in the claim) or recompute from `profiles_var_mad` (12.2 GB).

## 7. Pre-registration for a floors-only stage 1 ("JS-1", style of SC-K0..K3)

Budget: <= 0.5 GPU-h (attackers, optional scVI), 3-4 CPU-h, ~3.3 GB download (2.81 GB assembled table + 0.49 GB for R0), 1 builder-day, one attempt, no learned arm. Reusable code: `a_pathorob/adapters.py` (floors, `lda`, `lda_pca`, `pca`, `randproj`, `centre_whiten`), `attackers.py` + `suite.py`, the dither hook; new: loader, md5 split, copairs wrapper, folds, Harmony/Combat wrappers.

**Frozen before any metric.** Data: sha256 of the three metadata files (section 3); the assembled `profiles_var_mad_int_featselect.parquet` pinned by ETag `32fc704d018039d48503588aac65aa8c-336`. Compound universe: production ids with wells in >= 2 sources, minus controls, `JCP2022_UNKNOWN`, `JCP2022_033954`. Split: md5(JCP2022 + salt) parity -> train T / held-out H, 20% of T as validation V; 3 salts = 3 "seeds". Source folds for the secondary protocol: leave one source out over the 9 eligible sources.
Protocols: **P-C** (primary): floors and adapters fit on T wells of all sources; U and L measured on H. **P-S** (secondary, 9 units): fit on T wells of sources != s; queries are H wells of s with positives in other sources; L = source-vs-rest balanced attacker retrained on a plate-disjoint half of s.
Utility U: held-out-compound cross-source retrieval mAP = copairs `average_precision` with `pos_sameby=[JCP2022]`, `pos_diffby=[Metadata_Source]`, `neg_sameby=[Metadata_Plate]`, `neg_diffby=[JCP2022]`, then `mean_average_precision(0.05, null 10000, seed 0)`; report mean AP and fraction below corrected p. Secondary: the benchmark's own negcon/nonrep (pos_diffby=[]).
Leakage L: attackers (linear, MLP, kNN, LightGBM) retrained on plate-disjoint halves of H wells to predict source; worst-of-four; L_norm = (acc - chance) / (raw - chance).
Arms (floors, inductive, fit on T only): raw737; PCA-d, dithered PCA-d (H2-014), random projection-d, d in {16, 32, 64, 128, 301}; **label-only**: ldaT2 (LDA on the 302 Target2 classes pooled over training sources, 301 dims, plus shrinkage; this is the user's question), ldaP-d (PCA-256 then shrinkage LDA on T production compounds, d in {16, 32, 64, 255}). Transductive references (never wins): Harmony (harmonypy, batch = Metadata_Source, on PCA-100), Combat, Sphering (DMSO-fitted, epsilon chosen on V), scVI if time allows; JUMP's own `..._harmony.parquet` as a cross-check only. Controls: constant, noise-16, shuffled-source labels for the attacker (chance); shuffled-compound labels for mAP (chance); the production-compound split must put no id on both sides (F9: negligible skeleton overlap).
Code-validity gate R0 (run first, 0.49 GB): from the 37 S2 plates compute baseline mAP and compare with SI Sup. Fig. 3 baseline row (controls 0.90, nonrep 0.41) [units caveat in section 5].

Kill conditions (any one stops JS-1 and the JUMP line; text in the SC-K style, [I] = interpretation):
- **JK0 instrument.** (a) R0 differs from the published baseline row by > 0.03; or (b) fewer than 10,000 held-out compounds with >= 2 sources [observed about 40k Arevalo-5, about 56k 10-source]; or (c) the floor-vs-floor MDE80 on U (median over the six pairs among pca64, dith64, randproj64, ldaP64, ldaT2-301, raw737; paired within-compound permutation + compound bootstrap) is > 0.02 [= the stage-2 margin]; or (d) fewer than 8 sources with >= 30k eligible compounds [observed 9 in the 10-source table, 5 in Arevalo-5]. [I] (d) failing only makes P-S descriptive; it does not stop P-C. A K0 firing is reported "NO-GO (instrument)".
- **JK1 measurability.** Raw737 source attacker on H wells < min(3 x chance, 0.85) or < chance + 0.15, or any control > chance + 0.03. [I] same capped reading as SC-K1. Published support: baseline iLISI 0.01 / kBET 0.01.
- **JK2 under-determination (label-only code).** The V-selected best label-only arm (ldaT2-301, ldaT2-d, ldaP-d) has L_norm <= 0.25 (A's label-only LDA reached L_worst 0.00-0.20 with 2-4 classes, FOCUS.md) AND U >= U(R) - 0.03, in >= 2 of 3 splits, where R = the best transductive reference by U. Fires = a compound-supervised linear code already hides source and keeps held-out retrieval. This is the measurement for "does LDA-301 leak source": not answerable from published numbers.
- **JK3 headroom.** F* = the V-selected best inductive floor (all non-reference arms) at L_norm <= L(R) + 0.05. Fires if U(R) - U(F*) <= 0.03 in >= 2 of 3 splits (P-C) AND in >= 5 of 9 source folds (P-S). [I] If no inductive configuration reaches L <= L(R) + 0.05, JK3 does not fire (headroom exists). Published pressure: on S5 the baseline is within 0.03 of Harmony on nonrep, so the matched-leakage clause does all the work (Harmony iLISI 0.25 vs baseline 0.01).
- Sensitivity prints (do not change the verdict): JK3 with the benchmark's negcon/nonrep U; Hj on Target2-146 and on sources 5, 9, 11 (within-source mAP minus cross-source mAP); JK2 under "either"; Arevalo-5 only subset.
- **GO for stage 2** = none of JK0..JK3 fires. A pass buys one attempt, <= 5 GPU-h, and does not by itself cancel the stop rule (H2-019), exactly as RANKING_R2 section 7.

Stage-2 claim, recast (only after GO): a label-free AE+adversary / cVAE / dither adapter reaches, at U within 0.01 of the best inductive floor, L_norm lower by >= 0.15, or at matched L_norm higher U by >= 0.02; 95% CI (bootstrap over held-out compounds) above 0 in >= 2 of 3 splits, V-selected, power audit first. The +0.05 mAP margin of H2-011 is dropped because no published method moves benchmark mAP by that much.
Evaluation order (cheapest kill first): R0, JK1, JK3 with PCA/LDA/Harmony only, JK2, JK0(c). Expected to abort before any attacker sweep if JK3 fires.

## 8. What I could not establish (exactly)

- Whether a compound-supervised LDA to 301 dims leaks source: no published number; needs profiles (stage 1).
- Per-method numbers for S4 (figure only) and the unit of the SI mAP columns (fraction vs mean); the Source Data xlsx returned 403 from static-content.springer.com.
- The batch key and recipe of JUMP's `..._harmony.parquet`; why the assembled table holds about 90% of listed COMPOUND wells; whether the 2026-11-03 rewrite of the assembled objects changed their content.
- Seed behaviour of scVI, DESC, kBET in the benchmark; the 26k unread characters of the JUMP preprint; `metadata_harmonized/cpg0016-jump_draft.csv` (18.8 GB, unopened).
- Whether Arevalo's published scenario scores can be reproduced to the second decimal: R0 tests it on S2 only.
- Probabilities in section 0 and the oracle MDE are my estimates [E].

## Appendix: the computation (stdlib + pandas; inputs are the three CSVs of section 3)

```python
import sys, pandas as pd
d = sys.argv[1] + '/'
pl = pd.read_csv(d+'plate.csv.gz'); w = pd.read_csv(d+'well.csv.gz'); pc = pd.read_csv(d+'perturbation_control.csv')
RED = {"CP_32_all_Phenix1","CP_33_all_Phenix1","CP_34_mix_Phenix1","CP_35_all_Phenix1","CP_36_all_Phenix1","CP59","CP60"}
pl = pl[(~pl.Metadata_Batch.isin(RED)) | (pl.Metadata_PlateType == 'TARGET2')]
m = w.merge(pl, on=['Metadata_Source','Metadata_Plate'])
t = m[~m.Metadata_JCP2022.isin(set(pc.Metadata_JCP2022) | {'JCP2022_UNKNOWN','JCP2022_033954'})]
S5 = ['source_2','source_3','source_6','source_8','source_10']
x = t[t.Metadata_Source.isin(S5) & (t.Metadata_PlateType == 'COMPOUND')]
g = x.groupby(['Metadata_JCP2022','Metadata_Source']).size().unstack(fill_value=0)
ns = (g > 0).sum(axis=1); r = g.values; mm = r.sum(1)
print(len(g), ns.value_counts().sort_index().to_dict(), (ns >= 2).sum(),
      ((mm*(mm-1)/2 - (r*(r-1)/2).sum(1)).sum() / (mm*(mm-1)/2).sum()))   # cross-source share of positive pairs
```
Run on the metadata files in the scratchpad, output: 82,276 compounds; {1: 541, 2: 13917, 3: 34076, 4: 33341, 5: 401}; 81,735; 0.972.
