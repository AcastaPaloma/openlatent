# Round 2 scout: fields where the downstream use is open-ended / many-task (2026-11-04)

Question from the main agent: after four attempts killed by a trivial floor or headroom check (label-supervised LDA hides the nuisance;
retrieval secret overlaps the utility; robotics harm was clutter), where is "keep this label, hide that nuisance" NOT solvable by a
label-only projection, and a deterministic benchmark with trivial baselines already exists? The only unfalsified lead is a
task-agnostic adversarial autoencoder (AE, no label in training) that keeps held-out-class structure at matched nuisance leakage.

Evidence level. Every source listed in section 6 was opened with WebFetch (HTML or abstract page). WebFetch summarises pages with a small
model, so every number below is "as extracted"; I cross-checked only where two sources agree. Several PDFs (bioRxiv, Eta-WavLM, CellPainTR,
Springer/Nature) could not be read (compressed streams or login redirects); anything that rests only on a search-result snippet is marked
SNIPPET and is not relied on. Headroom figures are my priors, not measurements, and are labelled so.

## 0. Verdict and ranking

| Rank | Field | Benchmark (deterministic?) | Trivial floors with published numbers | Do label-only methods saturate it? | Headroom prior (NOT measured) | Compute (4x A6000) | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Single-cell batch integration (+ spatial transcriptomics as a second dataset family) | Open Problems batch_integration (MIT, Viash/Nextflow) + scib-metrics (BSD-3, JAX) + scIB-E fine-label tasks; datasets 16k to 800k cells; yes, given fixed embeddings | Unintegrated PCA, batch-centring, Harmony, scVI; built-in `embed_cell_types` (one-hot label) control; lung example: unintegrated total 0.548, scVI 0.648, scANVI 0.678 | YES on every label-based metric (control is a one-hot of the labels). NO evidence of saturation on intra-cell-type, fine-label, label-free metrics: scVI (no labels) beats scANVI on intra-type; a 2025 benchmark finds no semi-supervised method consistently beats the best unsupervised one | Leakage side large (iLISI 0.13 Harmony, 0.125 scVI on lung; scFM embeddings carry batch). Utility-at-matched-leakage side unknown; novelty versus scVI is the risk. Guess 0.03-0.08 | Under 10 GPU-h for 3 datasets x 3 seeds; HLCA 585k cells needs subsampling | GO-CHECK |
| 2 | Cell Painting / JUMP source removal (microscopy FM embeddings as the same family) | Arevalo et al. Nat Commun 2024: JUMP cpg0016, CC0, 5 scenarios, 10 metrics via scib, code BSD-3 (Snakemake); deterministic | MAD-normalised "no correction" baseline; Harmony, Seurat-RPCA top-3; sphering/Combat ~ baseline; "none adequately remove batch effects" in hardest scenarios | Partly. 302 Target2 compounds => LDA has 301 dims, so it does not hide source by dimensionality; but it fits the benchmark's own compounds. Held-out compounds (scenarios 3/5, 82k compounds) is the test it cannot pass trivially | Unknown; Harmony is a strong label-free floor. Guess 0.02-0.06. Replicate structure of non-Target2 compounds unverified | Under 5 GPU-h (tabular well-level profiles) | GO-CHECK |
| 3 | EEG foundation-model embeddings, subject removal across many tasks | OmniEEG-Bench (54 datasets, 10 FMs, 6 task families, github.com/ncclab-sustech/omni-eegbench) + FMScope (MIT) audit code; 3 seeds, licences per dataset not reported | LEACE on subject: linear subject probe 60-99% -> 0-5%; per-subject re-referencing; handcrafted band-power LR beats frozen FMs on EEGMAT (0.847 vs 0.708-0.755) | NO for LaBraM/CBraMod: LEACE leaves an MLP probe at chance AND raises label accuracy (+6 to +12 pp EEGMAT). Residual only for REVE | REVE only: MLP residual 56.4 (SleepDep), 40.0 (Stress), 11.5 (ADFTD) balanced accuracy after LEACE. Cohorts 17-71 subjects. Guess 0.0 for LaBraM/CBraMod, real for REVE | 5-10 GPU-h on 4 cohorts x 3 FMs; REVE access unverified | GO-CHECK (conditional, REVE only) |
| 4 | Astronomy instrument-agnostic latents (HSC x Legacy cross-matches) | No fixed benchmark; Platonic Universe cross-matches + embeddings on HF (UniverseTBD/pu-embeddings), probes on 45k COSMOS galaxies | Random ResNet-18 probe, AION-1 embeddings (Table 5 of 2604.09787) | Not applicable: no label-only floor defined | Already published as a label-free method; probes without CIs; survey ID is trivially decodable from pixel scale | About 8 h on 4 H100s in the paper | SKIP as a focus; keep as optional generality check |
| 5 | Speech SSL features with speaker removal (VoicePrivacy / SUPERB-style) | VoicePrivacy 2024 (LibriSpeech + IEMOCAP, retrained semi-informed attacker, public scripts) | Six baselines B1-B6 incl. McAdams; 36 participant systems | Only 2 utility tasks (WER, emotion UAR); not many-task; the programme's Focus C already covers the emotion half | Saturated field, 36 systems | n/a | SKIP |
| 6 | ECG/PPG/IMU, image-embedding fairness, protein/molecule assay bias | ECG: 15-186 subjects, single task; fairness: NH-Fair (supervised, attribute labels in training); protein/molecule: none found | ECG: Gaussian noise; fairness: well-tuned ERM | n/a | n/a | n/a | SKIP |

Top 3: single-cell integration, Cell Painting/JUMP, EEG (REVE-only). Section 4 gives a concrete 1-2 hour floor check for each, with a draft kill rule.

## 1. Structural finding that explains the four earlier failures and should steer the choice

1. A label-only projection (LDA) is only a strong "hide the nuisance" floor when the label set is tiny (C-1 = 1-3 dims for PathoROB). The programme's own
   protocol-2 result (journal, A, 11:50Z) already shows the matched-dimension LDA+PCA arm leaks L 0.5-0.9 at d >= 16. Fields with 16 to 300 classes
   (cell types, compounds) therefore do not hand the floor to LDA on the leakage axis; they hand it to LDA only on label-based utility.
2. The field-standard benchmarks in single-cell already contain the label-only floor as an explicit control: Open Problems `embed_cell_types`
   is "cells embedded as a one-hot encoding of celltype labels" (config text), plus a jittered variant. So every label-based bio metric (ARI, NMI,
   ASW-label, cLISI, isolated-label F1) is saturated by a label-only code by construction. A task-agnostic AE can only win on metrics the one-hot
   code cannot score: label-free structure (cell-cycle conservation, HVG overlap, kNN Jaccard, PCR), finer labels than were used to fit, held-out
   classes. Those must be pre-registered before any number is seen, otherwise it is metric shopping.
3. Evidence that label supervision is not an advantage when the downstream use is broader than the training label: scIB-E (Genome Biol 2025) reports
   unlabeled scVI had the highest intra-cell-type conservation and that label-constrained methods (scANVI, CellSupcon) "can compromise
   intra-cell-type information and lead to over-correction"; a 2025 semi-supervised benchmark concludes "none consistently outperform the
   strongest unsupervised method, scCRAFT". Both are the same shape as the unfalsified lead from A.
4. Counter-evidence to keep in view: in the lung example in the scib-metrics docs scANVI has the best total (0.678 vs scVI 0.648), driven by the
   label-based bio score (0.732 vs 0.662). Label supervision wins whenever the metric uses the training labels.

## 2. Field-by-field

### 2.1 Single-cell batch integration (rank 1)

Benchmarks (all opened):
- Open Problems task_batch_integration, https://github.com/openproblems-bio/task_batch_integration , MIT. 18 methods (incl. Geneformer, scGPT, SCimilarity,
  scPRINT, UCE, scVI, scANVI, Harmony), 7 controls (`embed_cell_types`, `embed_cell_types_jittered`, `no_integration`, `no_integration_batch`,
  `shuffle_integration`, `_by_batch`, `_by_cell_type`), 13 metrics (ARI, NMI, ASW label, cLISI, isolated label ASW/F1, ASW batch, iLISI, kBET, graph
  connectivity, PCR, cell-cycle conservation, HVG overlap). 6 CELLxGENE datasets (DKD, GTEx v9 209k, HypoMap 385k, Immune ~360k, mouse islet >300k, Tabula
  Sapiens ~500k). The leaderboard page shows no numeric scores; several methods are 76-100% missing, so use it for the control list, not for numbers.
  An older embed page lists Immune (33.5k cells in scIB, 10 batches), Lung 32,472, Pancreas 6 datasets.
- scib-metrics, https://github.com/YosefLab/scib-metrics , BSD-3, JAX/GPU. Warns that its values are not comparable with the original scIB repo.
- scIB-E (Yi et al., Genome Biol 2025, PMC12636214; code github.com/Chenxin-Yi/scIB-E, AGPL-3.0, Zenodo 10.5281/zenodo.17481059). Datasets: Immune 33,506
  cells / 16 types, Pancreas 16,382 / 14, BMMC 90,261 / 45, HLCA 584,944 cells with 5 annotation levels, human fetal lung (14 broad and 144 fine types),
  HBCA >800k cells (11 broad labels, 41 subclusters; a differential-abundance task trained on 11 broad labels and tested on 41 subclusters). Level-1
  methods include a GAN (adversarial batch) arm: a label-free adversarial VAE already exists as a comparator. Immune/pancreas on figshare, BMMC GEO
  GSE194122, atlases on CELLxGENE. Data licences not verified.
- Published trivial-floor numbers (scib-metrics lung example, 2,000 HVG, cell count not shown on page): Unintegrated bio 0.667 / batch 0.370 / total 0.548;
  Harmony 0.630 / 0.544 / 0.596; scVI 0.662 / 0.627 / 0.648; scANVI 0.732 / 0.597 / 0.678. cLISI is 1.000 for unintegrated and scANVI (saturated). iLISI
  is 0.008 unintegrated, 0.125 scVI, 0.130 Harmony, so batch mixing is far from saturated in absolute terms.
- Foundation-model embeddings: bioRxiv 2025.12.19.695371 (8 scFMs, 6 CELLxGENE benchmarks, logistic-regression batch probes, 5-fold) reports that every
  scFM except scCello trails Harmony and scVI on scIB, some fall below PCA, and post-hoc batch-mean centring "partially improves" but does not close the
  gap. It lists adversarial or invariance objectives as untested future work. Exact scIB numbers are in figures only (not extractable).
  Genome Biol "Zero-shot evaluation reveals limitations of single-cell foundation models" and the "Biology-driven insights" paper were NOT opened (login
  redirects); the claim that Geneformer trails HVG on batch metrics is SNIPPET only.
- Spatial transcriptomics: Owkin benchmark (bioRxiv 2026.01.12.699017, 11 methods, MOSAIC 60-patient subset, code github.com/owkin/st-benchmark) is SNIPPET
  only; reported that scVI/scVIVA lead batch and bio conservation. Same metrics and codebase as single-cell, so treat as a second dataset family, not a field.

Where the label-only floor stops working (what a GO claim would have to be): fit on broad labels, score on fine labels (HLCA levels 1-5, fetal lung 14 -> 144,
HBCA 11 -> 41); fit on a subset of types, score on held-out types; label-free structure metrics. This is the same held-class protocol as A protocol 2.

Risks: (a) scVI is already a label-free, batch-conditioned latent and wins intra-type, so the AE is closer to "scVI plus an adversary"; the
novelty bar is the retrained-attacker leakage measurement, which scIB metrics (kNN-based iLISI/kBET) do not provide; (b) batch is confounded with biology
in several datasets, so leakage cannot go to chance; the controls `no_integration_batch` and `shuffle_*_by_batch` calibrate that; (c) scFM embeddings
are worse than HVG-PCA, so an adapter on scFM embeddings starts from a weak encoder; run PCA-HVG and scFM both; (d) scIB-E code is AGPL, reimplement the
metrics from scib-metrics (BSD-3) instead.
Compute: scVI/AE training is minutes per run on 33k cells; scib-metrics on GPU is minutes; HLCA needs a 100k-cell subsample. Total under 10 GPU-h for
3 datasets x 3 seeds x ~12 arms (my estimate).
Verdict: GO-CHECK. Closest analogue of A that has an explicit label-only control, hierarchical labels, and published signal that label supervision costs
intra-class structure.

### 2.2 Cell Painting / JUMP (rank 2)

- Benchmark: Arevalo et al., "Evaluating batch correction methods for image-based cell profiling", Nat Commun 2024 (PMC11297288). cpg0016-jump, 13 sources,
  well-level profiles with 4,762 CellProfiler features, five scenarios (302 Target2 compounds x 13 batches single lab up to 82,412 compounds x 60
  batches, 5 labs, 3 microscope types), 10 metrics normalised 0-1 (4 batch, 6 bio incl. replicate-retrieval mAP; implemented in scib; total = 0.4 batch +
  0.6 bio). Data CC0 1.0 on the Cell Painting Gallery (S3, unsigned); code github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection (BSD-3,
  Snakemake, Zenodo 10.5281/ZENODO.11644046). The repo's download script fetches 5 sources of profiles; byte size not stated anywhere I opened.
- Baseline: preprocessing only (low-variance removal, per-plate MAD, rank-based inverse normal transform, correlation feature selection at 0.9), no batch
  correction. Methods: Combat, Sphering, scVI, DESC, Harmony, MNN, fastMNN, Scanorama, Seurat-CCA, Seurat-RPCA. Published ordering: Harmony and Seurat-RPCA
  consistently top 3; Sphering/Combat/MNN comparable to the baseline in scenario 3; DESC (the only one needing biological labels) and sphering
  underperform the baseline in scenario 4; authors state "none of the methods are able to adequately remove the batch effects" in the hardest scenarios.
  Main-text gives no numeric scores (figures only).
- Newer multi-task view: "Same Encoder, Different Winner" (arXiv 2609.12761) evaluates LoRA-adapted DINO/OpenPhenom/SubCell on JUMP-CP (1,466 perturbations,
  28,003 wells, 10 sources), RxRx1, RxRx3-core with replicate mAP, scIB, CellProfiler-feature prediction, cross-batch recall; classical CellProfiler
  features score replicate mAP 0.205 on JUMP-CP versus 0.209-0.285 for the fine-tuned (label-supervised) encoders, i.e. small headroom there, and Harmony adds
  scIB total +0.070 on JUMP-CP. Code, checkpoints and the paired-view JUMP-CP set are announced but not released at the time of reading.
  BatchBench (arXiv 2610.30104, opened here; earlier dropped by the programme) is confirmed unusable for us: hidden private test (sources 4, 6, 9, 13), docker submissions, no licence stated;
  its baselines (ResNet18 0.175, ARM-BN 0.260 overall on 2,000 compounds) show how far from solved it is but cannot be scored offline.
  CellPainTR (arXiv 2509.06986) abstract opened only (PDF unreadable): claims to beat ComBat and Harmony on JUMP with source-specific tokens; code
  github.com/CellPainTR/CellPainTR. BALANS (arXiv 2601.22331) abstract only: faster Harmony-style batch kernels, no numbers.
- Why the label-only floor may fail here: 302 compound classes => 301 LDA dims that can carry source-specific directions; many-class label sets are
  where A's LDA arm leaked. Why it may not: the benchmark's bio mAP is computed on the same compounds, and Harmony (label-free) is already the strong floor.
- Unverified and decisive: for the 82k non-Target2 compounds, whether replicates exist across sources (otherwise "held-out compound mAP" is within-source and
  batch is confounded with compound). I could not confirm this from the pages I opened.
- Compute: tabular well-level profiles; Harmony/PCA/AE on CPU plus one GPU, under 5 GPU-h (my estimate); Seurat-CCA takes about 17 days on the full set per the paper,
  skip it.
- Verdict: GO-CHECK, second to single-cell because (a) the label-only evidence is thinner (only DESC is label-using and it underperforms), and (b) the
  held-out-compound replicate structure is unverified.

### 2.3 EEG foundation-model embeddings, subject removal (rank 3, conditional)

- Audits opened: "Identity Trap" (arXiv 2606.06647): LaBraM/CBraMod/REVE frozen embeddings on EEGMAT (36 subjects), ADFTD (65), SleepDep (71 source, 36 used),
  Stress (17, not public). Linear subject probe pre-erasure 59-99.5% balanced accuracy; after LEACE 0.0-5.3%; one-hidden-layer MLP residual 1.4-6.9% for
  LaBraM and CBraMod but 3.3 (EEGMAT), 11.5 (ADFTD), 56.4 (SleepDep), 40.0 (Stress) for REVE. Within-subject label accuracy rises after erasure (EEGMAT
  +12.0 +- 7.4 LaBraM, +7.4 +- 0.7 REVE). Handcrafted band-power LR (101 features) beats frozen FMs on EEGMAT (0.847 vs 0.708-0.755) and is near chance on the others.
  Trait labels (ADFTD, Stress) are subject-level constants, so erasure removes them by definition. Code FMScope, github.com/Jimmy110101013/fmscope (MIT;
  `erase` = LEACE with Ledoit-Wolf, `rereference` = per-subject mean). It ships no embedding caches.
  "Pretrained, Frozen, Still Leaking" (arXiv 2606.09189): BIOT/LaBraM/EEGPT; Gaussian noise, bottleneck, dropout and DP-SGD heads all fail a
  privacy-utility gate against a noise-aware adaptive attacker; subject-disjoint attribute transfer gains 0.24-0.45 with CIs; datasets 13-23 subjects.
- Many-task benchmark: OmniEEG-Bench (arXiv 2606.00815): 54 datasets (abstract) / 58 tasks (conclusion; the paper is internally inconsistent), 10 FMs
  (BENDR, BIOT, LaBraM, CBraMod, BrainOmni, FEMBA, Neuro-GPT, NeuroLM, EEGMamba, REVE), six task families, linear probing (up to 40 samples per subject per
  class), 8:1:1 subject-level splits, 3 seeds, balanced accuracy; code github.com/ncclab-sustech/omni-eegbench; dataset licences and total hours not reported;
  no handcrafted-feature baseline reported. EEG-FM-Compass (arXiv 2601.17883): 12 FMs, 13 datasets, 9 paradigms; finds linear probing often inadequate and
  from-scratch specialists competitive. Neither benchmark evaluates subject removal; the combination (OmniEEG tasks as the many-task utility, FMScope-style
  retrained probes as leakage) would be ours to build.
- Headroom read: LEACE is already enough for two of three FMs; only REVE (and possibly newer FMs) leaves nonlinear subject information. That is exactly the
  "linear erasure fools linear probes, a retrained nonlinear probe keeps it" pattern recorded for A, but on cohorts of 17-71 subjects the MLP-residual
  numbers carry wide intervals (not reported in the extract).
- Risks: tiny cohorts; subject-level trait tasks are incompatible with subject removal, so the many-task suite must be restricted to within-subject state
  tasks; some OmniEEG datasets are gated (TUH needs registration; not verified for others); REVE weights access not verified; reported "MLP residual" may
  be a train-size artefact.
- Compute: 5-10 GPU-h for 4 cohorts x 3 FMs (my estimate), embedding extraction dominated.
- Verdict: GO-CHECK only as a 1-hour REVE check; otherwise SKIP.

### 2.4 Astronomy instrument-agnostic latents (rank 4, SKIP as a focus)

- "Learning What's Real" (arXiv 2604.09787, Apr 2026) already does a label-free physics/instrument split with cross-matched supervision: ~100k HSC x Legacy
  pairs from the Multimodal Universe, two ResNet-18 encoders, conditional flow matching decoder, 75k steps, 4 H100s ~8 h. Not adversarial. Physics latent
  MLP-probe R^2: redshift 0.801 (AION-1 0.857), log M* 0.755 (0.818), age 0.257 (0.353); instrument latent decodes PSF/depth (0.44-0.68). No instrument
  classifier, no retrained attacker, no CIs or seeds, qualitative retrieval and outliers. Code github.com/pablomerc/galaxy-counter ("simplified").
- Platonic Universe (arXiv 2509.19453): cross-matches JWST-HSC 1.67k, Legacy-HSC 102k, DESI-HSC 18.6k; probes on 45k COSMOS galaxies (LePHARE labels);
  embeddings on HF (UniverseTBD/pu-embeddings), code github.com/UniverseTBD/platonic-universe; no random-init or pixel-PCA baseline in the text I read.
- Not a fit: no fixed benchmark with floors; survey identity is decodable from pixel scale in any embedding; label sets are continuous regressions, so
  "label-only LDA" is not defined. Keep as the generality check already planned in FOCUS A; do not spend a focus slot.

### 2.5 Speech SSL / VoicePrivacy (rank 5, SKIP)

- VoicePrivacy 2024 overview (arXiv 2601.11846): LibriSpeech dev/test (29+40 speakers each), IEMOCAP (10 actors, 4 classes), semi-informed attacker retrained
  on anonymised train-clean-360, utility = WER and emotion UAR only, privacy = EER; baselines B1-B6 including McAdams (signal processing) and a Whisper+VITS cascade
  reaching EER ~48% with notable emotion loss; scripts public (github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024); 36 participant systems.
  Per-baseline numbers not in the extracted text. Corpus licences not stated; IEMOCAP is licence-restricted (from memory, not verified).
- No SUPERB-style many-task evaluation of speaker-removed SSL features found. Eta-WavLM (arXiv 2505.19273) PDF unreadable, metadata only; SEF-MK
  (2508.07086) and 2507.17851 seen as SNIPPET only. Skip: crowded, two utility tasks, and C already covers the emotion half.

### 2.6 Others checked, all SKIP

- ECG: REAN (arXiv 2607.06037) is label-supervised against frozen classifiers, one task (4-class arrhythmia), 186 subjects, and its own retraining-aware
  attacker brings re-identification back to 0.59-0.89; linkage-attack paper (arXiv 2508.15850) 106 subjects, no defence. Tiny cohorts, raw-signal level.
  Other ECG foundation-model leakage papers (2604.10424 membership inference, SHINE inversion) are SNIPPET only. IMU/PPG: nothing found.
- Image-embedding fairness: NH-Fair (arXiv 2602.03895, abstract only) finds many debiasing methods do not beat a well-tuned ERM baseline; methods use
  attribute labels in training; no task-agnostic group. CLIP debiasing papers (FairerCLIP, Fair PCA 2310.11867, 2411.12785) are SNIPPET only.
- Protein/molecule assay bias: one extended search found only single-cell/proteomics batch papers and a small-molecule compendium (arXiv 2605.31522, SNIPPET);
  no embedding benchmark with an assay-bias nuisance. Not verified either way.
- Microscopy foundation models: folded into 2.2.

## 3. What is already published (so we do not claim novelty we lack)

| Item | What it is | Relevance / gap |
|---|---|---|
| scIB-E Level-1 GAN arm (Genome Biol 2025, PMC12636214) | Label-free adversarial batch removal inside a VAE, 16 methods compared | Adversarial label-free arm exists in single-cell; no GAN-vs-scVI numbers in the text I read |
| scDREAMER (Nat Commun 2023, SNIPPET) | Unsupervised adversarial VAE for atlas integration | Predates 2024; precedent for the AE arm |
| Semi-supervised scRNA-seq benchmark (bioRxiv 2025.08.23.671952) | scANVI, scGen, ssSTACAS, scDREAMER, ItClust vs scVI, Harmony, Seurat RPCA, scCRAFT on 6 datasets, labels missing/wrong/mixed | "None consistently outperform the strongest unsupervised method"; at 70% boundary-mixed labels scANVI 7.55% below scVI, scGen 22.12% below. Code github.com/RainySheena/benchmark_semi |
| scFM batch paper (bioRxiv 2025.12.19.695371) | Frozen scFM embeddings carry batch; batch-centring partial fix | Lists adversarial/invariance objectives as untested; this is the exact opening for an adapter |
| Unsupervised IRM (arXiv 2505.12506, Norman and Meir) | Label-free invariance by distribution alignment across environments; PICA (linear) and VIAE (VAE); no adversary | Toy data (synthetic, S/C-MNIST), no LEACE/PCA baseline, no held-out task; shows the idea is not yet benchmarked at scale |
| SPLINCE (arXiv 2506.10703, NeurIPS 2025) | Oblique projection removing a concept while preserving covariance with the TARGET label | Needs the task label (so not task-agnostic); only Bias in Bios and Winobias; LEACE not mentioned in the abstract |
| Eguard 2411.05034, SPARSE 2602.07090, Denoising-Aware Inversion, ALGEN | Embedding-privacy defences/attacks (from contrarian file; not re-opened) | Already in Focus D context |
| EEG privacy audits (2606.09189, 2606.06647) | No tested defence passes a privacy-utility gate; linear erasure leaves MLP-recoverable identity for REVE | Adaptive (noise-aware) and nonlinear attackers are standard in this subfield |
| REAN (2607.06037) | Adversarial anonymiser vs frozen classifiers; retraining-aware attacker restores re-ID to 0.59-0.89 | Another instance of Finding 1 (frozen attackers understate leakage) |
| Astronomy label-free split (2604.09787) | Cross-match supervised, generative, no adversary | Occupies the astronomy version of the idea |
| Adversarial removal leaks to a post-hoc probe | Elazar and Goldberg 2018 (cited by 2603.16377, SNIPPET) | Old result; confirms why retrained probes are mandatory |

I found no 2025-2026 paper that combines (task-agnostic AE) + (adversary on a batch/source/subject label) + (attackers retrained on the latent) +
(held-out-class utility) on one benchmark. This is a negative search result from about 20 queries, not proof of absence.

## 4. Concrete floor checks (draft pre-registrations; each 1-2 hours; main agent to freeze margins)

Common arms (floors): unintegrated/raw; PCA-d; per-batch (source, subject) mean-centring; LEACE on the nuisance; Harmony where available; LDA(C-1) on the
fit labels; LDA then centring/LEACE; label one-hot control (Open Problems style). Oracle/ceiling: unintegrated PCA for utility (max leakage), the
jittered one-hot for label utility. Attackers retrained on the latent: linear, MLP, kNN, with an independent-backbone probe where possible. Leakage L
normalised by chance. 3 seeds, bootstrap over donors/batches/subjects, validation split for configuration search, sealed test for the number.

1. Single-cell (rank 1). Data: Immune (33,506 / 10 batches / 16 types) and Pancreas (16,382 / 9 / 14) from the scIB figshare links, plus HLCA or fetal lung
   subsample for fine labels. Utility = (a) kNN/probe accuracy on held-out cell types (fit labels exclude them), (b) fine-label accuracy when fit on broad
   labels, (c) scIB-E kNN-Jaccard and PCR (label-free). Headroom stage H1: retrained batch probe on the best floor exceeds 3x chance. H2: the best floor loses at
   least 0.10 of (a) or (b) relative to unintegrated PCA at leakage within +0.05 of the best floor. Kill if either fails on 2 of 3 datasets, or if scVI matches the
   AE within 0.02 on (a)-(c) at matched leakage (then the contribution is only the measurement protocol).
2. Cell Painting (rank 2). First confirm from the profile metadata that non-Target2 compounds have replicates in more than one source (if not, drop the
   held-out-compound arm and keep only scenario 1-2). Then: fit LDA/AE on Target2 + controls, score replicate mAP on non-Target2 compounds, with a retrained
   source probe. Kill if Harmony or LDA + source-LEACE is within 0.02 mAP of the AE at equal source leakage.
3. EEG (rank 3). REVE embeddings on EEGMAT and SleepDep only. Kill if the best of {LEACE, per-subject centring, LDA} leaves an MLP probe within 2x chance
   (then no headroom), or if an AE with adversary does not reduce the REVE MLP residual below 2x chance with within-subject label accuracy at or above the
   LEACE level.

## 5. What I could not verify

- Open Problems control scores (leaderboard shows no numbers); the "label-only control saturates label metrics" claim rests on the control definitions
  plus a tentative reading of the chart ordering, not on observed values.
- Dataset licences for scIB figshare data, CELLxGENE atlases (CC BY 4.0 is typical but not checked here), OmniEEG datasets, IEMOCAP.
- Byte sizes of the JUMP profile tables; replicate-across-source structure for non-Target2 compounds; which exact scenario numbers each Arevalo method scored.
- REVE/LaBraM/CBraMod weight access from the VM; OmniEEG-Bench split files.
- All scFM benchmark claims from Springer pages (login redirect), the 2026.04.17 bioRxiv PDF (unreadable), CellPainTR body text, Eta-WavLM body text.
- Headroom priors in the table are guesses informed by sections 1-2, not measurements.

## 6. Sources opened

Single-cell: https://pmc.ncbi.nlm.nih.gov/articles/PMC12636214/ (scIB-E, text through char 100k plus Methods tail);
https://github.com/Chenxin-Yi/scIB-E ; https://github.com/openproblems-bio/task_batch_integration (+ /src/control_methods, /src/metrics,
raw embed_cell_types config, raw README); https://openproblems.bio/results/batch_integration_embed ; https://openproblems.bio/benchmarks/batch_integration ;
https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html ; https://github.com/YosefLab/scib-metrics ;
https://www.biorxiv.org/content/10.64898/2025.12.19.695371v1.full ; https://www.biorxiv.org/content/10.1101/2025.08.23.671952v1.full .
Cell Painting: https://pmc.ncbi.nlm.nih.gov/articles/PMC11297288/ ; https://github.com/carpenter-singh-lab/2023_Arevalo_BatchCorrection (+ raw download_data.sh) ;
https://github.com/jump-cellpainting/datasets ; https://broadinstitute.github.io/jump_hub/explanations/data_description.html ;
https://arxiv.org/html/2609.12761v1 ; https://arxiv.org/html/2610.30104v1 ; https://arxiv.org/abs/2509.06986 ; https://arxiv.org/abs/2601.22331 .
EEG: https://arxiv.org/html/2606.06647 ; https://arxiv.org/html/2606.09189 ; https://arxiv.org/html/2606.00815 ; https://arxiv.org/abs/2601.17883 ;
https://github.com/Jimmy110101013/fmscope .
Astronomy: https://arxiv.org/html/2604.09787 ; https://arxiv.org/html/2509.19453 .
Speech/ECG/fairness/invariance: https://arxiv.org/html/2601.11846v1 ; https://arxiv.org/pdf/2505.19273 (metadata only) ; https://arxiv.org/html/2607.06037 ;
https://arxiv.org/html/2508.15850 ; https://arxiv.org/abs/2602.03895 ; https://arxiv.org/html/2505.12506 ; https://arxiv.org/abs/2506.10703 .
