# Scout report: medical imaging and biosignals

Scout cluster: CT/MRI/X-ray/pathology, ECG/EEG/EMG, wearable IMU/PPG, clinical video, shared medical latents, de-identification, harmonisation, shortcut removal.
Written 2026-11-04. Every paper below was opened (arXiv abstract/HTML page, GitHub, PhysioNet or PMC) and the numbers are what that page said. Many 2026 IDs (2604-2610) are unreviewed preprints; treat them as leads, not as established results. "Not stated" means the page I opened did not say, not that the paper does not.

## 0. Cross-cutting findings (these matter more than any one paper)

1. **The frozen-attacker problem reproduces in other modalities.** REAN (ECG) reports re-identification 0.964 -> 0.000 against frozen classifiers, but its own retraining-aware attacker recovers 0.590 at a 20% anonymised mix and 0.886 at 100%. CLOAK (sensor HAR) evaluates with utility and intrusive models "pre-trained on the raw sensor data"; its one retrained attacker (MotionSense gender, CNN 54.0%, Transformer 57.0%) "struggled to converge". A 2026 review of privacy-preserving action recognition found under 10% of papers test adaptive attackers. Our SUMMARY finding 1 is a real, general gap, not a KTH quirk.
2. **Hand-built or training-free transforms keep matching learned ones, again.** EEG: Riemannian mean alignment plus scatter whitening, no learned model, takes worst-case attacker accuracy 90.8% -> 15.6% (BCI IV-2a) for a 7-point motor-imagery loss. Pathology: Reinhard stain normalisation (+16.2% relative robustness index) and ComBat (+27.4%) beat domain-adversarial training (improved only 12 of 20 models, by 0.23 points on average). This echoes the canonical stick figure result.
3. **"Where the secret lives decides what hides it" shows up everywhere.** EEG subject identity sits in the aperiodic 1/f component and covariance geometry (removing 1/f cut subject probes 9-19 points on two models, none on a third). Pathology centre signature sits in stain/colour statistics, but ComBat also deletes biology when centre is correlated with class. ECG identity should sit in QRS morphology while arrhythmia sits in rhythm (my hypothesis; no paper I opened tests it).
4. **Metric traps to avoid.** REAN's gender "accuracy" of 0.142 is not protection (flip the sign: about 0.86). Any sign-flip-invariant, chance-normalised score fixes this. We already do this.
5. **Removing a nuisance can help the task, not just privacy.** Erasing the subject axis from frozen EEG foundation-model embeddings improved label decoding by +6 to +12 points (primary cells). Debiased latents raise accuracy under inverted correlations in X-ray/OCT. This makes the "isolate the nuisance" story stronger than pure privacy.

## 1. Key papers (20)

Method families: **ADV** adversarial/gradient-based learned, **MI** mutual-information or dependence penalty, **DIS** disentangled latent split, **TF** training-free hand-built transform, **GEN** generative (diffusion/GAN), **AUD** audit/benchmark only.

| # | Paper | ID / URL | Takeaway | Family |
|---|---|---|---|---|
| 1 | REAN: Reconstruction-aware ECG Anonymization (Jul 2026) | arXiv 2607.06037 | 1-D U-Net trained against frozen identity/gender/age + arrhythmia classifiers; 4 PhysioNet DBs, 186 subjects, 1.16M 8 s windows; ReID 0.964 -> 0.000, macro-AUROC 0.9982 -> 0.9991; deterministic anonymiser means retrained attacker recovers ReID 0.886 at 100% anonymised training. No code link seen. | ADV (frozen-attacker losses) |
| 2 | NeuroPriv (Aug 2026) | arXiv 2609.00390 | Adversarial representation on compact wearable-EEG features (EEGMAT): cognitive state 0.788 -> 0.781, subject ID 0.692 -> 0.206, gender 0.858 -> 0.563, age 0.789 -> 0.467. Attacker protocol not described on the abstract page. | ADV |
| 3 | Riemannian Suppression of Cross-Session Identity Leakage (Oct 2026) | arXiv 2610.15626 | Training-free covariance alignment + whitening; worst of five attackers 90.8% -> 15.6% (BCI IV-2a) and -> 9.9% (OpenBMI); MI accuracy -7.0 / -2.1 points. Sizes and chance level not stated. | TF |
| 4 | The Identity Trap in EEG Foundation Models (Jun 2026) | arXiv 2606.06647 | FMScope: frozen subject variance 13-89x random null in 12/12 model-dataset pairs; subject-axis erasure +6 to +12 pp label decoding; code github.com/Jimmy110101013/fmscope. Models LaBraM, CBraMod, REVE; datasets not named on page. | AUD + erasure |
| 5 | Pretrained, Frozen, Still Leaking (Jun 2026) | arXiv 2606.09189 | Cross-encoder ridge decoder + linear bridge leaks spectral attributes from BIOT/LaBraM/EEGPT (EEGMMI etc.); DP-SGD at epsilon 4, 8 leaves the attribute channel essentially unchanged. No code/benchmark release. Does not claim held-out-subject identity recovery. | AUD |
| 6 | SuLoRA: Subject-Specific Low-Rank Adapters (Oct 2025) | arXiv 2510.08059 | Models subject dependency instead of erasing it (shared + low-rank per-subject weights); better than subject-agnostic and subject-specific baselines on MEG speech and EEG motor imagery. Numbers not on page. | Param. factorisation |
| 7 | Membership Inference on ECG Foundation Encoders (Apr 2026) | arXiv 2604.10424 | SimCLR/TS2Vec/MAE encoders leak participation; strongest in small institution-specific cohorts; hiding raw signals and labels is not enough. No numbers on page. | AUD |
| 8 | Re-identification of patients from foundation-model features (npj Digit Med 2025, Nebbia et al.) | doi 10.1038/s41746-025-01801-0 (opened via OHSU Pure record) | Frozen FM features re-identify patients: 40.3% fundus, 46.3% OCT, 25.9% chest X-ray; task-trained model 82.3 / 93.9 / 63.7%. Internal data; no mitigation. | AUD |
| 9 | Keyed Nonlinear Transform (May 2026) | arXiv 2605.14123 | Key-conditioned obfuscation of intermediate features for split inference; ReID AUC 0.635 -> 0.586 (weak), classification within 1.0 pp, skin-lesion segmentation -4.4 Dice. Attacker adaptivity not stated. | TF/learned-free |
| 10 | Pitfalls of defacing whole-head MRI (Gao et al., CBM 2025) | arXiv 2501.18834 | Cascaded diffusion models rebuild faces from defaced MRI (train 180, test 484 incl. 469 from another dataset); recovered faces closer to the original than a population-average face (p<0.05); defacing also destroys facial-voxel signal (shin-muscle radiodensity correlation lost). | GEN attacker |
| 11 | Towards Robust Foundation Models for Digital Pathology / PathoROB (Jul 2025) | arXiv 2507.17845; github.com/bene-knz/PathoROB | 20 models, 4 datasets, 28 classes, 34 centres, 99,392 patches; metrics: Robustness Index (kNN), APD (linear probe under increasing centre-class correlation), clustering score. Deterministic, scripted, HF-hosted. | AUD + TF/ADV baselines |
| 12 | Current Pathology FMs are unrobust to Medical Center Differences (Jan 2025) | arXiv 2501.18055 | All 10 models encode centre strongly; embedding space organised more by centre than biology. | AUD |
| 13 | Robustifying pathology foundation models via fine-tuning (Jul 2026) | arXiv 2607.22861 | Fine-tuning recipe on 10 FMs: PathoROB RI 0.72 -> 0.87 average, no downstream trade-off; weights released on HF (Phaet, Mascaret). Recipe not described on the page. | Learned fine-tune |
| 14 | Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark (Feb 2026, Muller and Berens) | arXiv 2602.18502; github.com/berenslab/medical-shortcut-mitigation | 10 configs (ERM, rebalance, adversarial, dCor, MINE, MMD) on Morpho-MNIST, CheXpert (sex confound), OCT (synthetic notch filter); Original/Balanced/Inverted tests; dCor+Rebal and MINE best, dCor 100x cheaper than MINE; suppressing one cue can push the model to another. | MI / ADV / DIS |
| 15 | MIMM-X (Nov 2025) | arXiv 2511.22990; github.com/lab-midas/MIMM-X | Multi-factor MI minimisation, brain MRI age (NAKO/UKB) and CheXpert effusion; target subvector predicts sex/dataset at 51.8/50.0% vs ~70% baseline; validation-to-inverted drop is the shortcut metric. | MI + DIS |
| 16 | Mask of truth (Dec 2024, J Imaging Inform Med 2025) | arXiv 2412.04030 | Masking out the clinical ROI in PadChest and Chaksu fundus: every PadChest model stays above chance, i.e. classifiers rely on non-ROI regions. A simple deterministic shortcut probe. | AUD |
| 17 | Efficient Unsupervised Shortcut Detection and Mitigation in Transformers (Jan 2025) | arXiv 2501.00942 | Unsupervised detection+mitigation, gains in worst-group accuracy; datasets and numbers not on page. | Learned/unsup. |
| 18 | HACA3+ (Apr 2026, MIDL) | arXiv 2604.19474; github.com/shays15/haca3-plus | MRI harmonisation across 100+ scanners / 64 sites, T1/T2/PD/FLAIR, validated on a traveling subject across sites; weights and code public. | DIS |
| 19 | DIST-CLIP (Dec 2025) | arXiv 2512.07674 | Anatomy/contrast disentanglement guided by DICOM metadata via CLIP; code "upon publication". | DIS |
| 20 | CLOAK (Dec 2025; IMWUT 2026), with PrivDiffuser (arXiv 2412.14499, PETS 2025) | arXiv 2512.12086; github.com/sustainable-computing/CLOAK | Latent-diffusion sensor obfuscation: MotionSense activity 97.20% with gender privacy loss 0.02; MobiAct, WiFi-HAR, AudioMNIST too. Attackers are pre-trained on raw data (frozen protocol); 5 runs. PATN (arXiv 2511.07242, AAAI 2026) is the real-time perturbation variant; its datasets and attacker setup are not on the abstract page. | GEN / contrastive |

Also opened and used as context: CARE-PD (NeurIPS 2025, arXiv 2510.04312); Privacy-Preserving OR Workflow with Digital Twins (MIDL 2025, arXiv 2504.12552: segmentation+depth twins "on par with" RGB on 38 simulated trials, 5 events, internal data); Privacy-Preserving Action Recognition review (arXiv 2608.04501: 32 papers, under 10% use adaptive attackers, 40% report inconsistently defined cMAP); WristPrint (CCS 2021, PMC9988376, old: 353 users, ~96% re-identification risk from one day of wrist data; used only for scale); ON-Harmony (Sci Data 2025, PMC11992253).

Not opened, so not relied on: PP-VAE (medRxiv), Nature Medicine 2024 "limits of fair medical imaging AI", Lancet eClinicalMedicine 2024 MRI re-identification (403), Gao et al. full text, SSRN head-CT de-identification paper.

## 2. Candidate fields

Scores 1-5, 5 is best (for compute 5 = easy in days on 4x A6000). Order is my ranking by total.

### A. Pathology foundation-model embeddings: remove centre signature, keep tissue/cancer (total 26)

- Task kept: tissue/cancer type. Removed: medical centre (stain, scanner, lab procedure). Latent: frozen FM embedding -> learned mask/projection, or a learned head over layer-wise features.
- Benchmark: **PathoROB**, existing (RI, APD, clustering score; scripted, deterministic). Data: 99,392 patches, 28 classes, 34 centres (Camelyon CC0; TCGA CC-BY-NC-SA 4.0; Tolkach ESCA CC-BY-SA 4.0), about 2 GB of images on HF `bifold-pathomics/pathorob`; code BSD-3. Leaderboard RI: Atlas 2 0.928, RudolfV 2-S 0.917. Large FMs (UNI2-h, Virchow2) are gated on HF; I did not verify access terms.
- Scores: mask/latent value 5, data+licence 4 (TCGA is non-commercial), deterministic benchmark 5, compute 5, fit for learned 4, novelty 3.
- Case: the centre signature is a stated blocker for clinical use (all 20 models show it) and the test is exactly "keep label, remove nuisance on a frozen z". The headroom is the regime where **centre is correlated with class**, where ComBat deletes biology and DANN barely helps, which is where our "retrained probe + task-label-only baseline" normalisation adds something PathoROB lacks (RI is a kNN statistic, not a retrained centre classifier).
- Biggest risk: crowded and near ceiling (RI 0.89-0.93 for top models with Reinhard+ComBat up to 0.92; Filiot et al. report 0.87 average from fine-tuning), so a learned mask may win by 1-2 RI points that nobody cares about.
- Kill test (hours): learn a linear/sparse mask on frozen UNI/Phikon embeddings with a retrained centre probe; if it cannot beat ComBat+Reinhard on RI **and** cut the retrained-probe centre accuracy toward chance on the TCGA 2x2 correlated splits, drop it.

### B. EEG foundation-model embeddings: remove subject identity (and recording site), keep task (total 26)

- Task kept: motor imagery / cognitive state / sleep stage / seizure. Removed: subject identity (cross-session linkage), then site. Latent: frozen LaBraM/CBraMod/REVE/BIOT embeddings or covariance features.
- Benchmark: close to existing. FMScope (github.com/Jimmy110101013/fmscope, five diagnostics) and the two 2026 audits give protocols but no leaderboard. We would build a fixed splits file: subject-disjoint task evaluation, cross-session identity linkage among enrolled subjects with **retrained** attackers, chance and label-only baselines, bootstrap over subjects. Data: PhysioNet EEGMMI, 109 subjects, 64 ch, 160 Hz, 14 runs each (about 1,526 recordings, inferred), ODC-By 1.0, 3.4 GB; BCI IV-2a and OpenBMI (sizes not verified); Sleep-EDF, CHB-MIT used by the audits.
- Scores: value 4, data+licence 5, benchmark 3, compute 4 (small data; encoder weight licences unverified), fit for learned 5, novelty 5.
- Case: a young field (all papers June-Oct 2026) with two competing recipes, a trained adversary (NeuroPriv, subject ID 0.692 -> 0.206 at -0.7 point task) and a training-free geometry transform (Riemannian, 90.8 -> 15.6), and nobody compares them under the same retrained attackers. It directly tests our "where the secret lives" finding (aperiodic component vs covariance vs learned subspace), and erasure improves decoding, so utility is on our side.
- Biggest risk: identity is only meaningful with enrolled candidates and few subjects (9 in BCI IV-2a), so attackers saturate or CIs are huge, and nearly all supporting papers are single-group unreviewed preprints that someone may scoop before we finish.
- Kill test: reproduce the 90.8 -> 15.6 Riemannian number and NeuroPriv's 0.692 -> 0.206 with our retrained-attacker suite on EEGMMI; if both collapse to near-baseline leakage under retraining, the field is "evaluation correction only" and weaker than it looks.

### C. Clinical gait from pose/SMPL (Parkinson's): remove identity and cohort, keep UPDRS-gait / freezing of gait (total 24)

- Task kept: UPDRS-gait (0-3), FoG, medication state, PD vs healthy. Removed: subject identity (shape, style) and cohort/modality (RGB vs MoCap). Directly reuses our stick-figure/canonical-skeleton pipeline.
- Benchmark: **CARE-PD**, existing (NeurIPS 2025): 362 subjects, 8,477 walks (18.66 h), 9 cohorts; protocols within-dataset (LOSO), cross-dataset, leave-one-dataset-out, multi-dataset adaptation; macro-F1. Only SMPL pose is released, shape parameters withheld "because they could be identity-revealing", but **no re-identification test is reported on the page I read**. Release is non-commercial research use via the project site; the data-access section was truncated, so the DUA terms are unverified.
- Scores: value 4, data+licence 3 (NC, access unverified, only about 35% of walks carry UPDRS-gait, score 3 rare), benchmark 4, compute 5, fit for learned 4, novelty 4.
- Case: the dataset's anonymisation claim is untested, and its cross-cohort generalisation is poor (LODO F1 about 0.50 on BMClab/PD-GaM, at or below 0.18 on 3DGait), so a mask that removes cohort and identity but keeps severity serves both privacy and generalisation. We already know plain stick figures keep 53% KTH identity while a canonical redraw drops it, so the question is whether severity survives canonicalisation.
- Biggest risk: severity and gait style are the same signal, so there may be no operating point where UPDRS survives and identity is hidden; per-cohort subject counts (14-97) give wide CIs.
- Kill test: probe walk-level identity (multiple walks per subject, about 23 on average) from the released SMPL pose with a retrained classifier; if identity is already near chance, there is nothing to remove.

### D. ECG: remove identity, sex, age and site; keep rhythm/diagnosis (total 24)

- Task kept: arrhythmia / diagnostic superclass. Removed: identity, sex, age, recording site/device. Latent: waveform-level anonymiser (REAN) or an ECG-FM embedding.
- Benchmark: to build, from existing parts. **PTB-XL** (PhysioNet): 21,799 10 s 12-lead ECGs, 18,869 patients, age/sex/site/device/patient_id metadata, CC BY 4.0, 3.0 GB, patient-grouped 10 folds. REAN's four DBs: MIT-BIH Arrhythmia (47 subjects), MIT-BIH Long-Term (7), INCART (32), SHDB-AF (100). I did not open MIMIC-IV-ECG.
- Scores: value 4, data+licence 5, benchmark 3, compute 5, fit for learned 4, novelty 3.
- Case: the cleanest wall-to-wall evidence that frozen-attacker evaluation overstates privacy (0.000 vs 0.886) sits in the most active sub-field here. A hand-built "rhythm-only" or canonical-beat representation is the natural ECG analogue of our stick figure and would test learned vs hand-built directly.
- Biggest risk: PTB-XL has 1.16 records per patient (21,799 / 18,869), so closed-set identity linkage inside it is thin; a real identity task needs long single-subject databases with only 7-100 subjects, and the clinical motive for hiding ECG identity is weaker than for demographic/site shortcuts.
- Kill test: train a rhythm-only (RR-interval) baseline on the REAN splits; if it already matches REAN on arrhythmia and kills identity under retrained attackers, the learned story is dead.

### E. Wearable IMU / sensor HAR: remove identity and gender, keep activity (total 23)

- Benchmark: existing but fragmented. MotionSense (24 subjects, 6 activities x 15 trials, 50 Hz, MIT licence, gender/age/height/weight attributes), MobiAct (36 participants), WiFi-HAR (10), as used by CLOAK/PrivDiffuser; WISDM (51 participants, 18 activities) and mORAL (25) from WristPrint. We would unify: trial-disjoint splits (MotionSense: first 9 trials train, last 6 test), **retrained** identity and attribute attackers, hand-built baselines (downsample, low-pass, gait-cycle canonicalisation).
- Scores: value 4, data+licence 4, benchmark 3, compute 5, fit for learned 4, novelty 3.
- Case: this is the field that invented "keep the task, hide the attribute" (2018 onward) and 2025-26 papers (CLOAK, PrivDiffuser, PATN) all still report privacy as deviation from chance of attackers trained on raw data. An evaluation-correction paper with hand-built controls is cheap and mirrors our headline finding.
- Biggest risk: 24-36 subjects and 4-6 activities saturate (utility 97%+ for everyone), so utility will not discriminate methods and identity classes are trivially small; reviewers will call it incremental.
- Kill test: take CLOAK's released code and re-attack its output with a retrained attacker (CNN + Transformer, 5 seeds, trial-disjoint); if privacy loss stays near 0, the field is solved for these datasets and not worth our time.

### F. Head MRI/CT defacing and face reconstruction: NOT PURSUING (total 19)

- What it is: learned face-identity removal that keeps brain anatomy and, ideally, facial-region information (muscle radiodensity signal lost by defacing).
- Scores: value 5, data+licence 2, benchmark 2, compute 2, fit for learned 4, novelty 4.
- Why I would not pursue: (i) **Rule 2 conflict**: weights and datasets go only on HuggingFace and identifiable head MRI/CT cannot be redistributed; models trained on faces are also a risk. (ii) No open ground truth: face-photo/scan pairs are not public, so "re-identification" must be a surface-distance proxy against a population-average face, the metric Gao et al. used, which is a custom, not a standard, test. (iii) Compute: cascaded 3D diffusion on whole-head volumes is days to weeks, not days. (iv) The 2023 MIDI report (via search, not opened) found no known successful real-world facial-recognition attack, so the threat is mostly demonstrated by the authors' own attackers. (v) The CT-specific literature I found is a preprint at 85% -> 64% re-identification, not mature.

### Considered and folded in or dropped

- **MRI scanner/site harmonisation** (HACA3+, DIST-CLIP; ON-Harmony: 20 participants x 6 of 8 scanners, 3 vendors, 5 sites, 165 sessions, ~58 GB on OpenNeuro ds004712, article CC BY 4.0, dataset licence unchecked). Scores would be 4/4/4/3/4/2 = 21. Crowded image-to-image field, 3D compute, N=20, and the real test is downstream segmentation volumes, not a mask. Keep ON-Harmony as a fallback benchmark only.
- **Chest X-ray/MRI shortcut latents** (Muller and Berens, MIMM-X): methods are already well benchmarked (dCor+Rebal about as good as MINE, 100x cheaper); scores ~4/3/4/5/4/2 = 22. CheXpert is a research-use licence, shortcuts are injected by 95% subsampling, so it is a synthetic test. Use it as a **calibration benchmark** for whatever mask method we build for A, not as its own field.
- **Clinical video (OR/ICU)**: depth/segmentation twins exist (arXiv 2504.12552) but the data is internal and small (38 simulated trials); no ICU work found. Not benchmarkable by us.
- **sEMG gesture (NinaPro)**: subject-invariant methods exist (arXiv 2607.03783 snippet: 61.44% vs 41.30% on DB5) but only seen as a search snippet; privacy angle absent. Not pursued.
- **PPG/wearable physiology**: PRI-Attack and PhysioGait show identity in PPG-derived features, but I found no subject-invariant PPG representation paper; unverified lead.

## 3. Recommendation

1. **A (pathology) and B (EEG)** tie at 26: A has the best benchmark and compute, B the best novelty and strongest link to our "where the secret lives" result. Run both kill tests first; they cost hours.
2. **C (CARE-PD gait)** is third because it reuses our pose pipeline and has an existing benchmark with a missing identity test, but its licence terms and severity-vs-identity entanglement are the open questions.
3. **D (ECG)** is the fallback if B or C fail their kill tests; **E (wearable HAR)** is a cheap evaluation-correction side paper, not a focus field.
4. **Do not pursue F (defacing)**, and do not make MRI harmonisation or X-ray shortcuts their own fields.
5. The strongest contrarian argument against the whole cluster: medical datasets are small in subjects (7-109 for identity tasks), so identity is a weak, easily saturated nuisance. The nuisances with real scale are **centre/site/scanner** (A, MRI) and **cohort** (C). If lock-in needs a large-N nuisance, prefer A.
