# Hypotheses, programme 2 (generator round 1, 2026-11-04; not ranked)

22 falsifiable hypotheses against the locked FOCUS.md: instrument S (5), R-CXR (7), R-IMU (6), conditional G (2), cross-focus X (2). Written from CHARTER.md, FOCUS.md, SYNTHESIS.md, the contrarian files (`contrarian_missed.md`, `contrarian_stress_test.md`), the six scouts, and programme 1's lessons (floors beat learned adapters; the adversary adds nothing beyond a bottleneck or noise code; seeds share test units unless folds rotate; label-only LDA is a strong floor). Each file has: statement, mechanism, exact experiment (data, defence, attacker suite, metric, normalisation, units, seeds AND test-unit design, cross-fit), numeric prediction and falsifier, GPU-h, dependencies, prior with reasoning, cheapest killing floor. No hypothesis has been run. No git command was used.

Evidence labels used inside the files: MEASURED (stress test, on real artefacts), ANALYTIC (formula), SIMULATED, ESTIMATED, scout-read (abstract-level literature, to be verified against the primary source before any write-up).

## Shared definitions (referenced by the files)

- Normalisation. Per attacker k, L_k = (acc_k(release) - chance) / (acc_k(raw, retrained) - chance); L_worst = max over the suite, selected on validation units, scored once on the sealed units; the naive max over test is shown beside it. For AUC: (AUC - 0.5) / (AUC_raw - 0.5). Chance: MotionSense identity 1/24 = 4.17%; CXR AUC 0.5; Rank-1 1/(gallery size).
- SUITE-IMU-8 (retrained on every release; trial score = mean log-prob over the trial's windows; hand features = per channel std, MAD, percentiles 10/50/90, 16 FFT bins): LR; LDA; kNN (cosine and Canberra); random forest (500); LightGBM (300); MLP (2x256); 1D-CNN and GRU on raw 128x12 windows. Families for the G gate: F1 linear/MLP, F2 distance, F3 trees, F4 sequence/deep conv.
- SUITE-CXR-8: F1 Siamese-ResNet50 (PriCheXy's attacker); F2 ConvNeXt-T verifier; F3 ViT-S verifier; F4 frozen DINOv2-L kNN (cosine / L2 / Canberra) + InfoNCE head (RAD-DINO optional; its pretraining set includes NIH-CXR, a disclosed exposure); F5 registered NCC/SSIM (non-learned, HELD-OUT); F6 demographic + device GBM with image-estimated age/sex (cross-fitted); F7 defence-agnostic pretrained Siamese (random-obfuscation mixture); F8 acquisition-fingerprint attacker.
- MN-v1 (matched negatives, CXR): for each positive pair (patient p, two visits) one negative pair with the same sex, same view, identical (OriginalImageWidth, Height, PixelSpacing), and an age difference matched to the positive's age gap within +-1 y by stratified sampling. Validity control before any image number: a GBM on TRUE age / sex / device reaches AUC <= 0.56. PriCheXy's own random-negative protocol is always reported beside it.
- Test-unit rule (every file restates it). IMU: unit = trial, 3 rank folds (fold r tests the rank-r trial of each subject x activity, all 360 trials tested once; fold 1 is the 144-trial "last trial" analogue); seeds = inits inside folds, so seeds replicate training noise and folds replicate test units; bootstrap over trials, subject clusters (24) reported beside; identity is same-session. CXR: unit = patient; floors and latent adapters rotate over 5 patient-hash folds of the 13,302 multi-visit patients; the released PriCheXy-Net weights were trained on its own train patients, so that arm has fixed test units (its 1,424 test patients), seeds replicate training noise only, and this is stated every time (optional second PriCheXy retrain, ~8 GPU-h, for an independent test population).
- MDE reference (stress test): IMU paired MDE80 ~7.5 points at n = 144, ~4.8 at n = 360 (ANALYTIC, rho 0.5, ~12% leak); CXR paired MDE80 ~0.02 AUC at 1,424 patients, ~0.011 at 4,707 (Hanley-McNeil, rho 0.6); certificate floor at zero leak: chance +3.7 to +7.1 points at 120-150 units, +1.0 to +1.8 at 1,424 patients (K = 1..30).
- Data gates: ChestX-ray14 download (45 GB HF mirror) held until Auditor-2 clears terms and the main agent says so; MotionSense licence check by Builder H first; CT-RATE, NLST/TCIA, Capture-24, BOXRR-23 excluded from identity work. PriCheXy-Net code is GPL-3 (separate process).

## Table (alphabetical by file; dependencies are listed, no priority is implied)

| File | Short claim | Kind | GPU-h | Prior | Depends on |
|---|---|---|---:|---:|---|
| H-S-001-planted-leak-dose-response | zero-leak null holds; MDD at oracle acc 18-30% (IMU, n=144) and AUC 0.55-0.58 (CXR) | calibration | 4 | 0.70 | Builder S schema; CXR arm: download gate |
| H-S-002-selection-bias-of-the-maximum-and-search-loop-null | max over K=10 inflated 2-4 pt at n=144; sealed split removes >=80%; cross-fit beats more attackers; shuffled search-loop null | premise | 1 | 0.75 | IMU-002, CXR-004 predictions |
| H-S-003-evidence-budget-independent-windows-or-session-offset | pooling within a recording ~flat, across recordings decisive; N* vs ln C / I | measurement | 0 | 0.50 | HF `preds.npz` (78 files), MotionSense |
| H-S-004-suite-saturation-and-view-vs-architecture | <=3 members saturate; view > architecture; LOFO coverage | coverage | 4 inc. | 0.55 | IMU-002, CXR-003/-004 fits |
| H-S-005-one-line-statistic-matches-the-suite | one line matches the suite on high-leak, not low-leak releases; metadata beats image attackers on random negatives | boring control | 0.5 | 0.60 | IMU-002; CXR-001 |
| H-CXR-001-demographics-and-device-explain-published-linkage-auc | real age/sex estimators + device give AUC >=0.80 on PriCheXy-defended random negatives, <=0.56 matched | premise/protocol | 4 | 0.65 | download gate |
| H-CXR-002-identity-beyond-demographics-on-matched-negatives | raw Siamese beats best demographic line by >=0.03 on MN-v1 (stage-0 gate) | headroom/gate | 3 | 0.85 | CXR-001, download |
| H-CXR-003-floors-match-pricheXy-net | equal-strength random flow / blur / downsample / PCA within 0.03 AUC of PriCheXy-Net at matched utility | floors vs learned | 20 | 0.55 | CXR-001/-002, PriCheXy weights |
| H-CXR-004-held-out-attacker-family-moves-the-defence-verdict | suite worst case >= +0.08 AUC over PriCheXy's own attacker; ranking flips; pretrained agnostic attacker works | attacker re-audit | 12 | 0.75 | CXR-001/-003 |
| H-CXR-005-boring-control-acquisition-fingerprint-attacker | anatomy-free fingerprint AUC >=0.65, untouched by PriCheXy, ~chance when acquisition-matched | boring control | 1.5 | 0.55 | download, CXR-003 |
| H-CXR-006-pooled-visit-attacker-restores-linkage | pooling P=3 spread visits lifts +0.10; adjacent visits <=+0.05 | learned attacker | 4 | 0.55 | CXR-002/-003 fits |
| H-CXR-007-boring-control-visit-gap-and-near-duplicates | defended linkage concentrated in adjacent visits; raw flat | boring control | 0.5 | 0.40 | CXR-002/-003/-004 scores |
| H-IMU-001-activity-lda-floor-vs-privdiffuser | PrivDiffuser sits >=15 pt above the activity-LDA-k + dither identity frontier; identity-guided retrain does not beat it | floors vs learned | 6 (+8 arm B) | 0.70 / 0.70 | MotionSense, PrivDiffuser LFS |
| H-IMU-002-retrained-suite-vs-raw-trained-attacker | retrained suite reads >=20 pt more than the raw-trained CNN; worst family varies | attacker re-audit | 3 | 0.70 | IMU-001 releases |
| H-IMU-003-boring-control-window-split-memorisation | window split inflates gender (>=20 pt) and identity (>=10 pt) | boring control | 0.5 | 0.75 | MotionSense |
| H-IMU-004-quantiser-alone-share | quantiser share <=0.35 vs trees, >=0.6 vs linear | attribution | 4 | 0.50 | IMU-002 suite |
| H-IMU-005-learned-obfuscator-beyond-lda-held-out-utility | learned obfuscator beats floor envelope on held-out-activity / cadence utility at matched identity | learned method | 7 | 0.12 | IMU-001, IMU-003, X-001 |
| H-IMU-006-boring-control-static-windows-posture-signature | static windows identify subjects (>=30%) via orientation; de-meaned <=12% | boring control | 0 | 0.70 | MotionSense |
| H-G-001-lofo-gap-vs-mde-imu-gate | LOFO gap exists (>=10 pt) but population defender ~ single-strongest-with-restarts ~ dither | gate | 3 | 0.12 open | IMU-001, S-002 |
| H-G-002-lofo-gap-vs-mde-cxr-gate | same on CXR latents at MDE ~0.02 AUC with held-out-finding utility | gate | 3 | 0.15 open | CXR-001/-002/-004, download |
| H-X-001-boring-control-dither-rate-envelope | adversary-free dithered code matches every learned defence at matched utility | boring control | 4 inc. | 0.70 | IMU-005, G-002 encoders |
| H-X-002-stop-programme-2-rule | pre-registered decision table; stop if no learned line and no instrument justification | stop rule | 0 | 0.75 / 0.25 | all gates above |

## Dependency structure (for scheduling, not a ranking)

- No data download, CPU only: S-003 (HF `preds.npz`), S-005 IMU half, IMU-003, IMU-006, CXR metadata half of S-005 (CSV already fetched in the stress test).
- Needs only MotionSense and the PrivDiffuser checkpoint: IMU-001 arm A, IMU-002, S-001 IMU arm, S-002 IMU half, then IMU-004, IMU-005, G-001 (conditional on IMU-001), X-001.
- Needs the ChestX-ray14 download (gate held): CXR-001, -002 (stage 0), then -003, -004, -005, -006, -007, S-001 CXR arm, S-002 CXR half, G-002.
- Chain inside R-CXR: CXR-001 -> CXR-002 (stop R-CXR if it fails) -> CXR-003 / -004 / -005 -> CXR-006 / -007 -> G-002.
- Chain inside R-IMU: IMU-003 and IMU-006 (protocol controls) -> IMU-001 -> IMU-002 / -004 -> IMU-005 (only if arm B or the frontier leaves headroom) -> G-001.
- Cheap-gate subset (S-003, S-005, S-001, S-002, IMU-001A, IMU-002, IMU-003, IMU-006, CXR-001, CXR-002, CXR-005): ~25 GPU-h; the rest is conditional on those.

## GPU totals (stated, not additive with certainty because fits are shared)

S 9.5 (4 of it incremental), CXR 45 stated / ~35 after sharing fits (FOCUS envelope 35, range 25-50), IMU 20.5 (+8 for PrivDiffuser arm B; envelope 8-12 holds for IMU-001A, -002, -003, -006 only; IMU-004/-005 are stage-2), G 6 (the FOCUS gate), X 4 incremental. Programme total about 70-85 GPU-h if everything is run; the stop rule in H-X-002 is built to prevent that.

## Requirement coverage

(a) S instrument: S-001 planted-leak dose-response; S-002 selection bias of the maximum (also the shuffled-label null of the search loop and cross-fit vs more attackers); S-003 evidence-budget N* with the session-offset vs independent-windows question; S-004 saturation and view vs architecture; S-005 one-line statistic.
(b) R-CXR: CXR-001 matched negatives + demographic attacker explaining the published AUC; CXR-003 random warp / blur / downsample / PCA / JPEG floors vs PriCheXy-Net; CXR-004 held-out attacker family; CXR-005 device / acquisition confound; plus CXR-002 (stage 0), -006, -007.
(c) R-IMU: IMU-001 activity-LDA code floor vs PrivDiffuser; IMU-002 retrained suite incl. RF / LightGBM / sequence model; IMU-003 subject-level vs window-level split; IMU-004 quantiser-alone share; plus IMU-005, -006.
(d) G gate: G-001 and G-002, each against a single strongest attacker with restarts and a dither / LDA envelope, with the MDE stated.
(e) Boring controls that can kill an exciting result: S-005, CXR-005, CXR-007, IMU-003, IMU-006, X-001 (six). Stop hypothesis: X-002.
Learned methods and attackers: PrivDiffuser identity-guided retrain (IMU-001 B), AE+quantiser and adversarial obfuscators (IMU-004, -005), population-trained defenders (G-001, G-002), retrained Siamese / ConvNeXt / ViT / DINOv2-head / pretrained-agnostic attackers (CXR-004), learned set aggregator (CXR-006), 1D-CNN / GRU attackers (IMU-002).

## Things to verify before any write-up (carried into the files)

- LAPIS (2610.40722) numbers and the C-AAE "ADPCM-only 32/41/83%" figures are scout-read or unverified; C-AAE has no code, so IMU-004's learned arm is a re-implementation in spirit.
- PrivDiffuser's released MotionSense checkpoint was trained with a window-level split over all 24 users: its generated "test" windows are contaminated by training; IMU-001 arm B retrains with trial-disjoint folds.
- ChestX-ray14 has no acquisition dates: visit gap is an index gap. RAD-DINO pretraining includes NIH-CXR. MotionSense trial numbering for static activities and the absence of cross-session data are from the dataset description, to be checked against the files.
- All hypotheses were written before any programme-2 number exists; any threshold may only be tightened after data, with both verdicts reported.
