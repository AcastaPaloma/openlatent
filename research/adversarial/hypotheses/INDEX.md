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

---

# Round 3 (generator round 3, 2026-11-05; 18 hypotheses, not ranked)

Written after both programmes' builders reported (SUMMARY.md, FOCUS outcome blocks, both AUDIT_findings.md, the 4-5 Nov journals, results_notes A / E / F / H / N / S / T). Round 3 is built on six constraints from the evidence: (i) the only learned-over-floor result that survived a same-supervision comparison is the supervised nonlinear adapter on JUMP once about 10% of production labels exist (F3: dU_H +0.027 to +0.032 against the full envelope at 10%, +0.135 at 100%, margin 0.010); (ii) every published privacy figure that was re-measured depended on how the attacker was trained (AAE 4% frozen vs 83% retrained; PriCheXy 57.7 published vs 67.0 replicated, and a linkage worst case of 0.887); (iii) cheap floors (label-supervised linear codes, 4x downsample, dither, a 2-bit predicted label) matched or beat the learned defences tested; (iv) adversaries mostly did not converge (memorisation floor about 0.67 at n = 1,600, d = 32; the T.md attacker-scaling ladder was never run); (v) the sealed-suite instrument v3 detects leak types it was built or retrained to see and misses unseen ones; (vi) power limited many comparisons, so every file prints its MDE. Each file has: statement, mechanism, exact experiment (dataset, models, floors, attacker suite, metric, splits by independent unit, seeds AND test-unit design, cross-fit), numeric prediction and falsifier, GPU-h, dependencies, prior with reasoning, cheapest floor that could kill it. No hypothesis has been run. No git command was used. Numbers quoted from earlier results are copied from the notes named in each file; anything marked UNVERIFIED or ESTIMATED must be checked before pre-registration.

## Relation to work already ordered on 5 Nov (do not duplicate)

- F4 (F.md section 10: leave-source-out on the 5-source universe, "objects" view, 1/3/10/30/100% grid, <= 3 GPU-h) is the base for H3-001 (which uses the 10-source table F4 explicitly did not run), H3-002, H3-004, H3-005, H3-006, H3-007.
- A3 (adversary on half the PathoROB slides, confusion on the other half, floor calibrated on held-out slides) is the small-n counterpart of H3-006, which restates the design where n is large enough to control the memorisation floor.
- T3 (T.md section 8: full 37-floor pool, second seed at mu 0.01, mu 0.005, checkpoints C1 / C2, rotations 0-1 of the t3 salt = "roles re-drawn over the same patients", D4 control tolerance) is the base for H3-008 to H3-013. The CXR files below start where T3 stops: floors outside the frozen list, fresh patients (S-role bucket F), larger mu, attacker scaling, the G clause, learned compression, the label-persistence confound.
- H (WISDM replication and the sealed retrained wearable defender, ordered <= 4 GPU-h) is pre-registered in H3-014 / H3-015 (5.0 GPU-h together; trim by dropping OBF-A seeds if the cap is binding).

## Table (alphabetical by file; dependencies are listed, no priority is implied)

| File | Short claim | Kind | GPU-h | Prior | Depends on |
|---|---|---|---:|---:|---|
| H3-001-label-budget-unseen-sources-ten-source-table | F3 break-even <= 30% in >= 7 of 9 held-out sources and across a microscope-model block on the 10-source JUMP table | generalisation | 6 / 11 | 0.55 | F4 code; 10-source table download |
| H3-002-label-space-size-sets-the-break-even-and-privacy-label-budgets | dU over same-label LDA rises with log10(classes) at 0.025-0.07 per decade; wearable activity labels (K = 4-6) never break even | mechanism / new idea (label budgets for privacy defenders) | 2.5 | 0.60 | F3 arms; H development folds |
| H3-003-hlca-second-dataset-label-budget | on HLCA (E's dataset-grouped folds, K about 60) the adapter shows no break-even, testing the K-rule outside imaging | second-dataset replication, predicted negative | 3 | 0.25 (win) | E harness and cache |
| H3-004-test-time-source-recentring-stacked-on-adapter-and-floors | per-source test-time recentring of the low-d output lowers leakage >= 0.10 at |dU| <= 0.01 for adapter and floors alike | NEW idea (transductive calibration) | 2 | 0.45 | F3 adapters |
| H3-005-cross-source-mutual-nn-pseudo-positives-from-few-labels | MNN pseudo-positives from the b1 or Target2-only adapter recover >= half the gap to b10 | NEW idea (semi-supervised) | 3 | 0.20 | F3 b1 / a adapters |
| H3-006-held-out-unit-adversary-that-converges-jump | a convergent held-out-unit adversary adds < 0.010 over output dither at matched leakage; criterion C is attainable at n = 130k | instrument gate + learned test | 4.5 | 0.20 (win) / 0.60 (C attainable) | F3 code; A's memorisation calibration |
| H3-007-boring-control-jump-well-position-near-duplicate-shortcuts | the F3 gain survives position-matched negatives, near-duplicate pruning and source-level negatives | boring control | 1.5 | 0.75 (survives) | compound metadata |
| H3-008-cxr-composite-floors-on-fresh-units | composite and fine-grid floors on a fresh S-role bucket move the envelope >= 0.02; no image floor <= 0.75 at raw utility | floors vs learned, fresh units | 6 | 0.40 | T3; custodian salt after list v2 |
| H3-009-cxr-high-mu-and-same-spectrum-random-warp | pcxr at mu 0.02 / 0.04, last epoch, stays on the floor frontier and no better than a same-spectrum random warp | learned defender dose-response | 5 / 9 | 0.12 | T3 frontier and base |
| H3-010-cxr-attacker-scaling-ladder-convergence | tripling the attacker training set raises down_f4 linkage >= 0.03; the defender-versus-floor ordering survives | attacker convergence | 4.5 | 0.50 | T ladder never run |
| H3-011-cxr-population-defender-learned-residual-on-floor-held-out-view | a population-trained learned residual on down_f4 does not generalise to a held-out attacker view better than a single attacker, and not beyond down_f4 + matched noise | G clause, floors as the bar | 12 / 18 | 0.08 | H3-008, H3-010 |
| H3-012-cxr-learned-compression-rate-as-the-knob | linkage is one function of bits for classical and generic learned codecs; a task-aware learned code beats the log-bits interpolation to labels14 by >= 0.03 | NEW idea (learned compression) | 10 | 0.25 | H3-008 |
| H3-013-boring-control-cxr-label-persistence-no-finding-stratum | raw linkage survives in No-Finding pairs and with label-matched negatives; labels14 floor falls to ~0.55 there | boring control | 1.5 | 0.80 (survives) | T2 / T3 score files |
| H3-014-imu-sealed-retrained-defender | retrained AAE-R and an in-loop-adversary obfuscator are dominated by the activity-label floors on the salted sealed folds | sealed learned defender | 3.5 | 0.08 (learned win) | H salt and floors |
| H3-015-wisdm-replication | the floors-beat-defenders direction and the frozen-vs-retrained gap replicate on WISDM (51 subjects) | replication | 1.5 | 0.70 | H3-014 code; phone-model check |
| H3-016-boring-control-dithered-floors-and-evidence-budget | account-level pooling at least doubles the leakage of dithered floors (act60_r1, lda3_r1); predlabel stays flat | boring control | 0.5 (+1.5 CPU) | 0.50 | H generators, S evidence module |
| H3-017-boring-control-imu-adjacency-and-leave-activity-out | raw identity falls from >= 0.95 (interleaved) to <= 0.60 under leave-activity-out; floors barely move | boring control | 1 | 0.70 | H suite |
| H3-018-stop-for-real | four-leg decision table; stop only if the positive does not generalise, floors dominate defenders, new ideas are exhausted, power is adequate, and the four boring controls have run | stop hypothesis | 0 | 0.25 (full stop) | all |

## Requirement coverage

- Generalisation of the label-budget result: second dataset / encoder family H3-003 (and the "objects" view in F4); other sources / leave-source-out H3-001 (with F4); mechanism for where it should and should not hold H3-002.
- Adversary scored on held-out units that converges, memorisation floor controlled: H3-006 (criterion gate with noise-latent and planted-signal calibration, JUMP), H3-010 (CXR attacker convergence), H3-011 (convergence check in the loop).
- Chest X-ray: further floors H3-008 (blur, quantisation, JPEG, PCA, noise, composites, ideal low-pass; T3 covers the frozen 37), second operating point and seed H3-009 (T3 covers mu 0.005 / 0.01 and two seeds; this adds 0.02 / 0.04, C2 and a same-spectrum random control), population-trained defender with a held-out attacker family and dither / downsample as the bar H3-011.
- Wearable: sealed retrained defender H3-014, WISDM replication H3-015.
- NEW ideas (>= 3): transductive per-source calibration as a defence H3-004; learned compression with rate as the knob H3-012; label-budget curves for privacy defenders H3-002 (part B); cross-source MNN pseudo-positives H3-005; a learned residual on top of the strongest floor H3-011.
- Boring controls that could kill an exciting result (4+): H3-007 (JUMP shortcuts), H3-013 (CXR label persistence), H3-016 (dither does not compose with evidence), H3-017 (same-session adjacency / leave-activity-out).
- Stop: H3-018.

## Dependency structure (for scheduling, not a ranking)

- Start immediately, independent of in-flight work: H3-007, H3-013 (arm A, zero GPU), H3-016, H3-017, H3-002, H3-004, H3-006 (stage 0 then stage 1), H3-005, H3-003, H3-010 (T ladder rungs on T3 units).
- After F4 has produced one real aggregate: H3-001. After T3 has produced its nomination and frontier: H3-009; H3-008 needs the list-v2 hash and a custodian salt (floors first, salt second); then H3-011 and H3-012 (both use the fresh bucket F). H3-015 after H3-014's code runs on MotionSense. H3-018 last.
- Cheap-gate subset (H3-007, H3-013, H3-016, H3-017, H3-002, H3-004, H3-010): about 14 GPU-h; the four boring controls alone are about 4.5 GPU-h and are a precondition of H3-018.

## GPU totals (stated, not additive with certainty because fits are shared)

About 68 GPU-h in the reduced tiers, 83 in the full tiers (6 / 11, 5 / 9, 12 / 18 for H3-001, -009, -011). On the 4 idle A6000s that is roughly 17-21 hours of wall-clock if packed; CPU-bound pieces (attackers, metadata, tabular IMU suites) are extra and capped at 8 threads per job.

## Things to verify before any pre-registration (carried into the files)

- JUMP: whether the same compound occupies the same well position across sources on production plates (H3-007 stage 0); whether the 10-source assembled table carries Target2 plates and what its preprocessing already fitted on (H3-001); `compound.csv.gz` InChIKey / SMILES availability; the realised per-fold MDE80.
- WISDM biometrics: subject count, activities, duration, sampling rate, phone models, licence terms (H3-015 cites the dataset card from memory); CompressAI code and pretrained-weight licences, and single-cell foundation model weight licences (H3-012, H3-003); both are optional arms that are dropped if not cleared.
- ChestX-ray14: the S-role bucket F is fresh only with respect to the evaluation attackers and floors (the PriCheXy verifier, T2 base weights and R0 retrains saw S pairs); the utility classifier for any F-based claim must be refit on S \ F.
- The AAE training loss is not public (weights and evaluation notebooks only); AAE-R is a re-implementation in spirit.
- All thresholds above are written before any Round-3 number exists; they may be tightened after data with both verdicts reported, never loosened.
