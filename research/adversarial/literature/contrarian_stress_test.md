# Contrarian stress test of S, R and G (programme 2, 2026-11-04)

Scope: try to break S (sealed attacker suite + certificate), R (floors-first re-audit of published defences) and G (defender trained on an attacker population, tested on a held-out attacker family) before GPU-days are spent.
Method: read CHARTER, SYNTHESIS, the HAR / CT / leakage / suites scouts, FOCUS.md, programme-1 power-audit notes (A.md) and SUMMARY.md; then probed the rented VM (ubuntu@192.222.52.237: HEAD / GET tests, small downloads, CPU only, no GPU, ~/queue untouched, total download about 1.4 GB: HF snapshot 38 MB, UCI-HAR 61 MB, MotionSense 74 MB, ChestX-ray14 CSV and lists 11 MB, HHAR 822 MB and WISDM 310 MB streamed and discarded). Everything marked MEASURED below was computed in this session; scripts are in `research/adversarial/src/stress_test/` (copies of `~/work/stress_probe/scripts/` on the VM). Everything marked SIMULATED or ASSUMED says so. Nothing here used a GPU.

## 0. Bottom line

| Candidate | Verdict | One-line reason |
|---|---|---|
| S as a focus (suite + certificate) | **NO-GO as a standalone focus; GO as a 2-day, zero-GPU instrument inside R** | MEASURED on programme-1 harness predictions: KTH has 150 independent test recordings, paired MDE80 5-8 points vs programme-1 margins of 2-3 points; certificate at zero observed leak is chance + 3.7 pt (K=1) to + 7.1 pt (K=30). Only CXR (>= 1,424 patients) gives it resolving power. The "sealed suite / locked audit" framing was published for LLMs and tabular data in Aug-Oct 2026. |
| R-CXR (re-audit of chest X-ray linkage defences, with a changed protocol) | **GO-WITH-CHANGES** (best single choice) | Best power (1,424-4,707 patients), code and weights for PriCheXy-Net are retrievable, and a MEASURED floor shows the field's pair protocol is solvable from age + sex + device metadata alone (AUC 0.98 true, about 0.89 with image-estimated age and sex, SIMULATED). |
| R-IMU (re-audit of wearable IMU obfuscators, with a changed secret) | **GO-WITH-CHANGES** | Data are open and small, PrivDiffuser ships code and a checkpoint, and the cheap floors from the HAR scout turn out NOT to be floors for identity (MEASURED), so headroom is large; but only 24 subjects / 144 test trials, so only large claims are resolvable. |
| G (population-trained defender, held-out attacker family) | **NO-GO as a standalone start; conditional third focus behind a <= 6 GPU-h floors-first gate** | Novelty is real (no leave-attacker-family-out study for HAR or CXR found), but at n = 144-150 units its MDE80 (7-8 points) exceeds the effect programme 1 leads us to expect, and a published speech result shows held-out verifiers can be near-family. |

Recommended lock-in:
- **Exactly 2:** (1) R-CXR, (2) R-IMU (with the programme-1 video re-attack as an optional arm, not a focus). S is built into both as the reporting layer (about 2 engineer-days, 0 GPU).
- **Exactly 3:** the two above plus (3) G, conditional, GO-WITH-CHANGES, only after R shows a defender class with headroom over the floor envelope.
- S stays NO-GO as a standalone focus in both lists.

Things the main agent should correct in SYNTHESIS / scouts (all evidence below):
1. "Programme 1's result files stored only top-1": true for the 454 old `runs/*/result.json`; FALSE for the harness evaluations. HF `harness/<ds>/eval/<release>/preds.npz` holds per-clip class probabilities for retrained and frozen attackers, with `test_ids` (78 files).
2. C-AAE's "ADPCM-only supplies 32 / 41 / 83%" is NOT verified. The arXiv HTML has only a "Technical Validation of ADPCM" table with "(Before)/(After)" rows of undefined condition; and the user-ID split is "randomly split such that all users were included in both ... 4:1", not stated as by session. Drop the percentage until the PDF is checked.
3. PrivDiffuser's MotionSense "gender" attacker is trained and tested with all 24 users in both sets (`dataset_loader.py`: per-user groups over `range(24)` in train and test): it can memorise users.
4. The HAR scout's cheap IMU floors (per-window de-meaning, accelerometer magnitude, downsampling) do not remove identity (MEASURED below). The floors that do bite are label-supervised low-dimensional codes.
5. PriCheXy-Net's 5,000 test pairs involve only 1,307 distinct patients (492 patients behind the 2,500 positive pairs), so the effective unit count is about 500-1,300, not 5,000.

## 1. DATA: what the VM can actually fetch

### 1.1 ChestX-ray14 (the CXR testbed)
| Item | Result |
|---|---|
| Original NIH Box page `nihcc.app.box.com/v/ChestXray-NIHCC` | HTTP 404 from the VM |
| GCS bucket `gcs-public-data--healthcare-nih-chest-xray` | anonymous object GET 403, listing 401 |
| **Working route** | HF dataset `alkzar90/NIH-Chest-X-ray-dataset`, ungated: `data/images/images_001..012.zip` (2.0, 3.9, 3.9, 3.8, 3.9, 4.0, 4.0, 4.0, 4.1, 4.2, 4.2, 2.9 GB; repo total 45.1 GB), `data/Data_Entry_2017_v2020.csv` (9.0 MB, downloaded), `train_val_list.txt`, `test_list.txt`. Do not run the repo's loading script (arbitrary code); download the zips. VM has 881 GB free. |
| Licence | Card: "There are no restrictions on the use of the NIH chest x-ray images", attribution required (link NIH site, cite Wang et al. CVPR 2017, acknowledge NIH Clinical Center). HF sidebar says "unknown". Public data are not re-uploaded to HF (rule 2). |
| Patient IDs | CSV has Patient ID, Follow-up #, age, gender, view, width, height, pixel spacing. **No acquisition dates**: only the visit index, so time gaps cannot be computed. |
| Counts (MEASURED on the CSV) | 112,120 images, 30,805 patients, median 1 image per patient (mean 3.64, max 184). Patients with >= 2 images: **13,302** (43.2%); >= 3: 9,189; >= 5: 5,759; >= 10: 2,545. Views: PA 67,310, AP 44,810; PA-only multi-visit patients 10,688. |
| Official split | patient-disjoint: train_val 28,008 patients, test 2,797 patients (25,596 images); **1,424 test patients have >= 2 images**. |
| LAPIS split check | release 7,588 + test 4,707 = 12,295 patients, about 92% of all multi-image patients, so 4,707 is close to the largest test n obtainable. |

### 1.2 IMU HAR datasets with user IDs
| Dataset | Fetch result | Size | Licence | Users | Independent units / notes |
|---|---|---|---|---|---|
| MotionSense | GitHub raw `A_DeviceMotion_data.zip`, 200; downloaded and unzipped | 73.7 MB | MIT | **24** (360 trial files = 24 x 15) | 6 activities, 2-3 trials each; trials appear to be one session per subject (dataset description; not verifiable from files). Hold out the last trial per subject x activity = **144 test trials**. |
| UCI-HAR | UCI static zip, 200; downloaded | 61 MB | CC BY 4.0 (UCI page) | 30 (official split 21 train / 9 test, subject-disjoint) | Official split makes identity impossible; must re-split inside bouts. 7,352 + 2,947 windows, 281-409 per subject, 50% overlapping windows (near-duplicate leakage if split by window). |
| PAMAP2 | UCI id 231, 200 (not downloaded) | about 656 MB (scout) | CC BY 4.0 | 9 | one protocol session per subject |
| HHAR | UCI id 344, 200; streamed | 822 MB | CC BY 4.0 | 9 users, 8 phones + 4 watches | device-fingerprint confound |
| WISDM biometrics | UCI id 507, 200; streamed | 310 MB | CC BY 4.0 | 51 | 18 activities x 3 min, one session per subject |
| Capture-24 | not tested | 6.4 GB | CC BY, asks users not to identify individuals | 151 | licence conflict, skip |

No open IMU set has several independent sessions per subject; identity tests are same-session, time-blocked. State this in every claim.

### 1.3 HAR video
| Item | Result |
|---|---|
| KTH | `csc.kth.se/cvap/actions/*.zip` reachable (walking.zip 242 MB; about 1.1 GB for six), `00sequences.txt` 38 KB. 25 people. |
| HMDB51 | Brown host serves an HTML page (dead); HF `MichiganNLP/hmdb` `hmdb51_org.zip` 4.25 GB OK |
| UCF101 | `crcv.ucf.edu` no response; HF `quchenyuan/UCF101-ZIP` `UCF-101.zip` 6.96 GB OK |
| Charades | `ai2-public-datasets.s3...Charades_v1_480.zip` 16.3 GB OK |
| Identity labels | only KTH (25), Charades (about 90 homes) and UTD-MHAD (8) have any; UCF101 / HMDB51 have none. NTU RGB+D is gated and bars derived data. |
| Keypoint R-CNN weights | `download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`, 237 MB, 200 |
| On this VM now | `~/work/data` holds b_mw, cremad, d_embedpriv, e_singlecell, f_jump, jump_meta, pathorob; **no video**. Programme-1 ran on other hosts. |

### 1.4 Published defences: released and runnable?
| Defence | Code | Weights / data | Verdict |
|---|---|---|---|
| PrivDiffuser (PoPETs 2025) | `sustainable-computing/PrivDiffuser`, MIT, pushed 2025-08-05; artifact badges Available, Functional, Reproduced | Git LFS retrievable via `media.githubusercontent.com` (HEAD 200): `models/privdiffuser_motion_1/ckpt_80_checkpoint.pt` 853 MB, gender / activity classifiers, MotionSense and MobiAct arrays. No synthesised MotionSense set shipped: DDIM sampling must be run (authors: 1-2 days on a 2080 Ti for all three datasets; MotionSense cheapest). | Runnable. |
| PriCheXy-Net (MICCAI 2023) | `kaipackhaeuser/PriCheXy-Net`, GPL-3 (keep it as a separate repo or process), pushed 2023-10-05 | LFS retrievable (HEAD 200): `generator_lowest_total_loss_mu_{0.001,0.005,0.01}.pth` (31 MB), `pretrained_generator_prichexy_net.pth` (the pre-adversarial flow-field generator, a ready-made random-flow floor), `pretrained_verification_model.pth`, `pretrained_classifier.pth`. Pair lists: 10,000 train / 2,000 val / 5,000 test pairs and the official train/test lists. | Runnable once ChestX-ray14 is at 256 px. |
| C-AAE (2507.18072) | none on the arXiv page | none | Re-implementation only. |
| LAPIS (2610.40722, 29 Oct 2026) | none; paper states "no anonymized derivative images were distributed" | none | Do not re-implement as the first step. |
| CLOAK (2512.12086) | no GitHub link on the abstract page | not found | Unverified. |
| ObscureNet / replacement AE | `mmalekzadeh/replacement-autoencoder` MIT (2021) | n/a | Old, simple baseline. |

### 1.5 What survived of programme 1 (HF `KuanKuanKuan/openlatent-privacy`, model repo, 2,030 files, 14.35 GB)
- `runs/`: 458 runs (454 `result.json` with top-1 only, 454 `run.log`); 166 `anonymizer.pt` (1.8 KB to 14.3 MB: learned BDQ / pool parameters, so the learned releases can be regenerated). No retrained-attacker weights.
- `harness/`: 431 files, 14.17 GB. 128 `.pt` = **cached frozen attackers trained on raw frames** (name pattern `{secret}_{protocol}_{arch}_e{epochs}_s{seed}.pt`, confirmed in `privharness/evaluate.py`); 78 `preds.npz` + `result.json` across kth_ilic (16 releases), kth_scenario (15), utdmhad (16), charades (15), hmdb51_vp (16). Each `preds.npz` stores `(n_test, n_classes)` probabilities for `retrained` and `frozen` attackers (frame_r50, r3d; dino_probe / dino_ft on the DINO variants) and `result.json` stores `test_ids` (e.g. `person01_boxing_d3_1`, so the recording id is recoverable). Not stored: the released representations (stick figures, motion maps).
- `record/`: journal, results, dashboard.
- Regenerating a release for new attackers: deterministic function of raw video + pose model. Needs KTH (about 1.1 GB) and the 237 MB weights (both reachable) and a GPU pose pass; programme-1 logs show 65-300 min per release including attackers. Estimated 3 GPU-h for KTH pose extraction alone (ASSUMED).

## 2. POWER: effective independent units and what S can resolve

### 2.1 MEASURED on real programme-1 predictions (KTH identity, retrained attackers, best of frame_r50 and r3d)
KTH ilic: 598 test clips = **150 recordings** (25 persons x 6), about 4 clips per recording. Recording-cluster bootstrap (1,000-2,000 resamples).

| release | best-of-2 acc | clip-level binomial SE | recording-cluster SE | design effect |
|---|---:|---:|---:|---:|
| black (control) | 0.040 | 0.8 pt | 1.6 pt | 4.0 |
| raw | 0.992 | 0.4 | 0.5 | 1.9 |
| pose (stick figure) | 0.557 | 2.0 | 3.4 | 2.8 |
| pose_canon | 0.304 | 1.9 | 3.0 | 2.5 |
| pose_canon_down16 | 0.139 | 1.4 | 1.9 | 1.8 |
| pose_canon_down32 | 0.110 | 1.3 | 1.9 | 2.2 |
| bdq_down64 (learned) | 0.156 | 1.5 | 2.3 | 2.4 |
| bdq_down32 (learned) | 0.413 | 2.0 | 3.4 | 2.8 |
| motion_down32 | 0.642 | 2.0 | 3.3 | 2.8 |

Scenario split: also 150 recordings, design effect 2.0-3.0.

Paired differences between releases (best-of-2 each, recording-cluster bootstrap), MDE80 = 2.8 x SE:

| comparison (kth_ilic) | diff | SE | MDE80 | 95% CI |
|---|---:|---:|---:|---|
| canon x16 minus learned bdq x64 (programme-1 headline) | -1.7 pt | 2.7 | **7.6 pt** | [-7.2, +3.5] |
| canon x16 minus canon x32 | +2.8 | 1.8 | 5.0 | [-0.7, +6.2] |
| canon x16 minus canon x8 | -4.5 | 1.9 | 5.2 | [-8.3, -1.0] |
| canon x16 seed 0 minus seed 1 | +0.3 | 1.25 | 3.5 | [-2.0, +2.8] |
| canon x16 vs 24-epoch attackers | -0.2 | 1.7 | 4.7 | [-3.7, +3.0] |
| kth_scenario: canon x16 minus canon x32 | +4.3 | 2.1 | 5.9 | [+0.2, +8.3] |

Reading:
- Programme 1's 2-3 point margins are inside one standard error. Only differences of about 5 points or more resolve.
- Seed spread of the same release is SD 0.26-0.29 pt versus test-unit SE 1.8-2.5 pt: seeds replicate training noise only, as the charter says.
- Winner's curse for K = 2 (pick the better of two attackers on half the recordings, score on the other half): mean optimism **0.5-0.6 pt**, worst release 2.8 pt. Selection-bias machinery is moot until K >= 10, and the harness stores only 2-4 families per release, so K = 10 is not testable on existing files.
- Attacker family matters: frame_r50 vs r3d differ by up to 25 points on one release (bdq_down32 16.1 vs 41.3; pose 32.1 vs 55.7; motion_down32 52.0 vs 64.2) and the better attacker flips in 5 of 15 releases (gray, e24, s2, canon x32, raw). This supports the premise of S (family matters) but also shows that, per release, two families already capture most of the spread.
- Cross-fit (rotate the 150-recording test set over all 600 recordings, 4x attacker cost) would take the paired MDE80 from about 5-8 points to about 3-4 points; the cheapest way to buy power.

### 2.2 Resolving power of the S certificate (ANALYTIC, Clopper-Pearson one-sided 95%, Bonferroni over K attackers; zero leak means observed rate equals chance)
| independent units | 25-way identity, chance 4%: UCB (excess over chance) at K = 1 / 10 / 30 | binary attribute, chance 50%: at K = 1 / 30 |
|---:|---|---|
| 24 | 18.3% (+14.3) / 27.1% (+23.1) / 30.9% (+26.9) | +18.1 / +28.8 |
| 51 | 11.8% (+7.8) / 16.9% / 19.1% (+15.1) | +13.2 / +21.3 |
| 120-150 | 7.7-8.6% (+3.7 to +4.6) / 10.1-11.3% / 11.1-12.5% (+7.1 to +8.5) | +7.0 to +7.9 / +12.1 to +13.5 |
| 360 | 6.0% (+2.0) / 7.3% / 7.9% (+3.9) | +4.5 / +7.8 |
| 1,424 | 5.0% (+1.0) / 5.5% / 5.8% (+1.8) | +2.2 / +3.9 |
| 4,707 | 4.5% (+0.5) / 4.8% / 4.9% (+0.9) | +1.2 / +2.2 |

Real-data example (MEASURED): canonical stick figure x16 on KTH, best-of-2 13.9%, cluster SE 1.9 pt, so a one-sided 95% upper bound of 17.0% (K = 1) or 19.5% (K = 30, z = 2.94). Chance is 4%, so the certificate can only say "at most about 4-5x chance". It cannot certify "private" on KTH, MotionSense or UCI-HAR at any K.

### 2.3 Units and MDE80 per candidate testbed
| testbed | independent units | MDE80 |
|---|---|---|
| KTH ilic / scenario | 150 recordings (25 persons) | 3.5-7.6 pt paired between releases (MEASURED); about 3-4 pt with cross-fit (ESTIMATED) |
| MotionSense identity | 144 held-out trials (24 subjects); 360 with leave-one-trial-out cross-fit | about 7.5 pt (n = 144) and 4.8 pt (n = 360) for two defenders at about 12% leak (ANALYTIC, error correlation 0.5) |
| MotionSense / PAMAP2 / UCI-HAR / WISDM / HHAR subject-level attribute (gender, weight) | 24 / 9 / 30 / 51 / 9 subjects | 0.29 / 0.47 / 0.26 / 0.20 / 0.47 balanced accuracy: useless for comparing defenders |
| UTD-MHAD | 8 persons | unusable |
| CXR PriCheXy official test | 1,424 patients with >= 2 images | AUC 0.021 paired (rho 0.6, ANALYTIC Hanley-McNeil), Rank-1 3.2 pt |
| CXR LAPIS-size test | 4,707 patients | AUC 0.011, Rank-1 1.8 pt |
| PriCheXy's own 5,000-pair list | 1,307 patients overall; 492 behind the positive pairs | AUC about 0.035 paired: pairs do not add independent units |

Conclusion: S's certificate has resolving power only on CXR. On HAR testbeds its honest output is "not resolved" for every pair of defenders programme 1 cared about.

## 3. SATURATION / SCOOP (arXiv API queries and searches; window Aug-Nov 2026)

| Idea | Newest items found | What they do | Verdict |
|---|---|---|---|
| S (sealed suite, certificate) | LLM-Attack Suite 2610.22722 (Oct 2026); SynthGuard-ReleaseBench 2608.14753 (Aug 2026); IACR ePrint 2026/1990 "Bounded Information" (Sep); Picture the Epsilon 2608.17147 (Aug); VoicePrivacy Attacker Challenge 2504.14183 and Third Challenge 2601.11846; 2609.03055 (Sep, robot perception: "frozen attacker selection and representation-aware held-out attacks") | LLM-Attack Suite records attacker access, saves predictions for independent scoring, requires a linked undefended reference, pre-fixes a 2-point threshold, lists untested attacks. SynthGuard locks use, candidate panel, tolerances and audit schedule before evaluation, gives simultaneous finite-sample bounds, requires controls, excludes a case whose negative control passes, and finds transparent baselines pass while compact learned models fail. BI is a PAC union-bound certificate for a fixed attacker suite. | Crowded as method. HAR and CXR instances not found, but the framing is no longer novel. |
| R, wearable IMU | no re-audit of PrivDiffuser, ObscureNet or C-AAE after 2026-07-15 (queries on "PrivDiffuser", "ObscureNet", IMU / wearable anonymisation + attacker returned nothing); related: 2609.02971 (federated client-identity inference on UCI-HAR, five attack classes near 1.000), 2610.33206 A-GE (IMU age assurance) | n/a | Open. |
| R, CXR | **LAPIS 2610.40722 (29 Oct 2026)**: re-scores PriCheXy-Net with three retrained families (ResNet-50, ConvNeXt-T, ViT-S) plus SSIM / NCC structural retrieval; worst-case cross-study AUC 0.826 for PriCheXy-Net (own paper 0.577); no blur, downsample, PCA, random-warp, label-only or demographic baselines; no code | Partial scoop of the "rescore PriCheXy" half. Floors and protocol remain open. |
| R, video | 2610.34130 StickFigure (Oct 2026): pose-only IoT cameras export skeletons for HAR as a privacy mechanism, no attack audit (a parallel to our canonical-stick-figure finding, which says a skeleton alone leaks 53% KTH identity) | n/a | Open; useful citation. |
| G | 2610.36768 The Content Leash (Oct 2026): trains against a stronger speaker-verification representation; EER 2.8% floor to 26.3% on 38 unseen speakers and 27.9% against a held-out verifier; GaitProtector 2605.12431 (ensemble surrogates, held-out backbone, attackers pretrained not retrained); Wu et al. 2019 (sampled attacker set, unseen attackers) | positive transfer to held-out verifiers in speech, plausibly near-family | No leave-attacker-family-out study for IMU, video HAR or CXR found. Novel but priors split: frozen or near-family attackers transfer, retrained structurally different families unknown. |
| Floor-type results elsewhere | 2610.15626 (Oct 2026): a deterministic Riemannian transform removes cross-session EEG identity; "mean alignment and scatter whitening carry nearly all of the suppression" | the learned-vs-floor pattern of programme 1 recurs | context |
| Data | 2609.01036 MultiGait (199 people, multi-session, thermal / depth / lidar) | a multi-session identity dataset, not IMU; licence not checked | possible later arm for cross-session claims |

Current best published numbers to beat or reproduce: PriCheXy-Net ChestX-ray14 re-ID AUC 81.8 (real) to 57.7 (mu = 0.01), utility 80.5 to 76.2; LAPIS worst-of-3 AUC 0.826 for PriCheXy-Net and 0.791 for LAPIS, Rank-1 22.4% / 34.7%; PrivDiffuser MotionSense gender 93.52% (raw attacker) to 49.96%, HAR 96.32%; C-AAE MotionSense CNN user F1 0.532 to 0.073, PAMAP2 all-IMU CNN 0.990 to 0.157, LightGBM 0.717 to 0.323 (all as reported; C-AAE splits by random window).

## 4. FLOORS: the cheapest floor that could make each moot, and what is already measured

### 4.1 CXR linkage (MEASURED / SIMULATED on ChestX-ray14 metadata; scripts `cxr_meta_floor3.py`, `cxr_pricheXy_pairs.py`)
Gradient-boosted classifier on pair features (|age difference|, same sex, same width, same height, pixel-spacing difference, same view); no image used.

| protocol | attacker | AUC |
|---|---|---:|
| random negatives, official test list (4,000 pos / 4,000 neg) | true age + sex | **0.984** |
| same | age + sex as an image-based estimator might supply (SIMULATED: true age + N(0, 5 y), sex flipped 3%) | **0.884-0.887** |
| same | weaker estimator (sd 8 y, 95% sex) (SIMULATED) | 0.82-0.84 |
| same | device fields only (image size, pixel spacing, view) | **0.735-0.739** |
| same | true age + sex + device | 0.988-0.989 |
| **PriCheXy-Net's own 5,000 test pairs** (negatives: same-sex rate 0.513, median age gap 16 y; positives: same sex 1.00, median age gap 0) | true age + sex | **0.983** |
| same | estimated age + sex (SIMULATED) | 0.885 |
| same | device only | 0.743 |
| same | device + estimated age + sex (SIMULATED) | 0.921 |
| matched negatives (same sex, |age gap| <= 3 y) | true age + sex | 0.843 |
| same | estimated age + sex (SIMULATED) | 0.49-0.51 |
| same | device only | 0.733 |
| same | true age + sex + device | 0.900 |

Reading:
1. PriCheXy-Net's "undefended Siamese" AUC (0.818) is below what age + sex alone gives on the same pairs (0.983), and below a simulated image-estimated demographic attacker (0.885). Its defended 0.577 therefore shows that its own Siamese attacker is weak and/or removes demographic cues, not that no attacker can link. A demographic-plus-device attacker trained on the DEFENDED images (age and sex regressors, about 2 GPU-h) is the floor-attacker nobody has run. The 0.885 figure is a projection from assumed estimator error; it must be measured before it is quoted.
2. LAPIS's acquisition-matched AUC of 0.742 for its own release numerically equals the device-only metadata AUC measured here on PriCheXy's pairs (0.743). Possibly a coincidence; check on day 1.
3. The correct protocol is matched negatives (same sex, age within 3 y, same device fields). Under it, demographic estimators drop to chance (simulated 0.49-0.51) and any residual linkage is identity beyond demographics and device. Whether that residual is large on real images is the stage-0 question.
4. Other cheap CXR floors still to run: random smooth flow of equal mu (use `pretrained_generator_prichexy_net.pth`), Gaussian blur, x2 / x4 downsample, JPEG, and for latent-release variants PCA-k, random projection and label-only LDA on the 14 findings (utility the LDA cannot hold: held-out findings, region probes).

### 4.2 Wearable IMU (MEASURED on MotionSense; scripts `ms_floor.py`, `ms_floor2.py`)
Setup: 24 subjects, windows of 128 samples (stride 64), hand features (std, mean absolute difference, percentiles, 16 FFT bins), train on all trials except the last trial per subject x activity, test on those **144 trials**; trial score = mean class probability over its windows. Chance 4.2%. Single seed, single split, so each number is +-4 pt (50%) to +-2.5 pt (10%).

| release (cheap transform) | RF | LR | other |
|---|---:|---:|---|
| all 12 channels incl. means | 73.6% | 72.9% | LDA 66.0% |
| no attitude / gravity (rotationRate + userAcceleration), de-meaned per window | **78.5%** | 70.1% | LDA 62.5% |
| userAcceleration only | 74.3% | 60.4% | |
| acceleration magnitude only (1 channel) | **54.9%** | 31.2% | LDA 24.3% |
| downsample x4 / x8 / x16 (12 / 6 / 3 Hz), rot + acc | 76.4% / 76.4% / 72.2% | | |
| activity-LDA code, 5 dims (activity accuracy 92.3%) | 34.0% | 7.6% | kNN 31.2% |
| activity-LDA code, 3 dims (activity 82.2%) | 26.4% | 6.2% | kNN 23.6% |
| activity-LDA code, 2 dims (activity 73.9%) | 16.0% | 6.2% | kNN 11.8% |
| activity-LDA code, 1 dim (activity 59.8%) | 9.0% | 4.2% | kNN 11.1% |

Reading:
- Per-window de-meaning, magnitude-only and 16x downsampling are NOT floors against a nonlinear identity attacker: identity is in per-user amplitude statistics of gait, which is also the activity signal ("the task implies the secret").
- Label-supervised low-dimensional codes are the floors that bite, and they are leaky for nonlinear attackers (a 5-dim activity code leaves RF identity at 8x chance), so there is real headroom for a learned defender over the LDA floor at 3-5 dims; a linear-only attacker would have reported 6-8% and called it private. This is Finding 1 of programme 1 on IMU.
- Any published IMU defender reported at near-chance identity at high activity accuracy is therefore remarkable and worth auditing; but with 144 trial units only recoveries of about 15 points or more are resolvable.
- Not yet run: quantiser / noise dither, a 1D-CNN attacker, WISDM.

### 4.3 KTH / programme-1 video releases
The floors are already programme 1's own (canonical stick figure, fixed motion map). The re-attack question is whether new families (pose-sequence GCN on the redrawn skeletons, silhouette gait, OSNet, fine-tuned VideoMAE-v2 / UniFormer, VLM) move the best-of-2 numbers by more than the MDE (5-8 points). Controls exist (black 4.0%, shuffled labels at chance).

### 4.4 S
Floors that kill it: (a) effective units (Section 2: collapses at n <= 150), (b) a one-line statistic: for K <= 4 the best-of-2 and the val-selected pick differ by 0.5 pt on average (MEASURED), and per release two families capture most of the spread, so a 10-30 member suite adds little before the sample size is fixed. A cross-fit protocol (rotating test units) buys more resolution than any additional attacker.

### 4.5 G
Floors that kill it: noise dither / rate bottleneck at matched utility and leakage (programme 1: the adversary added nothing in four settings, including a dither AE matching an adversarial AE); one strongest attacker with restarts; activity-LDA-k code (Section 4.2). Published numbers already show the shape for frozen attackers (The Content Leash: held-out verifier 27.9% vs 26.3%), but for retrained structurally different families (tree models, metadata / structural attackers) nothing is published. Also, at 144-150 units a held-out-family difference below 7-8 points cannot be called.

## 5. The single likeliest way each wastes two weeks

- **S:** Building the general machinery (KL-binomial / union-bound certificate, locally simultaneous intervals, EVT stopping rules, a 10-30 member suite) on KTH or IMU data, and getting "not resolved" for every defender comparison (paired MDE80 5-8 pt vs 2-3 pt effects), while LLM-Attack Suite, SynthGuard-ReleaseBench and BI already hold the "sealed / locked audit" framing.
- **R:** Re-implementing the unreleased defences (C-AAE, LAPIS, CLOAK), or running the full PrivDiffuser sweep over MobiAct and WiFi-HAR, plus 42 GB of ChestX-ray14 plumbing, while the decisive result (the shortcut in the pair protocol, window-level user memorisation, same-session identity) was decidable on CPU in a day. Concretely: do stage 0 (Section 6) before any download beyond the CSV.
- **G:** Training attacker populations inside a defender loop (3-5 GPU-days) before any floor gate shows headroom, and ending in a tie with noise dither inside the 7-8 point MDE: an unresolved null that programme 1 already predicted (about 15% success prior).

## 6. Recommendation and first benchmark per focus

Pre-registration items common to all (charter): floors named before models, kill rule and margin fixed from the real MDE, cross-fit for test units, constant and shuffled controls at chance, raw reference above the label-only baseline, attackers retrained and worst case reported, predictions saved per unit.

### Focus 1: R-CXR. GO-WITH-CHANGES
Changes from the scout's D1: (1) primary protocol = demographic- and acquisition-matched negatives (same sex, |age gap| <= 3 y, same width / height / pixel spacing / view); random-negative AUC secondary only; (2) add a demographic attacker (age and sex regressors trained on the released images) and the metadata floors to the suite; (3) floors: random flow from `pretrained_generator_prichexy_net.pth`, blur, x2 / x4 downsample, JPEG, then PCA / LDA for latent variants; (4) unit = patient, cluster bootstrap; (5) do not re-implement LAPIS first; (6) CT proper stays out (LIDC-IDRI has 8 patients with two timepoints, NLST is CDAS-gated and bars identifying participants, CT-RATE is form-gated and forbids re-identification and redistribution, DeepLesion licence unverified), so say plainly that "CT" is served by chest radiographs as a proxy.
- Stage 0, CPU, <= 1 day: matched-negative metadata floor on the exact PriCheXy pair lists (extend `cxr_pricheXy_pairs.py`); download ChestX-ray14 and resize to 256 px; real-image Siamese on matched negatives. **Kill:** real-image matched-negative AUC does not exceed the metadata floor by >= 0.03 (the paired MDE80 at 1,424 patients), because then there is no identity beyond demographics to defend.
- **First benchmark to replicate:** PriCheXy-Net on ChestX-ray14 at 256 px with the repo's pair lists and released generator weights: real 81.8 / mu = 0.01 57.7 AUC, utility 80.5 / 76.2. Then re-score with three retrained families + SSIM / NCC + demographic attacker + the floors above, 1 seed first, 3 seeds if it survives.
- **Estimated GPU:** about 35 GPU-h (range 25-50): PriCheXy-Net mu = 0.01 retraining about 8 (range 4-12, ASSUMED), 4 learned attacker families x 7 releases x about 0.7 = about 20, demographic regressors about 4, plus floors. Wall time about 1-1.5 days on 4 GPUs after about 3 h of CPU data preparation.

### Focus 2: R-IMU. GO-WITH-CHANGES
Changes: (1) headline secret = trial-held-out identity on MotionSense (144 test trials; use leave-one-trial-out cross-fit for 360), WISDM (51 subjects, time-blocked) as a second set; subject-level gender / weight only as user-disjoint and never as a defender comparison (MDE80 0.29); (2) state that identity is same-session; (3) floors = activity-LDA-k codes, quantiser, noise dither (NOT de-meaning, magnitude or downsample, which are measured non-floors); (4) defenders = PrivDiffuser released checkpoint, a replacement-AE baseline, floors; C-AAE only as a cheap re-implementation; (5) only recoveries >= about 15 points are claimed; (6) optional arm: programme-1 video re-attack on the stored `preds.npz` (the baseline is already free) plus 2-3 new attacker families.
- Stage 0, CPU, <= 1 day: re-run PrivDiffuser's own raw attacker under a user-disjoint split vs its window split (quantifies memorisation); LDA-k floors with a 1D-CNN and LightGBM attacker (extend `ms_floor2.py`); WISDM block split.
- **First benchmark to replicate:** PrivDiffuser MotionSense (released `ckpt_80`): gender attacker 93.52% (raw) to 49.96%, HAR 96.32%; then retrained-attacker trial-level identity. 
- **Estimated GPU:** 8-12 GPU-h (DDIM sampling of the MotionSense test windows about 2-4 on an A6000, ASSUMED from the authors' 1-2 days for three datasets on a 2080 Ti; attackers about 3; floors on CPU). The video arm adds about 20-30 GPU-h (KTH pose regeneration about 3, new families 1-2 each) and should not start before the IMU stage 0 passes.

### Focus 3 (only in the three-focus plan): G. Conditional GO-WITH-CHANGES
Start only when R shows at least one defender class with headroom over the lower envelope of all floors. Families must be structurally different and fixed in advance (linear / kNN | tree | deep conv | non-learned metadata or structural attacker); rotate leave-one-family-out; comparators = noise dither, rate bottleneck, activity-LDA-k, single strongest attacker with restarts; margin >= the measured MDE (about 7-8 pt at 144-150 units, about 2-3 pt on CXR).
- **First benchmark:** there is no published HAR / CXR leave-attacker-family-out number to replicate (closest: The Content Leash held-out verifier, speech). The first benchmark is the stage-0 gate itself on cached MotionSense windows or CXR-matched latents: defenders trained against {LR, MLP, kNN}, held out {tree family}, and the rotation; expected pass probability about 15%.
- **Estimated GPU:** <= 6 GPU-h for the gate (small adapters, 3 seeds x 4-5 rotations); the full study only after it passes (3-5 GPU-days).

### S (both plans): instrument, not focus. About 2 engineer-days, 0 GPU
Deliver a saved-prediction schema (`unit_id, cluster_id, y, probs` per attacker), cluster bootstrap with design effect, paired MDE80 per comparison, Clopper-Pearson / Bonferroni suite upper bound, planted-leak and constant / shuffled controls, and cross-fit support. Prototype numbers in Section 2 came from `kth_power.py` and `cert.py`.
- **Stop rule:** if, after R-CXR stage 0, the matched-negative CXR residual is within MDE of the metadata floor, stop R-CXR and write the negative with the protocol finding as the by-product; if R-IMU stage 0 shows PrivDiffuser's release already sits at the LDA-k floor under retrained attackers, report that and do not start G.

## 7. Caveats
- The simulated demographic attacker (Section 4.1) assumes age error sd 5 y and 97% sex accuracy; real image-based estimators have image-dependent errors. The 0.885 figure is a projection, not a measurement.
- MotionSense numbers: one split, one seed, hand features, no CNN attacker; they are lower bounds on leakage.
- KTH numbers use the programme-1 harness predictions as stored; recordings are the cluster unit; persons (25) as clusters would be more conservative.
- Licence readings come from the pages opened (HF card, UCI pages, LICENSE file size); "same-session" for MotionSense is from the dataset description and cannot be verified from the files. DeepLesion and MultiGait licences were not checked.
- The arXiv API search covered titles / abstracts after 2026-07-15 with a limited set of queries; absence of a scoop is not proof.
- The VM was shared with queue workers (GPUs busy); nothing in ~/queue was touched and no GPU job was run. Probe data (about 700 MB) is in `~/work/stress_probe` on the VM and can be deleted.
