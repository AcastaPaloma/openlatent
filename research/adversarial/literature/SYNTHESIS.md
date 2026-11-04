# Phase-1 synthesis, programme 2 (2026-11-04, after 6 scouts; scores are scouts' own priors /30, mostly abstract-level reads)

## Convergence (the same four ideas surfaced in 3-6 scouts independently)
| # | Direction | Scouts | Scores | Kill floor named by scouts |
|---|---|---|---|---|
| S | **Sealed fixed-code attacker suite + certificate**: pre-declared attackers, saturation/coverage curves, selection-bias correction of the maximum (Bonferroni/cluster bootstrap), planted-leak calibration, smallest resolvable effect | suites A 26; leakage D1 25, D2 25, D3 23; agentic D1 25, D2 24; FM D6 23 | 23-26 | suite worst case within the MDE of the best of {LDA, kNN, linear probe}; one-line statistic matches the suite at every dose; effective n too small for any interval to resolve |
| R | **Floors-first re-audit of published defences under a retrained suite**: wearable IMU obfuscators (27), programme-1 HAR releases (24), CXR linkage defences (26; LAPIS 2610.40722 got AUC 0.826 vs PriCheXy's own 0.577) | HAR, CT, suites | 24-27 | label-only LDA, per-window de-meaning, downsampling, quantiser alone (supplies 32-83% of one IMU defence's claimed drop), random smooth warp, blur, PCA |
| G | **Defender trained on an attacker population, tested on a held-out attacker FAMILY** (charter's second falsifiable clause) | suites C 26; CT D2 27; HAR IMU 27 (second half); agentic D4 24 | 24-27 | rate bottleneck / noise dither at matched leakage; one strongest attacker with restarts; programme 1 already predicts failure (~15%) |
| L | **Attacker ladder**: knowledge (ignorant / retrained / defence-aware / fine-tuned FM), pretrain x adaptation with a random-init copy, agent vs hand-built vs Optuna at equal compute | suites B 25; FM D1 24; agentic D3 24 | 24-25 | random-init copy or scratch R3D matches the FM; Optuna ties the agent |

## Facts that shape the choice
- No "AutoAttack for privacy" exists outside speech (VoicePrivacy Attacker Challenge is the closest); no 2024-26 leave-attacker-family-out study for privacy defences; <10% of action-recognition privacy papers use adaptive attackers.
- 2024-26 HAR defences (PrivDiffuser, ImmCOGNITO, PMR, NTU120-Privacy-21K, GaitProtector) are scored against ONE attacker trained on raw data; the exception C-AAE (IMU) shows CNN vs LightGBM differ 2x on one release.
- Attacker strength in programme 1 came from the INPUT VIEW, not the architecture; foundation-model attackers are not reliably stronger (small CNN beat RETFound 72 vs 39 R@1; DINOv2 trails R3D on BDQ/stick figures).
- With 12-48 test units, best-of-N search is a winner's-curse machine; the bias of the maximum at 10 attackers is ~2-6 points, the size of programme-1 effects. Programme 1's result files stored only top-1: attackers must be rerun with predictions saved.
- Data: CXR longitudinal re-ID is open (ChestX-ray14); CT is not (LIDC 8 two-timepoint patients; NLST CDAS-gated and forbids identifying participants; CT-RATE forbids re-identification and redistribution; DeepLesion licence unverified). IMU: CC BY / MIT datasets exist; Capture-24 asks users not to identify individuals. Licence flags for the owner are in each scout file.

## Provisional focus (not locked)
S (suite + certificate) is the shared instrument; R (re-audit, IMU + programme-1 HAR releases + CXR) applies it and is where headline findings come from; G (population-trained defender, held-out family) is the learned-method core and depends on S for its evaluator.
Proposed lock: S + R as the two focuses, G as a conditional third behind a floors-first gate.

## Contrarian round (must fail to beat S, R, G before lock-in)
1. Missed-directions scout.  2. Stress test: data access from the VM (ChestX-ray14, IMU sets, HAR video, programme-1 artefacts on HF `KuanKuanKuan/openlatent-privacy`), effective n, saturation, licences.
