# H3-015: WISDM biometrics replication of "floors beat identity-targeted defenders" and of "frozen raw-trained attackers understate leakage", on 51 subjects and a different sensor setup

- Focus: R-IMU replication | Kind: replication of direction (not magnitude) | GPU-h: 1.5 | Prior: 0.70 that the floor-versus-defender direction replicates in >= 2 of 3 time folds
- Depends on: H3-014 (AAE-R, OBF-A, floor code); WHITELIST (WISDM biometrics, UCI 507: CC BY 4.0, explicit biometric use; "check the phone-model confound before use"); Auditor-2 F17 (WISDM is one 3-minute block per subject-activity, adjacent windows leak; may mix phone models). Dataset facts below are from the dataset card as I recall it and MUST be verified by Builder H before pre-registration: 51 subjects, 18 activities, about 3 minutes per subject-activity, 20 Hz, phone (pocket) and smartwatch (dominant wrist) accelerometer and gyroscope.

## Statement
On WISDM (phone accelerometer + gyroscope, six activities chosen before any number from the locomotion / posture set: walking, jogging, stairs, sitting, standing, and one non-locomotion hand activity such as typing) with a TIME-BLOCKED cross-fit (each subject-activity recording cut into 3 contiguous 55-second folds with 5-second gaps; defenders and floors trained on two folds, released on the third):
(a) the floors beat AAE-R and OBF-A as in H3-014: Delta_L <= -0.15 or Delta_U <= -0.0285 (units re-derived for WISDM from its own floor pair) on >= 2 of 3 folds;
(b) the frozen raw-trained identity CNN reads <= 15% on AAE-R (windows) while the retrained suite reads >= 0.45 trial-segment-level: gap >= 0.30;
(c) the static-posture identity premise (MotionSense: sit / stand windows identify subjects at 7-8x chance after de-meaning) holds in WISDM: de-meaned static identity >= 5x chance (chance 1/51 = 1.96%);
(d) identity from the phone MODEL alone (the confound check) explains < 25% of the retrained suite's raw accuracy; if >= 25%, stratify by model and report both.

## Mechanism
Same as H3-014. WISDM adds more subjects (51, so the subject-cluster interval is tighter), a different sampling rate and placement, longer single blocks (stronger time-block leakage inside a recording, which the gap and the cross-fit partly control), and a possible secondary source (phone model), which is a confound to be measured rather than assumed away.

## Experiment (exact)
- Data and units: units = (subject, activity, time-fold) segments, 51 x 6 x 3 = 918 segments; cluster = subject (51). Windows of 128 samples at 20 Hz, stride 64; the hand-feature view (44 features per window, same function as H) and raw windows; no resampling to 50 Hz unless pre-registered (state either way).
- Pre-registration before any number: the six activities, the fold cut points, the floor list (copied from FLOOR_LIST.json, recomputed for the 6 channels; no new floors), the margins re-derived from the WISDM floor pair by the H-A3 rule and the realised MDE80 (printed first; if MDE80_L > the margin, outcome INCONCLUSIVE). The sealed-split logic: the cut-points are fixed by a salt committed AFTER the floor list (custodian), as for H.
- Defenders: AAE-R and OBF-A of H3-014 retrained on WISDM folds (the shipped MotionSense weights cannot be applied to another sampling rate / placement; the R0 gate is the same: AAE-R vs its own frozen raw-trained CNN). Floors per H3-014.
- Attackers: H suite retrained; frozen raw-trained CNN for the protocol gap; L trial-segment-level worst case (segment = 55-second fold of one subject-activity, about 40 windows), U window-level activity accuracy; held-out family (SVM, transformer).
- Confound check (before the defender runs): phone model per subject from the dataset metadata; accuracy of a model-only classifier; stratified results if needed.
- Seeds vs test units: test units = segments (918) in a 3-fold time-blocked cross-fit (every segment scored once); seeds: 2 for AAE-R, 1 for OBF-A per lambda; the subject bootstrap (B = 2000) carries test-unit variation; folds replicate time blocks within the same sessions, NOT new sessions (stated: "same-session identity" as in MotionSense).

## Prediction and falsifier
- Predicted: (a) Delta_L(AAE-R) <= -0.20 on all 3 folds, Delta_L(OBF-A best) in [-0.3, +0.05]; (b) frozen 2-12%, retrained 0.45-0.80; (c) static de-meaned identity 0.10-0.30 (5-15x chance); (d) phone model explains <= 15%.
- FALSIFIED (the direction does not replicate) if on >= 2 of 3 folds a retrained defender is within 0.05 of the floor envelope or above it in L at equal U (with MDE80 below that margin), OR the frozen-versus-retrained gap is < 0.15, OR the model confound explains >= 50% of the retrained accuracy so that identity is a device label (then the WISDM result is withheld).

## Cost / dependencies
1.5 GPU-h (small data: 51 subjects x 6 activities x 3 min x 20 Hz is ~2 M samples; AAE-R 3 folds x 2 seeds + OBF-A 3 lambdas x 3 folds at ~3 min each, attackers mostly CPU). Download 0.3 GB, CC BY 4.0 with attribution. Starts after H3-014's code runs on MotionSense.

## Prior
0.70. The mechanism is not dataset-specific, and the floor advantage on MotionSense was large (0.28-0.78 in L). Risks: WISDM's coarser sampling and per-block structure may make the floors' dither variants behave differently, and 51 subjects with phone-model structure can make identity a device label.

## Cheapest floor that could kill it
`predlabel` and the activity-LDA floors refit on WISDM; for the confound, the phone-model-only classifier.
