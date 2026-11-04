# H-IMU-001: On user IDENTITY, PrivDiffuser's release is dominated by the activity-LDA-k + dither frontier; an identity-guided retrain does no better than that frontier

- Focus: R-IMU | Kind: floors-first re-audit of a learned defence (two arms) | GPU-h: arm A ~6 (DDIM sampling of the MotionSense test windows 2-4, attackers ~2-3); arm B ~8 (identity-guided PrivDiffuser retrain, one fold) and optional | Prior: arm A 0.70; arm B 0.70 (that no learned advantage appears)
- Depends on: Builder H licence check (MotionSense, MIT per README); PrivDiffuser repo (MIT, Git LFS `models/privdiffuser_motion_1/ckpt_80_checkpoint.pt` 853 MB, gender / activity classifiers, MotionSense arrays); stress-test floors `ms_floor.py`, `ms_floor2.py`.

## Statement
Secret = user identity (24 subjects, chance 4.17%), NOT gender (PrivDiffuser's gender split is window-level over 24 users and cannot carry a claim; MDE80 0.29 balanced accuracy at 24 subjects). Utility = activity (6 classes). Under a retrained suite and trial-level scoring:
- Arm A (released gender-guided checkpoint, DDIM-sampled release): at activity accuracy within 1 point of PrivDiffuser's reported 96.3% (activity accuracy measured by an RF / 1D-CNN retrained on the release), worst-suite trial identity is >= 55% (raw retrained 74-78%, MEASURED single split), while an activity-LDA-5 code (activity accuracy 92.3% window-level LDA, 34.0% RF identity MEASURED) with dither tuned to the same activity accuracy sits >= 15 points below it: PrivDiffuser lies above the floor frontier on identity.
- Arm B (PrivDiffuser retrained with user ID as the guided secret, trial-disjoint): at matched activity accuracy its worst-suite identity is within +-8 points of the lower envelope of {activity-LDA-k + dither, quantiser, per-window de-mean + PCA}: no learned advantage.

## Mechanism
PrivDiffuser is a conditional diffusion model trained to remove a chosen attribute (gender) while the activity is preserved; its release still follows the input window, and per-user amplitude statistics of gait (which are also the activity signal; stress test 4.2: de-meaning, magnitude-only and x16 downsampling leave RF identity at 55-78%) are not what a gender-guidance term removes. A label-supervised low-dimensional code discards everything the activity labels do not need; with 3-5 dims it still leaves RF at 8x chance but far below the raw signal.

## Experiment (exact)
- Data / units: MotionSense (24 subjects, 6 activities, 15 trials each, 50 Hz; windows of 128 samples, stride 64). Unit = trial. Cross-fit: 3 rank folds (fold r tests the rank-r trial of every subject x activity; all 360 trials tested once; fold 1 = the stress-test "last trial" analogue with 144 trials for the headline); seeds = 3 attacker inits inside each fold.
- Releases: raw; PrivDiffuser arm A (released checkpoint; NOTE contamination: its training windows include our test trials because its split is window-level; disclosed, effect on identity unknown); PrivDiffuser arm B (retrained on training-fold trials with user ID guidance; one fold first); floors: activity-LDA-k (k = 1..5) with dither sigma in {0, 0.25, 0.5, 1} x code SD; uniform quantiser b in {2, 3, 4, 6} bits with dither; per-window de-mean (expected non-floor).
- Suite (retrained on each release): LR, LDA, kNN (cosine, Canberra), random forest (500), LightGBM (300), MLP on hand features; 1D-CNN and GRU on raw windows (shape-preserving releases only); trial score = mean log-prob over windows. Worst over the suite with validation-selected member scored once.
- Metrics: trial-level identity top-1; activity accuracy from a retrained RF on the release; frontier = lower envelope of floors in (activity accuracy, worst-suite identity); normalised L = (acc - chance) / (acc_raw - chance); trial-cluster and subject-cluster bootstrap B = 2000; paired MDE80 ~7.5 points at n = 144, ~4.8 at n = 360 (ANALYTIC).
- Replication arm (cheap): same floors with a 3-member suite on WISDM phone accelerometer (51 subjects, time-blocked split, CC BY 4.0), floors only (no PrivDiffuser).
- Controls: constant and shuffled-label inputs at chance (4.2%); raw reference above the activity-label-only baseline.

## Prediction and falsifier
- Predicted arm A: PrivDiffuser RF / LightGBM identity 55-70%; LDA-5 + dither at the same activity accuracy 25-40%; gap >= 15 points (resolved at both n). Arm B: within +-8 points of the envelope; probability that it is >= 10 points BELOW the envelope (learned wins) about 0.20.
- FALSIFIED (arm A) if PrivDiffuser's worst-suite identity at matched activity accuracy is <= LDA-5's + 5 points (the diffusion release hides identity by accident: a surprising and publishable surprise). FALSIFIED (arm B, a learned win) if the identity-guided retrain is >= 10 points below the floor envelope under cross-fit and 3 seeds: opens H-IMU-005 and the G gate.

## Cost / dependencies
Arm A ~6 GPU-h (inside the R-IMU 8-12 envelope); arm B ~8 additional and conditional on arm A showing a gap.

## Prior
Arm A 0.70 (the release follows the input, which carries gait amplitude); arm B no-advantage 0.70 (programme 1: adversary adds nothing beyond a bottleneck in four settings).

## Cheapest floor that could kill it
The activity-LDA-k + dither frontier itself (CPU, `ms_floor2.py` extension): if PrivDiffuser lands on that frontier, the exciting reading ("diffusion defence works") is dead.
