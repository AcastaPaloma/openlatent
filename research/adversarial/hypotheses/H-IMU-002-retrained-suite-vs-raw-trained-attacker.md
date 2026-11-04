# H-IMU-002: A retrained multi-family suite (trees, deep, sequence) reads at least 20 points more identity than the raw-trained CNN that IMU defence papers use, and the worst family changes across releases

- Focus: R-IMU | Kind: attacker-side re-audit | GPU-h: ~3 (1D-CNN and GRU members; trees and linear on CPU) | Prior: 0.70
- Depends on: H-IMU-001 releases (raw, PrivDiffuser, quantiser, dither, de-mean, downsample); MotionSense; C-AAE (2507.18072) reported a 2x spread between CNN and LightGBM on PAMAP2 (0.157 vs 0.323) and its ADPCM-only percentages are UNVERIFIED (stress test correction 2).

## Statement
PrivDiffuser, CLOAK and PATN score privacy with one CNN trained on RAW data (PrivDiffuser: "we train this model on raw sensor data"; threat model says the adversary does not know the data was obfuscated). On six shape-preserving MotionSense releases (raw, PrivDiffuser released checkpoint, uniform quantiser b = 3 + dither, noise dither sigma = 0.5 SD, per-window de-mean, x8 downsample-and-hold): (1) the median over releases of [worst-suite RETRAINED trial identity minus the FROZEN raw-trained 1D-CNN's identity] is >= 20 points (programme 1: frozen understated KTH identity by a median 47 points); (2) the spread max - min over the retrained families {LR, kNN, RF, LightGBM, 1D-CNN, GRU} is >= 15 points on >= 50% of releases; (3) at least two different families are the worst on >= 2 releases each (no single dominant attacker); (4) trees on hand-feature views are the worst family on the quantiser and dither releases, the 1D-CNN or GRU on the PrivDiffuser release (a prediction about which view wins, not required for the main claim).

## Mechanism
A raw-trained classifier applied to a release learns raw-data cues; a retrained attacker adapts to whatever the release preserves. Tree models on summary statistics read amplitude statistics that quantisation and dither preserve; sequence models read temporal structure that diffusion generation may preserve. The IMU defence literature scores one family, so ranking of defences by that attacker is not a ranking of leakage.

## Experiment (exact)
- Data / units: MotionSense, unit = trial; 3 rank folds (all 360 trials tested once), seeds = 3 inits inside each fold; paired trial-cluster bootstrap B = 2000, subject-cluster bootstrap beside.
- Frozen reference: 1D-CNN (4 conv + 3 FC, PrivDiffuser style) trained on raw windows of the training trials, identity labels, frozen, applied to each release; and the same architecture retrained on the release. Frozen applies only to shape-preserving releases.
- Retrained suite: LR, LDA, kNN (cosine / Canberra), RF (500), LightGBM (300), MLP on hand features; 1D-CNN and GRU on raw windows; validation-selected member scored on the sealed fold; naive max reported beside.
- Metrics: trial-level identity top-1 (chance 4.17%), normalised L, family spread, worst-family histogram; Kendall tau between the frozen-attacker ranking and the retrained-suite ranking of the six releases.
- Controls: constant / shuffled at chance for every member; raw reference far above activity-only baseline.

## Prediction and falsifier
- Predicted: median understatement 20-35 points; spread >= 15 points on 4 of 6 releases; two or more distinct worst families; tau <= 0.6.
- FALSIFIED if: median understatement < 8 points (a frozen raw-trained CNN is an adequate privacy proxy on IMU; retrained suites are not needed here), or one family is worst on >= 5 of 6 releases with a margin over the next family >= MDE80 (the suite is one attacker).

## Cost / dependencies
~3 GPU-h; needs H-IMU-001 releases (the PrivDiffuser sampling is the long pole). Its predictions feed H-S-002, H-S-004, H-S-005.

## Prior
0.70 for the median-understatement claim; 0.5 for the specific family prediction (4).

## Cheapest floor that could kill it
A single retrained random forest on hand features reproducing the suite worst case on all six releases within MDE80 (CPU, hours): then the "suite" is one tree model.
