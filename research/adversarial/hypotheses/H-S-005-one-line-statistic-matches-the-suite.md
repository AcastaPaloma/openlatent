# H-S-005: A one-line statistic matches the full suite on high-leak releases but not on low-leak releases; on CXR random negatives a metadata line beats every image attacker

- Focus: S (instrument) | Kind: boring control that can kill the instrument | GPU-h: ~0.5 (CXR metadata arm and IMU statistics are CPU) | Prior: 0.60
- Depends on: IMU releases and suite predictions (H-IMU-002); CXR metadata CSV (11 MB, already downloaded in the stress test) and, for the image-estimated arm, H-CXR-001.

## Statement
Define the one-line statistics: IMU = per-trial per-channel mean and SD (12 x 2 numbers) -> LDA / nearest centroid; for low-dimensional code releases = per-trial mean code -> nearest centroid. CXR = pair features (|age gap|, same sex, same width / height / pixel spacing, same view) -> gradient boosting; second CXR line = 16x16 raw-pixel NCC. Claims:
1. IMU: the one-line statistic reaches >= 85% of the suite worst case (accuracy minus chance) on high-leak releases (suite worst >= 40%: raw, de-mean, downsample, quantiser) and <= 60% of it on low-leak releases (suite worst <= 20%: activity-LDA-3/5 + dither, PrivDiffuser if it lands there). So the line matches the suite exactly where nobody needs a suite.
2. CXR on random negatives: the metadata line with TRUE age and sex reaches AUC 0.98 (MEASURED 0.984 official test list, 0.983 PriCheXy's 5,000 pairs) and exceeds the suite worst case for every defended PriCheXy release; with image-ESTIMATED age and sex it stays above 0.80 (simulated 0.885; to be measured). On matched negatives (MN-v1, see INDEX) the line falls to <= 0.56 and the image suite exceeds it by >= 0.03 on the raw release.
3. S is only justified where the line fails: S is worth building iff the line matches the suite within MDE80 in <= 30% of low-leak cells.

## Mechanism
Programme 1: LDA was the strongest identity attacker on CREMA-D; a six-number prosody line set the floor; brightness-only gave 22.4% identity. When the secret is carried by first and second moments, the suite is redundant with a summary statistic. The CXR numbers come from the stress test (sec 4.1), which showed that PriCheXy's pair protocol is solvable from metadata.

## Experiment (exact)
- IMU releases and SUITE-IMU-8 as in H-IMU-002; the one-line statistics are computed per trial on the same released windows and scored on the same test trials (3 rank folds, all 360 trials tested once; trial and subject bootstrap; paired difference line minus suite worst with MDE80).
- CXR: PriCheXy's 5,000 test pairs and MN-v1 matched pairs built on the 1,424 official-test patients; true-metadata line and image-estimated line (H-CXR-001 estimators, cross-fitted); NCC line on 16x16 pixels of raw and defended releases. Patient-cluster bootstrap.
- Classification of cells as high-leak / low-leak is fixed in advance by the suite worst case on the raw reference (threshold 40% / 20% trial accuracy; AUC 0.80 / 0.65).
- Controls: constant and shuffled inputs at chance for the line itself; the line must not exceed the raw suite on the raw release by more than MDE80 on IMU (if it does, the suite is broken).

## Prediction and falsifier
- Predicted: claim 1 holds in >= 70% of high-leak cells and fails (line < 60% of suite) in >= 70% of low-leak cells; claim 2 as stated; claim 3 holds.
- FALSIFIED if the line is within MDE80 of the suite in >= 70% of IMU low-leak cells (S adds nothing exactly where it is needed; programme 2 keeps only the floors-first audit), or if on matched negatives the metadata line is within 0.03 AUC of the raw-image Siamese (no identity beyond demographics, which also stops R-CXR: see H-CXR-002).

## Cost / dependencies
~0.5 GPU-h; zero for IMU. Depends on H-IMU-002 predictions, the MN-v1 construction, and H-CXR-001 for the estimated line.

## Prior
0.60. Strong for the high-leak half; the low-leak half rests on the stress-test observation that a 5-dim activity code still leaves RF identity at 8x chance, which a per-trial mean-code centroid will not see.

## Cheapest floor that could kill it
This hypothesis IS the floor for S. Its negative (line matches the suite in low-leak cells) kills the instrument's justification; it costs one CPU afternoon.
