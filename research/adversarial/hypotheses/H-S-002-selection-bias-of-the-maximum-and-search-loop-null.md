# H-S-002: The maximum over K attackers is inflated by 2-4 points at 144 trials and ~1 point at 1,424 patients; the sealed split removes most of it; cross-fit buys more resolution than more attackers

- Focus: S (instrument) | Kind: premise / selection-bias budget | GPU-h: ~1 (1D-CNN and Siamese members only; the rest is CPU on saved predictions) | Prior: 0.75
- Depends on: Builder S schema; saved per-unit predictions from the IMU and CXR suites (H-IMU-002, H-CXR-004 fits are reused); the 78 harness `preds.npz` files give K = 2-4 per release only, so K = 10 needs the new suites.

## Statement
(1) Selecting the best of K attackers on the same test units that score it inflates the reported leakage by +2 to +4 points of trial-level identity accuracy at K = 10 (and +3 to +5 at K = 30) on MotionSense (144 test trials), and by +0.006 to +0.015 AUC at K = 8 on 1,424 CXR patients. (2) Selecting on a validation split of the attacker-training units and scoring once on the sealed test removes at least 80% of that bias. (3) A shuffled-label run of an Optuna-style attacker-design search (N = 100 designs, best picked on the test units) reaches 7-10% trial accuracy on 24 subjects (chance 4.2%); the val-selected design scored once on the sealed split reads 4.2% +- 1.7. (4) Cross-fit over 3 rank folds lowers the paired MDE80 of two real releases by at least 25% (analytic: 7.5 to 4.8 points at 12% leak), whereas adding 5 more suite members lowers it by less than 5% relative.

## Mechanism
Expected maximum of K standard normals is 0.56, 1.16, 1.54, 1.87 sigma for K = 2, 5, 10, 20. At n = 144 and p = 0.12 the trial-level SE is 2.7 points; with error correlation rho = 0.5 between attackers the independent part is 1.9 points, so K = 10 gives about 2.9 points. This is the size of programme-1 effects (2-3 points). Measured on KTH at K = 2 the optimism was 0.5-0.6 points (worst release 2.8), which matches the formula, so the arithmetic is credible; K >= 10 has never been measured.

## Experiment (exact)
- Real data: IMU releases (raw, activity-LDA-3/5 codes, quantiser, dither, PrivDiffuser, de-mean; 8) x SUITE-IMU-8 members x 3 restarts x 2 views (hand features / raw) = K up to 48 attackers; CXR floor releases (6) x SUITE-CXR-8 members x 3 restarts. Saved probabilities only.
- Selection simulation: 2000 random splits of test units (selection half / scoring half; the stress test used recordings, here trials / patients). Optimism = score of the max on the selection half minus its score on the scoring half. Report K in {2, 5, 10, 20, 30, 48}.
- Synthetic truth: Gaussian class-mean generator, 24 classes, 144 test units, K attackers = LDA on random feature subsets with error correlation 0.3-0.7, true Bayes accuracy known. Coverage over 1000 replications of: percentile bootstrap of the naive max, val-selected sealed + cluster bootstrap, Bonferroni Clopper-Pearson, locally simultaneous interval (Zrnic-Fithian).
- Loop null: shuffled subject labels, Optuna TPE over a broad attacker template grid (family, depth, width, regularisation, view), 100 trials, 3 repeats; the same loop under a hidden-label evaluator (best design scored once on sealed units).
- Cross-fit vs more attackers: paired difference between two real releases (e.g. activity-LDA-5 vs quantiser b=3) by (a) single split n = 144, (b) 3-fold rank cross-fit n = 360 pooled, (c) single split with the suite grown from 4 to 9 members; paired trial-cluster bootstrap MDE80 = 2.8 x SE.
- Test-unit design: IMU unit = trial (3 rank folds, all 360 trials tested once, seeds inside folds); CXR unit = patient (official-test 1,424 headline; floor-only 5-fold rotation).

## Prediction and falsifier
- Predicted: optimism at K = 10 of 2-4 points IMU, 0.006-0.015 AUC CXR; val-selected sealed bias <= 0.5 point; loop-null max 7-10%; naive-max percentile bootstrap covers <= 88% (nominal 95%), val-selected sealed >= 92%, Bonferroni >= 98%; cross-fit MDE80 reduction >= 25%, suite growth < 5%.
- FALSIFIED if: optimism at K = 10 is below 1 point at n = 144 in at least 80% of releases (selection machinery unnecessary, keep only the sealed split; the scouts' D3 floor), or the loop-null maximum is below chance + 1.5 sigma (search is harmless at these sizes).

## Cost / dependencies
~1 GPU-h plus CPU; depends on saved predictions from H-IMU-002 / H-CXR-004; independent of any defence result.

## Prior
0.75. Formula and K = 2 measurement agree; main uncertainty is the true inter-attacker error correlation (0.3-0.7 range).

## Cheapest floor that could kill it
The sealed split alone: if val-selection + single scoring already gives an unbiased number, no Bonferroni / winners'-curse correction adds anything, and the only reported correction is the cluster bootstrap.
