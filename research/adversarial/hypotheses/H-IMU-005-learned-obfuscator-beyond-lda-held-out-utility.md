# H-IMU-005: A learned adversarial / VIB obfuscator keeps held-out-activity and cadence utility that an activity-LDA code cannot, at matched worst-suite identity

- Focus: R-IMU | Kind: LEARNED method vs floor envelope (the "exciting" hypothesis) | GPU-h: ~7 (3 folds x grid of d in {4, 8, 16}, lambda in {0, 0.3, 1, 3}, 3 seeds, ~3 min each, plus attacker retraining) | Prior: 0.12
- Depends on: H-IMU-001 (the activity-LDA-k + dither frontier and the stage-0 headroom), H-IMU-003 (protocol), H-X-001 (dither control). Programme-1 precedent: Focus A and C, where no learned adapter beat the label-only LDA arm and the held-out-emotion criterion failed (dU_HE -0.21).

## Statement
Because an activity-LDA code collapses to the 5-dim subspace of the six training activities, the only utility a learned encoder can add is utility the labels did not select: (U1) held-out activity classes (encoder fitted on 4 of 6 activities, linear-probe accuracy on the other 2), (U2) cadence / step-intensity regression (continuous, derived from FFT peak frequency of the window), (U3) activity accuracy at 6 classes. Hypothesis: a validation-selected learned obfuscator (conv encoder d = 8-16, activity CE + identity adversary lambda, trained against in-loop MLP and 1D-CNN adversaries, no peeking at the held-out family) beats the lower envelope of {activity-LDA-k + dither, quantiser, per-window de-mean + PCA-k, dither-only label-free AE} by DeltaU >= +0.05 on U1 and DeltaR2 >= +0.05 on U2, at worst-suite trial identity within +-3 points (RF, LightGBM, 1D-CNN retrained), in >= 2 of 3 rank folds and 3 seeds.

## Mechanism
A bottleneck trained on identity-adversarial signal can retain structure orthogonal to the labels (cadence, gait regularity) at the cost of identity-correlated structure; a label-supervised projection keeps none of it. Programme 1 found the adversary-free bottleneck, not the adversary, supplying whatever gain existed.

## Experiment (exact)
- Data / units: MotionSense, trial unit, 3 rank folds, all 360 trials tested once; seeds 3 inside folds; activities for U1: rotate which 2 of the 6 are held out (3 rotations: {dws, ups}, {wlk, jog}, {std, sit}).
- Learned arms: (i) adversarial AE (lambda in {0.3, 1, 3}); (ii) VIB (beta grid, no adversary); (iii) lambda = 0 AE; (iv) VQ code with codebook dropout. Selection by validation trials only, by the pre-declared rule "highest U1 subject to worst-suite validation identity <= the floor's at the same activity accuracy + 3 points".
- Floors (lower envelope): activity-LDA-k (k = 1..5) padded to the encoder width with dither, uniform quantiser, per-window de-mean + PCA-k, noise-dithered label-free AE (H-X-001), raw.
- Suite: RF, LightGBM, kNN, LR, MLP on features; 1D-CNN and GRU on raw windows. Held-out family for the final report (not used in selection): GRU / sequence model.
- Metrics: DeltaU1, DeltaR2 vs the envelope with paired trial-cluster bootstrap; identity matched within +-3 points (leakage-matched comparators per programme-1 practice); MDE80 printed (4.8 points at n = 360).
- Controls: constant / shuffled at chance; raw above LDA-only baseline.

## Prediction and falsifier
- Predicted: DeltaU1 -0.05 [-0.15, +0.03]; the adversary-free AE is the only positive but leaks 14-35x chance; success probability 0.12.
- FALSIFIED (a learned win) if DeltaU1 >= +0.05 with CI excluding 0 in >= 2 of 3 folds and 3 seeds at matched identity, and the dither control (H-X-001) does not match it. The hypothesis is KILLED by: the dithered label-free code matching the adversarial AE at equal leakage (adversary adds nothing), or DeltaU1 < +0.05 / CI containing 0.

## Cost / dependencies
~7 GPU-h; starts only after H-IMU-001 shows the LDA-k frontier is not already below every learned release.

## Prior
0.12. Four closed focuses in programme 1 point the same way; the IMU case differs only in the possible cadence / held-out-activity utility.

## Cheapest floor that could kill it
Activity-LDA-k + dither on U1 / U2 (CPU): LDA's held-out-class utility is the comparator; if a plain PCA-k of de-meaned features already holds U1 and U2 at the same identity, nothing learned is needed.
