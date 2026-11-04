# H-IMU-004: A plain quantiser + dither supplies at most one third of a learned compressive anonymiser's identity drop against tree attackers, but most of it against linear attackers

- Focus: R-IMU | Kind: floors-first attribution (rate bottleneck without learning) | GPU-h: ~4 (re-implemented AE + ADPCM-style quantiser, 3 folds, small attackers) | Prior: 0.50
- Depends on: H-IMU-002 suite and splits; C-AAE (2507.18072) has no public code, so the learned arm is a re-implementation (autoencoder + ADPCM-style quantiser), NOT a reproduction; the paper's "ADPCM-only supplies 32 / 41 / 83% of the drop" is UNVERIFIED (arXiv HTML shows only an undefined before / after table, and the user-ID split is random, not by session).

## Statement
Define the share Q_f = (L_f(raw) - L_f(quantiser alone)) / (L_f(raw) - L_f(AE + quantiser)) for attacker family f, where L_f is the retrained family-f trial identity at MATCHED activity accuracy (the quantiser strength is tuned on validation to the strongest setting keeping activity accuracy within 1 point of the AE+quantiser release, which is >= 90%). Hypothesis: Q_tree (RF, LightGBM) <= 0.35 while Q_linear (LR) >= 2 x Q_tree and >= 0.6. So the quantiser is a strong floor only against attackers that cannot read amplitude statistics; against trees the learned part carries most of the drop (or the AE's bottleneck does; see the dither control in H-X-001).

## Mechanism
Identity in MotionSense trials is carried by per-user amplitude statistics of gait (stress test: x16 downsample leaves RF at 72.2%, de-meaned acceleration + rotation RF 78.5%, acceleration magnitude alone RF 54.9%). A uniform or adaptive-step quantiser coarsens amplitude resolution but preserves its distribution shape; activity also depends on amplitude, so a quantiser cannot be pushed far at matched utility. Linear models on raw windows see exact sample values, which coarse quantisation corrupts.

## Experiment (exact)
- Data / units: MotionSense, trial unit, 3 rank folds, 3 seeds inside folds.
- Releases: raw; uniform quantiser b in {1, 2, 3, 4, 6} bits with per-window range normalisation and optional dither; ADPCM-style (step-adaptive differential) quantiser; AAE-style autoencoder (conv encoder, d = 16-32, adversary on user ID with weight lambda in {0, 0.3, 1, 3}) followed by the quantiser; dither-only (Gaussian noise on codes); all selected on validation.
- Suite: LR, kNN, RF, LightGBM, MLP (hand features), 1D-CNN; worst per family and overall.
- Metrics: Q_f with paired trial-cluster bootstrap on the numerator and denominator (ratio CI by bootstrap); frontier plots of identity vs activity accuracy; MDE80 printed.
- Controls: AE with lambda = 0 (no adversary) and dither-only are in the grid (H-X-001 uses them); constant / shuffled at chance.
- Test-unit design: 3 rank folds, all 360 trials tested once; learned arms retrained per fold (defender sees only training-fold trials).

## Prediction and falsifier
- Predicted: Q_RF, Q_LightGBM in [0.15, 0.35], Q_LR in [0.6, 1.0]; AE + quantiser at >= 90% activity accuracy reaches RF identity 20-35% vs raw 74-78%.
- FALSIFIED if Q_tree >= 0.5 (the quantiser alone is a strong floor and the learned part is not needed against trees), or Q_LR < 1.5 x Q_tree (no attacker-family dependence, contradicting the C-AAE pattern as relayed).

## Cost / dependencies
~4 GPU-h; the learned arm is a faithful-in-spirit but unverified re-implementation, say so in any write-up.

## Prior
0.50. The family-dependence prediction rests on a relayed, unverified table; the tree-share bound rests on MEASURED amplitude-statistics persistence.

## Cheapest floor that could kill it
The activity-LDA-k + dither frontier (H-IMU-001) at matched activity accuracy: if that floor alone beats both the quantiser and the AE+quantiser by >= MDE80, the attribution question is moot (the right comparator was never the quantiser).
