# H-X-001: Boring control (both focuses): a noise-dithered, adversary-free code matches every learned defence at matched leakage; the adversary adds nothing beyond a rate bottleneck or noise

- Focus: X (cross-focus: R-IMU and R-CXR) | Kind: boring control that can kill any learned win | GPU-h: ~4 (reuses the encoders of H-IMU-005 and H-G-002; incremental dither / lambda = 0 arms) | Prior: 0.70
- Depends on: H-IMU-005 and H-G-002 encoders; H-IMU-001 / H-CXR-003 frontiers; programme-1 results: dithered no-adversary AE matched or beat the adversarial AE at equal leakage in Focus C (one seed, not replicated), A, D, and the stress test's four settings.

## Statement
In every cell where a learned defence is reported in programme 2 (PrivDiffuser identity-guided retrain, H-IMU-005 AE / VIB / VQ, H-G-002 latent adapter, PriCheXy-Net), the paired difference in worst-suite leakage between the learned defence and a noise-dithered, adversary-free bottleneck of equal width (lambda = 0 plus Gaussian noise on the code, strength swept) at matched utility (activity accuracy +-1 point IMU; held-out-finding AUC +-0.01 CXR) is within +-MDE80 in >= 75% of cells, and the adversary-vs-no-adversary contrast (lambda > 0 versus lambda = 0 at the same noise) is within +-MDE80 in >= 80% of cells. Numerically: |Delta L| <= 5 points of normalised identity leakage (IMU, cross-fit n = 360) and <= 0.02 AUC (CXR, n = 1,424-13,302).

## Mechanism
What leaves identity out of a code is a limit on the rate of nuisance-correlated information, and dither lowers the effective SNR of every feature including identity-bearing ones. A learned adversary can only remove what its training attackers can read, so against retrained, different attackers its added effect is small beyond the bottleneck. Programme 1: adversary vs rate (A, C), dither vs adversarial AE (C, one seed), reconstruction-only AE as the weakest family.

## Experiment (exact)
- Arms per learned defence: lambda = 0 (no adversary) and lambda in {0.3, 1, 3}; code noise sigma in {0, 0.25, 0.5, 1, 2} x code SD; width d in {4, 8, 16}; VIB beta grid. All selected on validation by the same rule as the learned arm (the leakage-matched pick, never by test).
- Matched comparison: for each learned defence point, interpolate the dither frontier at its utility; Delta L = L_learned - L_dither(utility); pair on test units.
- Suite: SUITE-IMU-8 / SUITE-CXR-8 as in the host hypotheses; held-out family reported separately.
- Test-unit design: IMU: trial unit, 3 rank folds (all 360 trials tested once), 3 seeds inside folds; CXR latent: patient unit, 5 patient folds, 3 seeds. Image-space PriCheXy-Net: fixed 1,424 test patients (stated).
- Controls: constant / shuffled at chance; raw reference above LDA-only baseline.

## Prediction and falsifier
- Predicted: |Delta L| <= 5 points / 0.02 AUC in 70-85% of cells; adversary contrast within MDE80 in >= 80%.
- FALSIFIED if the adversarial arm beats the dither frontier at matched utility by >= MDE80 in >= 2 independent cells (different dataset or family rotation) over 3 seeds with the CI excluding 0 on cross-fit: then the adversary has a real, replicated effect and programme 1's mechanism does not transfer.

## Cost / dependencies
~4 GPU-h incremental; must run alongside, not after, any learned-method hypothesis (its verdict is a condition of reading them).

## Prior
0.70. Four consistent programme-1 settings plus the speech-side counter-evidence being frozen-attacker only.

## Cheapest floor that could kill it
None cheaper: this control is the floor. A single cell where the adversarial arm wins by more than MDE80 on cross-fit with 3 seeds ends it.
