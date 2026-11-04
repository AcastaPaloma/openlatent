# H-B-002: A latent action model trained with action-correlated sprites (rho_eff ~ 0.9) loses >= 2x on decorrelated sprites and encodes the sprites (probe R^2 >= 0.5)

- Focus: B' (conditional on H-B-001 part (b)) | Kind: measurement of the failure mode | Cost: ~5 GPU-h | Prior: 0.55
- Depends on: H-B-001 (verdict that masks do not remove `corr_*` sprites); Builder-B data (`distractors.py` kinds `none`, `indep`, `corr_noisy`, `corr_det`), LAM trainer.

## Statement
For kind `corr_noisy` with the `SIGMA_CORR` sweep giving rho_eff in {0.2, 0.6, 0.9, 0.97} (plus `corr_det` as rho = 1), train the LAM on those frames and measure the linear-probe NMSE of the macro action on a decorrelated test (`indep` sprites with the same appearance). Then:
(a) NMSE(rho_eff = 0.9) >= 2.0 x NMSE(trained on `none`) and NMSE(0.97) >= 3.0 x; NMSE(0.2) <= 1.3 x;
(b) a retrained ridge probe decodes the sprite velocity from z with R^2 >= 0.5 at rho_eff = 0.9 and <= 0.1 when trained on `indep`;
(c) the shortcut grows with latent size: ratio(z-dim 8) - ratio(z-dim 3) >= 0.5.

## Mechanism
Reconstruction objectives push latent actions to absorb exogenous information from the future frame (2605.20223, abstract opened); when the exogenous signal also carries the action in training, the cheapest predictor of the next frame can be the sprite motion rather than the arm, which breaks when the correlation is broken. Dueling World Models name the same regime as their boundary (2608.06706, abstract opened). The z-dim effect is the capacity argument: with z-dim = 3 (the action dimension) z must prioritise one channel; with 8 it can carry both and the probe may pick the wrong one. Nikulin et al. state that the action probe does not show whether exogenous noise is also encoded (scout-reported, literature/contrarian_stress_test.md 2.2), so the distractor probe is the missing measurement.

## Experiment (exact)
- Data: Builder-B reach-v3 first (then push-v3); sprites as in `distractors.py` (3 opaque discs over the arm, velocity = R_k a_xyz + sigma * eps). Test set: `indep` sprites (AR(1) velocities, same palette and radius).
- Models: LAPO-style LAM, z-dim {3, 8}; no-mask arm (the failure mode) and GT-mask arm (H-B-001 says it does not fix it).
- Probes: ridge probe (macro action <- z) trained on 20% of train episodes under the training sprite kind, tested on the `indep` test; ridge probe (sprite velocity <- z), retrained per model.
- Seeds: 3 LAM seeds x 5 rho levels x 2 z-dims x 2 mask arms = 60 runs (~0.1 GPU-h each at 128x128 with a small CNN).
- Metric: NMSE (1.0 = predict mean), ratio to the `none`-trained model, R^2 of the sprite probe.

## Prediction and falsifier
- Prediction: ratios 1.0-1.3 (rho 0.2), 1.3-2 (0.6), 2-4 (0.9), 3-6 (0.97 and `corr_det`); sprite-probe R^2 0.5-0.9 at 0.9; ratio(z=8) - ratio(z=3) in 0.5-2; GT-mask arm within 20% of the no-mask arm (masks do not help).
- Falsified (no failure mode to fix) if the ratio at rho_eff = 0.9 is <= 1.3 for both z-dims (the LAM ignores correlated sprites), or the sprite probe R^2 <= 0.2 at 0.9.

## Cost
~5 GPU-h.

## Dependencies
H-B-001 outputs; `SIGMA_CORR` parameter; sprite-velocity labels (`vres`, `vtot` are already returned by `distractors.sample`).

## Prior and why
0.55. Shortcuts of this kind are the norm in spurious-correlation benchmarks; the unknown is whether a small CNN LAM on 128x128 frames prefers sprite motion (small discs, little pixel mass) to arm motion (large, strongly action-dependent) when both are available.

## Sources
[O] 2605.20223; [O] 2608.06706; [S] 2601.22714 (twin-render linear-probe protocol; literature/robotics_embodied.md #13).
