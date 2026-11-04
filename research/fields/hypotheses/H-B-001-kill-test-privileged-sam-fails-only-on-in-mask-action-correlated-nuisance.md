# H-B-001: Oracle-prompted zero-shot SAM masks stop working on action-correlated sprites that overlap the arm; they still solve independent ones. B' lives or dies on this contrast

- Focus: B' (robotics, conditional; this is the one-day kill test) | Kind: kill test / baseline | Cost: ~15 GPU-h | Prior: 0.35 that B' survives
- Depends on: Builder-B code as written at 2026-11-04 ~03:30Z: `src/b_robotics/envgen.py` (reach-v3, push-v3, pick-place-v3; 128x128; macro actions; exact arm/object silhouettes; EE pixel), `distractors.py` (kinds `none`, `indep`, `corr_det`, `corr_noisy`: three opaque discs that start near the end effector, drawn over the arm, half the palette arm-like), `sam_masks.py` (SAM ViT-B prompted with 3 points from the *clean* arm silhouette excluding sprite-covered pixels: the most generous zero-shot baseline).

## Statement
Train latent-action models (LAPO-style inverse + forward model) on the clean frames plus each distractor kind and score by the linear-probe NMSE of the macro action from z (probe trained at the training distribution, tested on `indep` sprites, i.e. decorrelated). Then:
(a) on kind `indep`, SAM-masked LAM NMSE <= 1.10 x GT-mask NMSE and <= 1.25 x clean-oracle NMSE (masks solve action-independent distractors: replication of MaskLAM's finding in our env);
(b) on `corr_noisy` (rho_eff ~ 0.9, see below) and `corr_det`, the SAM-masked and the GT-masked LAM NMSE (at the decorrelated test) are >= 1.5 x the clean oracle, because the sprites' pixels overlap the arm region and their motion is a function of the action, so masking the arm does not remove them and the model can use them as a (spurious) action readout;
(c) the distractor-motion probe (retrained, ridge on z -> sprite velocity) has R^2 >= 0.5 on SAM-masked z for `corr_*` and <= 0.1 for `indep`.
If (a) holds and (b) fails (masks within 1.25 x oracle on `corr_*` too), B' is dead.

## Mechanism
MaskLAM with SAM masks already beats the label-supervised reference on Distracting Meta-World (NMSE 0.090 vs 0.142; GT mask 0.088; scout-reported project-page numbers, literature/contrarian_stress_test.md 2.1). Masks work when nuisance and task occupy different pixels and fail when they overlap (pattern across the robotics papers, literature/robotics_embodied.md). Dueling World Models name "distractors whose motion tracks the action" as the boundary of their method (2608.06706, abstract opened); Why Latent Actions Fail assumes exogenous changes are unrelated to actions (2605.20223, abstract opened). Whether an unlabeled LAM can tell the arm from an arm-coloured, action-driven sprite is the open question; with `corr_det` the two are *not identifiable* from video alone, so the claim is about `corr_noisy` where the arm's residual is exactly explained by the action and the sprite's is not.

## Experiment (exact)
- Data: Builder-B episodes, 600 train / 100 val / 100 test per task, 16 macro steps; start with reach-v3, add push-v3 if time. rho_eff = sd(drive)/sqrt(sd(drive)^2 + sigma^2): sweep `SIGMA_CORR` in {0.5, 1.5, 4.5, 15} px (rho_eff ~ 0.97, 0.9, 0.6, 0.2 for a drive of ~1.5 px; compute from data) plus `corr_det`. Test kind: `indep` (decorrelated, same visual statistics).
- Models: LAPO-style LAM (CNN inverse model (o_t, o_t+1) -> z in R^8; forward model (o_t, z) -> o_t+1 in pixels) with arms: no mask; SAM-masked loss (Builder-B masks); GT arm mask; clean oracle (kind `none`); label-supervised reference (500 labelled transitions); optional extension arm: per-frame photometric normalisation.
- Probes: ridge probe of the 3-D macro action from z (trained on 20% of train episodes), evaluated on `indep` test; ridge probe of sprite velocity from z (distractor probe). Report chance as NMSE = 1.
- Seeds: 3 LAM seeds x 4 kinds x 4 arms ~ 48 runs of ~0.3 GPU-h.
- Metric: NMSE ratio to the clean oracle; distractor-probe R^2; SAM mask quality (IoU vs silhouette, decoy pixels inside the mask, already stored by `sam_masks.py`).

## Prediction and falsifier
- Prediction: (a) ratios as above; (b) ratios 1.5-3.0 for `corr_noisy` at rho_eff ~ 0.9 and >= 2.5 for `corr_det`; label-supervised reference within 1.3 x oracle; (c) R^2 0.5-0.9 vs <= 0.1. SAM masks include 5-25% decoy pixels on `corr_*` kinds because the sprites sit on the arm, so (b) is partly a mask-quality effect; report masked vs unmasked arms to separate it from the shortcut effect.
- Falsified (B' dead) if (b) fails, i.e. SAM-privileged or GT masks are within 1.25 x oracle on `corr_noisy`; or if (a) fails (SAM can't even solve `indep`: then the baseline is broken and the test is void, fix first). If (b) holds, B' continues with H-B-002/003. Extension: if the global photometric variants (flicker) are added, a per-frame normalisation should remove them (<= 1.15 x oracle); that tells which nuisances are really in B' scope.

## Cost
~15 GPU-h (48 runs x ~0.3 h) + SAM mask generation (Builder-B, ~1 GPU-h per kind).

## Dependencies
Builder-B `distractors.py`, `sam_masks.py`, LAM trainer; a `SIGMA_CORR` argument; ridge probes.

## Prior and why
0.35. (a) is very likely (0.85). (b) is the real risk: if the LAM's forward model can already tell sprites from the arm by appearance (sprites are discs, the arm is not), the shortcut may not materialise, and then masks are not the limiting factor either way. Identifiability caveat for `corr_det` stated above.

## Sources
[O] 2608.06706; [O] 2605.20223; [S] 2601.22714, 2602.02259, 2502.09680 (literature/contrarian_stress_test.md 2.1 and robotics_embodied.md); Builder-B source files named above.
