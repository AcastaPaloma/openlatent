# H-CXR-007: Boring control: defended-release linkage is concentrated in adjacent (near-identical) visits, while raw linkage barely depends on the gap

- Focus: R-CXR | Kind: boring control (near-duplicate / visit-gap confound; Engelmann et al., npj Digit Med 2026, on fundus) | GPU-h: ~0.5 (stratified re-scoring of H-CXR-002/-003 predictions; NCC on CPU) | Prior: 0.40
- Depends on: saved pair scores from H-CXR-002/-003/-004; the CSV Follow-up # (ChestX-ray14 has NO acquisition dates, so the gap is an index gap, not days).

## Statement
Stratify positive pairs by follow-up-index gap g in {1, 2-3, >= 4}, each stratum paired with MN-v1 negatives matched to that stratum's age-gap distribution. (1) On RAW images the Siamese AUC drops by <= 0.03 from g = 1 to g >= 4 (transfer over long gaps, consistent with Packhauser's ten-year claim). (2) On PriCheXy mu = 0.01 and the best floor the worst-suite AUC drops by >= 0.08 from g = 1 to g >= 4: defended-release linkage lives mostly in adjacent visits. (3) Pairs whose raw 256-px NCC (after affine registration) exceeds 0.95 (near-duplicates) make up >= 15% of g = 1 positives, and removing them lowers the defended-release AUC by >= 0.03. If (2) and (3) hold, any headline AUC averages a near-duplicate regime where defences fail with a far-gap regime where they look strong.

## Mechanism
Consecutive visits in a short interval produce near-identical radiographs; a smooth small warp cannot hide a pair that is almost pixel-identical (NCC with registration undoes the warp). Far-gap pairs differ in positioning, disease and weight, so a learned verifier has less to hold on to once the warp perturbs fine detail. The fundus re-identification critique showed 16x16 raw pixels already link 42.5% of patients because of near-identical images.

## Experiment (exact)
- Data / units: official-test patients with >= 2 images (1,424), strata by follow-up index gap; patients with several gaps contribute their closest pair to g = 1 and their furthest to g >= 4 (reported separately, no double counting inside a stratum).
- Scores: Siamese-R50 (H-CXR-002/-004 fits), registered NCC / SSIM, DINOv2 kNN; releases raw, PriCheXy mu = 0.01, random flow mu = 0.01, blur sigma 4.
- Metrics: stratified AUC with patient-cluster bootstrap (per-stratum n reported; the g >= 4 stratum is the smallest, state MDE80); near-duplicate flag NCC > 0.95 after affine registration on raw; AUC with and without flagged pairs.
- Test-unit design: no new fits; units and folds inherited (fixed 1,424 for PriCheXy; floors rotated).
- Controls: strata with matched ages; same analysis on random pairs (no gap dependence expected -> flat).

## Prediction and falsifier
- Predicted: raw drop <= 0.03; defended drop >= 0.08; near-duplicate share 15-35% of g = 1 positives.
- FALSIFIED if: raw drop >= 0.05 (raw linkage is itself partly near-duplicate matching: then a "far-gap-only" protocol is needed for every claim) OR defended drop <= 0.03 (no gap dependence; a single headline number is representative).

## Cost / dependencies
~0.5 GPU-h plus CPU; free once H-CXR-002/-003/-004 predictions exist.

## Prior
0.40. The two-part pattern (flat raw, steep defended) is specific; either part failing is also informative, which is why it is a control and not a claim.

## Cheapest floor that could kill it
Registered NCC alone (CPU) on g = 1 pairs: if it links raw pairs at AUC >= 0.95 the near-duplicate share is large and every raw-image attacker number is partly a duplicate-image number.
