# H-CXR-002: On matched negatives a retrained image Siamese beats the best demographic + device attacker by at least 0.03 AUC on RAW images (stage-0 gate of R-CXR)

- Focus: R-CXR | Kind: headroom check / stage-0 kill gate | GPU-h: ~3 (Siamese ResNet-50 at 256 px, 3 seeds) | Prior: 0.85
- Depends on: H-CXR-001 (estimators, MN-v1), the ChestX-ray14 download gate, PriCheXy pair lists / training recipe. This is the stress test's stage 0: "kill if real-image matched-negative AUC does not exceed the metadata floor by >= 0.03 (paired MDE80 at 1,424 patients)".

## Statement
There is identity signal beyond demographics and acquisition to defend. On MN-v1 matched negatives, over the 1,424 official-test patients with >= 2 images, a Siamese ResNet-50 retrained on raw 256-px images reaches AUC 0.88-0.97 and exceeds max(true-metadata GBM, estimated-demographic GBM from H-CXR-001) by >= 0.03 AUC (paired patient-bootstrap CI excludes 0.03) in 3 of 3 seeds. Rank-1 in a 1,424-patient gallery is >= 20x the chance rate of 1/1,424 (0.07%) and >= 3x the metadata line's Rank-1.

## Mechanism
Packhauser et al. report verification AUC 0.994 on the full ChestX-ray14 and transfer across datasets and time gaps over ten years, which cannot be demographics; chest anatomy, spine curvature, vascular and clavicle geometry are individual. Under MN-v1 the demographic line is removed by construction; a residual Siamese AUC is identity (plus acquisition residuals not matched by the CSV fields, which H-CXR-005 and H-CXR-007 audit).

## Experiment (exact)
- Data / units: ChestX-ray14 256 px, unit = patient, 1,424 official-test patients with >= 2 images; MN-v1 pairs (one positive, one matched negative per patient; positives = two distinct visits; Follow-up index differs).
- Attackers: Siamese ResNet-50 (two shared-weight trunks, BCE on pair embedding, PriCheXy recipe, 3 seeds, training on the official train patients with MN-v1-style matched pairs AND a second variant trained on random pairs (the standard protocol) evaluated on MN-v1, to show training-protocol sensitivity); true-metadata GBM; estimated-demographic GBM (H-CXR-001); acquisition fingerprint (H-CXR-005).
- Metric: AUC, patient-cluster bootstrap; paired difference to the best demographic line; Rank-1 / Rank-5 in the full 1,424 gallery (matched on sex and view for the gallery: report both unconstrained and constrained galleries).
- Test-unit design: no defender is trained here, so rotation is free: 5 patient-hash folds over the 13,302 multi-visit patients (Siamese retrained per fold, 3 seeds inside each fold), pooled out-of-fold n = 13,302 as the primary number, with the official-test 1,424 as the headline fold; paired MDE80 ~0.02 at 1,424 and ~0.011 at 4,707 patients (Hanley-McNeil, rho = 0.6, ANALYTIC).
- Controls: constant image and shuffled-patient-label Siamese -> AUC 0.5; metadata line on MN-v1 <= 0.56 (validity).

## Prediction and falsifier
- Predicted: raw Siamese on MN-v1 AUC 0.92 [0.85, 0.97]; best demographic line <= 0.56; gap >= 0.30, far above the 0.03 gate; Rank-1 >= 30%.
- FALSIFIED (STOP R-CXR at this gate) if the Siamese-minus-best-demographic gap is < 0.03 AUC: then no identity beyond demographics exists to defend, and the stress test's by-product (the protocol shortcut) is the result. Also falsified if the random-pair-trained Siamese on MN-v1 collapses toward the demographic line (< 0.65): it learned the shortcut, not identity, and the PriCheXy protocol is uninformative.

## Cost / dependencies
~3 GPU-h; needs the download gate and H-CXR-001; uses the PriCheXy recipe (GPL-3 code in a separate process).

## Prior
0.85. The founding result (0.994 AUC) is hard to explain by demographics; the risk is acquisition and near-duplicate effects not removed by MN-v1 (H-CXR-005, H-CXR-007).

## Cheapest floor that could kill it
The metadata + device GBM on MN-v1 (CPU, minutes, the CSV already on hand). If it exceeds 0.60 the matching is leaky and the gate is not yet valid.
