# H2-016: BORING CONTROL. Power audit: at the present sealed-unit counts the minimal detectable difference in leakage-matched held-class utility is >= 0.10 on TCGA cells and ~0.05-0.06 on Tolkach cells, and matched leakage itself is resolved only to +-0.10 to +-0.20

- Focus: A + C (instrument validity) | Kind: boring control / decision instrument | Cost: 0 GPU-h analysis, ~5 GPU-h for the cross-fitted remedy | Prior: 0.70
- Depends on: stored per-unit predictions (`results/a_p2h_*/preds`, HF `runs/a_*`, C per-actor counts); then H2-001 for the remedy.

## Statement
From the intervals already printed in results_notes/A.md (paired unit bootstrap, sealed C = 20 slides Camelyon, 48 cases TCGA, 12 cases Tolkach; C: 91 actors), define MDE80 = 2.8 x (paired bootstrap SE of the difference). Then:
- midnight/TCGA leakage-matched Delta_held (+0.164 [0.040, 0.284], SE ~0.06): MDE80 ~0.17; midnight/Tolkach (+0.083 [0.048, 0.126], SE ~0.02): MDE80 ~0.06; dinov2-L cells are in the same range (Delta CI half-widths 0.03-0.10);
- the L_worst CI half-width is >= 0.15 in the Tolkach cells (e.g. 0.27 [0.05, 0.47]) and ~0.10 in TCGA cells, so a leakage-matching tolerance of 0.05 is below the resolution in every A cell;
- consequently a claim of "+0.04 over a leakage-matched floor" cannot be adjudicated from one split in any A cell, and only repeated unit-disjoint cross-fitting (H2-001's remedy: 4-5 folds so every non-training unit is a test unit once) brings MDE80 to <= 0.04 (Tolkach) and <= 0.10 (TCGA).
In C (91 actors, 4 folds, bootstrap over actors) MDE80 for U_HE differences between a learned and a LDA comparator is ~0.04-0.05, so the pre-registered +0.05 criterion is at the detection boundary.

## Mechanism
Held-out-class cells split a small number of independent units (cases or slides) into train, probe, validation and test; the unit, not the patch, is the independent observation, and the held-class probe is evaluated on a subset of those units; matching on L adds a second noisy coordinate. This is arithmetic, not a model claim, and it bears on how every earlier "pass" and "fail" in A is read (all single seed).

## Experiment (exact)
- Step 1 (CPU): for every learned-vs-reference pair in A.md PROTOCOL 2 (seed 0 and, when present, seeds 1-2) compute the paired-bootstrap SE of Delta_held, Delta_retention and Delta_L; compute MDE80; also simulate power by injecting a known shift delta in {0.02, 0.04, 0.06, 0.08, 0.10} into the held-class probe correctness at the unit level and recording how often the SAME decision rule (lower CI > 0) fires.
- Step 2: repeat for C (U_HE differences, paired over actors) and for D's retention CIs (queries 323).
- Step 3 (remedy, GPU): cross-fitted evaluation with K = 4 unit-disjoint folds (sealed test rotates; adapters refit per fold; selection on V inside the fold) for midnight and phikon-v2 x {TCGA, Tolkach}.
- Reported: MDE80 table, the delta at which power reaches 0.8, and which previous pass/fail statements are inside the undetectable zone.

## Prediction and falsifier
- Prediction: MDE80 >= 0.10 in both TCGA cells, 0.05-0.07 in both Tolkach cells; L half-width >= 0.15 in Tolkach; cross-fitting (K = 4) reduces MDE80 by 35-50%.
- Falsified if MDE80 <= 0.05 for the held-class Delta in >= 3 of 4 cells and L half-width <= 0.10 (the instrument is adequate and the single-split cells can adjudicate).

## Cost
Analysis: ~2 CPU-h. Cross-fit remedy ~5 GPU-h (4 cells x 4 folds x 3 seeds ~0.1 GPU-h per AE grid slice).

## Dependencies
Per-unit predictions on disk (they exist for the p2 runs).

## Prior and why
0.70. The intervals in A.md already imply it; this hypothesis makes the implication explicit and sets the decision threshold before more data arrives. Ceiling cells (held-class near 1.0) have small SE, so the Tolkach MDE may be lower than I state.

## Sources
results_notes/A.md (all PROTOCOL 2 intervals; cell sizes in the gate section item 6); results_notes/C.md (bootstrap over 91 actors); results_notes/D.md (323 test queries).
