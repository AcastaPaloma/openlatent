# H2-011: NEW FIELD N2. Cell Painting (JUMP): a label-free adapter on well-level profiles retrieves replicates of HELD-OUT compounds across laboratories better than inductive floors, at equal plate/source leakage

- Focus: candidate field (not previously killed) | Kind: headroom check first, learned arm second | Cost: ~1 CPU-h + 1 GPU-h check, ~4 GPU-h full | Prior: 0.15 (full claim), 0.70 (headroom check passes)
- Depends on: Arevalo et al. pipeline and well-level profiles (public, CC0 on the AWS Cell Painting Gallery, `cpg0016-jump`); H2-001 AE template; no image extraction (BatchBench images, 14-119 GB per subset, are deliberately not used).

## Statement
Use the five scenarios of the opened batch-correction benchmark (Arevalo et al.): scenarios 4 and 5 (5 laboratories, 3 imaging systems; 302 and ~82k compounds). Protocol: split compounds 50/50 into adapter-train and held-out compounds; fit the adapter on wells of train compounds only (label-free; the adversary targets plate id); evaluate on held-out compounds with (i) cross-source replicate-retrieval mAP (replicates of the same compound in OTHER laboratories), (ii) leakage = attackers (linear, MLP, kNN, LightGBM, retrained on the released profiles) predicting source (seen sources) and plate (held-out source, leave-one-source-out). Headroom check (floors and published methods only):
- Hj: within-source mAP of the held-out compounds (the no-cross-source-shift oracle) minus cross-source mAP of the best transductive method (Harmony, Sphering with negative controls, Seurat, Scanorama) >= 0.10;
- Hk: the best INDUCTIVE floor (PCA-d, per-profile robust standardisation, dithered PCA, LDA on train-compound identity) is >= 0.05 below that transductive method in cross-source mAP at plate-leakage <= L(method) + 0.05.
If Hj and Hk pass, the claim: the label-free adversarial AE (or cVAE) exceeds the inductive floors by >= +0.05 cross-source mAP at matched leakage and is within 0.03 of the transductive best, 3 seeds, bootstrap over compounds.

## Mechanism
Profiles are consumed for an open-ended list of tasks (replicate retrieval, mechanism of action, target, gene function) and the compound identity "label" has hundreds to tens of thousands of classes with 2-8 replicates, so a label-supervised projection is metric learning on seen compounds only and does not define the content needed for new compounds. The opened benchmark paper states that no method adequately removed batch effects once there were more than a few hundred compounds and more than one microscope type, and that Sphering needs negative controls in every batch while Harmony/MNN/fastMNN must reprocess all data to add new batches: i.e. exactly the transductive limitation an inductive adapter would remove. The risk is that Harmony/Seurat already sit close to the replicate-retrieval ceiling in the easier scenarios (scenario 1 showed no clear winner among ten methods).

## Experiment (exact)
- Stage 0: download profiles for scenarios 4 and 5 (size to be checked; the repository documents the Snakemake pipeline; R-only methods Seurat RPCA/fastMNN can be skipped, Python methods Harmony, Scanorama, Sphering, scVI run); fix compound split, source folds and the held-out-source plate definition before any metric.
- Stage 1 (check): references and floors above; metrics: mAP (replicate retrieval, cross-source, held-out compounds), kBET/LISI printed for context; leakage as above with plate-level and source-level attackers.
- Stage 2: AE+adversary, cVAE (H2-007), contrastive (H2-003) on the standardised profiles; dz in {16, 32, 64}; SEL-V on a validation slice of held-in compounds; 3 seeds; leave-one-source-out for the 5 sources.
- Controls: shuffled plate labels at chance; random profiles at chance mAP; negative-control wells excluded from the mAP positives; near-duplicate wells across splits removed.
- Comparators named in advance: inductive floors above (kill) and transductive Harmony/Sphering (reference only).

## Prediction and falsifier
- Prediction: Hj passes (within-source minus cross-source mAP >= 0.10 for the hard scenarios); Hk passes (0.05-0.15); learned arm +0.03 to +0.07 over inductive floors, 0.02-0.06 below the transductive best.
- Falsified (NO-GO at the check) if Harmony/Sphering are within 0.05 of the within-source oracle, or the inductive floors are within 0.03 of them. Falsified (full claim) if the learned arm gains < +0.03 mAP over the best inductive floor in >= 3 of 5 leave-one-source-out folds.

## Cost
Profiles are tabular: the check is CPU-bound ~1 h plus Harmony/scVI references; adapters ~4 GPU-h across folds and seeds.

## Dependencies
Data download (CC0 per the opened page; per-scenario sizes unverified); the Arevalo scripts for metric definitions.

## Prior and why
0.70 that headroom exists (the opened paper's own conclusion), 0.15 for the claim: the same programme pattern would put a learned adapter at or below Harmony/Sphering-class methods, and replicate retrieval for held-out compounds is dominated by perturbation strength, not source removal.

## Sources
[O] PMC11297288 Arevalo et al. (scenarios, methods, metrics, conclusions on batch removal; code carpenter-singh-lab/2023_Arevalo_BatchCorrection); [O] arXiv 2610.30104 BatchBench abstract (JUMP-CP leaderboard, no numbers on the abstract page); [S] literature/other_vision.md and contrarian_stress_test.md.
