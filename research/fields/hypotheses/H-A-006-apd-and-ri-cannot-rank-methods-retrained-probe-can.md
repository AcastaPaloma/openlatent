# H-A-006: APD and RI cannot separate raw / LEACE / ComBat / learned adapters on good FMs; the retrained centre probe separates them by >= 0.4 and disagrees with RI on rank

- Focus: A | Kind: metric / protocol hypothesis (justifies the whole headroom framing) | Cost: ~2 GPU-h + ~10 CPU-h | Prior: 0.60
- Depends on: adapters from H-A-002, H-A-003, H-A-004 (any 6+ distinct methods); Builder-A's re-implementation of RI, APD, clustering score (checked against the leaderboard first).

## Statement
Take at least 6 transformed versions of the same embeddings (raw, LEACE, ComBat, per-centre z-score, best bottleneck, best adversarial adapter, best stain-pair adapter) on the top-3 ungated FMs. Then:
(a) APD (ID and OOD) differs by < 1.5 points between the best and worst method, and the 95% CIs (20 repetitions, bootstrap over slides) overlap for >= 80% of method pairs;
(b) RI differs by < 0.08 across methods on Tolkach and TCGA (saturated: top models are at 0.88-0.96);
(c) L_worst (retrained probe) spans >= 0.4 across the same methods; and Kendall tau between the RI ranking and the L_worst ranking is <= 0.5.

## Mechanism
Published mitigation effects on APD are small: Reinhard +1.11 points, DANN +0.23, Reinhard+DANN +1.30, with "no method fully eliminated generalization drops" (2507.17845, HTML opened). RI is a kNN neighbourhood statistic whose cross-model comparability is disputed (CRoMa, 2607.25497, abstract opened: "structural limitations that make cross-model comparison unreliable"); top models are at ceiling (leaderboard in literature/contrarian_stress_test.md 1.1). The retrained-probe leakage measures a different quantity (capacity-bounded recoverability). The stress-test synthetic showed RI and a retrained probe can rank methods differently (LDA beat linear erasure on RI, 0.629 vs 0.582, while its retrained centre accuracy stayed 0.49 vs chance 0.25).

## Experiment (exact)
- Data: PathoROB, 3 datasets, encoders phikonv2/phikon/midnight (UNI2-h if available). First reproduce the published RI of Phikon, Phikon-v2 within 0.01-0.02 per dataset (stress-test replication check).
- Methods: those listed in the statement, produced by H-A-001..004 plus ComBat (neuroCombat/pycombat, centre = batch) and per-centre z-score (centre ID oracle).
- Metrics: RI (PathoROB code, k as in the repo), APD ID/OOD with the repo's 20 repetitions and seeds, clustering score (50 k-means inits), L_worst from A-SUITE (slide-disjoint 60/10/30).
- Statistics: bootstrap over slides inside the 20 repetitions; Kendall tau over methods per encoder with CI by bootstrapping methods x datasets.
- Normalisation: APD in the paper's relative-percentage form; L_worst vs raw/chance as in H-A-001.

## Prediction and falsifier
- Prediction: max APD spread < 1.5 points with overlapping CIs; RI spread < 0.08 on Tolkach and TCGA, up to 0.2 on Camelyon for the weakest FM only; L_worst spread >= 0.4 (raw 1.0 vs best <= 0.6); tau(RI, L_worst) between -0.2 and 0.5.
- Falsified if APD separates the best from the worst method by >= 2.5 points with non-overlapping CIs on >= 2 FMs (APD is informative; the probe is not the only axis), or tau >= 0.8 (RI agrees with the probe; no new information in the probe).

## Cost
~2 GPU-h for adapters (shared with other hypotheses) + ~10 CPU-h for 20 repetitions x 8 methods x 3 FMs x 3 datasets of logistic regressions.

## Dependencies
Builder-A: metrics port verified against leaderboard values; H-A-001..004 outputs.

## Prior and why
0.60. APD saturation is documented; the uncertain part is (c) tau: with only 6-8 methods tau has wide CIs and RI may correlate with the probe more than the synthetic suggested.

## Sources
[O] 2507.17845; [O] 2607.25497 CRoMa; [O] 2609.28231 (new AUCC metric, ComBat inconsistent); literature/contrarian_stress_test.md 1.1, 1.5.
