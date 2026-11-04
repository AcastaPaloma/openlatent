# H2-015: BORING CONTROL. The "best of 10 AE variants picked at test, leakage-matched to the best reference" procedure inflates a null family by as much as +0.05 to +0.09 on 12-24 test units

- Focus: A (applies to any post hoc pick in the programme) | Kind: boring control, calibrates the lead | Cost: 0 GPU-h (~2 CPU-h) | Prior: 0.35 that the null's 95th percentile reaches the observed gain on midnight/Tolkach
- Depends on: saved predictions in `results/a_p2h_*/preds`; H2-014 families for the null configurations.

## Statement
Apply the exact procedure that produced the seed-0 lead (A.md PROTOCOL 2 item 4: pick among 10 configurations of one family at test; for each, compare held-class utility with the best reference having L <= L_config + 0.05) to a NULL family in which no member is better than any other: ten dithered-PCA configurations (H2-014) whose (U, L) points lie on one frontier. Bootstrap the sealed units (B = 1000, units resampled jointly for all configurations and references, so correlation between arms is preserved), recompute the statistic M = max over the 10 configurations of Delta_c, and take its distribution. Then:
- on midnight/Tolkach (12 test cases) the 95th percentile of M is >= 0.04, so the observed +0.083 [0.048, 0.126] is only about twice the selection artefact;
- on midnight/TCGA (24 test cases) the 95th percentile of M is >= 0.08, so the observed +0.164 [0.040, 0.284] is within a factor of two of the artefact;
- the V-selected number of H2-001 (no best-of-N on C) is NOT subject to this inflation and is the one that counts.

## Mechanism
With 12 test cases and held-class probes that differ by single-case flips, the paired difference of two points on the same frontier has a standard deviation of several points; taking the maximum over 10 configurations with a baseline that is itself the max over many references (a favourable-to-baseline matching) is a biased-up statistic whose bias scales as sd x sqrt(2 ln 10) ~ 2.1 sd. The existing CIs are bootstrap intervals for a fixed pair, not for the selected pair. This is the Focus A error we have already made once (picking the best test point), now bounded numerically.

## Experiment (exact)
- Null family: ten (d, r) pairs of dithered PCA/LDA+PCA with L from 0.1 to 0.5 (H2-014), same cells as A.md PROTOCOL 2 (dinov2-L and midnight x {TCGA, Tolkach}, seed 0 preds) and the new cells when they exist.
- Procedure P (as written above) is coded once; the references for the null are the OTHER nine members plus `lda`, `pca__d`, `centre_whiten`; M computed on 1000 joint bootstrap resamples of sealed units.
- Also compute P on the real AE family (10 variants) to get the distribution of its M, and the V-selected single-config Delta as in H2-001.
- Reported: 50th/95th percentile of M for the null; observed Delta; the fraction of null resamples with M >= observed.

## Prediction and falsifier
- Prediction: null M 95th percentile 0.04-0.09 (Tolkach) and 0.08-0.16 (TCGA); p(null M >= observed) = 0.05-0.30 in both cells.
- Falsified (control fails, lead stands) if the observed Delta exceeds the null 95th percentile by >= 0.03 in both cells. Confirmed (control kills the seed-0 reading) if p(null M >= observed) >= 0.2 in either midnight cell.

## Cost
No GPU; CPU bootstrap ~2 h.

## Dependencies
Per-unit prediction files (exist for seed 0 in `preds`); H2-014 null configs.

## Prior and why
0.35. The observed intervals are narrow in Tolkach because both arms are near the ceiling (held-class 0.99 vs 0.91), which makes the null spread smaller than my arithmetic suggests; TCGA is where the lead is least secure.

## Sources
results_notes/A.md PROTOCOL 2 item 4 (post hoc, 10 AE variants, single seed); FOCUS.md Focus A (selection on validation, sealed test).
