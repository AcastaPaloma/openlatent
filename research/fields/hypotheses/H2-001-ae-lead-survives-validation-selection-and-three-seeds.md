# H2-001: The task-agnostic adversarial autoencoder gain over leakage-matched inductive floors survives validation-only selection, 3 seeds and two more encoders

- Focus: A (PathoROB) | Kind: replication / strengthening of the only unfalsified lead | Cost: ~5 GPU-h (+~30 CPU-h attacker evaluation) | Prior: 0.30
- Depends on: `a_p2h_*_s1|s2` (queued), `run_sweep.py --grid p2 --holdout`, `aggregate_p2.py`; H2-014 (dithered floors) for the comparator set; phikon-v2 / phikon embedding caches (exist).

## Statement
Take the `ae` family of Builder A (autoencoder through a rate-limited Gaussian channel, no class labels, centre adversaries retrained from scratch each round). Select ONE configuration per (encoder, dataset, seed) on the validation split V only (rule SEL-V in INDEX.md), then report it on the sealed split C. Then, on the held-out-class protocol (encoder sees half the classes; utility is a linear probe on the unseen classes):

- in >= 3 of 6 test cells (midnight x {TCGA, Tolkach}, phikon-v2 x {TCGA, Tolkach}, phikon x {TCGA, Tolkach}) the 3-seed mean of Delta_held = U_held(ae) - U_held(best inductive floor at L <= L_ae + 0.05) is >= +0.04 with a unit-bootstrap 95% lower bound > 0, AND the retention-xc difference is >= 0;
- the gain over the transductive per-centre whitening reference (needs centre ids at release, a different threat model) is >= +0.015 in midnight/Tolkach;
- the V-selected configuration lies in the same family (same dz and beta, lam within one grid step) in >= 2 of 3 seeds in each winning cell;
- dinov2-L (negative encoder) shows Delta_held <= +0.02 in both datasets, as in seed 0.

## Mechanism
Seed 0 (results_notes/A.md, PROTOCOL 2, post hoc, best of 10 AE variants picked at test): midnight/Tolkach `ae__beta0.001_dl32_lam10.0` at L 0.27 held-class 0.993 vs 0.910 for plain LDA (+0.083 [0.048, 0.126]), retention +0.243; midnight/TCGA +0.164 [0.040, 0.284]; dinov2-L none. The label-supervised LDA projection collapses onto its training-label subspace (held-class 0.910 vs raw 1.000 on midnight/Tolkach), the reconstruction term of the AE does not, and the adversary removes centre directions that reconstruction would otherwise copy. Where the raw held-class ceiling is already low (dinov2-L/TCGA raw 0.633, floors 0.647) there is nothing to win. Selection on test and a single seed can manufacture a +0.08 on 12 test cases, hence this replication.

## Experiment (exact)
- Data/encoders: PathoROB TCGA (4 classes, seen = 2, held = 2) and Tolkach (6 classes, seen = 3, held = 3); encoders midnight_clsmean (replicate), phikonv2_clsmean, phikon_clsmean (new), dinov2L_clsmean (negative control). Splits, A/B/V/C unit-disjoint, sealed C: exactly as A.md pre-registration.
- AE grid (all run, none dropped after looking): dz in {8, 16, 32}, beta in {1e-3, 1e-2}, lam in {0, 1, 3, 10} = 24 configs, 12 adversary rounds as `ae.py`.
- Seeds: encoder seeds 0, 1, 2 (jobs `a_p2h_*_s1|s2` already queued for the four midnight/dinov2 cells; add phikon-v2 and phikon). Attacker suite A-SUITE6 (linear, MLP, kNN, FT-transformer, LightGBM, RFF-SVM), retrained on B; L_worst normalised to raw as in A.md.
- Comparators (rule LM-rule): inductive floors = PCA-d, random projection-d, `lda_pca__d` (d in 4..128), noise-dithered PCA-d (H2-014); reported separately: `centre_whiten` (transductive) and the label-supervised learned families `vib, vq, dann, adv_retrain, supcon_xc` at their V-selected configs.
- Metrics: U_held (balanced accuracy of a linear probe retrained on B restricted to held classes, scored on C and on the benchmark's unseen centres), retention-xc, L_worst, slide-fingerprint leak (H2-005 definition) reported as a side column. Bootstrap over sealed units (cases/slides), B = 1000, seed-mean inside each resample; seed spread reported separately.
- Controls that must hold: constant and noise embeddings at chance, orthogonal shuffled-centre control at chance, raw above chance (as in A.md).

## Prediction and falsifier
- Prediction: Delta_held(I-floor) >= +0.04 with lower CI > 0 in 3-4 of 6 test cells, concentrated on midnight and phikon-v2; whitening gap +0.015 to +0.03 on midnight/Tolkach; same-family selection in >= 2 of 3 seeds; dinov2-L <= +0.02.
- Falsified (lead dead, stop AE work in A) if the 3-seed mean Delta_held(I-floor) < +0.02 in midnight/Tolkach and midnight/TCGA, or <= 1 of 6 cells passes. Two passing cells of six is inconclusive and routes to the cross-fitted design of H2-016, not to a claim.

## Cost
Seeds 1-2 midnight/dinov2 (8 jobs, ~0.2 GPU-h each, CPU-bound attackers) ~1.6; phikon-v2 + phikon 4 cells x 3 seeds ~2.4; widened AE grid +~1 GPU-h. Total ~5 GPU-h and ~30 CPU-h.

## Dependencies
Builder A harness only. Needs H2-014's dithered floors for the stricter comparator; if absent, report both with and without.

## Prior and why
0.30. The seed-0 effects are large relative to their CIs on midnight, but the gain was found after picking among 10 variants at test, Tolkach has 12 test cases, and the T-reference (whitening) is only 0.021 behind. Requiring V-selection, 3 seeds and two further encoders conjoins four ~0.7 events.

## Sources
results_notes/A.md (PROTOCOL 2 seed 0), FOCUS.md (A kill rule, revised), journal 2026-11-04 ~11:50Z.
