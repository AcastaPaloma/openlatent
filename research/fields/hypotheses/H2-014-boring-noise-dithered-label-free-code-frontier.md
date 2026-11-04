# H2-014: BORING CONTROL. A noise-dithered label-free code (PCA-d or LDA+PCA-d plus Gaussian noise, or PCA-d followed by closed-form centre erasure in the low-dimensional space) matches the AE at equal leakage

- Focus: A (applies to every field with a label-free latent) | Kind: boring control, can kill the AE lead | Cost: ~1 GPU-h (+~8 CPU-h) | Prior: 0.35 that it kills (the dithered code is within 0.03 of the AE in >= 3 of 4 cells)
- Depends on: `adapters.py` floors (PCA-d, `lda_pca__d`, random projection exist); no learned model needed.

## Statement
Release z = z0 + sigma * eps, eps ~ N(0, I), sigma_j = r x std(z0_j) per dimension, where z0 is PCA-d (d in {16, 32, 64}), `lda_pca__d` (d in {16, 32, 64}) or random projection-d, r in {0.25, 0.5, 1, 2, 4}, fitted on adapter-train units only; noise is applied to every released vector (attacker training, utility-probe training and test; 4 draws per training unit so the attacker can average). A second deterministic stack: PCA-d (d in {16, 32}) then LEACE of the centre fitted on A units in the d-dimensional space (n_A >> d, unlike the 2048-d fit that failed out of sample). On midnight and phikon-v2 x {TCGA, Tolkach}, held-out-class protocol, the best such floor selected on V by SEL-V satisfies |U_held(floor) - U_held(V-selected AE)| <= 0.03 at L within +-0.05 of the AE, in >= 3 of 4 cells, and the same for retention-xc within 0.05.

## Mechanism
D's headroom stage found that the strongest floor in the vector-space class was Gaussian noise (r = 1: L 0.633 at retention 0.805), ahead of PCA and random projection, and that LEACE was the weakest; in A the existing label-free references (PCA-d, `lda_pca__d`, whitening) are pure projections without dither, so the leakage-matched comparison offers the AE a curve with few points near L 0.2-0.4 and nothing in between them. Dithering moves a rich code (d = 32, retention-xc 0.6-0.8) down the leakage axis at a cost in utility that is not obviously larger than the AE's. A rate-limited Gaussian channel is, in effect, what the AE's beta term is; the question is whether training adds anything to "PCA + the same channel".

## Experiment (exact)
- Cells, splits, encoders, attackers (A-SUITE6, retrained on B with noisy releases), L_worst, U_held, retention-xc, bootstrap over sealed units: exactly H2-001.
- Arms: (i) dithered PCA-d, (ii) dithered `lda_pca__d`, (iii) dithered random projection, (iv) PCA-d + low-dimensional LEACE(centre) (ridge chosen on V), (v) PCA-d + low-dimensional `centre_whiten` computed on A units only (inductive version), each with the r ladder; AE (V-selected) from H2-001.
- Comparison: the monotone envelope of each family on (L, U_held); selection of one point per cell by SEL-V; paired bootstrap difference at the AE's L +- 0.05.
- Controls: dithered constant vector at chance; dither applied only to attacker input but not to utility (an illegal asymmetric variant) is reported only to show the size of the artefact.

## Prediction and falsifier
- Prediction (control kills): U_held(best dithered) within 0.03 of the AE in 3 of 4 cells; retention gap <= 0.05; the low-dimensional LEACE stack reduces linear leakage to chance but leaves L_worst >= 0.5 (nonlinear attackers), so it is not the winner.
- Falsified (the control fails and the AE lead stands against the strongest floor class) if the AE exceeds every dithered family by >= +0.04 U_held with lower CI > 0 in >= 3 of 4 cells.

## Cost
~1 GPU-h for the PCA/embedding side; the attacker evaluation is CPU-bound (~8 h for 4 cells x 3 seeds x ~60 configs).

## Dependencies
Builder A harness; independent of the AE (can run first and is a prerequisite for the strongest comparator in H2-001 and H2-006).

## Prior and why
0.35. Noise was the winning floor in D and the A+C literature calls a rate bottleneck without adversary the mandatory control; but the A seed-0 gain is concentrated on midnight where the label-free curve between L 0.2 and 0.4 is empty, so some of the AE gain is probably curve-fill, and some of it nonlinear structure that Gaussian dither cannot imitate.

## Sources
results_notes/D.md section 5.1 (noise as strongest floor at tau 0.70; LEACE weakest); results_notes/A.md PROTOCOL 2 items 3-4 (references used for leakage matching).
