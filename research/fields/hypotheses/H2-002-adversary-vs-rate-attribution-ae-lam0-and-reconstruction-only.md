# H2-002: The adversary is not the active ingredient of the task-agnostic AE: a rate-limited reconstruction code with lam = 0 recovers >= 80% of the gain, and beats PCA-d only by nonlinearity

- Focus: A + C | Kind: attribution control for the AE lead | Cost: ~3 GPU-h (A 2, C 1) | Prior: 0.35
- Depends on: H2-001 grid (lam = 0 rows are in it), C's `c_ae_d{16,32}_l{0,3,10}_s{0,1,2}` (queued, 3 GPU-min each), H2-014 for the PCA/dither comparators.

## Statement
At matched leakage, the area under the (L, U_held) frontier of the AE with lam in {3, 10} exceeds that of the same AE with lam = 0 by <= 0.02 (adversary adds nothing beyond the rate limit) in >= 3 of 4 A cells; and the lam = 0 AE exceeds PCA-d (the linear, rate-matched reconstruction code) by >= 0.03 AUC in >= 3 of 4 cells (the nonlinear decoder, not the adversary, is what keeps held-out structure). The same ordering holds on CREMA-D held-out emotions (C): U_HE(AE lam 3/10) - U_HE(AE lam 0) <= 0.02 at equal identity leakage.

## Mechanism
FOCUS lesson from KTH and every later field: rate does most of the work and the adversary only adds a bit (programme finding 2; C: adversarial d16 vs rate d16 identity -0.013 [-0.027, +0.003], not significant). The AE with lam = 0 is a beta-VAE-like reconstruction code with a Gaussian rate channel: capacity is the only privacy mechanism. Seed 0 of A already contains one such point (midnight/TCGA `ae__beta0.001_dl8_lam0.0`, L 0.29, +0.146 [0.010, 0.271] over `lda_pca__d8`) and the dinov2-L/Tolkach V-selected AE had lam = 0. If the adversary is a placebo here, the "adversarial" framing of the lead is wrong and the right comparator family is "nonlinear PCA + dither" (H2-014).

## Experiment (exact)
- A cells: midnight and phikon-v2 x {TCGA, Tolkach}, held-out-class protocol, seeds 0-2.
- Arms (all label-free): R0 AE lam = 0 (beta in {1e-3, 1e-2}); R1 AE lam in {3, 10}; R2 denoising AE (input dropout 0.3, lam = 0); R3 linear AE = PCA-d; R4 deterministic bottleneck (beta -> 0, no noise). Ladder dz in {4, 8, 16, 32, 64} to trace the frontier in L.
- Metric: area under the monotone upper envelope of U_held vs L over L in [0.1, 0.5] (trapezoid, per seed), plus retention-xc AUC; bootstrap over sealed units, paired across arms (shared resamples). Normalisation: L as in A.md (chance and raw), U_held as balanced accuracy, raw held-class accuracy and the LDA arm printed alongside.
- C: CREMA-D WavLM-Base+ mean+std, `c_ae_*` configs (dz 16/32, lam 0/3/10), U_HE = recall on never-seen DIS+FEA, identity T1 worst attacker, 4 actor-disjoint folds, 3 seeds, bootstrap over 91 actors.

## Prediction and falsifier
- Prediction: AUC(R1) - AUC(R0) in [-0.01, +0.02]; AUC(R0) - AUC(R3) in [+0.03, +0.08]; R2 within 0.02 of R0; C: |U_HE(lam 10) - U_HE(lam 0)| <= 0.02.
- Falsified (a) if AUC(R1) - AUC(R0) >= +0.05 in >= 2 A cells (then the adversary is real and H2-003/H-X-002 on adversary design become worth running) or (b) if AUC(R0) - AUC(R3) < +0.01 in >= 3 cells (reconstruction nonlinearity adds nothing over PCA; then the AE lead reduces to PCA + noise, which H2-014 would confirm).

## Cost
A: 4 cells x 3 seeds x (24 configs ~12 min) ~2 GPU-h plus the extra dz ladder; C: 18 AE jobs x ~3 GPU-min ~1 GPU-h.

## Dependencies
Builder A p2 harness; Builder C `ae.py` (exists).

## Prior and why
0.35 for the conjunction (each half ~0.55-0.6). The adversary-adds-little half agrees with C's seed-0 result and with the programme's repeated finding; the nonlinear-beats-PCA half is the one the A seed-0 numbers support (+0.146 at lam 0) but they are post hoc.

## Sources
results_notes/A.md PROTOCOL 2 item 4; results_notes/C.md 2c and section 3 (adv vs rate); literature/methods_transfer.md (sceptic paragraph: rate-matched no-adversary control).
