# H-X-002 (A and C): Minimax against a restarted, diverse population of attackers lowers worst-case leakage on held-out attacker families by >= 0.10 vs a single adversary

- Focus: A + C | Kind: learned method (population minimax) | Cost: ~6 GPU-h | Prior: 0.40
- Depends on: shared adversary module (X-001); attacker families held out of the training population implemented in the probe suites.

## Statement
Train the encoder against a population P of K = 6 adversaries (logistic, MLP-2x512, MLP-4x1024, soft-kNN with learned temperature, RFF-kernel logistic, and a per-focus sequence/attention head for C) with the encoder loss a log-sum-exp (temperature tau) of the adversaries' accuracies, and each adversary re-initialised with probability 0.1 per epoch plus a full warm-up epoch on the frozen encoder. At d = 64 and U >= 0.97 the population-trained encoder has L_worst on *held-out* attacker families (LightGBM, RBF-SVM with different gamma, 1-NN, a Transformer or x-vector TDNN not in P) at least 0.10 lower than a single-MLP adversary, in both A and C; the single adversary shows the fool-the-adversary pattern and the population one does not.

## Mechanism
Our video journal: a VAE "fooled" frequently-restarted adversaries while fresh attackers reached 75.6% (H18, journal 13:20-13:55), and a fix (adversary kept near convergence, full warm-up) helped partly. Fairness-adversary literature uses ensembles of diverse discriminators with orthogonality penalties for stability and reduced bias (Han, Baldwin, Cohn, 2101.10001, abstract opened); intermittent re-initialisation to decorrelate trajectories appears in null-sampling fair representations (2008.05248, search snippet only, not opened). Kernelised erasure protects only against the adversary kernel it was trained with and "does not transfer to different nonlinear adversaries" (2201.12191, abstract opened): the held-out-family evaluation is exactly the right test of whether diversity generalises or merely enlarges the training family.

## Experiment (exact)
- Data: A: PathoROB phikonv2_clsmean, secret = centre; C: CREMA-D wavlm-base-plus pooled (or frames for the sequence head), secret = actor. Training secret labels from train centres/actors; evaluation on unseen centres/actors (source-probe in A; T1 on test-fold actors in C).
- Arms: single MLP adversary (X-001 recipe); population P (K=6) with tau in {0.1, 1}; population with random-label "decoy" adversaries removed; LEACE; rate-matched bottleneck (no adversary).
- Held-out attacker families for evaluation only: LightGBM, RBF-SVM (RFF, gamma != the one in P), 1-NN, extra-trees, Transformer (A: tokens = 8 chunks of z) / x-vector TDNN (C). Report the in-P attackers separately.
- Seeds: 3 encoder x 3 attacker; A: 3 datasets; C: 4 folds; bootstrap over slides/actors.
- Metric: L_worst(held-out families), L_worst(in-P), the in-training population accuracy, U.

## Prediction and falsifier
- Prediction: held-out-family L_worst: single adversary ~0.45-0.6, population ~0.3-0.45 (difference >= 0.10 in both foci); in-P L_worst similar for both (the benefit is generalisation, not training-family fit); compute 2.5-3.5x single adversary. The rate-matched bottleneck at d=64 sits between the two or equals the population.
- Falsified if the held-out-family difference is < 0.04 in either focus (diversity only enlarges the training family; leakage is attacker-family specific) or if the population arm destabilises (U < 0.9 U_ref).

## Cost
~6 GPU-h: A 2, C 4 (population training ~3x single, 3 seeds, d=64 and one lower d).

## Dependencies
Builder-A, Builder-C adversary loop; a differentiable soft-kNN and RFF logistic head.

## Prior and why
0.40. Known gain direction in fairness (2101.10001, in-family), but the held-out-family claim is the hard one and the speech/pathology embeddings are high-capacity enough that a deeper held-out model can always find residual structure.

## Sources
[O] 2101.10001; [O] 2201.12191; [snip] 2008.05248; our journal H18; [S] 2609.00390 NeuroPriv (adversarial identity 0.692 -> 0.206, still 7x chance).
