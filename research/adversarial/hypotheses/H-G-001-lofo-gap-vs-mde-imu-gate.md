# H-G-001: IMU gate: a leave-one-attacker-family-out gap exists, but a defender trained on a population does not beat one trained on the single strongest attacker with restarts

- Focus: G (conditional third; gate <= 6 GPU-h shared with H-G-002) | Kind: gate / learned method vs three controls | GPU-h: ~3 (cached MotionSense features; 36 defender trainings x 4 methods ~1 min each, attacker retrains) | Prior: 0.12 that G opens (population defender beats the floor envelope by >= MDE on the held-out family in >= 3 of 4 rotations); 0.70 that the gap Gamma exists
- Depends on: H-IMU-001 (activity-LDA-k + dither frontier, RF identity at 8x chance on LDA-5 as MEASURED headroom), H-S-002 (MDE), H-IMU-003 (trial-held-out protocol). G does not start until these pass the charter's rule: "if the gap is inside the MDE, G stays closed".

## Statement
Fix four structurally different attacker families in advance: F1 linear / MLP on hand features {LR, LDA, MLP}; F2 distance {kNN cosine, kNN Canberra, nearest centroid}; F3 trees {RF, LightGBM}; F4 sequence / deep conv on raw windows {1D-CNN, GRU}. Train a small learned encoder (hand features 252-d -> z of d = 8; activity CE + in-loop adversaries) against three families and test against the fourth, rotating (4 rotations). Define the LOFO gap Gamma = L_HO(D_pop) - L_in(D_pop) (held-out-family normalised leakage minus in-family leakage). Predictions: (1) Gamma exists: >= 10 points of trial-level identity in >= 2 of 4 rotations (largest when trees are held out: in-family leakage at chance, RF reads 20-30%); (2) the population defender D_pop minus the single-strongest-attacker-with-5-restarts defender D_single (compute-matched) on L_HO is within +-MDE80 (4.8 points cross-fit): the population adds nothing over restarts; (3) a noise-dithered label-free code D_dither at matched activity accuracy (+-1 point) and matched in-family leakage (+-3 points) matches D_pop on L_HO within MDE80. G OPENS only if D_pop is below min(D_single, D_dither, activity-LDA-k envelope) on L_HO by >= 7 points in >= 3 of 4 rotations; P(open) = 0.12.

## Mechanism
In-training discriminators at chance do not predict a fresh probe (Elazar and Goldberg 2018; programme 1; stress-test floor: a 5-dim activity code that a linear attacker calls private leaves RF at 8x chance). Whether diversity in the training population rather than a bottleneck helps is the open question; programme 1's adversary-adds-nothing-beyond-bottleneck pattern, and The Content Leash (speech) reporting held-out verifiers near family (27.9% vs 26.3%), argue for failure; GaitProtector trains against surrogate ensembles and tests a held-out backbone but never retrains.

## Experiment (exact)
- Data / units: MotionSense cached hand-feature windows, trial unit, 3 rank folds (defender and attackers see only training-fold trials), 4 family rotations, 3 seeds inside each (fold, rotation) cell; all 360 trials tested once per rotation. Cross-fit pooled MDE80 4.8 points (ANALYTIC; recomputed from the real paired SE).
- Defenders: D_pop (adversaries from 3 families sampled uniformly each step, uniform population as in Vinitsky et al. robust RL); D_single (strongest in-family attacker chosen on validation, 5 restarts, fresh re-initialisations); D_dither (rate bottleneck + Gaussian noise, no adversary, strength tuned to matched in-family leakage); D_LDA-k (closed form). Non-differentiable members (trees) enter D_pop by alternating retrain-and-reweight every 5 epochs (stated approximation: they supply a penalty through sample-weights on the encoder loss, not gradients).
- Evaluation: attackers retrained on z from scratch per family, trial-level identity, held-out family never used in selection or training; naive and val-selected both reported.
- Matching: activity accuracy (RF on z) within 1 point; in-family leakage within 3 points.
- Controls: constant / shuffled at chance; D with lambda = 0 as the no-adversary arm.

## Prediction and falsifier
- Predicted: Gamma(F3 held out) 10-20 points; D_pop - D_single within +-4; D_pop - D_dither within +-4; G opens with probability 0.12.
- FALSIFIED (G stays closed) if Gamma < MDE80 in all four rotations (nothing to close) OR D_pop does not beat min(D_single, D_dither, LDA-k) by >= 7 points on L_HO in >= 3 of 4 rotations. FALSIFIED the other way (G opens) if that win occurs with the CI excluding 0 and 3 seeds agreeing.

## Cost / dependencies
~3 GPU-h; shares the 6 GPU-h gate with H-G-002.

## Prior
0.12 (open) / 0.70 (gap exists). Reasoning above; the MDE is the binding constraint (7-8 points at 144 units, stress test).

## Cheapest floor that could kill it
Activity-LDA-k + dither, then D_single with restarts, then D_dither; any one of them within MDE80 closes G.
