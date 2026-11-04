# H-A-005: Label-free per-centre whitening is a hard floor on unseen centres; Tent-style entropy adaptation of the adapter adds >= 2 points biology where class priors are skewed

- Focus: A | Kind: learned (test-time) + deterministic floor | Cost: ~2 GPU-h | Prior: 0.30 (compound), 0.60 for the floor half
- Depends on: Builder-A cache; trained adapters from H-A-002/003.

## Statement
At test time, an unseen centre provides an unlabeled set of embeddings (>= 256 patches from >= 3 slides). Then:
(a) T2, per-centre mean + ridge-regularised covariance whitening of the (adapter) embeddings, with no training, reduces the retrained-probe leakage among unseen centres to L_worst <= 0.30 (relative to raw), and
(b) T3, updating only the LayerNorm affine parameters of the adapter for 20 steps by entropy minimisation of its tissue head (Tent, 2006.10726) yields >= 2 points higher OOD biology accuracy than T2 when the unseen centre's class prior is skewed (Cramer's V >= 0.5 within the test centre pair), at equal L_worst (within 0.05).

## Mechanism
Normalisation-statistics adaptation (AdaBN, 1603.04779) is parameter-free and in BatchBench it beat adversarial removal (training-free adaptation +7 points over a ResNet18 baseline while DANN -0.5, numbers from literature/contrarian_stress_test.md; the BatchBench abstract page itself shows no numbers, so treat as scout-reported). Per-centre standardisation removes first and second moments of the centre offset, but when a centre's class mix differs from the training mix it also deletes class-prior-induced mean shifts, i.e. biology (ComBat pattern). Entropy minimisation (Tent: channel-wise affine updates at test time, 2006.10726 abstract) keeps the decision structure while moving the affines, so it should not delete class-correlated mean shifts.

## Experiment (exact)
- Data: PathoROB; adapters trained on ID centres only (H-A-002 MLP d=128 and H-A-003 M2). Test: OOD centres (Camelyon 3, TCGA 4, Tolkach 1) plus held-out ID slides as reference. Class-prior skew: build test centre subsets at V in {0, 0.5, 0.8} against the training centres using PathoROB's splitter.
- Variants: T0 none; T1 per-centre z-score (mean/std); T2 per-centre full whitening with ridge (PCA-64 first); T3 Tent-style affine update on the adapter (Adam lr 1e-3, 20 steps, batch = all centre patches in 256-chunks); T4 T2 then T3; T5 ComBat reference (centre ID known); T6 oracle: adapter retrained with the unseen centre's unlabeled embeddings included in a domain-adversarial term.
- Centre ID is assumed known at test time (batch-correction regime); a centre-ID-free variant uses k-means (k = 2..5) on embeddings to cluster batches and is reported separately.
- Probes: A-SUITE source-probe among all centres (slide-disjoint) on the adapted embeddings; biology LR probe trained on ID training embeddings (not adapted) and applied to adapted OOD embeddings (this is the deployment case).
- Seeds: 3 adapter x 3 TTA seeds (random chunking) x attackers; bootstrap over slides.
- Metric: L_worst, U (ID, OOD), and the gap closed to T6: (U_T - U_T0)/(U_T6 - U_T0).

## Prediction and falsifier
- Prediction: T2 gives L_worst <= 0.30 (OOD) in >= 7/9 cells and costs <= 1 point biology at V = 0 but >= 2 points at V >= 0.5; T3 loses <= 1 point at V >= 0.5 and gives L_worst reduction >= 20% relative to T0; T4 is best on both axes; T3 closes >= 50% of the gap to T6.
- Falsified if T2 is within 1 point of T3 on biology at V >= 0.5 (the learned part adds nothing; keep only the floor), or if T2 fails to cut L_worst below 0.5 (second-order statistics are not the centre signal).

## Cost
~2 GPU-h (adaptation steps are cheap; 20 steps x 99k patches x 9 cells x 3 seeds).

## Dependencies
Builder-A: adapters from H-A-002/003 (checkpoint format), known centre IDs per patch at test time.

## Prior and why
Floor half 0.60 (the stress-test synthetic found centre-mean standardisation was the only transform that also brought the nonlinear probe near chance). The learned half 0.35: entropy minimisation can collapse on 2-class Camelyon (degenerate solutions), and a 2-point gain over a floor that already works is small. Compound 0.30.

## Sources
[O] 1603.04779 AdaBN; [O] 2006.10726 Tent; [O] 2609.28231 (ComBat gives no consistent slide-level gain); [S] BatchBench 2610.30104 numbers via literature/contrarian_stress_test.md (its abstract page shows none); literature/contrarian_stress_test.md 1.5.
