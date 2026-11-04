# H-X-008 (BORING CONTROL, A and C): Per-source mean/whitening with oracle source IDs reaches L_MLP <= 0.35 at <= 3 points utility cost, matching the best learned per-sample encoder

- Focus: A + C | Kind: boring control (group-aware floor, "tier 2") | Cost: ~1 GPU-h (CPU mostly) | Prior: 0.50
- Depends on: Builder-A / Builder-C caches; best learned adapters for the comparison column.

## Statement
Given oracle source IDs at release time (centre in A; actor in C; i.e. the data holder normalises its own sources before release), subtracting each source's mean embedding (and optionally whitening with a ridge-regularised covariance on a PCA-64 projection) yields: C: L_worst(MLP/kNN/GBT) <= 0.35 for mean subtraction and <= 0.15 for whitening on T1, with emotion UAR within 3 points of raw (the balanced design means every actor's mean equals the same average-emotion point); A: at V = 0 (balanced), per-centre z-score gives L_worst <= 0.30 with biology within 1 point, but at V = 0.5 loses >= 2 points of biology (ComBat effect). If so, these per-source floors match or beat the best learned per-sample encoder, which is *not comparable* (it needs no source IDs) but is the reference any group-level release must be compared with.

## Mechanism
The stress-test synthetic found centre-mean standardisation was the only transform that also brought the nonlinear probe near chance (0.27-0.33 with chance 0.25), whereas linear erasure left 2.2x chance (literature/contrarian_stress_test.md 1.5). In BatchBench, training-free adaptation (AdaBN, +7 points) beat adversarial removal (DANN, -0.5) (scout-reported; the abstract page shows no numbers). In CREMA-D every actor delivers all six emotions in the same sentences, so the actor mean is a clean estimate of the actor offset and the within-actor deviations are exactly the emotion contrasts. Residual risk: actor-specific covariance and style (expressiveness scale) remain.

## Experiment (exact)
- C: CREMA-D wavlm-base-plus best layer, pooled mean+std -> PCA-64 (fit on train actors); per-actor mean over *all* the actor's clips (oracle, unlabeled), subtract; whitening variant with ridge 1e-2; evaluate speaker attackers (C-SUITE) on T1/T2/T3 with the attacker trained on the transformed S_TR x E_TR clips of the enrolled actors, emotion UAR with the probe trained on transformed encoder-train actors and tested on test-fold actors.
- A: PathoROB phikonv2/phikon/midnight; per-centre z-score and PCA-64 whitening with centre ID; splits at V in {0, 0.5}; A-SUITE source-probe; biology accuracy.
- Variants: group size ablation (per-source mean estimated from 5, 20, 80 clips/patches); per-source mean on the *class-balanced* subset only; adapter-then-normalise.
- Seeds: 3 attacker seeds x 4 folds / 3 datasets; bootstrap over sources.
- Metric: L_worst, U; comparison to H-A-001/002/003 and H-C-001/002/003/007 best adapters.

## Prediction and falsifier
- Prediction: C: mean subtraction L_MLP 0.2-0.4 (residual dynamics), whitening 0.05-0.2, U drop <= 0.03; with only 5 clips/source the effect shrinks by about half. A: V = 0 L_worst 0.15-0.35; V = 0.5 biology -2 to -6 points.
- Falsified if mean subtraction leaves L_MLP > 0.6 in C (actor offset is not the leak; style is), or if A's per-centre z-score leaves L_worst > 0.6 (the stress-test synthetic fails to transfer to real FMs).

## Cost
~1 GPU-h (+ CPU probes).

## Dependencies
Builder-A / Builder-C caches; group indices.

## Prior and why
0.50 joint (C 0.6; A 0.5). It is a floor for a different problem (group-aware release); the surprise would be if it fails.

## Sources
literature/contrarian_stress_test.md 1.5 (synthetic) and 1.1 (BatchBench numbers, scout-reported); [O] 1603.04779 AdaBN; [O] 2609.28231.
