# H-A-001: Linear centre erasure zeroes the linear probe but leaves a retrained nonlinear probe >= 0.30 of the raw leakage, on real pathology FMs

- Focus: A (PathoROB) | Kind: premise / headroom check (deterministic baseline, no learned method) | Cost: ~2 GPU-h + ~6 CPU-h | Prior: 0.75
- Depends on: Builder-A embedding cache (`src/a_pathorob`, encoders phikonv2_clsmean, phikon_clsmean, midnight_clsmean, dinov2L_clsmean) and the shared probe suite.

## Statement
On frozen PathoROB embeddings, a centre eraser fitted with the closed-form LEACE solution on training slides drives a retrained *linear* centre probe to chance (normalised leakage L_LR <= 0.10), but a retrained MLP / kNN / gradient-boosted probe still recovers at least 30% of the raw centre signal (L_worst >= 0.30) in at least 7 of 9 (3 encoders x 3 datasets) cells. The same LEACE fitted on ID centres also fails to erase *unseen* centres (L_LR on OOD centres >= 0.30). This is the headroom premise that Focus A rests on; if it is false, A has no job left.

## Mechanism
LEACE guarantees that no *linear* classifier trained on the fit distribution can beat a constant predictor (2306.03819, abstract opened). It removes at most (n_centres - 1) directions (1 direction for Camelyon, 3 for TCGA 4x4, 2 for Tolkach). Centre signatures in a ViT embedding (stain, scanner blur, compression, tissue prep) are spread over many nonlinear and higher-order statistics, so a model with any capacity re-finds them. Separately, the concept-subspace framework reports that LEACE "has difficulty generalizing to unseen data" (2605.01381, abstract opened), and ISP/LEACE guarantees are conditioned on the fit distribution. The stress-test synthetic (literature/contrarian_stress_test.md 1.5) saw exactly this pattern (linear erasure: linear probe 0.25 = chance, retrained nonlinear probe 0.55 = 2.2x chance) but on random Gaussians, so it is not evidence about real FMs.

## Experiment (exact)
- Data: `bifold-pathomics/PathoROB-{camelyon,tcga,tolkach_esca}` with the repo's shipped splits and seeds. Centres: Camelyon 2 ID + 3 OOD (chance 0.50 on ID), TCGA 4x4 4 ID + 4 OOD (chance 0.25), Tolkach 3 ID + 1 OOD (chance 0.33). 99,392 patches total (2507.17845 HTML, opened).
- Encoders (frozen, builder cache; feature = concat(CLS, mean patch tokens)): phikonv2_clsmean, phikon_clsmean, midnight_clsmean; dinov2L_clsmean as a non-pathology control (its biology accuracy will be lower; its centre leakage should be at least as high as the pathology models). UNI2-h only if gated access arrives. Optional arm: CLS-only vs mean-patch-only vs concat; one extra extraction of 6 equally spaced intermediate layers (~1 GPU-h) for a "where in the network is centre" curve.
- Ladder: raw | global z-score | LEACE(centre) fitted on the probe-train slides | INLP/ISP iterated 2 and 5 rounds | LEACE fitted on ID centres, probed on OOD centres (generalisation arm) | LEACE fitted on the probe-train slides *including* the OOD centre (oracle upper bound).
- Attacker suite (A-SUITE): logistic regression, MLP (2x512, dropout 0.2, AdamW, early stop on validation slides), kNN (k=20, cosine), LightGBM (300 trees), RFF-SVM (random Fourier features, 4096). Slide-disjoint 60/10/30 train/val/test (the PathoROB probe protocol), hyperparameters picked on validation slides only, retrained on every transformed embedding.
- Seeds: 3 attacker seeds x 3 slide-partition seeds. LEACE is deterministic.
- Metric: per attacker k, L_k = (acc_k(z) - chance) / (acc_k(raw) - chance); L_worst = max_k L_k. Biology utility U = (acc_bio(z) - chance) / (acc_bio(raw) - chance) with a slide-disjoint logistic-regression probe. 95% CI: bootstrap over slides within centre (1000 resamples), not over patches.
- Protocol sub-control: repeat the probes with a *patch*-disjoint split instead of slide-disjoint.

## Prediction and falsifier
- Prediction: raw centre LR accuracy 0.88-0.98 (replicates 2507.17845 Sec. 2.2). After LEACE: L_LR <= 0.10 in all 9 cells; L_worst >= 0.30 in >= 7/9 cells (median ~0.5); U >= 0.97; INLP x5 lowers L_worst by < 0.15 versus LEACE; on OOD centres LEACE-ID leaves L_LR >= 0.30; patch-disjoint probing inflates L_worst by >= 0.05 over slide-disjoint in >= 6/9 cells (slide fingerprints survive erasure).
- Falsified if L_worst(LEACE) < 0.15 in >= 5 of 9 cells (linear erasure is enough; no retrained-probe headroom; A reduces to chasing RI, which is saturated), or if L_LR after LEACE exceeds 0.20 on the fit slides (implementation or distribution problem; fix before anything else).

## Cost
~2 GPU-h (feature extraction is already the builder's job; intermediate-layer arm ~1 GPU-h) + ~6 CPU-h for 5 attackers x 9 cells x 3x3 seeds.

## Dependencies
Builder-A: embedding cache and slide IDs/centre labels per patch; probe suite with slide-disjoint splits. LEACE/INLP implementations (`concept-erasure` package or a 30-line covariance solve).

## Prior and why
0.75. Linear-probe centre accuracy is 88-98% across FMs (2507.17845), centre signal is huge, and LEACE removes <= 3 directions. The risk is that kNN/GBT on 2048-d with ~30k test patches under-perform the raw reference so L_worst is dominated by noise; then the "worst" attacker choice makes the headline noisy. The OOD-generalisation clause is the softer one (0.6).

## Sources
[O] 2306.03819 LEACE; [O] 2605.01381 concept-subspace framework (LEACE generalisation weakness); [O] 2507.17845 PathoROB (HTML read: RI/APD definitions, centre counts, 88-98% centre probe); [S] 2604.05296 ISP (literature/methods_transfer.md); literature/contrarian_stress_test.md section 1.5.
