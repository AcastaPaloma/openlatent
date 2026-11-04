# H2-004: Dose-response: the AE gain over the label-supervised arm grows as the number of classes the encoder sees shrinks

- Focus: A | Kind: mechanism test (is the gain really "where the label does not determine the content"?) | Cost: ~2 GPU-h (+~10 CPU-h) | Prior: 0.40
- Depends on: AE embeddings from H2-001 (the AE is label-free, so one training serves every class split); LDA/label-supervised arms re-fit per split.

## Statement
On Tolkach (6 classes), let the encoder see k in {2, 3, 4} classes and let the held-out utility be a probe on the remaining 6 - k classes. Averaged over 10 random class subsets per k, 3 seeds, and for midnight and dinov2-L (positive and negative encoder), define Delta(k) = U_held(AE, V-selected) - U_held(best label-supervised arm: `lda`, `lda_pca__d`, `vib`, `vq`, `supcon_xc` at L <= L_AE + 0.05). Then Delta(2) > Delta(3) > Delta(4) on the seed-mean, Delta(2) - Delta(4) >= +0.03, and Delta(4) <= +0.02.

## Mechanism
Label-only projection keeps k - 1 dimensions (LDA) or a k-way code (vib/vq). With k = 2 that is ONE direction and the unseen classes lose almost everything that distinguishes them; with k = 4 it is three directions, and A's seed-0 table already shows `lda_pca__d8` reaching 0.973 held-class on midnight/Tolkach. If the AE advantage is a property of task under-determination, it must vanish as the label code becomes rich. If it does not shrink, it is something else (dimension mismatch AE d = 32 vs LDA d <= 4, or an artefact of the probe), and H2-014 becomes the main suspect.

## Experiment (exact)
- Data: PathoROB Tolkach (6 classes, 12 sealed cases; low power, see H2-016) and TCGA (4 classes: only k = 2 available, reported as the k = 2 anchor; k = 3 leaves one held class and is reported as one-vs-rest AUC only).
- Encoders midnight, dinov2-L; AE configs from H2-001 grid selected on V per (seed, k) by SEL-V computed on held-out classes of V.
- For each (k, subset, seed): refit LDA (k - 1 dims), `lda_pca__d` (d in 4..64), vib, vq, supcon_xc on A units with only the k seen classes; evaluate with A-SUITE6 (L_worst) and the held-class linear probe retrained on B; LM-rule comparator.
- Metrics: Delta(k) per subset; mean over subsets and seeds; paired bootstrap over sealed cases; Spearman correlation of Delta with 1/(k - 1) across subsets.
- Controls: constant/noise/orthogonal-shuffle at chance for every subset; subset-wise raw held-class accuracy printed (ceiling at 1.0 limits Delta on midnight).

## Prediction and falsifier
- Prediction: midnight: Delta(2) ~ +0.06 to +0.10, Delta(3) ~ +0.03 to +0.06, Delta(4) ~ 0 to +0.02; Spearman(Delta, 1/(k - 1)) >= +0.4; dinov2-L Delta <= +0.02 for all k (raw held-class ceiling lower and floors already match).
- Falsified if |Delta(2) - Delta(4)| < 0.01 (flat), or Delta increases with k, or Delta(2) < +0.03 on midnight. A flat Delta says the AE gain is not about label under-determination.

## Cost
AE is trained once per (encoder, seed): 2 x 3 fits; the per-subset supervised arms are small. Attacker evaluation dominates: 30 subsets x ~10 configs x 2 encoders x 3 seeds x ~20 s ~ 10 CPU-h. ~2 GPU-h.

## Dependencies
Builder A harness (`--holdout` accepting an explicit seen-class list); H2-001 for AE embeddings.

## Prior and why
0.40. The mechanism is the very premise of the revised FOCUS A kill rule; seed-0 numbers (LDA 2 dims 0.910 vs d8 0.973 vs AE 0.993) are consistent with a dose-response but also with a ceiling effect that compresses it, and Tolkach has only 12 test cases.

## Sources
results_notes/A.md PROTOCOL 2 (midnight/Tolkach table), FOCUS.md Focus A (revised kill text).
