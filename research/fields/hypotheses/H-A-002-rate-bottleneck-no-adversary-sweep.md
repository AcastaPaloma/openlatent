# H-A-002: A biology-only rate bottleneck (no adversary) removes >= 40% of centre leakage at d=8 for free, but no rate reaches L_worst <= 0.25

- Focus: A | Kind: learned method (mandatory control per FOCUS) | Cost: ~4 GPU-h | Prior: 0.50
- Depends on: Builder-A embedding cache + probe suite; H-A-001 numbers for LEACE/raw references.

## Statement
A small adapter g: R^2048 -> R^d trained with the tissue-label cross-entropy only (no centre labels, no adversary) and then frozen reduces the worst-case retrained-probe centre leakage L_worst by at least 40% relative to raw at d = 8 while holding biology utility U >= 0.99 on ID and OOD centres. No rate d in {2..256} gets L_worst <= 0.25 at U >= 0.98. At equal d, the biology bottleneck leaks less than PCA-d and than a random orthogonal projection to d dimensions (by >= 0.15 at d = 8).

## Mechanism
Our own KTH result: at 1-2 numbers per frame the adversary adds nothing and rate does the work (results/SUMMARY.md finding 3; journal 2026-10-28 "no adversary needed"). In pathology the tissue label needs <= log2(28) ~ 5 bits per dataset, so a supervised bottleneck may be forced to drop most stain/scanner directions that are not useful for tissue. But stain correlates with tissue appearance (ComBat deletes biology when centre and class correlate: 2507.17845 mitigation section), so the bottleneck can keep centre through class-correlated colour. PCA-d keeps the highest-variance directions, where centre signal sits (centre is the dominant variance axis: "embedding space organised more by centre than biology", literature/medical_biosignal.md #12), so PCA should leak more than the supervised bottleneck.

## Experiment (exact)
- Data/encoders/probes: as H-A-001 (PathoROB camelyon, tcga 4x4, tolkach; phikonv2/phikon/midnight clsmean; A-SUITE slide-disjoint 60/10/30).
- Methods, all trained on ID training slides only (probe-train slides excluded from adapter training to stop the adapter memorising probe slides): (i) linear bottleneck; (ii) MLP 2048-1024-d with LayerNorm; (iii) product VQ bottleneck with M sub-codebooks x K=64 (rate 6M bits), M in {1,2,4,8}; each with tissue-CE only. Controls at equal d: PCA-d (fit on train slides), random orthogonal projection (seeded), LDA-d (task-label-only projection, d <= n_classes - 1). References: raw, LEACE (from H-A-001).
- Ladder: d in {2, 4, 8, 16, 32, 64, 128, 256}.
- Seeds: 3 adapter seeds x 3 attacker seeds x 3 slide partitions (headline cells only); bootstrap over slides.
- Metric: L_worst as H-A-001; U on ID held-out slides and on OOD centres (balanced); report the full (U, L_worst) frontier and its area. Normalisation: raw = (U=1, L=1); chance L=0; task-label-only reference = LDA-d.

## Prediction and falsifier
- Prediction: at d = 8 the MLP bottleneck has U >= 0.99 and L_worst(bottleneck) <= 0.6 x L_worst(raw) (= relative reduction >= 40%) on >= 2 of 3 encoders in each dataset; PCA-8 L_worst - bottleneck-8 L_worst >= 0.15; random-projection-8 loses U (<= 0.90); min over d with U >= 0.98 of L_worst >= 0.25 (never reaches the "convincing" target of 0.2 from the stress test); VQ M=2 ties the continuous bottleneck within 0.05 in L_worst at equal numbers of bits.
- Falsified if (a) L_worst(bottleneck, d=8) > 0.85 x raw (rate does not help; centre rides along with tissue), or (b) some d reaches L_worst <= 0.20 at U >= 0.98 (then a rate knob alone solves A and every adversary/MI hypothesis is moot), or (c) PCA-8 leaks no more than the bottleneck (no supervision effect).

## Cost
~4 GPU-h: adapters train on cached 2048-d vectors in 1-3 minutes each; 8 d x 3 encoders x 3 datasets x 3 seeds x 3 variants ~ 650 runs.

## Dependencies
Builder-A: cache + dataset/slide/centre/label arrays; shared training loop for adapters. None on B or C.

## Prior and why
0.50 for the compound statement (the 40% reduction half ~0.7, the "never below 0.25" half ~0.65). Rate-vs-privacy is our one robust finding in video, but pathology centre signal is high-variance and class-entangled; transfer of the KTH regularity is exactly what is being tested.

## Sources
[O] 2507.17845 (ComBat vs biology; centre dominates); [S] Privacy PORCUPINE, Interspeech 2024 (VQ occupancy; literature/methods_transfer.md); our results/SUMMARY.md and journal.
