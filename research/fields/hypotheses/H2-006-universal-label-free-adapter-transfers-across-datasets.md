# H2-006: A label-free adapter trained once on other datasets transfers its source invariance to an unseen dataset (the "embedding released for unknown future use" scenario)

- Focus: A + C | Kind: learned method, cross-dataset transfer | Cost: ~3 GPU-h | Prior: 0.20
- Depends on: H2-001 AE implementation; A embedding caches (Camelyon, TCGA, Tolkach in the same 3072-d midnight space); for the C arm, RAVDESS audio features (not yet extracted; licence and size not verified by me).

## Statement
Leave-one-dataset-out on PathoROB with midnight (and phikon-v2): train ONE label-free AE (dz 32, beta 1e-3, lam 3 or 10) on the adapter-train units of the two other datasets, with the adversary targeting the union of those datasets' centre ids; apply it frozen to the third dataset, retrain attackers and probes on that dataset's B units, score on its sealed C. In >= 3 of 6 (target x encoder) cells:
- L_cross <= L_within + 0.15, where L_within is the AE trained on the target dataset itself (H2-001, same selection rule);
- U_cross >= U_within - 0.03, with U = full-label probe accuracy (every label is "unseen" by construction) on TCGA/Tolkach and tumour-vs-normal on Camelyon;
- U_cross >= best inductive floor fitted on the same two source datasets (PCA-d, `lda`-free; LEACE-on-centre fitted on the source datasets) + 0.03 at matched L.
C arm (secondary): AE trained on CREMA-D actors, applied to RAVDESS (24 actors, 2 sentences per the public description, to be verified): identity leakage of the RAVDESS actors (attackers enrol on sentence 1, tested on sentence 2) <= 0.3 of raw RAVDESS identity.

## Mechanism
A foundation model embedding is released once and used on data nobody has seen. A closed-set centre adversary is organised around the centres it saw; LEACE-type closed forms fail on unseen data (C: LEACE fitted on 56 actors leaves 53x chance on unseen ones, L 0.73; 2605.01381 per scouts: "struggles to generalise to unseen data"). A reconstruction-regularised latent may carry less dataset-specific structure and a rate-limited code removes high-frequency, source-specific texture independent of which centres were seen. Whether the removed directions generalise across tissues and stains is exactly the unknown.

## Experiment (exact)
- Folds: target in {Camelyon, TCGA, Tolkach}; source = the other two; encoder in {midnight, phikon-v2}. Source units pooled with dataset-balanced sampling; the target's units and OOD centres never enter any fit (fit-distribution rule).
- Arms: AE-cross (V-selected on a held-out slice of the SOURCE datasets only, because the target's V is unavailable in a truly unknown-future-use setting; a variant selected on target V is reported as an oracle), AE-within (H2-001), PCA-d and LEACE fitted on sources, raw, `centre_whiten` on the target (transductive reference).
- Attackers/metrics: A-SUITE6 retrained on target B, L_worst normalised to raw target; slide-disjoint bootstrap on target C; U as above. Report seed spread (3 seeds) separately.
- Controls: constant/noise/orthogonal shuffle at chance; AE trained on shuffled embeddings (feature-permuted) must give chance utility.

## Prediction and falsifier
- Prediction: L_cross - L_within in [+0.05, +0.25] (partial transfer); U within 0.03 of within-dataset AE in half of the cells; advantage over PCA-d/LEACE fitted on sources +0.03 in 2 of 6 cells.
- Falsified (invariance does not transfer) if L_cross - L_within >= +0.30 in >= 4 of 6 cells; or if PCA fitted on the sources matches the AE-cross within 0.02 at matched L in >= 4 of 6 cells (then a universal learned adapter has no edge over a universal PCA).

## Cost
Leave-one-out AE fits (6 cells x 3 seeds x ~24 configs, ~3 min each) ~2 GPU-h plus the attacker evaluation; C arm +0.5 GPU-h plus RAVDESS extraction (~0.3 GPU-h, if the licence allows download).

## Dependencies
H2-001 code; shared centre-id vocabulary across datasets; RAVDESS download only after the licence check by the programme owner.

## Prior and why
0.20. Dataset shift between tissues is far larger than between centres within one dataset, and every closed-form eraser we measured fails out of sample; the only reason the prior is not lower is that a pure rate bottleneck has no centre-specific component to overfit.

## Sources
results_notes/C.md (LEACE trained actors vs unseen actors); [S] 2605.01381 and literature/methods_transfer.md (generalisation of linear erasure to unseen data; abstract-level only, not opened by me).
