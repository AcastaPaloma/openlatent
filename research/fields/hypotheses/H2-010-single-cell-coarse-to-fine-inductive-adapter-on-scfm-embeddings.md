# H2-010: NEW FIELD N1. Single-cell batch removal: an inductive, label-free adapter on frozen single-cell foundation-model embeddings keeps fine cell-type structure under unseen studies where coarse-label projections collapse

- Focus: candidate field (not previously killed) | Kind: headroom check first, learned arm second | Cost: ~2.5 GPU-h check, ~5 GPU-h full | Prior: 0.15 (full claim), 0.45 (headroom check passes)
- Depends on: embedding extraction for 100k cells with frozen scGPT / Geneformer / UCE (weights and licences not verified by me); H2-007 cVAE as the scVI-like arm.

## Statement
Data: a human atlas with hierarchical cell-type annotation and many studies (CELLxGENE Census subset, e.g. the lung atlas; ~100k cells, >= 10 studies; the hierarchy levels, the licence per dataset and the study count must be verified before use). Threat model I (inductive): the adapter is fitted on 70% of the studies without fine labels; the deployed embeddings come from UNSEEN studies, source id not available at release. The adapter sees only coarse labels (level 1-2, <= 10 classes) or none (AE arms); the downstream tasks are fine cell types (level 4-5, 30-60 classes) and rare types. Headroom check (floors + reference methods, no learned arm):
- Hh: Harmony (transductive T reference; scVI as the learned reference) on the held-out studies exceeds the best inductive floor (PCA-d of the scFM embedding, coarse-label LDA + PCA padding, noise-dithered PCA, per-cell instance normalisation) by >= 0.08 in fine-label macro-F1 of a probe trained on train studies and scored on held-out studies, at study-leakage L <= L(Harmony) + 0.05;
- Hc: the coarse-label LDA projection keeps <= 70% of raw's above-chance fine-label macro-F1 (the task under-determines the content).
If Hh and Hc pass, the claim: the label-free adversarial AE / cVAE adapter exceeds the best inductive floor by >= +0.05 fine-label macro-F1 at matched study leakage and is within 0.03 of Harmony, 3 seeds, bootstrap over donors/studies.

## Mechanism
Single-cell foundation-model embeddings are released as general-purpose cell representations but retain batch information: the opened comparison reports that scFMs mostly score below Harmony and scVI, some below PCA, and that probes predict batch better from scFM embeddings; per-batch centring helps but does not reach Harmony/scVI (Wang et al., bioRxiv 10.64898/2025.12.19.695371, opened). Every established integration method uses batch ids at inference (transductive), so the inductive setting is the one a released embedding actually faces. Cell-type annotation is hierarchical and users query any level, so a coarse-label projection collapses (the same mechanism as held-out classes in A), while a label-free latent is the natural release. The counter-risk is the scIB literature itself: methods are mature, scVI/Harmony are strong, and the zero-shot benchmarks disagree on whether FMs help (stress-test scout, not opened by me).

## Experiment (exact)
- Stage 0: choose atlas and verify licence, study count, annotation depth; fix train/validation/test studies (unseen-study split) before any metric.
- Stage 1 (check, ~2.5 GPU-h): extract frozen embeddings (scGPT or Geneformer or UCE; at least two) plus an FM-free baseline (log-normalised 2000 HVGs -> PCA-50); compute the floors and references above; fine-label probes (logistic regression, 5-fold by donor) and study-leakage attackers (linear, MLP, kNN, LightGBM retrained on the released embeddings of held-out studies, cell-disjoint within study); scIB aggregate (NMI/ARI/silhouette label, graph connectivity, kBET/iLISI) printed as context only.
- Stage 2: AE + adversary (H2-001 template, adversary targets study id), cVAE (H2-007), contrastive (H2-003); dz in {16, 32, 64}; SEL-V on validation studies only.
- Controls: constant/noise embeddings at chance for both tasks; batch-id-shuffled attacker at chance; near-duplicate cells across splits removed; the FM-free baseline reported as a separate encoder, never merged.
- Kill floors named in advance: best of {PCA-d, dithered PCA, coarse-LDA+PCA}; the transductive Harmony/per-batch centring are references and are never counted as inductive wins.

## Prediction and falsifier
- Prediction: Hc passes (coarse LDA keeps ~40-60% of raw fine-label F1); Hh passes by 0.08-0.15; learned arm +0.03 to +0.08 over inductive floors but still 0.02-0.05 below Harmony.
- Falsified (NO-GO at the check) if the best inductive floor is within 0.05 of Harmony at matched leakage, or if coarse LDA keeps > 85% of raw (no under-determination). Falsified (full claim) if the learned arm gains < +0.03 over the floors in all three encoders.

## Cost
Embedding 100k cells with an FM on an A6000 ~1-2 GPU-h per model; adapters ~1 GPU-h; Harmony/scVI references CPU or ~1 GPU-h.

## Dependencies
Dataset/licence verification (owner); FM weights; H2-001/H2-007 code.

## Prior and why
0.45 that headroom exists (FM embeddings are known to leak batch and to lag Harmony), 0.15 for the full claim: the strongest trivial floors here are not label-only projections but the non-FM PCA pipeline and transductive Harmony, and the field's own benchmarks are crowded.

## Sources
[O] bioRxiv 10.64898/2025.12.19.695371 (scFM batch effects: models, datasets, Harmony/scVI/PCA comparison, batch-centring result); [S] scIB, scIB-E, Kedzierska and CellBench-LS via literature/other_modalities.md and contrarian_stress_test.md (not opened).
