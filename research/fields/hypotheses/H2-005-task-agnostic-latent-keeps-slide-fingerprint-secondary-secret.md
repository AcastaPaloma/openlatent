# H2-005: What the task-agnostic AE keeps beyond the label is disproportionately slide-identifying: at equal held-class utility it leaks more slide/case identity than PCA or the LDA family

- Focus: A (also the C analogue: speaker vs channel) | Kind: stress test of the AE lead on a second secret | Cost: ~1 GPU-h (+~6 CPU-h) | Prior: 0.60
- Depends on: AE embeddings from H2-001; the patch-disjoint diagnostic already in `run_sweep.py`.

## Statement
Define the slide-identity attack: on the sealed test units, split each slide's patches into two disjoint halves; attackers (A-SUITE6, retrained from scratch) are trained on half 1 and predict the slide (or case) of half-2 patches among the n_C test slides (chance 1/n_C: 20 slides Camelyon, 48 cases TCGA, 12 cases Tolkach). Leak_slide = (acc - chance) / (acc_raw - chance). At the V-selected AE configuration (H2-001) in midnight x {TCGA, Tolkach} and phikon-v2 x {TCGA, Tolkach}:
1. Leak_slide(AE) >= 0.5 while the label-supervised family at the same centre leakage (+-0.05) has Leak_slide <= 0.25;
2. at equal held-class utility (+-0.02), Leak_slide(AE) - Leak_slide(best inductive floor: PCA-d, `lda_pca__d`, dithered PCA) >= +0.10 in >= 3 of 4 cells.

## Mechanism
A already measured (gate table, item 4) that slide fingerprints survive every transform: patch-disjoint worst-case centre attack 0.78-1.00 after LEACE, ISP, centre mean/z-score, 0.36-0.66 after LDA. The AE is trained to reconstruct the whole embedding minus the directions the adversary can use to name the centre; texture and staining details that are slide-specific but not centre-predictive across slides are exactly the "non-label structure" the lead is credited for keeping. So the retention win and a patient/slide re-identification risk may be the same quantity. This is the pathology form of "identity is the scene" (Charades) and of CREMA-D's channel confound.

## Experiment (exact)
- Cells and encoders as H2-001 (midnight, phikon-v2, dinov2-L as negative); arms: raw, AE (V-selected), `lda`, `lda_pca__d`, PCA-d, dithered PCA (H2-014), `centre_whiten`.
- Attackers: linear, MLP, kNN, LightGBM on the released representation; hyper-parameters on an inner patch split of half 1. Bootstrap over slides/cases is not possible for an identification task whose classes are the units, so the bootstrap is over patches within slide with the split fixed (stated limitation); the cross-encoder/seed spread is the real error bar.
- Report the 2-D plane (U_held, Leak_slide) and (L_centre, Leak_slide) for every arm; claim only about the plane, not single points.
- Controls: constant/noise at chance; shuffled slide labels at chance.

## Prediction and falsifier
- Prediction: Leak_slide raw = 1; AE 0.5-0.8; `lda` family 0.1-0.3; PCA-4 0.2-0.4. At equal held-class utility AE exceeds the floors by +0.10 to +0.30.
- Falsified if Leak_slide(AE) <= Leak_slide(inductive floor at equal U_held) + 0.05 in >= 3 of 4 cells (the AE is no worse on the second secret, which strengthens the lead).
- Consequence if confirmed: every future AE claim must be reported in the (U_held, L_centre, Leak_slide) triple; a lead that wins on centre leakage only is a statement about one secret.

## Cost
No new training beyond H2-001 (AE embeddings reused); attacker evaluation ~6 CPU-h; ~1 GPU-h for extra encoder cells.

## Dependencies
Builder A: expose the patch-disjoint split for slide ids, not only centre ids.

## Prior and why
0.60. Reconstruction objectives keep slide-level texture, and the existing diagnostics show slide ids are the hardest thing to remove; the competing floors (PCA-4, LDA-2) are so low-dimensional that they cannot hold many slide identities, which already favours the prediction.

## Sources
results_notes/A.md gate summary, item 4 and "patch-disjoint worst" columns; results_notes/C.md Gate 2 (channel/session signature as the analogous second secret in audio).
