# H3-003: Second dataset and encoder family: on HLCA single cells (E's dataset-grouped folds, so every test cell comes from an UNSEEN dataset) the supervised nonlinear adapter shows no label-budget break-even over the label-supervised linear floors, which tests the K-rule of H3-002 outside imaging

- Focus: single-cell (programme-1 E data) as a second dataset for the supervised-adapter result | Kind: replication attempt, predicted negative on the registered arm; a win would be the first independent replication | GPU-h: 3 | Prior: 0.25 that a break-even <= 30% appears; 0.55 that dU stays <= +0.015 at every budget; 0.20 INCONCLUSIVE at the donor-cluster MDE
- Depends on: Builder E's cached HLCA subsample and `e_singlecell` harness (units, folds, attackers, L definition, SC-K0 power audit reused UNCHANGED); H3-002 for the K-rule; licence CC BY 4.0 via the CELLxGENE policy (E.md); no new download.

## Statement
On the E HLCA subsample (97,232 cells, 107 donors, 14 datasets in 11 studies, 60 supported finest types) with E's 3 dataset-grouped folds (F1 = 2 datasets / 40 donors, F2 = 5 / 37, F3 = 7 / 30; the held-out datasets are never in any fit), a supervised MLP adapter (CE + SupCon on PCA-50 of log-normalised HVGs, output d in {8, 16, 32}) with a fraction f in {1, 3, 10, 30, 100}% of training cells labelled never beats the full inductive floor envelope (E2's registered floors plus label-supervised LDA to d fitted on the SAME labelled cells and on 100%) by >= 0.02 macro-F1 of held-out-dataset fine-type utility at matched batch leakage, at any f, under either label regime:
- R1 (E's regime): labels = the coarse annotation; utility = U_fine and U_hold (held-out finest types) as E defines them.
- R2 (the JUMP analogue): labels = the finest annotation (K about 60); utility = U_fine on the held-out datasets.
An exploratory arm holds out 20% of finest types from every fit and asks the retrieval question on them (about 12 types, MDE80 of order 0.05-0.10 so it can only be INCONCLUSIVE or large).

## Mechanism
K = 60 (R2) or fewer (R1) is of the same order as d = 16-32, so the linear label-supervised code already spans the label information (H3-002). In single cells the nonlinear structure that survives (continuous trajectories, within-type gradients) is not what the label asks for; and with COARSE labels SupCon collapses the fine structure the utility needs, so R1 is predicted to lose to ldaC_pca16 (E: r_fine 0.77). Contrast with JUMP, where K is 32k and the held-out classes are the evaluation target. If the adapter wins here, the K-rule is incomplete and the supervised-adapter result is more general than argued; E2's reading "the single-cell headroom is transductive" would then be wrong for supervised inductive maps.

## Experiment (exact)
- Data and units: as above. Independent unit = donor (107), bootstrap over donors with E's study-cluster bootstrap (resample the 11 studies, then donors inside) as the conservative second interval. Source = dataset (batch); leakage L = E's definition: attackers (linear, MLP, kNN, LightGBM) retrained on the released vectors of held-out cells, donor-disjoint halves within each held-out dataset, balanced accuracy over the held-out batches present, worst case over the four, normalised by raw. Do not edit `attackers.py`.
- Floors: E2's registered inductive list (pca16, rp16, dith16_r0.5 and the noise / dithered grid, `ldaC_pca16`, `ldaC`), plus `ldaM_f_d` (PCA-50 then shrinkage LDA on exactly the labelled cells, d in {4, 8, 16, 32}). Transductive references (scVI U 0.816 at L 0.80; Harmony U 0.668 at L 0.37) are context only, never floors.
- Encoders (the second-encoder axis): (i) PCA-50 (E's input); (ii) an inductive scVI latent (10 dims) trained on training-fold cells only; (iii) a single-cell foundation model's frozen embedding ONLY if the steward clears its weights licence (not assumed; skip otherwise).
- Adapter: MLP in -> 512 -> 256 -> d, LayerNorm, GELU, dropout 0.1; loss = CE on a linear head + SupCon (tau 0.1; positives = same label across datasets), 6,000 steps, AdamW 1e-3 cosine, output dither r in {0, 0.5}; no adversary. Label budget: nested random subsets of training-fold cells by md5 (2 salts).
- Matched leakage and V-selection exactly as E2/F3 (V = a donor-disjoint validation slice of the training folds; held-out fold read once). Trained check: loss ratio, effective rank >= d/4, and U_V(arm) - U_V(permuted-label arm) >= margin (memorisation-calibrated, as F3 8.5; the E2 lesson that in-sample criteria are unattainable at small n is applied).
- Margin: delta_U = 0.02 macro-F1, set before the run; if the realised floor-vs-floor MDE80 (E: 0.028-0.057, median 0.035) exceeds 0.02 the cell is INCONCLUSIVE, not NO-GO.
- Seeds vs test units: the 3 dataset-grouped folds rotate the test units (every donor tested exactly once); pooled out-of-fold across folds; 3 adapter seeds per cell replicate training only; 2 label salts replicate the labelled subset. Cross-fit is the fold rotation.

## Prediction and falsifier
- Predicted: R2 dU(adapter - envelope) in [-0.03, +0.015] at every f and d, best at f = 100% (+0.005), break-even censored above 100%; R1 dU in [-0.06, 0.0] (collapse of fine structure); exploratory held-out-type arm in [-0.02, +0.06].
- FALSIFIED (the K-rule does not explain the nulls) if the adapter beats the full envelope by >= 0.02 macro-F1 with donor-cluster lower bound > 0 at some f <= 30% in >= 2 of 3 folds for encoder (i) or (ii) under R2. A win under R1 only is reported as a different finding (coarse-to-fine transfer).

## Cost / dependencies
~3 GPU-h: 5 budgets x 3 d x 2 r x 3 folds x 2 seeds x 2 regimes = 360 fits at ~30 s on cached PCA features, plus E's CPU attackers and a scVI fit per fold (~0.3 h). Needs Builder E's cache paths (`~/work/cache/e_singlecell`).

## Prior
0.25 for a replicated win. E2's registered inductive encoders sat -0.037 below the floor envelope (MDE80 0.020); label supervision helps utility but `ldaC_pca16` already keeps r_fine 0.77 and the coarse-label code collapses fine structure on two datasets.

## Cheapest floor that could kill it
`ldaC_pca16` and `ldaM_f_16` (same labels) in the envelope, and the dithered PCA / noise codes of E2. A "win" that disappears against `ldaM` fitted on the same labelled cells is the same-supervision comparison F3 required.
