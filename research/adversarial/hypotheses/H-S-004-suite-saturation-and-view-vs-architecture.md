# H-S-004: The suite saturates after about three members, and the input VIEW moves the worst case more than the architecture does

- Focus: S (instrument) | Kind: premise / coverage | GPU-h: ~4 incremental (shares fits with H-IMU-002 and H-CXR-004; ~10 if run standalone) | Prior: 0.55
- Depends on: H-IMU-002 releases and fits; H-CXR-003 / H-CXR-004 releases (CXR arm after the download gate).

## Statement
Over 8 IMU releases and 6 CXR releases, (1) greedy forward selection of suite members reaches the full-suite worst case within one MDE80 with at most 3 members in >= 90% of cells; (2) variance of cell-level attacker accuracy explained by INPUT VIEW (IMU: hand features / raw sequence / per-subject statistics; CXR: raw pixels / downsampled / frozen-FM embedding / metadata) is >= 2x the variance explained by ARCHITECTURE within a view (programme-1: attacker strength came from the view; brightness-only 22.4% identity; LDA strongest on CREMA-D); (3) leave-one-family-out coverage: removing any one family lowers the suite worst case by <= MDE80 in >= 90% of IMU cells, but removing the non-learned structural family (NCC / SSIM with registration) lowers it by more than MDE80 in >= 30% of CXR cells (LAPIS: PriCheXy wins on structural Rank-1 22.4% vs 34.7% while losing on AUC_cross, so the ranking depends on the family); (4) the stop rule "worst-case gain from the next rung < MDE80 over two consecutive rungs" fires in >= 75% of cells.

## Mechanism
With 24 subjects or 1,424 patients the attacker space is bounded by the data, not by capacity; small, tuned tabular / feature attackers sit near the Bayes limit and deep nets cannot add much on 144 units (leakage scout D2 risk). The family that can add is the one that changes the VIEW or the matching principle (non-learned registration, metadata), not another MLP.

## Experiment (exact)
- Ladder axes per release: training-data fraction {1/8, 1/4, 1/2, 1}; capacity (LR, MLP, 1D-CNN / GRU; CXR: ResNet-18, ResNet-50, ConvNeXt-T); restarts {1, 3, 10}; family; view. IMU views: hand features (std, MAD, percentiles, FFT) with LR / LDA / kNN / RF / LightGBM / MLP; raw 128x12 windows with 1D-CNN and GRU; per-trial summary statistics with nearest centroid. CXR views: raw 256 px (Siamese-R50, ConvNeXt-T, ViT-S), 64 px downsample, frozen DINOv2-L embedding (kNN Canberra, InfoNCE head), registered NCC / SSIM, demographic + device metadata GBM. Optional tuning-free member: TabICLv2 (code open; weight licence and class limit unverified in the contrarian file, check before use).
- Releases: IMU: raw, activity-LDA-3, activity-LDA-5, quantiser b=3, dither sigma=0.5 SD on LDA-5, per-window de-mean, PrivDiffuser release, downsample x8. CXR: raw, PriCheXy mu=0.01, random flow mu=0.01, blur sigma=4, downsample x4, PCA-64.
- Analysis: (a) greedy forward selection curve of worst-case vs members; (b) two-way variance decomposition (view, architecture-within-view) on cell accuracy, mixed model with release as random effect; (c) LOFO table; (d) stop-rule evaluation.
- Units / splits: IMU unit = trial, 3 rank folds, seeds inside folds, trial and subject bootstrap; CXR unit = patient, official-test 1,424 headline, floors also rotated over 5 patient folds; PriCheXy arm has fixed test units (stated; seeds replicate training noise only).

## Prediction and falsifier
- Predicted: forward selection needs 2-3 members (typically RF or LightGBM on hand features plus the 1D-CNN on IMU; Siamese-R50 plus NCC / SSIM plus the demographic attacker on CXR); view explains 2-4x the variance architecture does; dropping NCC / SSIM matters in 30-60% of CXR cells.
- FALSIFIED if: >= 5 members are needed in >= 25% of cells (the suite is not small), or architecture explains at least as much variance as view, or the worst case is still rising by more than MDE80 per doubling at the top rung in >= 25% of cells (the ladder has not saturated, no complete suite can be declared).

## Cost / dependencies
Incremental ~4 GPU-h on top of H-IMU-002 and H-CXR-004 fits.

## Prior
0.55. The view-over-architecture claim is a direct carry-over of programme 1; saturation at n = 144 is likely but the CXR structural-family claim is based on one unreviewed paper (LAPIS, unverified against the PDF).

## Cheapest floor that could kill it
A retrained {LDA, kNN-Canberra, RF on hand features} trio equals the full suite within MDE80 in >= 90% of cells: then the suite is those three plus the metadata floor, and "robust suite" becomes a three-line recommendation (still a valid by-product).
