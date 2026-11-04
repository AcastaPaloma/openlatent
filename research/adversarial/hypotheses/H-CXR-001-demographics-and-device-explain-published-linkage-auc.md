# H-CXR-001: A demographic + device attacker built from REAL image estimators explains most of PriCheXy-Net's published linkage AUC under the field's random-negative protocol, and none of it under matched negatives

- Focus: R-CXR | Kind: premise / mandatory protocol change (stress test sec 4.1) | GPU-h: ~4 (age / sex regressors ~2 x 1, PriCheXy release generation 0.5, GBM on CPU) | Prior: 0.65
- Depends on: ChestX-ray14 download (HF `alkzar90/NIH-Chest-X-ray-dataset`, 45 GB, gate: Auditor-2 licence clearance + main agent); PriCheXy-Net repo (GPL-3, keep as a separate process) with `generator_lowest_total_loss_mu_{0.001,0.005,0.01}.pth` and the 10,000 / 2,000 / 5,000 pair lists; CSV metadata (Patient ID, Follow-up #, age, gender, view, width, height, pixel spacing).

## Statement
MEASURED (stress test): on PriCheXy's own 5,000 test pairs a gradient-boosted classifier on TRUE age + sex reaches AUC 0.983 (negatives: same-sex rate 0.513, median age gap 16 y; positives: same sex 1.00, median age gap 0), 0.743 from device fields alone, and a SIMULATED image-estimated demographic attacker (age sd 5 y, sex flipped 3%) 0.885. Hypothesis: with real estimators (ResNet-50 age regressor + sex classifier trained on training-split images) applied to (i) raw 256-px images and (ii) PriCheXy-Net mu = 0.01 outputs, the demographic + device attacker reaches AUC >= 0.85 on raw and >= 0.80 on the defended release under PriCheXy's random negatives (so its published 81.8 -> 57.7 measures how weak its own Siamese is, not how unlinkable the images are), and <= 0.56 under matched negatives MN-v1 on both. Defined share E = (AUC_demo - 0.5) / (AUC_reported - 0.5): E >= 1 on the undefended 0.818 and >= 4 on the defended 0.577.

## Mechanism
PriCheXy's deformation is a smooth, small flow (mu <= 0.01 of image size): it moves anatomy a few pixels but leaves global shape, heart size, thoracic width, bone density cues from which age and sex are estimated (CXR age MAE of a few years, sex accuracy above 97% in the literature). Positive pairs have age gap about 0 and sex agreement 1.0, random negatives often differ in sex and decades of age. The shortcut lives in the pair protocol, so any defence evaluated with random negatives is partly a demographic test.

## Experiment (exact)
- Data / units: ChestX-ray14 at 256 px, unit = patient. PriCheXy split (70:10:20 patient-disjoint) for estimator training; evaluation on its test patients: 5,000 pair list (1,307 patients, 492 behind the positives) and our own MN-v1 pairs built on the 1,424 official-test patients with >= 2 images (one positive per patient, one matched negative).
- MN-v1 (matched negatives, defined in INDEX): same sex, same view, identical (OriginalImageWidth, Height, PixelSpacing), |age difference| matched to the positive pair's age gap within +-1 y by stratified sampling. Validity control before any image number: GBM on TRUE age / sex / device reaches AUC <= 0.56 on MN-v1 (otherwise tighten the caliper).
- Estimators: ResNet-50 (ImageNet init), age L1 regression + sex head, trained on PriCheXy-train patients; estimates for training patients are produced by 5-fold patient-hash cross-fitting so the downstream GBM sees realistic error; estimates for test patients from the model not trained on them. Trained on raw images; also retrained on mu = 0.01 outputs (the attacker knows the defence). 3 seeds.
- Pair attacker: LightGBM on |age_hat gap|, P(same sex), sex-confidence, device equality flags (from CSV; pixel-only device recovery is H-CXR-005), trained on the 10,000 PriCheXy train pairs (with cross-fitted estimates).
- Releases: raw, PriCheXy mu in {0.001, 0.005, 0.01} (released weights). Optional: random flow of equal RMS displacement (H-CXR-003).
- Metrics: AUC with patient-cluster bootstrap (B = 2000), share E; age MAE and sex accuracy on defended vs raw images (does the defence remove demographics?).
- Test-unit design: PriCheXy-Net arm uses fixed test units (its 1,424 test patients; the released weights were trained on its train patients so rotation would leak); seeds replicate estimator training noise only and this is stated; test noise is represented by the patient bootstrap. The estimators and GBM rotate over 5 patient folds inside the PriCheXy-train pool for their own validation.
- Controls: constant image -> estimators predict the mean (AUC 0.5); shuffled patient labels -> 0.5.

## Prediction and falsifier
- Predicted: raw 0.85-0.92; mu = 0.01 0.80-0.90 (age MAE rises < 1.5 y, sex accuracy > 95%); MN-v1 0.49-0.56; E >= 4 on the defended release.
- FALSIFIED if: (a) real-estimator AUC on PriCheXy random negatives < 0.75 on raw (simulation overstated, the shortcut is weaker than stated); (b) PriCheXy outputs destroy demographic estimability (age MAE doubles, sex accuracy < 90%), giving demographic AUC < 0.65 on the defended release (the defence does remove demographics); (c) MN-v1 AUC stays > 0.60 with estimated values (matching leaky or estimators carry non-demographic identity).

## Cost / dependencies
~4 GPU-h. Feeds H-CXR-002, -003, -004 and H-S-005. Gated by the download.

## Prior
0.65. Stress-test measurement is on true metadata and a simulation of the estimator; realistic estimator errors are image-dependent and may be larger on a defended image (then the E >= 4 claim weakens but (a)-(c) can still fail either way).

## Cheapest floor that could kill it
The same pair protocol with true metadata is already 0.983: the cheapest "kill" is simply that any Siamese number on random negatives is dominated by the metadata line. The only way this hypothesis dies is a measured estimator AUC far below the simulated 0.885.
