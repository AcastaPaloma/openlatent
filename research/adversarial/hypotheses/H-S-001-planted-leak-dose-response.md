# H-S-001: Planted-leak dose-response: the sealed suite's minimum detectable leak is set by independent units, not by attackers

- Focus: S (instrument) | Kind: calibration / premise | GPU-h: ~4 (IMU 1.5, CXR 2.5; frozen-feature members plus small CNN / ResNet-18 Siamese at 128 px, calibration only) | Prior: 0.70
- Depends on: Builder S saved-prediction schema and locked-evaluator interface; MotionSense windows (Builder H licence check first); the CXR arm needs the ChestX-ray14 download gate (Auditor-2 clearance + main agent release).

## Statement
Into a release whose identity information is zero by construction, plant a leak of known, dialled dose. Then (1) the dose-0 null sits at chance: the suite's val-selected attacker exceeds the Bonferroni upper bound over K members in at most 5% (accept 2-8%) of 200 independent null replications; (2) detection is monotone in dose; (3) the minimum detectable dose (MDD: 80% power, one-sided alpha 0.05, Bonferroni over K) corresponds to a matched-filter oracle identity accuracy of 18-30% (4-7x chance) on MotionSense (144 test trials, 24 subjects) and to an oracle AUC of 0.55-0.58 on ChestX-ray14 pairs (1,424 test patients); (4) for at least one planted channel type the suite sits at least 5 points of accuracy (IMU) or 0.03 AUC (CXR) below its own oracle at the top dose, i.e. the suite has a blind spot. Below the MDD the certificate must print "not resolved", never "private".

## Mechanism
Certificate width is a unit-count effect (stress test 2.2: at 144-150 units a zero observed leak reads chance +3.7 points at K=1 and +7.1 at K=30; at 1,424 patients +1.0 and +1.8). A planted leak with a key-aware oracle separates "suite too weak" (suite below oracle at high dose) from "test too small" (oracle itself unresolved at low dose). Nobody has such a calibration for HAR or CXR releases; the agentic scout (D1) proposes it, the leakage scout flags construct validity as the main risk.

## Experiment (exact)
- Null releases. IMU: donor swap, each window of (subject s, activity a) replaced by a window of a randomly drawn different subject doing the same activity (fresh draw per window), so activity and marginals survive and identity is zero. CXR: per-image independent donor draw from the same (sex, view, age-decile) cell, so two images of one patient are independent; utility is destroyed on purpose, this is a calibration set, not a defence.
- Planted channels, 6 doses plus dose 0. IMU: (a) subject-specific spectral line, f_s = 0.8 s Hz (s = 1..24), random phase per trial, amplitude {0.02, 0.05, 0.1, 0.2, 0.4, 0.8} x channel SD on userAcceleration; (b) subject-specific gain exp(tau z_s) on userAcceleration, tau in {0.02, 0.05, 0.1, 0.2, 0.4, 0.8}; (c) subject-specific constant offset on rotationRate (placement / session analogue), same SD grid. CXR: (a) patient-specific fixed low-pass random field (8 cycles/image), (b) patient-specific fixed white-noise texture; amplitudes {0.5, 1, 2, 4, 8}/255 added after the null draw, on 256-px images.
- Oracle arm: key-aware matched filter (spectral-line correlator, gain likelihood ratio, offset nearest-centroid; CXR: NCC of band-passed residuals of the two images). It gives the attainable accuracy / AUC per dose.
- Suite (retrained on the planted release, 3 restarts): IMU: LR, LDA, kNN (cosine and Canberra), random forest, LightGBM, MLP on hand features (std, MAD, percentiles, 16 FFT bins per channel), 1D-CNN on raw windows; trial score = mean log-prob over the trial's windows. CXR: ResNet-18 Siamese, frozen DINOv2-L kNN (Canberra) + InfoNCE head, NCC/SSIM, GBM on 16x16 raw-pixel difference features.
- Certificate: val-selected member scored once on the sealed test; one-sided Clopper-Pearson with Bonferroni over K (`src/stress_test/cert.py`); cluster bootstrap B=2000 over trials (IMU, subject clusters reported beside) and over patients (CXR).
- Test-unit design. IMU: unit = trial; 3 rank folds (fold r tests the rank-r trial of every subject x activity), so all 360 trials are tested once; seeds = 3 attacker inits inside each fold (they replicate training noise, the folds replicate test units). CXR: unit = patient; 5 patient-hash folds over the 13,302 multi-visit patients (no defender is trained here, so rotation is free); headline fold = the official-test 1,424 patients.
- Controls: constant input and shuffled subject labels at chance; raw reference above the donor-swap null; 200 null replications re-draw the donors.

## Prediction and falsifier
- Predicted: null violation rate 2-8%; IMU 80%-power dose has oracle accuracy 18-30%; CXR 80%-power dose has oracle AUC 0.55-0.58; the K=7 Bonferroni MDD is 1.2-1.6x the K=1 MDD in dose units; blind spot present for gain-type planting (tree and CNN members read amplitude statistics, a pure gain on a noisy donor is the hardest case) by >= 5 points.
- FALSIFIED if: (a) the null violation rate exceeds 10% (harness or split leak, stop and fix before anything else); (b) the suite resolves oracle accuracy below 10% at n = 144 (my MDE analysis was too pessimistic, the certificate is worth running everywhere); (c) a one-line statistic (per-subject mean/SD LDA; CXR 16x16 NCC) reproduces the suite's power curve at every dose (hands off to H-S-005).

## Cost / dependencies
~4 GPU-h; Builder S interface (saved `unit_id, cluster_id, y, probs` per attacker); CXR arm after the download gate.

## Prior
0.70. The unit-count arithmetic is analytic and measured on KTH (design effect 1.8-4.0); the uncertain parts are the exact location of the MDD band and whether donor-swapped real windows truly carry zero identity (checked by the shuffled/null control).

## Cheapest floor that could kill it
The one-line statistic (H-S-005) matching the suite at every dose, or a null above chance (harness bug). Both cost under an hour on CPU.
