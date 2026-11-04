# H-C-007: Speaker-adversarial emotion encoders (GRL / SISER-style entropy adversary) raise actor-disjoint emotion by 1-4 points but lower retrained-probe speaker leakage by < 0.10

- Focus: C | Kind: learned method (adversarial) under honest evaluation | Cost: ~5 GPU-h | Prior: 0.55
- Depends on: Builder-C frame features; C-SUITE including the x-vector TDNN attacker; H-C-002 CE baseline.

## Statement
On CREMA-D with frozen WavLM features and a trainable pooling+MLP adapter (d = 128), adding a speaker adversary on the training actors improves emotion UAR on unseen actors by 1-4 points over CE-only (it acts as a regulariser against speaker idiosyncrasy), but reduces L_worst of retrained speaker probes on the unseen test-fold actors by at most 0.10 (absolute) for gradient-reversal, entropy-maximisation (SISER-style) and a diverse-population adversary alike.

## Mechanism
SISER reports that an ECAPA-TDNN speaker discriminator with an entropy objective lifts IEMOCAP UA from 56.46 (wav2vec 2.0 without suppression) to 60.63 (2609.02941, abstract opened); it reports no retrained speaker probe on that page, so speaker removal is not shown. The same pattern elsewhere: NeuroPriv drops EEG identity 0.692 -> 0.206 with chance 0.028 (7x chance remains; 2609.00390, scout-opened); in our video work a VAE fooled its in-training adversary at chance while a fresh attacker reached 75.6% (journal 13:20, H18). Adversaries trained on 56 labelled actors constrain those 56 actors' directions only; the test-fold actors are new people.

## Experiment (exact)
- Data/protocol: CREMA-D, 4 folds; encoder trained on train actors with speaker labels; attackers enrol test-fold actors (T1 primary).
- Encoder: wavlm-base-plus best layer frames -> attentive pooling -> MLP 768 -> 512 -> 128; emotion head.
- Arms: (0) CE only; (1) GRL linear speaker head (alpha in {0.3, 1, 3}); (2) GRL with ECAPA-style small TDNN adversary, entropy-maximisation loss (SISER-style; weights of the adversary re-initialised every 3 epochs, warm-started for one epoch on a frozen encoder, as the journal's H18 fix); (3) population adversary from X-002; (4) LEACE; (5) rate-matched bottleneck d=6 (H-C-002).
- Attackers: C-SUITE (LR, MLP, kNN, LightGBM, x-vector TDNN on frames if sequence). In addition report the *in-training adversary accuracy* to quantify the fool-the-adversary gap.
- Seeds: 3 x 4 folds; bootstrap over actors.
- Metric: emotion UAR on test-fold actors (U), L_worst on T1/T2/T3, gap = L_worst(fresh) - L(in-training adversary).

## Prediction and falsifier
- Prediction: arms 1-3 give U +0.01 to +0.04 over arm 0; L_worst(arm i) >= L_worst(arm 0) - 0.10 for i = 1..3; in-training adversary accuracy at <= 1.5x chance while fresh MLP/TDNN probes reach >= 4x chance (gap >= 0.3); arm 5 (rate-matched, no adversary) has lower L_worst than arms 1-2 at |dU| <= 0.03.
- Falsified if any adversarial arm reduces L_worst by >= 0.25 vs CE-only at U within 0.03 (the adversary does hide speaker), or if no arm improves U (no regularisation effect).

## Cost
~5 GPU-h: 6 arms x 3 alpha/seed x 4 folds with the TDNN adversary at ~5 min per run.

## Dependencies
Builder-C: adversary module with restarts; TDNN attacker; H-C-002 baseline.

## Prior and why
0.55. The utility-gain half has literature support from SISER (self-reported, one dataset); the "privacy barely moves" half repeats a pattern we saw four times in video, but speech has a more homogeneous speaker space where a strong adversary could do better (population variant X-002).

## Sources
[O] 2609.02941 SISER; [S] 2609.00390 NeuroPriv (literature/methods_transfer.md); our journal H18; [O] 2101.10001 (diverse adversaries, orthogonality).
