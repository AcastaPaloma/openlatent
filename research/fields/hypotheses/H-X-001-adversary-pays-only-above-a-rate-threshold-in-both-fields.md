# H-X-001 (A and C): The adversary adds nothing at d <= 4 and >= 0.10 normalised leakage at d >= 32; the threshold rate is within 4x across pathology and audio

- Focus: A + C (cross-focus transfer test of our own KTH finding) | Kind: learned method (adversary x bottleneck grid) | Cost: ~10 GPU-h | Prior: 0.45
- Depends on: Builder-A and Builder-C adapter training loops, probe suites; H-A-002 and H-C-002 for the lambda=0 row.

## Statement
Cross a bottleneck dimension d in {2, 4, 8, 16, 32, 64, 128} with adversary weight lambda in {0, 1, 3, 10} (adversary = MLP, restarted every 3 epochs with a full warm-up epoch on the frozen encoder, kept near convergence) on Focus A (adapter on Phikon-v2 clsmean, secret = centre) and Focus C (adapter on WavLM-base+ pooled, secret = actor). Define the adversary gain Delta(d) = L_worst(lambda=0) - min over lambda>0 of L_worst(lambda) at matched utility (U within 0.02 of the lambda=0 model, by interpolation on the frontier). Then: Delta(d) <= 0.05 for d <= 4 in both foci; Delta(d) >= 0.10 for some d >= 32 in at least one focus; the smallest d* with Delta >= 0.10 differs by less than 4x between A and C. Additionally, the in-training adversary accuracy is at <= 1.5x chance for >= 30% of lambda>0 runs whose fresh MLP probe reaches >= 3x chance (the fool-the-adversary gap seen in the VAE runs).

## Mechanism
KTH: at 1-2 numbers per frame the adversary buys nothing (a0 81.8/10.7 vs a-adv ~81/12), at ~15 numbers per frame it buys ~15 identity points (89.5/53.2 vs 93.0/40.8) (results/SUMMARY.md, journal 2026-10-28 05:20). Rate caps what can leak; the adversary decides what the spare capacity carries. Methods consensus: rate bottleneck without adversary is the mandatory control, adversarial gain over it at matched rate is the headline number (literature/methods_transfer.md section (b)).

## Experiment (exact)
- Focus A: PathoROB (3 datasets), phikonv2_clsmean, adapters trained on ID training slides; secret labels = centre; probes A-SUITE slide-disjoint; seen-centre and unseen-centre (source-probe) leakage reported separately.
- Focus C: CREMA-D 4 folds, wavlm-base-plus best layer; adapter on encoder-train actors; attackers enrol test-fold actors (T1/T2/T3), C-SUITE.
- Seeds: 3 encoder seeds x 3 attacker seeds per cell (7 x 4 x 3 = 84 runs per dataset/fold, each 1-3 min).
- Metric: L_worst, U, Delta(d); frontier interpolation (linear in U at fixed d across lambda); bootstrap over slides (A) and actors (C).
- Normalisation: raw = (1, 1), chance L=0; task-label-only reference.

## Prediction and falsifier
- Prediction: A: Delta(2..4) <= 0.05, Delta(64) in 0.08-0.2; C: Delta(2..4) <= 0.05, Delta(64) in 0.08-0.25; d* (A) and d* (C) both in [16, 64]. In-training-adversary vs fresh-probe gap present in >= 30% of runs at d >= 32.
- Falsified if the adversary adds >= 0.10 already at d <= 4 in either focus (rate is not the main dial), or Delta < 0.05 at every d in both foci (adversary useless: then drop adversarial methods from the programme and keep rate + closed-form), or d* differs by > 8x between foci (the KTH regularity does not transfer).

## Cost
~10 GPU-h total (A ~4, C ~6).

## Dependencies
Builder-A and Builder-C training loops with the shared adversary module; H-A-002 / H-C-002 for the lambda = 0 rows.

## Prior and why
0.45. The compound statement has four parts; the "no gain at low rate" half is likely (0.75), the "gain at high rate" half ~0.65, and the cross-field d* equality half ~0.6.

## Sources
results/SUMMARY.md and docs/journal.md (KTH); [S] literature/methods_transfer.md; [O] 2101.10001 (diverse adversaries, instability of single adversaries).
