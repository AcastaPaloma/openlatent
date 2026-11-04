# H-X-003 (A and C): A batch distance-correlation penalty matches the adversary's worst-case leakage within 0.05 at <= 1/5 wall-clock; a CLUB mutual-information bound does worse

- Focus: A + C | Kind: learned method (dependence minimisation, non-adversarial) | Cost: ~4 GPU-h | Prior: 0.35
- Depends on: shared adapter training loop; X-001 / X-002 adversary baselines for the comparison.

## Statement
Replace the adversary by (i) a distance-correlation (dCor) penalty between z and the one-hot secret computed per batch, or (ii) the CLUB contrastive log-ratio upper bound of I(z; secret) with a variational q(secret | z). With lambda tuned on validation sources, dCor reaches L_worst (held-out attacker families, unseen sources) within 0.05 of the single-MLP adversary's at <= 1/5 of its wall-clock, in both A and C; CLUB is worse than dCor by >= 0.05 at the same lambda budget.

## Mechanism
Muller & Berens benchmark adversarial learning, latent-space splitting and dependence minimisation for shortcut mitigation in medical imaging and find that the best models combine rebalancing with disentanglement and that mitigation helps under strong spurious correlation (2602.18502, abstract opened; the "dCor+Rebal about as good as MINE and 100x cheaper" detail is from literature/medical_biosignal.md #14, not on the abstract page). dCor is a differentiable, parameter-free dependence measure with no inner optimisation, so it avoids the restart/warm-up problems of minimax (H18). CLUB (2006.12013, abstract opened) is an MI *upper* bound needing a variational conditional q, so it inherits the adversary's capacity dependence and has high variance with few samples per class. Risk: dCor against a one-hot with 22-56 classes per batch of 512 gives ~10 samples per class.

## Experiment (exact)
- Data/encoders: as X-001 (A: PathoROB phikonv2_clsmean; C: CREMA-D wavlm-base-plus pooled).
- Batching: class-balanced sampler (A: 8 centres x 64 patches; C: 32 actors x 16 clips) so dCor sees every secret class.
- Arms: lambda in {0.1, 0.3, 1, 3, 10} for dCor(z, onehot) with unbiased U-centering; CLUB (q = 2x256 MLP, 5 inner steps); HSIC (RBF, median heuristic) as a third; single adversary (X-001); CE only; LEACE.
- Evaluation: A-SUITE / C-SUITE on unseen sources; held-out attacker families as in X-002.
- Seeds: 3 x 3; wall-clock measured on the same GPU type.
- Metric: L_worst, U, wall-clock per run, and the lambda needed for L_worst <= 0.5.

## Prediction and falsifier
- Prediction: dCor L_worst within 0.05 of adversary in both foci at U >= 0.97; time ratio dCor/adversary <= 0.2; CLUB L_worst >= dCor + 0.05; HSIC between. A: dCor equal or better on unseen centres (no restarts to game).
- Falsified if dCor L_worst > adversary + 0.10 in both foci (penalty cannot replace the adversary), or CLUB <= dCor (the bound is not the bottleneck).

## Cost
~4 GPU-h (A 1.5, C 2.5; dCor adds an O(B^2) kernel per step).

## Dependencies
Builder-A / Builder-C loops with a balanced sampler; X-001 baseline.

## Prior and why
0.35. Cheaper loss, plausible parity in fairness benchmarks, but our own experience is that in-training dependence measures can be gamed by the encoder exactly as adversaries are (the VAE fooled both of its adversaries), and a dependence measure on one-hot labels is silent about nonlinear structure the held-out attackers use.

## Sources
[O] 2602.18502; [O] 2006.12013 CLUB; our journal H18.
