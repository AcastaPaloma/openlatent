# H-A-003: A class-conditional adversary preserves biology under centre-class correlation where marginal erasure (LEACE, plain adversary) destroys it

- Focus: A | Kind: learned method (adversarial, conditional MI) | Cost: ~3 GPU-h | Prior: 0.45
- Depends on: Builder-A cache; PathoROB split generator for Cramer's V levels (APD protocol); A-SUITE.

## Statement
Train adapters on splits where centre and tissue class are correlated (Cramer's V = 0.5 and 0.8 in training, as in the APD protocol). Evaluate on the balanced V=0 split (ID centres) and on OOD centres. A *class-conditional* adversary (the adversary receives (z, y) and predicts centre; the encoder maximises its loss, i.e. it minimises I(z; centre | y)) achieves at V = 0.8 at least +3 points more biology accuracy than both (a) a marginal adversary (predicts centre from z alone) and (b) LEACE(centre), at the same L_worst (within 0.05). At V = 0 the three are within 1 point (control).

## Mechanism
When centre and class are correlated in training, "centre information" includes a legitimate shortcut through the class. A marginal eraser must remove everything predicting centre, including class-bearing directions, so it deletes biology; this is the ComBat pattern ("removing biological signal correlated with centre", 2507.17845 mitigation section; DANN helps only +0.23 points). The conditional objective only penalises centre information *beyond* what the label explains, which is the quantity that stays a nuisance when the correlation is broken at test time. Conditional-MI losses exist in fairness VAEs (CAD-VAE 2503.07938, scout-opened, no numbers on its page), and Muller & Berens find that disentanglement helps under strong spurious correlation and that rebalancing + disentanglement beats rebalancing alone (2602.18502, abstract opened).

## Experiment (exact)
- Data: Camelyon (2 classes x 2 ID centres), TCGA 4x4 (4 classes x 4 centres), Tolkach (6 x 3). Training sets built with the PathoROB generator at V in {0, 0.5, 0.8} (V=1 is excluded: centre = class, non-identifiable). Test: balanced V=0 ID, and OOD centres.
- Encoders: phikonv2_clsmean, phikon_clsmean, midnight_clsmean.
- Methods (adapter 2048 -> 1024 -> 128, tissue head): M0 CE only; M1 marginal GRL adversary (MLP, restarted every 3 epochs, warm start, kept near convergence as in journal H18/b_steps fix); M2 conditional adversary (MLP over concat(z, one-hot y)); M3 LEACE(centre) fitted on the correlated training split; M4 conditional-LEACE baseline: per-class centre-mean subtraction on train, class estimated at test by M0 head (a hand-built conditional floor); M5 ComBat with class covariate as a reference.
- Attackers: A-SUITE retrained on the *balanced* ID slides' transformed embeddings (so correlation cannot be used by the probe). Seeds: 3 adapter x 3 attacker.
- Metric: biology accuracy on balanced ID and OOD (U normalised to raw trained at V=0); L_worst; the APD-style relative drop vs V=0-trained model.

## Prediction and falsifier
- Prediction: at V=0.8, U(M2) - U(M1) >= 0.03 and U(M2) - U(M3) >= 0.03 on >= 2 of 3 datasets (TCGA 4x4 largest, ~5 points), |L_worst(M2) - L_worst(M1)| <= 0.05, and M2 U within 1 point of M0 U trained on V=0 data. At V=0: all within 1 point. LEACE loses >= 5 points on Camelyon at V=0.8 (centre direction aligned with class).
- Falsified if U(M2) - max(U(M1), U(M3)) < 0.01 at V=0.8 on all datasets (conditioning buys nothing), or L_worst(M2) exceeds L_worst(M1) by > 0.1 (conditioning leaks centre).

## Cost
~3 GPU-h: 3 V-levels x 3 encoders x 3 datasets x 5 methods x 3 seeds ~ 400 adapter runs of ~1-2 min.

## Dependencies
Builder-A: split generator reproducing PathoROB's APD splits at chosen V (or the repo's own function); shared adversary loop with restarts.

## Prior and why
0.45. The mechanism is clean, but a conditional adversary can degrade to the marginal one when y is nearly constant within a centre (V near 1), and APD gains in the literature are tiny (Reinhard +1.11, DANN +0.23 points), so a 3-point gap needs strong correlation; at V=0.8 on 2-class Camelyon the effect may only show there.

## Sources
[O] 2507.17845 (APD, ComBat/DANN results); [O] 2602.18502 (shortcut mitigation benchmark); [S] 2503.07938 CAD-VAE; [O] 2006.07500 (class-conditional invariance is not sufficient; matching); our journal H18 (adversary fools restarting adversaries).
