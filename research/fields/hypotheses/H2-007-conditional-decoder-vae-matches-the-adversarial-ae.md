# H2-007: A conditional-decoder VAE (source id given to the decoder, no adversary) reaches the adversarial AE's leakage-matched utility with lower seed variance

- Focus: A (then single-cell, H2-010) | Kind: learned method, non-adversarial alternative | Cost: ~2 GPU-h | Prior: 0.30
- Depends on: H2-001 AE code (add a decoder input); centre one-hot available at training (the same information the adversary uses).

## Statement
A VAE on cached embeddings whose decoder additionally receives the centre one-hot (and whose encoder does not), with rate weight beta in {1e-3, 1e-2, 1e-1} and dz in {8, 16, 32}, and NO adversary, satisfies on midnight and phikon-v2 x {TCGA, Tolkach} (held-out-class protocol, 3 seeds):
1. at matched leakage (L within +-0.05 of the V-selected adversarial AE), U_held(cVAE) is within 0.03 of the adversarial AE in >= 3 of 4 cells;
2. the seed-to-seed s.d. of L at a fixed configuration is <= 0.6 x that of the adversarial AE (no minimax instability);
3. adding the retrained adversary to the cVAE (cVAE+adv, lam 3) improves L by <= 0.05 (the conditioning already did the job).

## Mechanism
The decoder can ignore z for anything the centre id explains, so there is no reconstruction pressure to encode the centre in z; this is the design of scVI-type models in single-cell integration (batch as a covariate of the decoder) and of fair-VAE baselines. It trades the adversary's moving target for a static, convex-ish objective. Known failure: nothing forbids z from encoding centre anyway, and z carries centre information that correlates with other directions (stain also correlates with tissue); leakage then stays high at any beta that preserves utility. The comparison with the adversarial AE says whether the adversary buys a real guarantee.

## Experiment (exact)
- Same grid, splits, attackers, selection and comparators as H2-001; cVAE arms: (i) decoder-conditioned VAE; (ii) cVAE with the centre id also dropped at decoder with p = 0.3 (to avoid a decoder that leans fully on the id); (iii) cVAE + adversary lam in {3}.
- Metrics: L_worst, U_held, retention-xc, and seed s.d. of L and of U_held over 3 seeds x 3 hyper-parameter-identical restarts; bootstrap over sealed units.
- Controls: same as H2-001; a cVAE with a shuffled centre id given to the decoder must leak like the lam = 0 unconditioned VAE (the conditioning, not extra capacity, is the cause).

## Prediction and falsifier
- Prediction: cVAE ties the adversarial AE on U_held (+-0.03) at matched L in 3 of 4 cells; s.d. of L 40-60% of the AE's; cVAE+adv adds <= 0.05.
- Falsified if L(cVAE) - L(adversarial AE) >= +0.15 at matched U_held in >= 3 of 4 cells (centre leaks through z; conditioning is not enough), or U_held(cVAE) <= AE - 0.05 at matched L.

## Cost
~2 GPU-h (small VAEs on cached vectors; attacker evaluation is CPU-bound).

## Dependencies
H2-001; if it passes it supplies the template for the scVI-like arm in H2-010 and H2-013.

## Prior and why
0.30. Conditioning only removes the incentive to encode the source; it does not remove the source from z. In single-cell integration this is nevertheless a strong baseline, so a negative result here would be informative about why scVI sometimes leaves batch structure.

## Sources
literature/other_modalities.md and the opened bioRxiv 10.64898/2025.12.19.695371 list scVI as a standard strong comparator; no claim about scVI internals relied on beyond that.
