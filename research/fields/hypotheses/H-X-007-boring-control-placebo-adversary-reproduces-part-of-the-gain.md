# H-X-007 (BORING CONTROL, A and C): An adversary trained against shuffled or irrelevant labels reproduces >= 40% of the real adversary's leakage reduction

- Focus: A + C | Kind: boring control that can kill an exciting result | Cost: ~2 GPU-h | Prior: 0.30 (that the placebo gets >= 40%)
- Depends on: X-001 grid (best real-adversary configuration per focus).

## Statement
At the d and lambda where the real adversary (X-001) gives its largest gain Delta over lambda = 0, a placebo adversary run with identical architecture, restarts, warm-up and gradient-reversal weight but with (a) secret labels permuted across *samples* (no source structure) or (b) an irrelevant real attribute (A: a random 50-way hash partition of slides unrelated to centre; C: sentence identity) produces at least 40% of Delta in at least one focus.

## Mechanism
The adversary term also acts as a regulariser (noise in the gradient, reduced effective capacity, early stopping through restarts). In our video runs, adding an adversary at high rate changed which information the bottleneck kept, and encoder bias mattered as much as the adversary (journal 2026-10-28 05:55: a U-Net encoder with adversary collapsed to chance while BDQ did not). Without a placebo, a gain from the adversary cannot be attributed to adversarial pressure on the secret. The same logic holds for the SISER-style utility gain in H-C-007.

## Experiment (exact)
- Setup identical to the best X-001 cell per focus (A: phikonv2, d = 64, lambda*; C: wavlm-base-plus, d = 64, lambda*), unseen-source evaluation.
- Arms: real adversary; placebo (a) label-permuted; placebo (b) irrelevant attribute; lambda = 0; GRL weight 0 but adversary still trained (a zero-gradient control); a noise-injection control (Gaussian noise on z with sd matched to the gradient norm of the real arm).
- Attackers: A-SUITE / C-SUITE.
- Seeds: 5 encoder seeds (this control must be tighter) x 3 attacker seeds; bootstrap over sources.
- Metric: share = (L(0) - L(placebo)) / (L(0) - L(real)); U.

## Prediction and falsifier
- Prediction: share(placebo-a) in 0.1-0.5 with median ~0.25; share(placebo-b) in 0.1-0.4; the noise control gets 0.1-0.3; U within 0.02 for all.
- Falsified (the control kills) if share >= 0.40 in either placebo for either focus: the adversary is mostly a regulariser and the programme should report "regularisation" and keep only the rate/closed-form story. Falsified the other way (the real adversary is real) if share <= 0.15 everywhere.

## Cost
~2 GPU-h.

## Dependencies
X-001 outputs; a shuffled-label adversary flag in both loops.

## Prior and why
0.30. I expect small placebo effects, but this control is rarely run and regularisation effects of a restarted adversary on a high-capacity bottleneck can be considerable; getting >= 40% in one of four (focus x placebo) cells is not unlikely.

## Sources
Our journal and SUMMARY (adversary effects); [S] literature/methods_transfer.md sceptic's paragraph.
