# H3-009: Chest X-ray: at larger displacement budgets (mu 0.02 and 0.04) and with the last-epoch checkpoint, the retrained PriCheXy-style defender still sits on or above the floor frontier and does no better than a RANDOM warp with the same measured spectrum

- Focus: R-CXR | Kind: learned defender dose-response with a same-mechanism random control | GPU-h: 5 reduced / 9 full | Prior: 0.12 that any variant beats the envelope by the margins; 0.85 that learning adds <= 0.02 over the same-spectrum random warp
- Depends on: T3 (second seed at mu 0.01 and mu 0.005, C1 and C2 checkpoints; its floor frontier and nominations; base weights from T2 `t_x_pcxr_base`); the frozen list entries `rw2_mu{0.02,0.04}`; a NEW list entry type `rw3_mu{0.02,0.04}` (random flow with the radially averaged power spectrum and per-image RMS measured on the new defenders), which requires a list-v3 hash and a custodian note before sealed use. Units: the T3 partition (rotations 0 and 1 of the t3 salt) so that every comparison is PAIRED with T3's floor results; the independence limit of T3 ("roles re-drawn over the same patients") is inherited and stated.

## Statement
T3 covers mu = 0.01 and 0.005. At those budgets the learned displacement is tiny (T2: RMS 0.66 px, max 1.27 px at mu 0.01, the bound is mu x 127.5 px) and every warp-pipeline release carries a zero-padding border artefact. Claim for the larger budgets:
(a) `pcxr_mu0.04` and `pcxr_mu0.02` at the last epoch (C2), single seed each, have worst-case linkage L (rotations 0-1 pooled) within [-0.03, +0.02] of the floor frontier interpolated at their utility, i.e. dL <= 0.02 and dU <= 0.01 (no crossing); the 99% lower bound of dL stays below the 0.03 margin and of dU below 0.02;
(b) the learned displacement at mu 0.04 / C2 has RMS >= 2.5 px (the family is no longer in its trivial regime);
(c) at equal measured spectrum the random warp `rw3_mu0.04` leaks as much as the learned defender: L(pcxr) - L(rw3) in [-0.02, +0.02], and U differs by < 0.01 (learning adds nothing over the mechanism it parameterises);
(d) lengthening the adversarial stage at mu 0.04 from 40 to 120 epochs (the paper used 250) changes L by < 0.02.

## Mechanism
A smooth bounded warp changes the registration between visits by at most a few pixels. Linkage models trained on matched (defended, REAL) pairs read interior texture and shape that survive small warps, which is why random warps of the same strength leak 0.853 vs 0.860 for the learned one at mu 0.01. As mu grows the warp starts to disturb anatomy as well and utility falls; a learned warp can place its distortion where the attacker looks, but the attackers are retrained on the final release, so that advantage is exactly what retraining removes. Nothing in T2/T3 tests whether the answer changes outside the mu = 0.005-0.01 regime where the generator barely moves.

## Experiment (exact)
- Defenders (recipe A3, all components trained on S only, T2 base weights): `pcxr_mu0.02` seed 0 and `pcxr_mu0.04` seed 0, 40 adversarial epochs, checkpoints C1 (lowest validation total loss) and C2 (last epoch) saved; plus, in the full tier, `pcxr_mu0.04` seed 0 at 120 epochs (C2 only; reserved cut: first to go). The A3 selection rule is not changed.
- Floors: the frozen list members `rw2_mu{0.02,0.04}`; the new `rw3_mu0.02`, `rw3_mu0.04` (flow statistics measured on the new defenders' applied displacement with the border artefact subtracted, T `flowstats.py`); plus T3's nominated floors (re-used, same units). The envelope is T3's frontier extended by the new floors; margins `margin_u = 0.02`, `margin_l = 0.03` fixed now; five defenders ->, for the two defenders x two checkpoints here, Bonferroni 99.5% lower bounds.
- Evaluation on T3 rotations 0 and 1: F1 (match), F3, F4, F5a / F5b, plus F2 and F6 for the two C2 defenders and their `rw3` partners; utility U (S-trained DenseNet-121 fine-tuned on the defender's own defended S images, as T2). Border-artefact control: `warp0` and a zero-flow release with the artefact corrected (mirror padding) so the share of leakage due to the border is reported.
- Controls: const and shuffled-label controls with T3's calibrated tolerance (delta_F1 = 0.06 for the Siamese family); the sealed-query ledger digest includes the salt id.
- Seeds vs test units: units = T3's 3,060 patients over rotations 0-1; single seed per defender (stated; a second seed of the highest-mu defender is run only if its C2 beats the frontier on either axis at the 95% level); training noise and test noise are therefore not separated for the defender, but the same-spectrum random warp provides a within-unit paired control. MDE80 printed (T2: dU 0.004, dL 0.0135 on 4,585 patients; about 1.2x larger on 3,060).

## Prediction and falsifier
- Predicted: L(pcxr_mu0.04, C2) 0.80-0.84 at U 0.755-0.775; L(pcxr_mu0.02, C2) 0.83-0.85; both on the frontier within the CIs; L(rw3_mu0.04) within 0.02 of the learned one; C1 and C2 differ by < 0.02 at mu 0.02.
- FALSIFIED (the learned family does cross the frontier) if either defender, at C1 or C2, has a 99.5% lower bound of dL >= 0.03 OR of dU >= 0.02 on the pooled rotations, in both rotations separately (sign agreement), with the same-spectrum random warp NOT achieving the same. FALSIFIED on (c) if the learned defender beats `rw3` at equal spectrum by >= 0.03 in L at U within 0.01 (learning helps beyond the mechanism; this alone would be a positive finding even if the frontier is not crossed).

## Cost / dependencies
Reduced 5 GPU-h = two adversarial runs (mu 0.02, 0.04) x ~1.4 h + evaluation of 2 defenders x 2 checkpoints + 2 `rw3` floors on 2 rotations (~0.2 h per release-rotation) + utility. Full 9 GPU-h adds the 120-epoch run (~4 h) and F2 / F6. Starts after T3 has produced its nomination and frontier.

## Prior
0.12. T2's defender barely differed from `rw3_mu0.01` (0.860 vs 0.853) and from raw (0.887); published results (57.7 vs 67.0 replicated here) suggest stronger effects only on a weaker verifier. A larger mu makes the utility cost real and puts the defender right on the random-warp curve.

## Cheapest floor that could kill it
`rw3` at the same mu (same spectrum, no learning) and `down_f4`, which already sits 0.055 lower than the mu = 0.01 defender at equal utility.
