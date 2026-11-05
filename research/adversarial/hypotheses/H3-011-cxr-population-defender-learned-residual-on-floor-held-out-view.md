# H3-011: The charter's G clause on chest X-ray, with the dither and downsample floors as the bar: a defender trained against a POPULATION of attackers on top of the 4x downsample does not generalise to a held-out attacker VIEW better than a single-attacker defender, and does not beat down_f4 plus matched noise

- Focus: G (conditional, reopened by the CXR power: paired MDE80 about 0.013-0.02 AUC at ~2,900 patients vs 7-8 points on IMU) | Kind: learned method, with population / single / matched-noise controls | GPU-h: 14 | Prior: 0.08 that G opens
- Depends on: H3-008 (fresh bucket F, floor frontier and the committed list-v2); H3-010 (attacker convergence, so that "held-out gap" is not an under-training gap); T3 (floor frontier, calibrated Siamese control tolerance); FOCUS A6 / AUDIT F16 (families must be VIEWS, partition frozen before training). Cost exceeds the G gate of FOCUS (6 GPU-h) because the CXR defenders are image-space; the gate rationale (MDE) no longer applies at this n and the main agent must re-authorise the budget.

## Statement
Defender D_theta(x) = down_f4(x) + delta_theta(x), delta_theta a small U-Net with ||delta||_inf <= eps = 8/255, trained so that a frozen DenseNet-121 (fine-tuned on down_f4 images of S2) keeps the 14-finding loss near the down_f4 level and ||D - down_f4||_1 is penalised. Attacker views (frozen partition): V1 raw-image Siamese ResNet-50; V2 frozen DINOv2-B pair head (differentiable through the encoder); V3 normalised cross-correlation (structural, differentiable conv at 128 px); V4 hand statistics LightGBM (F6; never in any defender training; always held out). Rotation h in {V1, V2, V3}: the population defender D_pop(h) trains against the other two (min-max, softmax-weighted worst case) and h is held out; D_single trains against V1 only with 5 restarts (same compute). All attackers are then RETRAINED on the final release (the programme's standing rule) and the worst case over views is reported. Define Gamma(D) = L_h(D) - L_in(D) with L_in = worst retrained in-population view and L_h the retrained held-out view.
(a) Gamma(D_pop) <= Gamma(D_single) - 0.03 in >= 2 of 3 rotations is the population-benefit claim; predicted FALSE: |Gamma(D_pop) - Gamma(D_single)| <= 0.02.
(b) D_pop beats the floor envelope {down_f4, down_f3, down_f4+noise, down_f4+jpeg, composites of H3-008} at matched utility by dL >= 0.03 on the held-out-view L in >= 2 of 3 rotations: predicted FALSE; the matched-noise control D_noise (down_f4 + a random delta of equal L2 and L-infinity budget) is within 0.01 of D_pop.
G opens only if (a) and (b) both hold with 99% lower bounds and D_noise is within tolerance.

## Mechanism
A defender starting from a strong floor can only add structure-specific distortion within an eps budget. The retrained attacker removes whatever the in-loop population used to chase; the population effect predicted by the minimax-over-attackers literature requires attackers that differ in the cue they read. V1, V2 and V3 all read the same shape and texture cues from the same pixels, so a defender fooling two may fool the third (small Gamma), but the retraining step erases most of the learned part either way. In programme 1, noise and rate controls matched every learned arm at equal leakage; the downsample is the stronger version of the rate control here.

## Experiment (exact)
- Data and roles: fresh bucket F of H3-008 is the SEALED test (defender trained on S2 = S \ F, attackers retrained on the A role of the T3 partition, rotation 0; validation V for early stopping / model selection of the defender's epoch; nothing tuned on F). Test pairs: N2 matched negatives built within F; G0 on F.
- In-loop attackers: V1 trained jointly (Adam 1e-4), V2 head trained on frozen DINOv2 features of the defended image (backprop through the encoder, bf16, 224 px resize), V3 a fixed NCC with a learned scalar calibrator; attacker updates 3 steps per defender step; the defender's loss = utility loss + lambda_g * worst-case-weighted adversarial loss + L1 term; lambda_g in {0.3, 1} (two settings, V-selected); 20 adversarial epochs (123 s per epoch in T2 with one verifier; ~3x with the population; the final checkpoint C2 and the lowest-validation-loss checkpoint C1 both saved, as T3).
- Convergence/collapse check for the in-loop attackers on held-out V pairs, relative to a frozen long-schedule retrain (H3-006 criterion C, gap <= 0.03), reported per rotation; a defender whose in-loop attackers never converge is INVALID for (a).
- Comparators at matched utility (14-finding AUC within 0.01 of the defender): `down_f4`; `down_f4` + Gaussian noise (levels V-selected); `down_f4__jpeg_q50`; the H3-008 envelope; `labels14` as a code reference.
- Evaluation: all four views retrained on the final release (V1 3 seeds, others 1), worst-case L; patient-cluster bootstrap B = 2000, shared resamples; margins `margin_l` 0.03, `margin_u` 0.02; 99% bounds (three rotations x population vs single).
- Seeds vs test units: units = F patients (about 2,900), a single sealed bucket; rotations are over the held-out VIEW, not over patients; 2 defender seeds for rotation V2 only (the one expected to matter most); seeds replicate defender training noise. The independence limit of the test patients (one bucket) is stated; the held-out-view claim is paired within F.

## Prediction and falsifier
- Predicted: Gamma(D_single) 0.02-0.05, Gamma(D_pop) 0.01-0.04; D_pop vs envelope dL in [-0.01, +0.015]; D_noise within 0.01 of D_pop; held-out V4 (hand statistics) L about 0.74-0.76 for every release (V4 is weak: raw 0.752).
- FALSIFIED (G opens) if (a) and (b) hold at 99% in >= 2 of 3 rotations and D_noise does not reproduce the gain; that single result would continue as the programme's one learned line. G stays closed if any condition fails; the in-loop non-convergence outcome is INVALID, not NO-GO.

## Cost / dependencies
14 GPU-h: 3 population defenders (~2 h each at ~3x epoch cost), 1 single defender (1.5 h), a second seed for V2 (2 h), evaluation of 5 defenders + 4 floors x ~0.4 h (3.6 h), utility, collapse and convergence checks (1 h). Needs: DINOv2-in-the-loop smoke test (0.2 h), H3-008's committed list, a ledger entry per release.

## Prior
0.08. Programme 1's four consistent settings and T2's defender at the floor frontier; the one thing that could differ here is that the base is already a strong floor and the n supports a 0.02 difference, which is the first time G can actually be tested.

## Cheapest floor that could kill it
`down_f4 + matched noise` and D_single (the compute-matched single-attacker defender). Either matching D_pop within 0.01 is enough.
