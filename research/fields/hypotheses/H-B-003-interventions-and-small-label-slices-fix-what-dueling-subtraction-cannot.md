# H-B-003: A 5% interventional slice or 200 action labels closes >= 40-60% of the shortcut gap; readout-time mean-over-actions subtraction closes < 10%

- Focus: B' (conditional) | Kind: learned remedies compared | Cost: ~8 GPU-h | Prior: 0.30
- Depends on: H-B-002 (a measured shortcut gap on `corr_noisy`).

## Statement
At rho_eff = 0.9 on the sprite kind `corr_noisy` with the largest gap from H-B-002, define gap closed G = (NMSE_none - NMSE_remedy) / (NMSE_none - NMSE_oracle) on the decorrelated test. Then:
(R1) Dueling-style readout subtraction (subtract the mean forward-model prediction over sampled z) closes G < 0.10, as its authors' stated boundary implies;
(R2) mixing p = 5% of transitions in which the executed action is random while the distractor still follows the expert command (an intervention slice, no labels needed) closes G >= 0.40;
(R3) adding an auxiliary action-prediction loss on N = 200 labelled transitions closes G >= 0.60, and the number N* needed to reach G = 0.6 grows with rho_eff: N*(0.97) >= 5 x N*(0.6).

## Mechanism
Dueling World Models cancel only variation shared across actions, hence cannot remove a distractor whose motion tracks the action (2608.06706, abstract opened). Why Latent Actions Fail proves that auxiliary action supervision keeps latent actions consistent across exogenous states (2605.20223, abstract opened). An intervention slice breaks the correlation in a fraction of the data without labels (the LAM sees the arm move randomly while the distractor does not follow), and about 2k labels already suffice for near-peak success in MaskLAM's data (project-page figure quoted in literature/contrarian_stress_test.md 2.1). If R2 works without labels, it is the learned, field-agnostic remedy; if only R3 works, the finding is a label-budget law.

## Experiment (exact)
- Data/models: as H-B-002 (reach-v3, then push-v3; LAPO-style LAM, z-dim 8; sprite kind `corr_noisy` at rho_eff = 0.9, plus 0.6 and 0.97 for R3; test on `indep`). R2 needs one extension of Builder-B's generator: drive the sprites from the scripted expert's *command* (noiseless, before the exploration noise `ACT_NOISE` is added) rather than the executed action, and let the intervention slice execute random actions; `distractors.sample` currently drives the sprites from the executed macro actions, so a `drive_from="command"` option must be added.
- Arms: R0 none; R1 readout subtraction applied post hoc to the trained forward model; R2 p in {1, 5, 20}% intervention slice; R3 N in {50, 200, 1000} labels with the auxiliary loss weight tuned on val; R4 R2 + R3; reference: clean oracle (kind `none`) and GT-mask LAM.
- Probes: action ridge probe at rho_test = 0 (primary), distractor probe R^2.
- Seeds: 3 LAM seeds x 12 arm-levels = 36 runs (~0.2 GPU-h each).
- Metric: G, NMSE ratio to oracle, label budget N*.

## Prediction and falsifier
- Prediction: G(R1) < 0.10; G(R2, 5%) 0.4-0.7 (1%: < 0.2; 20%: > 0.6); G(R3, N = 200) 0.6-0.9; N*(0.97)/N*(0.6) >= 5; R4 > each alone by >= 0.1.
- Falsified if G(R2, 5%) < 0.20 (interventional data does not help an unlabeled LAM), or G(R1) >= 0.40 (readout subtraction works on correlated distractors, contradicting the authors: check for a bug before believing it), or N* does not depend on rho_eff (no label-budget law).

## Cost
~8 GPU-h.

## Dependencies
H-B-002; Builder-B data generator with an intervention-slice option (executed action random, distractor driven by the expert command).

## Prior and why
0.30 compound: R1 failing is very likely (0.85); R3 working is likely (0.7); R2 is the genuinely uncertain, interesting part (0.4), because an unlabeled LAM can still encode both arm and distractor in a larger z.

## Sources
[O] 2608.06706; [O] 2605.20223; literature/contrarian_stress_test.md 2.1 (label budget, MaskLAM project page).
