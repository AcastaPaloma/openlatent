# H-X-002: What would make us stop programme 2: no learned method beats the lower floor envelope on a held-out family, and the instrument adds nothing over a one-line statistic

- Focus: X (programme-level) | Kind: stop hypothesis / decision table | GPU-h: 0 (decision logic over the gates; the cheap gates cost ~25 GPU-h in total) | Prior: 0.75 that no learned method beats the floor envelope; 0.25 that all four gates fail and the programme stops
- Depends on: H-CXR-002, H-IMU-001, H-G-001 / H-G-002, H-S-005, H-X-001; charter stop rule ("if cheap gates (floors, headroom, power) fail, stop the focus and write the negative result with its by-products; do not spend GPU to fill the queue").

## Statement
Programme 2 is a measurement and negative-result programme unless a learned method clears the floor envelope on a held-out attacker family under cross-fit. Pre-registered decision table (all four gates evaluated with 3 seeds, cross-fit, original-and-tightened verdicts reported side by side; no gate may be loosened after data):
Each gate has two readings: an AUDIT reading (the published claim changes; this is a write-up result, costs no further GPU) and a LEARNED-LINE reading (a learned method beats the floor envelope; this is the only thing that justifies further GPU). The STOP rule is about the second reading.
- Gate R-CXR. Audit survives if H-CXR-002 passes (Siamese - best demographic >= 0.03 AUC on MN-v1) and H-CXR-001 / -004 move PriCheXy's published number by more than the MDE. Learned line survives only if, in addition, H-CXR-003 shows PriCheXy-Net (or a G-002 adapter) below the floor envelope by >= 0.05 AUC at matched utility. Else R-CXR ends as: "the published linkage defence is a demographics + device + weak-attacker artefact", with MN-v1 and the demographic attacker as by-products.
- Gate R-IMU. Audit survives if H-IMU-001 arm A shows PrivDiffuser above the activity-LDA-k frontier by > MDE80 on identity (or H-IMU-003 shows the split inflation). Learned line survives only if arm B or H-IMU-005 shows a learned release below the envelope by >= MDE80 (7.5 points at n = 144, 4.8 at n = 360) and H-X-001 does not kill it. Else R-IMU ends as an audit result.
- Gate G. Opens only if H-G-001 or H-G-002 shows the population defender beating the envelope by >= MDE on the held-out family in >= 3 of 4 rotations. Else G stays closed.
- Gate S. Instrument justified if H-S-005 shows the one-line statistic within MDE80 of the suite in <= 30% of low-leak cells AND H-S-001 resolves a planted leak inside the MDD band stated there. Else S is reported as a reporting standard with "not resolved" outputs, not as a measuring advance.
STOP rule: if no learned line survives (R-CXR, R-IMU, G all fail their learned reading) AND the instrument's justification fails, programme 2 ends within ~3 working days and ~20 GPU-h: write the protocol findings (MN-v1, demographic attacker, window-split memorisation, floors), release the VM. If only the learned reading fails everywhere but the audits and S survive, programme 2 continues as a zero-new-GPU write-up. Any single surviving learned line continues alone, with at most one stage-2 attempt (<= 5 GPU-h) pre-registered before it starts, as programme 1 did for F. Predictions: P(no learned method beats the floor envelope on a held-out family) = 0.75 (P(one does) = 0.10 after allowing for the fraction that fails H-X-001); P(full stop, i.e. learned lines AND instrument all fail) = 0.25 (the gates are positively correlated through the shared floors-win mechanism; independence would give about 0.12); P(audit survives) R-CXR 0.80, R-IMU 0.70; P(G opens) 0.10; P(S justified) 0.55.

## Mechanism
Four settings of programme 1 ended with cheap floors winning (label-only LDA, per-source whitening, rate bottleneck / dither); the stress test measured MDEs above expected effects for IMU and G. The charter already says what to do on negatives; this file fixes the thresholds in advance so a negative cannot be quietly reinterpreted.

## Experiment (exact)
No new experiment (the cheap-gate hypotheses together cost about 25 GPU-h; the STOP clause above says "~20 GPU-h" as the budget at which the decision is taken, with the remainder conditional). The decision is a table read from the results of the named hypotheses at the end of the cheap-gate stage: each gate's verdict is "pass", "fail", "unresolved (inside MDE)", with the MDE printed. "Unresolved" counts as fail for the purposes of spending further GPU, but is reported as unresolved. Verdicts are stated with BOTH the original criteria and any tightened ones.

## Prediction and falsifier
- Predicted: R-CXR and R-IMU both end with "floors and protocol artefacts explain most of the published claim"; the audit changes at least one headline number (PriCheXy's random-negative AUC; PrivDiffuser's identity leakage) by more than its MDE.
- FALSIFIED (the programme should NOT stop) if any learned method (PrivDiffuser retrain, H-IMU-005 AE, H-G-001 / H-G-002 population defender, a PriCheXy-style defender) beats the lower envelope of ALL floors on the held-out family by >= MDE80 under cross-fit with 3 agreeing seeds and passes H-X-001. That single result becomes the programme's one continuing line.

## Cost / dependencies
Reads results only. Needs the gates above to be pre-registered before the first number (they are listed in INDEX.md).

## Prior
0.75 for "no learned win on a held-out family"; 0.25 for the full-stop outcome (learned lines and instrument all fail). Audit results are not failures of the programme: they are its likeliest publishable outputs.

## Cheapest floor that could kill it
A single learned win. The first place to look is H-IMU-001 arm B (identity-guided PrivDiffuser retrain below the floor envelope), the cheapest learned method with prior headroom (stress test 4.2).
