# AUDIT_results (Auditor-3): independent re-derivation of programme-2 headline numbers

Scope: wearable IMU (arm B, B2, H3-017) and chest X-ray (T3, T4a, T4b). I read only the result JSON, not the builders' aggregator output. My scripts are in the session scratchpad (audit3/). Read-only.

**Key limit:** no per-trial, per-patient or per-pair score is in the repo (198 files under results/; no .npz/.npy; no numeric array of length >= 20 in any h_imu or t_cxr JSON). AUCs, accuracies and bootstrap intervals CANNOT be recomputed. I re-derived everything that is a function of the reported arm-level values.

## Table

| item | claimed | checked | status |
|---|---|---|---|
| 1 arm B aae_re L / U | 0.948 [0.923,0.970] / 0.926 | armB and armB2 JSON agree on all 27 shared arms. Per-fold L (0.972/0.983/0.870) weighted 96/96/72 gives 0.9482 | CONSISTENT-ONLY |
| 1 B2 runs; predlabel | L 0.937, 0.914; U 0.934, 0.943; predlabel 0.042 / 0.937 | identical in armB2_result.json | CONSISTENT-ONLY |
| 1 Delta_L, Delta_U, label | -0.894/-0.029 (arm B); C1s1 -0.878/-0.020; C2 -0.111/-0.009 | my Pareto-interpolation code on the JSON arm (L,U) and floor_set lists reproduces all 8 point values to 4 dp | REPRODUCED (points); percentile bounds consistent-only |
| 1 NO-GO-MATCHED | m_L 0.150, m_U 0.0285 | every hi95 (max +0.057 L, +0.0054 U) is below its margin: MATCH on all axes; MDE80 = 2.556 x SE = 0.1006 / 0.0267; K1 not fired | REPRODUCED from JSON fields |
| 1 L range | 0.034 | 0.948232 - 0.914141 = 0.034091 (CI consistent-only) | REPRODUCED from reported L |
| R0-AAE msda frozen ID | 4.07% | r0_aae.json per-run values 4.06/5.28/2.89 give mean 4.07, SD 0.98 (SD is ddof=0; ddof=1 gives 1.20) | REPRODUCED |
| 2 H3-017 | 0.233, 0.93, 0.064/0.042 | JSON matches. "6 of 9 / 5 of 9" re-counted from claim rows; C9 ratio 0.1979/0.2604 = 0.76 | CONSISTENT-ONLY |
| 3 T3 raw, down_f4, defender, labels14 | 0.883; 0.806/0.785; 0.807/0.784; 0.688 | result.json, floors_table.md and T.md 8.12 table identical | CONSISTENT-ONLY |
| 3 KT2 | none beats envelope | dU/dL of all 5 defenders re-derived from releases and the 9-floor envelope equal the JSON to 4 dp. All 99% lower bounds are below the margins | REPRODUCED (points) |
| 3 G2' | 4 clauses pass | g2.json values match T.md 8.11; shuffled F1 0.5326 is inside 0.06, outside 0.021 as stated | CONSISTENT-ONLY |
| 4 ladder | 0.806/0.817/0.852; +0.045 [0.033,0.059]; raw +0.025; labels14 +0.013 | R4-R1 equals the difference of reported L for all 4 releases; CI half-widths are 0.93-0.99 of the iid Hanley-McNeil width at n=3016 | CONSISTENT-ONLY |
| 4 ordering R4 | -0.011 [-0.022,-0.001] | 0.85164 - 0.86263 = -0.01092 | REPRODUCED (point) |
| 4 T4b | 0.887 vs 0.886; labels14 0.654 | armb.json drops, gaps and the 20.5% drop share recompute exactly | CONSISTENT-ONLY |

The pre-registered rules applied to the stated numbers give the stated labels in every case. No numeric DISCREPANT item was found; the findings below concern wording, labels and unstated dependencies.

## Findings (by severity)

1. **T3 NO-GO depends on `down_f4`, which sits outside the nomination rule.** The 9-floor envelope includes `down_f4` as a "named extra" (T.md:264, 276). Without it my point dL for pcxr_mu0.01_s1_C2 is +0.061 and dU is +0.018; for s1_C1 dL is +0.041 (margins 0.03/0.02). I could not compute the 99% bounds. T.md:264 says "an envelope without it would favour the defender", but SUMMARY.md:27 and FOCUS.md:118-119 say only "none beats". T.md:276 calls the verdict "not post-hoc", but `down_f4` was chosen after T2 showed it beating the defender.
2. **"Ties ... held at the strongest rung" contradicts T4a.** SUMMARY.md:27 says the best defender ties the floor at the strongest rung. ladder.json "ordering" gives floor-minus-defender -0.0109 at R4, 99% [-0.0218, -0.0011]. FOCUS.md:123 and T.md:348 ("tie became a small floor lead") are accurate. SUMMARY also omits the defender-seen-S caveat.
3. **"Grow 3.5x" is mislabelled** (SUMMARY.md:27; journal 2026-11-05.md:79, 85). 8,714/2,490 = 3.5 is the ADDED pool. With A about 2,490 patients (T.md:22), R4 is about 11,200 (x4.5); ladder.json says triples grow x2 then x2.6 (x5.2).
4. **"Every chest X-ray figure is a LOWER BOUND"** (SUMMARY.md:27; journal :79) exceeds the pre-registered reading. Only down_f4 and the defender meet "LOWER BOUND". raw (+0.025) and labels14 (+0.013) are "PARTLY CONVERGED / unresolved" (ladder.json).
5. **"Still accelerating" (SUMMARY.md:27, T.md:348) is mostly attacker switching.** Per doubling, F1 rises 0.044 then 0.051 and F3 is flat (0.010, 0.010). The max switches from F3 to F1 between R2 and R4 (ladder.json argmax). The defender's own increments decelerate (0.030, 0.025). The rise is real; the "acceleration" claim is not supported.
6. **Mixed-stratum comparison** (SUMMARY.md:26): "falls from 0.93 on dynamic to 0.23 under LAO". The 0.93 is the dynamic-only P-int value. The 0.23 is pooled over six activities. Like-for-like values are dynamic 0.928 to 0.303, or pooled 0.717 to 0.233 (h3_017_result.json). The per-activity values in H.md:378 average 0.258, not 0.233, because each is a per-stratum max over attackers.
7. **"Beats it on both axes by far more than the margins"** (H.md:348; journal :27). Against real floors, predlabel exceeds aae_re's U by 0.011, below m_U 0.0285 with overlapping CIs. Delta_U is -0.029, -0.020, -0.009 (the last two inside the margin) and comes from interpolating between predlabel and `raw`. The leakage axis (-0.89) carries the verdict. Likewise "lda3 beats msda on utility" (0.935 vs 0.930; SUMMARY.md:26). FOCUS.md:110 and H.md:334 still say "21 floors" where H-B5 (H.md:367) says 20 + predlabel.
8. **labels14 "same utility" / "dominates the measured utility axis"** (SUMMARY.md:27; journal :16, :81). labels14 U is 0.775 (= raw), below down_f4 0.785 and the best defender 0.784. raw-minus-U for down_f4 is -0.0096 [-0.0137, -0.0055].
9. **Sign of the utility cost** (T.md:303, item 5): "moves U by -0.009 to -0.002; utility cost within one point". The figures are raw minus defender, so defenders have slightly HIGHER measured U than raw (0.776-0.784 vs 0.775), not a cost.
10. **Two metadata floors for T3** (0.533 in g2.json, T.md:272, pooled; 0.537 in result.json KT1, T.md:276, rotation 0 only). The unlabelled "raw 0.885" (FOCUS.md:96, SUMMARY.md:27) is T2 G2 on rotations 1-2 (T.md:204); the pooled T2 figure is 0.887.
11. **KT1 controls are vacuous in result.json.** `controls_failed: []`, while the logged `raw_shuf:f3_best` is 0.766. This is presumably label-independent retrieval scorers. g2.json uses `f3a_logreg` (0.508).
12. **Over-stated wording, minor.** (a) SUMMARY.md:26 "only the dither-free label stays at chance" and "undithered codes 0.92-0.96": act*_r1 account CIs include chance; undithered lda2 is 0.667 (h3_016_result.json). (b) "At chance between static and dynamic": P-S2D/P-D2S CIs reach 0.15-0.17. (c) "Margin above every printed MDE80" (T.md:276) holds only for primary rows (labels14 sensitivity: MDE80_dL 0.045-0.065, margin raised). (d) H.md:390 "every protocol where both exist" is a single protocol (LAOdyn). (e) SUMMARY.md:5 still says programme-2 numbers were not re-derived. (f) The B2 range CI is positive by construction (3 runs).

## What I could not check, and why

- Every AUC, accuracy, per-trial correctness and bootstrap interval. Per-unit scores live on the VM or HF and are not in the repo. Interval checks are consistency-only (Hanley-McNeil widths, per-fold weighting).
- T3 99% lower bounds and per-fold arm-B values (H.md:346, not in JSON); they need bootstrap resamples or unit scores.
- Evaluator code (attacker training, pair building, role splits) and the data (HF mirror vs official). I audited arithmetic downstream of the result files, not the pipeline.
- T2 numbers (67.0 vs 57.7 replication, T2 verdicts) and programme 1: outside scope.
