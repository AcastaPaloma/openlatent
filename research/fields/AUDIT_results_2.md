# AUDIT_results_2 (Auditor-4): programme 1 results after Auditor-1 (JUMP, HLCA)

Read-only. I read SUMMARY.md, FOCUS.md, F.md 11-24 and journal 2026-11-05, then recomputed from `results/f_jump/*.json|md` with my own scripts (scratchpad `audit4/`: a1_stage1.py, a1b.py, a1c_f4.py, a2_ksweep.py, a3_eng.py, a4_hlca.py, plus ad hoc JSON reads). Rules were re-implemented from the F.md pre-registration text, not from the aggregators.
**Key limit:** `results/f5_*/` holds only `job_result.json`; no per-compound, plate, donor or cell scores are in the repo (HF/VM only). Bootstrap CIs, MDE80, attacker leakage and V-selection therefore cannot be recomputed; I checked deltas, fits, rules and internal consistency.

## Table

| item | claimed | recomputed / checked | status |
|---|---|---|---|
| Stage 1, b10 vs full envelope | P0 +0.029, Pb +0.015, Pabc +0.014; >0.010 in 3/3 splits | s1.json U_H(adapter) - U_H(ldaP_8), mean of 3 splits: +0.0290 / +0.0150 / +0.0136; per-split min 0.0113 | REPRODUCED (CIs CONSISTENT-ONLY) |
| Stage 1 rules | no collapse; strict SURVIVES fails | rules i-iii not fired (Pb 0.0128/0.0152/0.0172); SURVIVES fails at Pa b100 (0.093/0.094/0.093 vs 0.095), Pb s0, Pabc s0, s1; S = 0.87/0.52/1.01/1.64/0.47 | REPRODUCED |
| F4 break-even | 2.9% [2.7,3.2]; 4.3% [4.0,4.8] | log-linear crossing of 0.010: 2.91%, 4.35% | REPRODUCED |
| F4 leave-source-out | no break-even in 5 folds; 0.013-0.038 below best floor | ceiling - floor -0.0136/-0.0130/-0.0385/-0.0250/-0.0295; lso10 unpowered, b1-b30 untrained | REPRODUCED |
| H3-002 law | means -0.009..+0.100; beta 0.0291 [0.0195,0.0395]; R2 0.76 | WLS (1/SE^2) on 21 cells: beta 0.02913, alpha -0.0638, R2 0.757 (unweighted 0.797); my cell bootstrap [0.0198,0.0385] | REPRODUCED |
| H3-002 wells/class | -0.036/-0.028/-0.019, threshold 0.015 | -0.0358/-0.0284/-0.0189; FALSIFIED 3/3 by rule | REPRODUCED |
| H3-003 s18 | refuted; +0.055..+0.204 | arithmetic and rule (refuted at f=1,3,10,30) | REPRODUCED |
| H3-003 replication | dU_same +0.140/+0.201; dU_label +0.060/+0.057 | exact; Q1 and Q2 SURV at 0.02 and 0.035 | REPRODUCED (points) |
| 20A | +0.012/+0.052; "+0.018..+0.034" | exact; interpolated: f=10% +0.026/+0.018, f=100% +0.034/+0.028 | REPRODUCED; label DISCREPANT (M3) |
| 20B / s21 | 0/24 cells; rho 1.18-1.77 | md parsed: 0/24, rho 1.18-1.77, flags match | REPRODUCED |
| s21 collapse | CORAL beta 3 collapses 3 of 6 | 2 of 6 (rank 3.7, 3.8); beta 30 6 of 6 | DISCREPANT (L1) |
| 20B in-loop accuracy | 0.34-0.35 | not in committed files | UNVERIFIABLE |
| H3-004 | leakage 0.19->0.52 etc.; falsified | md rows and rule (a) match | REPRODUCED |
| H3-005 | FAIL-PAIRS; 0.76/0.76/0.69%; oracle -0.0026/+0.0133 | match; G1 0/3, G2 0/3, G3 false at both thresholds | REPRODUCED |
| H3-001 staged | INVALID; -0.010..-0.090; margins +0.007,+0.007,-0.011,-0.017 | -0.0103..-0.0478 (P0), -0.0155..-0.0902 (Pabc); margins +0.0068/+0.0067/-0.0106/-0.0168 | REPRODUCED |
| H3-001 check | ref +0.0162/+0.0603, +0.0283/+0.1068; nulls 0.003/0.009/0.0155/0.0238 | ceiling - floor and stored-U differences exact | REPRODUCED |
| Spend | job-hours | `job_result.json` seconds match every section except stage 1; 49 f5_ jobs exit 0, uploaded | see L3 |

## Findings

No numeric error and no pre-registered verdict that fails to follow from its stated numbers (original and tightened). The issues below are wording, labelling and omissions.

**Medium**
1. **"4 of 4 folds" overstated.** SUMMARY.md:19 says the adapter "does not beat its permuted-label floor ... in 4 of 4 folds". In `h3_001_staged.json` the P0 b100 margins are +0.0068 (blk) and +0.0067 (s1): above the floor, below the required +0.010. Only s9, s11 are at or below it. F.md:636 and journal:60 state it correctly.
2. **H3-001 sentence is incomplete.** SUMMARY.md:19 omits the closing check (F.md 24.1: seen-source reference wins, R-A), which is what separates "table cannot show a gain" from "no unseen-source transfer"; journal:67 says SUMMARY is "current to H3-001". It quotes -0.090, which is the s9 Pabc b100 cell, where the Pabc control fails (0.0155) and the adapter is 0.10 below its permuted floor. The clause "0.013 to 0.038 BELOW the best floor" is F4's (five-source), placed after the H3-001 clause; H3-001's b100 gaps are 0.011-0.048 (P0). FOCUS.md has no H3-002/004/005/001 blocks.
3. **"Explained by normalisation" (SUMMARY.md:18, FOCUS.md:158) is the discrete-rule label only.** At f=10% the donor upper bound is 0.0185 vs margin 0.020, and the normalised r=0 CE enters the 0.05 tolerance by 0.01 (L 0.72 vs 0.73). Same data: interpolated matched-leakage dU +0.026 [0.018,0.033] (cenn) and +0.018 [0.012,0.026] (cenp); paired r=0 adapter minus cenn +0.0135 [0.0099,0.0194] with L lower by 0.043 (`h3_003_20a_extra.json`). F.md:562 labels the verdict RULE-DEPENDENT, but SUMMARY/FOCUS attach that only to the 100% number, and the quoted "+0.018 to +0.034" mixes budgets (100% is +0.028..+0.034).
4. **Small-K sentence (F.md:464, journal:17).** "Small-K cells sit within -0.014 to +0.005 ... not replicated": that range is draws d1-d2 only. All nine cells (K=30,100,300): -0.0182 to +0.0050; 7 of 9 negative with compound CIs excluding zero; draw means -0.009/-0.007/-0.006. Correct reading: small, negative on average, below the 0.010 margin.
5. **"Needs about 3.3 wells per class" (SUMMARY.md:19).** Only 2.0 vs ~3.36 was contrasted; the <=4-well control moved dU by -0.0002..-0.0015 (uninformative). "Needs" is untested; the same row says "one draw each" next to "3 label draws".
6. **Cross-cutting #2 (SUMMARY.md:32).** "The one exception" conflicts with the single-cell row (SupCon HLCA +0.052 over the normalised CE and the full envelope at f=100%). "About half the original gain" holds for b10 only; b100 is 1.4-2.4x P0 under Pb/Pd/Pabc.

**Low**
- L1. F.md:594 and journal:50: CORAL beta 3 collapse is 2 of 6 cells; "3.7-5.2" is the range over all six. 0-of-24 unaffected; SUMMARY's "CORAL collapsed" is true only for beta 30.
- L2. Plate-cluster CIs sit 0.003-0.012 below the compound-level point and exclude it in 12 of 54 stage-1 rows (all b100 under Pa/Pb/Pd/Pabc) and 3 of 4 reference rows; unexplained. "Above 0.010 in 3 of 3 splits" is point-level: Pabc s0 plate lower bound is 0.0086.
- L3. F.md:376 gives stage-1 spend as "0.35 worker-hours"; summed `seconds` is 4,604 s = 1.28 job-h (estimate was about 1); every other section uses summed seconds.
- L4. "Two predictions wrong or half right" (SUMMARY.md:18; FOCUS.md:158): F.md shows 18.2 wrong, 19.2 wrong, 20A half right, s21 partly wrong.
- L5. SUMMARY.md:19 lists Pb/Pabc as "+0.014 to +0.015" (Pb is 0.0150, Pabc 0.0136); FOCUS.md:104 "1.5-2.4" for b100 S is 1.44-2.44; journal:35 "lower bound > 0.125" is 0.1245 for seed 0.
- L6. `h3_003_20a.md` prints "GEN" and a "Controlled replication" title; 20B aggregator prints NOT SUPPORTED per budget beside INERT TERM. F.md subsections 15.1/16.1/17/15.2 and journal entries (10:05Z, 10:45Z) are out of order. H3-005 is absent from SUMMARY and FOCUS.

**Wording that holds:** "REPLICATED" (seeds/label draws, same three folds, PCA-50, d16) is scoped in the row; "closed" is scoped to two families on one architecture; "PARTIAL SURVIVAL" matches the failed strict rule.

## What I could not check, and why
- Any bootstrap CI, MDE80, plate/donor/study interval, or leakage value (no score arrays in repo).
- V-selection of arms and floors (needs per-arm V scores); the aggregators' code correctness.
- The ksweep d0 duplicate-job provenance (preds mtime is on the VM) and the 20B in-loop accuracy 0.34-0.35.
- Pabc is not neutral: position codes score below noise (0.016 vs 0.068), penalising position-aware embeddings. Validity rules test only an upper bound.
- E2, A3 and programme-2 results (out of scope).
