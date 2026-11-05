# Controlled replication (HLCA, d16, seeds [0], folds [0, 1, 2]); valid=True; CE control adequate=False
Q1 (replicates 18.2): original REPLICATES | tightened REPLICATES
Q2 (shortcut?): original PARTIAL | tightened PARTIAL  (by budget: {10: 'EFFECT_UNCONFIRMED'})

Selection: leave-one-fold-out. dU in macro-F1; intervals 95% donor bootstrap [study-cluster]; folds F1/F2/F3; seeds s0/s1/s2.

| f | n seeds | U adp | L adp | U same | dU_same | dU_label | B* (per fold) | dU_env | vs noise_r1 | folds dU_label | seeds dU_label | test-selected dU_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 1 | 0.777 | 0.56 | 0.601 | +0.176 [+0.165,+0.187] [+0.157,+0.200] | +0.076 [+0.066,+0.086] [+0.059,+0.114] | ce_f10_r0.5/ce_f10_r0.5/ce_f10_r0.5 | +0.076 [+0.066,+0.086] | +0.030 | +0.066, +0.121, +0.072 | s0:+0.076 | +0.076 |

Leakage axis (U, L; raw50 = 1, chance = 0), primary budgets:

| arm | n seeds | U [95%] | L [95%] |
|---|---|---|---|
| adp_f10_r0 | 1 | 0.777 [0.739,0.785] | 0.56 [0.50,0.61] |
| adp_f10_r0.5 | 1 | 0.760 [0.721,0.767] | 0.45 [0.34,0.48] |
| ce_f10_r0 | 1 | 0.757 [0.719,0.765] | 0.64 [0.58,0.69] |
| ce_f10_r0.5 | 1 | 0.702 [0.664,0.710] | 0.52 [0.38,0.54] |
| ldaX_f10 | 1 | 0.813 [0.770,0.822] | 0.82 [0.78,0.91] |
| ldaM_f10 | 1 | 0.601 [0.563,0.611] | 0.73 [0.66,0.78] |
| raw50 | 1 | 0.822 [0.780,0.831] | 1.00 [1.00,1.00] |
| pca16 | 1 | 0.646 [0.602,0.660] | 0.83 [0.79,0.89] |
| rp16 | 1 | 0.526 [0.495,0.538] | 0.68 [0.63,0.71] |
| noise_r0.5 | 1 | 0.804 [0.762,0.812] | 0.90 [0.85,0.93] |
| noise_r1 | 1 | 0.747 [0.706,0.755] | 0.74 [0.65,0.76] |
| dith16_r0.5 | 1 | 0.489 [0.455,0.501] | 0.58 [0.50,0.65] |
| dith16_r1 | 1 | 0.326 [0.302,0.338] | 0.42 [0.28,0.47] |
| ldaC_pca16 | 1 | 0.553 [0.525,0.563] | 0.79 [0.74,0.86] |

controls: {0: {'U': 0.013760214012973004, 'u0': 0.013400444732399847, 'worst': 0.2841784749629342, 'chance': 0.2809523809523809}}
MDE80 f=10 (seed 0): not run
'beats label floors' is not 'removes batch': L stays high at r = 0 for the adapter and the CE floor.