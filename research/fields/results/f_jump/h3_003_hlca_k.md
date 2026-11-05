# Reduced H3-003 (HLCA, R2, d16, 1 seed, 3 folds): verdict K-RULE REFUTED

control ctrl_noise16: U 0.014 vs chance 0.013, worst attacker 0.284 vs chance 0.281 -> ok; trained budgets 5/5

| f | trained | adapter arm | U_ad | L_ad | U_LDA(same) | L_LDA | dU_same [donor] [study] | folds | envelope | dU_full [donor] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | True (dU vs perm +0.328) | adp_f1_r0 | 0.693 | 0.67 | 0.638 | 0.81 | +0.055 [+0.045,+0.064] [+0.036,+0.084] | +0.043, +0.021, +0.118 | dith16_r0.5 | +0.216 [+0.202,+0.226] |
| 3% | True (dU vs perm +0.519) | adp_f3_r0 | 0.699 | 0.67 | 0.614 | 0.80 | +0.086 [+0.075,+0.099] [+0.064,+0.105] | +0.095, +0.065, +0.110 | dith16_r0.5 | +0.222 [+0.209,+0.234] |
| 10% | True (dU vs perm +0.632) | adp_f10_r0 | 0.733 | 0.70 | 0.595 | 0.82 | +0.138 [+0.124,+0.154] [+0.113,+0.158] | +0.142, +0.106, +0.153 | noise_r1 | +0.056 [+0.048,+0.068] |
| 30% | True (dU vs perm +0.643) | adp_f30_r0 | 0.765 | 0.67 | 0.585 | 0.81 | +0.180 [+0.165,+0.189] [+0.157,+0.193] | +0.165, +0.159, +0.167 | noise_r1 | +0.088 [+0.078,+0.097] |
| 100% | True (dU vs perm +0.478) | adp_f100_r0 | 0.789 | 0.68 | 0.586 | 0.81 | +0.204 [+0.191,+0.214] [+0.183,+0.220] | +0.187, +0.173, +0.212 | noise_r1 | +0.112 [+0.102,+0.123] |

refuted_at: [1, 3, 10, 30]; break-even vs full envelope at: [1, 3, 10, 30, 100]; MDE80 (f=10 adapter vs same-label LDA): 0.04575381036221016