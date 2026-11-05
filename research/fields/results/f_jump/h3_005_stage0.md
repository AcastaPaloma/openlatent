# H3-005 stage 0: cross-source MNN pseudo-pairs (JUMP S5, view A)

## split s0: pool 108713 production wells; labelled production wells {a: 0, b1: 1030, b3: 3235}

| embedding | mutual pairs | tau_cal (lab pairs in prefix) | G1 retention / hidden precision | G2 retention (tau*) | G2 at 0.85 retention | P1 random precision | P2 shuffled-source retention / precision / true cross-source share |
|---|---|---|---|---|---|---|---|
| a|raw256 | 4434 | 0.838 (33) | 0.0007 / 0.139 | 0.0000 (inf) | 0.0000 | 0.0000 | 0.0254 / 0.189 / 0.00 |
| a|adapter | 24715 | 0.997 (4031) | 0.0038 / 0.005 | 0.0000 (inf) | 0.0000 | 0.0000 | 0.0047 / 0.016 / 0.61 |
| b1|adapter | 25653 | 0.990 (4298) | 0.0579 / 0.113 | 0.0076 (0.997) | 0.0072 | 0.0000 | 0.0661 / 0.103 / 0.67 |
| b3|adapter | 26680 | 0.976 (5099) | 0.3027 / 0.070 | 0.0240 (0.995) | 0.0225 | 0.0000 | 0.3225 / 0.063 / 0.72 |

## split s1: pool 108347 production wells; labelled production wells {a: 0, b1: 1012}

| embedding | mutual pairs | tau_cal (lab pairs in prefix) | G1 retention / hidden precision | G2 retention (tau*) | G2 at 0.85 retention | P1 random precision | P2 shuffled-source retention / precision / true cross-source share |
|---|---|---|---|---|---|---|---|
| a|raw256 | 4470 | 0.838 (33) | 0.0007 / 0.158 | 0.0000 (inf) | 0.0000 | 0.0000 | 0.0260 / 0.200 / 0.00 |
| a|adapter | 24838 | 0.997 (4121) | 0.0038 / 0.005 | 0.0000 (inf) | 0.0000 | 0.0000 | 0.0044 / 0.008 / 0.66 |
| b1|adapter | 25284 | 0.980 (4360) | 0.2514 / 0.031 | 0.0076 (0.997) | 0.0071 | 0.0000 | 0.2826 / 0.028 / 0.67 |

## split s2: pool 109186 production wells; labelled production wells {a: 0, b1: 912}

| embedding | mutual pairs | tau_cal (lab pairs in prefix) | G1 retention / hidden precision | G2 retention (tau*) | G2 at 0.85 retention | P1 random precision | P2 shuffled-source retention / precision / true cross-source share |
|---|---|---|---|---|---|---|---|
| a|raw256 | 4476 | 0.838 (33) | 0.0008 / 0.190 | 0.0000 (0.893) | 0.0000 | 0.0000 | 0.0250 / 0.182 / 0.00 |
| a|adapter | 24773 | 0.998 (4054) | 0.0012 / 0.000 | 0.0000 (inf) | 0.0000 | 0.0000 | 0.0017 / 0.022 / 0.58 |
| b1|adapter | 25473 | 0.987 (4421) | 0.1210 / 0.052 | 0.0069 (0.997) | 0.0064 | 0.0000 | 0.1366 / 0.049 / 0.68 |

## oracle-pair ceiling (split s0, U_H = mean cross-source replicate AP over H compounds, P0)

U_H: {"mlp_a_d8": 0.1395, "mlp_b1_d8": 0.1576, "mlp_b3_d8": 0.1763, "orc_q1": 0.155, "orc_q3": 0.1709}

* orc_q1 - b1: -0.0026 [-0.0040, -0.0013]
* orc_q3 - b1: +0.0133 [+0.0119, +0.0147]
* b3 - b1: +0.0187 [+0.0174, +0.0200]
* orc_q1 - b3: -0.0212 [-0.0227, -0.0198]
* orc_q3 - b3: -0.0053 [-0.0067, -0.0039]

**original verdict: FAIL-PAIRS (G2: fewer than 2 of 3 splits reach the retention at the precision even with the hidden-label threshold)**  (G1 in 0/3 splits, G2 in 0/3, G3 False)
**tightened verdict: FAIL-PAIRS (G2: fewer than 2 of 3 splits reach the retention at the precision even with the hidden-label threshold)**  (G1 in 0/3 splits, G2 in 0/3, G3 False)