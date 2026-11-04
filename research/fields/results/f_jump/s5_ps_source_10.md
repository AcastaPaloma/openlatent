# JS-1 aggregate, scenario 5 (B=500); units: compounds (U), plates (L). Point estimates drive the registered decisions; intervals are conditional on V-selection.

## split s0: {'V': 8196, 'H': 40417} compounds (V/H), wells H 135685, V 27515, fit 143165

| arm | family | d | U_V | U_H xs [95% CI] | U_H nonrep | worst att H [CI] | L_V | L_H [CI] | lin / mlp / knn / lgbm (H) |
|---|---|---|---|---|---|---|---|---|---|
| raw | floor | 1040 | 0.130 | 0.038 [0.037, 0.039] | 0.039 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 0.96 / 1.00 / 0.98 / 1.00 |
| pca16 | floor | 16 | 0.156 | 0.048 [0.047, 0.050] | 0.050 | 0.87 [0.86, 0.87] | 0.83 | 0.84 [0.83, 0.84] | 0.61 / 0.87 / 0.82 / 0.82 |
| pca64 | floor | 64 | 0.144 | 0.044 [0.042, 0.045] | 0.045 | 0.99 [0.99, 0.99] | 0.99 | 0.99 [0.99, 0.99] | 0.84 / 0.99 / 0.98 / 0.96 |
| rp64 | floor | 64 | 0.138 | 0.040 [0.039, 0.041] | 0.041 | 0.96 [0.96, 0.96] | 0.94 | 0.95 [0.95, 0.95] | 0.70 / 0.96 / 0.90 / 0.89 |
| ldaT2_16 | labelonly | 16 | 0.190 | 0.067 [0.065, 0.069] | 0.068 | 0.54 [0.54, 0.55] | 0.41 | 0.43 [0.42, 0.44] | 0.37 / 0.54 / 0.50 / 0.51 |
| ldaT2_64 | labelonly | 64 | 0.141 | 0.044 [0.043, 0.045] | 0.045 | 0.88 [0.88, 0.88] | 0.83 | 0.85 [0.85, 0.86] | 0.65 / 0.88 / 0.82 / 0.78 |
| ldaT2_max | labelonly | 301 | 0.098 | 0.026 [0.025, 0.027] | 0.027 | 0.99 [0.99, 0.99] | 0.98 | 0.99 [0.98, 0.99] | 0.87 / 0.97 / 0.99 / 0.91 |
| ldaP_64 | labelonly | 64 | 0.130 | 0.044 [0.043, 0.045] | 0.045 | 0.94 [0.94, 0.95] | 0.92 | 0.93 [0.93, 0.94] | 0.53 / 0.94 / 0.89 / 0.78 |
| ldaP_255 | labelonly | 255 | 0.102 | 0.029 [0.028, 0.030] | 0.030 | 0.99 [0.99, 0.99] | 0.99 | 1.00 [0.99, 1.00] | 0.88 / 0.99 / 0.99 / 0.96 |
| dpca16_r1 | dither | 16 | 0.174 | 0.053 [0.052, 0.054] | 0.054 | 0.53 [0.53, 0.54] | 0.41 | 0.42 [0.41, 0.42] | 0.43 / 0.53 / 0.50 / 0.51 |
| dpca64_r1 | dither | 64 | 0.169 | 0.051 [0.050, 0.052] | 0.052 | 0.82 [0.81, 0.82] | 0.75 | 0.77 [0.77, 0.78] | 0.60 / 0.82 / 0.78 / 0.71 |
| noise_r1 | dither | 1040 | 0.142 | 0.046 [0.044, 0.047] | 0.047 | 0.97 [0.97, 0.97] | 0.95 | 0.96 [0.96, 0.97] | 0.84 / 0.97 / 0.92 / 0.92 |
| ldaT2_max_r1 | dither | 301 | 0.110 | 0.029 [0.029, 0.030] | 0.030 | 0.88 [0.87, 0.88] | 0.82 | 0.85 [0.84, 0.85] | 0.74 / 0.88 / 0.88 / 0.75 |
| ldaP_255_r1 | dither | 255 | 0.122 | 0.033 [0.032, 0.034] | 0.034 | 0.90 [0.90, 0.90] | 0.82 | 0.88 [0.87, 0.88] | 0.61 / 0.90 / 0.84 / 0.70 |
| T_centre | ref | 1040 | 0.145 | 0.043 [0.042, 0.045] | 0.045 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 0.55 / 0.99 / 0.96 / 1.00 |
| T_swhiten64 | ref | 64 | 0.155 | 0.050 [0.049, 0.052] | 0.051 | 0.79 [0.78, 0.79] | 0.66 | 0.74 [0.73, 0.74] | 0.24 / 0.79 / 0.60 / 0.41 |
| T_combat | ref | 1040 | 0.144 | 0.043 [0.042, 0.044] | 0.044 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 0.53 / 0.99 / 0.96 / 1.00 |
| T_harmony | ref | 1040 | 0.158 | 0.051 [0.050, 0.053] | 0.053 | 0.99 [0.99, 0.99] | 0.98 | 0.99 [0.99, 0.99] | 0.54 / 0.98 / 0.83 / 0.99 |
| T_sphering | ref | 1040 | 0.109 | 0.031 [0.030, 0.032] | 0.032 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 0.96 / 1.00 / 1.00 / 1.00 |

**JK1**: raw worst-of-4 0.997 [0.996, 0.998] (needs >= 0.60); controls {}; U(shuffled compound) nan vs U(noise16) nan -> pass v0-literal=True, pass JUMP0-text (D1, post-hoc sensitivity)=True

**JK3** (R = T_harmony: U_V 0.158, U_H 0.051, L_V 0.98, L_H 0.99): U-axis F* = ldaT2_16 (U_H 0.067, L_H 0.43), dU = -0.016 [-0.017, -0.015] -> headroom_U=False; L-axis F_L = dpca16_r1 (U_H 0.053, L_H 0.42), dL = -0.57 [-0.58, -0.57] -> headroom_L=False; fires=True

per-reference: T_centre: U_V 0.145, L_V 1.00, F* ldaT2_16, dU_H -0.024, headroom_U=False, headroom_L=False, fires=True; T_swhiten64: U_V 0.155, L_V 0.66, F* ldaT2_16, dU_H -0.017, headroom_U=False, headroom_L=False, fires=True; T_combat: U_V 0.144, L_V 1.00, F* ldaT2_16, dU_H -0.024, headroom_U=False, headroom_L=False, fires=True; T_harmony: U_V 0.158, L_V 0.98, F* ldaT2_16, dU_H -0.016, headroom_U=False, headroom_L=False, fires=True; T_sphering: U_V 0.109, L_V 1.00, F* ldaT2_16, dU_H -0.036, headroom_U=False, headroom_L=False, fires=True

**JK2** (constrained V-selection, L_V <= 0.25): best = None (no label-only arm with L_V <= 0.25; 0 candidates) -> fires=False; literal reading: best-by-U_V = ldaT2_16 (U_H 0.067, L_H 0.43) fires=False; with dithered LDA: None

**power (JK0c)**: median floor-vs-floor MDE80 = 0.0011 over 10 pairs; n_compounds_H = 28380

## Verdict table

```
{
 "JK1_v0_literal_pass_splits": 1,
 "JK1_v0_literal_pass": true,
 "JK1_jump0_text_pass_splits": 1,
 "JK1_jump0_text_pass": true,
 "JK2_fires_splits": 0,
 "JK2_fires": null,
 "JK3_fires_splits": 1,
 "JK3_fires": null,
 "median_mde80_per_split": [
  0.0010578565142984665
 ],
 "JK0c_pass": true
}
```
