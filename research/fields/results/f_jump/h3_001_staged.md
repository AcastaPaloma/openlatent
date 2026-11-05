# H3-001 staged aggregate (pre-registration: results_notes/F.md section 23); B=1000; held-out units: folds ['blk', 's1', 's9', 's11']

**Per held-out source inside the block (descriptive, plate-cluster 95%)**: {"b10": {"source_2": [-0.0132, [-0.0158, -0.0104]], "source_5": [-0.0227, [-0.0248, -0.0207]], "source_6": [-0.0193, [-0.0213, -0.0169]], "source_10": [-0.0355, [-0.0374, -0.0337]]}}

## P0 fold blk: hold_source=source_2,source_5,source_6,source_10, n_fit_wells=215612, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=True (U(noise16) 0.050 vs shuffled 0.051, attacker 0.102, max null 0.041); powered=True (MDE80 b30 0.0016, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | mlp_b3_d8_r0.5 | 0.063 [0.25] | -0.0177 [-0.0187, -0.0167] (-0.0192, -0.0167) | ldaP_4 | False | mlp_b3_d8_r0.5 | 0.0102 [0.0094, 0.0110] | True | -0.028 |
| b10 | mlp_b10_d8_r0.5 | 0.058 [0.28] | -0.0229 [-0.0238, -0.0219] (-0.0238, -0.0213) | ldaP_4 | False | mlp_b10_d8_r0.5 | 0.0006 [-0.0002, 0.0014] | False | -0.035 |
| b30 | mlp_b30_d8_r0.5 | 0.055 [0.31] | -0.0386 [-0.0397, -0.0375] (-0.0396, -0.0367) | ldaM_b10_8 | False | mlp_b30_d8_r0.5 | -0.0028 [-0.0036, -0.0019] | False | -0.032 |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -0.0177, "b10": -0.0229, "b30": -0.0386, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.24 | 7.5 | 0.197 | 0.167 | 0.045 | True |
| mlp_b3_d8_r0.5 | 0.24 | 7.5 | 0.195 | 0.167 | 0.045 | True |
| mlp_b10_d8 | 0.27 | 7.7 | 0.185 | 0.160 | 0.043 | True |
| mlp_b10_d8_r0.5 | 0.27 | 7.7 | 0.181 | 0.160 | 0.043 | True |
| mlp_b30_d8 | 0.33 | 7.6 | 0.171 | 0.155 | 0.041 | True |
| mlp_b30_d8_r0.5 | 0.33 | 7.6 | 0.172 | 0.155 | 0.041 | True |
| mlp_b100_d8 | 0.42 | 7.4 | 0.159 | 0.153 | 0.042 | False |
| mlp_b100_d8_r0.5 | 0.42 | 7.4 | 0.162 | 0.153 | 0.042 | False |

## P0 fold s1: hold_source=source_1, n_fit_wells=337754, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=True (U(noise16) 0.015 vs shuffled 0.017, attacker 0.102, max null 0.012); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | none trained (['mlp_b3_d8', 'mlp_b3_d8_r0.5']) | | | | | | | | |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -1.0, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.26 | 7.2 | 0.051 | 0.053 | 0.012 | False |
| mlp_b3_d8_r0.5 | 0.26 | 7.2 | 0.050 | 0.053 | 0.012 | False |
| mlp_b10_d8 | 0.31 | 7.1 | 0.053 | 0.053 | 0.013 | False |
| mlp_b10_d8_r0.5 | 0.31 | 7.1 | 0.052 | 0.053 | 0.013 | False |
| mlp_b30_d8 | 0.42 | 6.9 | 0.053 | 0.048 | 0.011 | False |
| mlp_b30_d8_r0.5 | 0.42 | 6.9 | 0.052 | 0.048 | 0.011 | False |
| mlp_b100_d8 | 0.56 | 6.5 | 0.054 | 0.047 | 0.011 | False |
| mlp_b100_d8_r0.5 | 0.56 | 6.5 | 0.054 | 0.047 | 0.011 | False |

## P0 fold s9: hold_source=source_9, n_fit_wells=297086, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=True (U(noise16) 0.015 vs shuffled 0.016, attacker 0.102, max null 0.013); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | mlp_b3_d8 | 0.023 [0.33] | -0.0136 [-0.0145, -0.0128] (-0.0146, -0.0125) | ldaM_b30_8 | False | mlp_b3_d8 | -0.0107 [-0.0116, -0.0100] | False | -0.011 |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -0.0136, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.23 | 7.3 | 0.066 | 0.051 | 0.012 | True |
| mlp_b3_d8_r0.5 | 0.23 | 7.3 | 0.063 | 0.051 | 0.012 | True |
| mlp_b10_d8 | 0.25 | 7.5 | 0.061 | 0.053 | 0.013 | False |
| mlp_b10_d8_r0.5 | 0.25 | 7.5 | 0.060 | 0.053 | 0.013 | False |
| mlp_b30_d8 | 0.35 | 7.5 | 0.057 | 0.053 | 0.013 | False |
| mlp_b30_d8_r0.5 | 0.35 | 7.5 | 0.058 | 0.053 | 0.013 | False |
| mlp_b100_d8 | 0.50 | 7.3 | 0.047 | 0.058 | 0.014 | False |
| mlp_b100_d8_r0.5 | 0.50 | 7.3 | 0.049 | 0.058 | 0.014 | False |

## P0 fold s11: hold_source=source_11, n_fit_wells=341183, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=True (U(noise16) 0.062 vs shuffled 0.056, attacker 0.102, max null 0.050); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | mlp_b3_d8_r0.5 | 0.072 [0.25] | -0.0352 [-0.0372, -0.0335] (-0.0366, -0.0317) | ldaM_b30_8 | False | mlp_b3_d8_r0.5 | -0.0290 [-0.0307, -0.0273] | False | -0.029 |
| b10 | mlp_b10_d8_r0.5 | 0.069 [0.27] | -0.0380 [-0.0399, -0.0362] (-0.0392, -0.0335) | ldaM_b30_8 | False | mlp_b10_d8_r0.5 | -0.0326 [-0.0344, -0.0308] | False | -0.033 |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -0.0352, "b10": -0.038, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.26 | 7.1 | 0.224 | 0.196 | 0.052 | True |
| mlp_b3_d8_r0.5 | 0.26 | 7.1 | 0.226 | 0.196 | 0.052 | True |
| mlp_b10_d8 | 0.31 | 7.2 | 0.216 | 0.203 | 0.052 | True |
| mlp_b10_d8_r0.5 | 0.31 | 7.2 | 0.219 | 0.203 | 0.052 | True |
| mlp_b30_d8 | 0.43 | 7.2 | 0.209 | 0.205 | 0.055 | False |
| mlp_b30_d8_r0.5 | 0.43 | 7.2 | 0.214 | 0.205 | 0.055 | False |
| mlp_b100_d8 | 0.57 | 6.8 | 0.182 | 0.199 | 0.056 | False |
| mlp_b100_d8_r0.5 | 0.57 | 6.8 | 0.191 | 0.199 | 0.056 | False |

## P0 cross-fold: **INVALID**; valid and powered folds []; f* <= 30% in 0, f* > 30% or censored in 0; block dU_H(b100) = None

**Per held-out source inside the block (descriptive, plate-cluster 95%)**: {}

## Pabc fold blk: hold_source=source_2,source_5,source_6,source_10, n_fit_wells=215612, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=False (U(noise16) 0.073 vs shuffled 0.051, attacker 0.102, max null 0.041); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | mlp_b3_d8 | 0.087 [0.33] | -0.0155 [-0.0168, -0.0142] (-0.0197, -0.0157) | ldaP_4 | False | mlp_b3_d8_r0.5 | 0.0086 [0.0075, 0.0098] | False | -0.025 |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -0.0155, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.24 | 7.5 | 0.243 | 0.230 | 0.068 | True |
| mlp_b3_d8_r0.5 | 0.24 | 7.5 | 0.242 | 0.230 | 0.068 | True |
| mlp_b10_d8 | 0.27 | 7.7 | 0.230 | 0.223 | 0.064 | False |
| mlp_b10_d8_r0.5 | 0.27 | 7.7 | 0.229 | 0.223 | 0.064 | False |
| mlp_b30_d8 | 0.33 | 7.6 | 0.216 | 0.215 | 0.061 | False |
| mlp_b30_d8_r0.5 | 0.33 | 7.6 | 0.221 | 0.215 | 0.061 | False |
| mlp_b100_d8 | 0.42 | 7.4 | 0.203 | 0.209 | 0.060 | False |
| mlp_b100_d8_r0.5 | 0.42 | 7.4 | 0.206 | 0.209 | 0.060 | False |

## Pabc fold s1: hold_source=source_1, n_fit_wells=337754, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=False (U(noise16) 0.231 vs shuffled 0.017, attacker 0.102, max null 0.012); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | none trained (['mlp_b3_d8', 'mlp_b3_d8_r0.5']) | | | | | | | | |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -1.0, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.26 | 7.2 | 0.447 | 0.550 | 0.212 | False |
| mlp_b3_d8_r0.5 | 0.26 | 7.2 | 0.460 | 0.550 | 0.212 | False |
| mlp_b10_d8 | 0.31 | 7.1 | 0.456 | 0.554 | 0.219 | False |
| mlp_b10_d8_r0.5 | 0.31 | 7.1 | 0.468 | 0.554 | 0.219 | False |
| mlp_b30_d8 | 0.42 | 6.9 | 0.437 | 0.527 | 0.200 | False |
| mlp_b30_d8_r0.5 | 0.42 | 6.9 | 0.452 | 0.527 | 0.200 | False |
| mlp_b100_d8 | 0.56 | 6.5 | 0.440 | 0.507 | 0.186 | False |
| mlp_b100_d8_r0.5 | 0.56 | 6.5 | 0.457 | 0.507 | 0.186 | False |

## Pabc fold s9: hold_source=source_9, n_fit_wells=297086, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=False (U(noise16) 0.133 vs shuffled 0.016, attacker 0.102, max null 0.013); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | none trained (['mlp_b3_d8', 'mlp_b3_d8_r0.5']) | | | | | | | | |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -1.0, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.23 | 7.3 | 0.314 | 0.354 | 0.111 | False |
| mlp_b3_d8_r0.5 | 0.23 | 7.3 | 0.324 | 0.354 | 0.111 | False |
| mlp_b10_d8 | 0.25 | 7.5 | 0.300 | 0.367 | 0.117 | False |
| mlp_b10_d8_r0.5 | 0.25 | 7.5 | 0.315 | 0.367 | 0.117 | False |
| mlp_b30_d8 | 0.35 | 7.5 | 0.284 | 0.360 | 0.114 | False |
| mlp_b30_d8_r0.5 | 0.35 | 7.5 | 0.301 | 0.360 | 0.114 | False |
| mlp_b100_d8 | 0.50 | 7.3 | 0.257 | 0.361 | 0.118 | False |
| mlp_b100_d8_r0.5 | 0.50 | 7.3 | 0.272 | 0.361 | 0.118 | False |

## Pabc fold s11: hold_source=source_11, n_fit_wells=341183, H/V compounds {'V': 11362, 'H': 56076}

**Headroom (G-H4)**: no trained b100 adapter; controls ok=False (U(noise16) 0.111 vs shuffled 0.056, attacker 0.102, max null 0.050); powered=False (MDE80 b30 nan, b100 nan); valid=False

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b3 | mlp_b3_d8_r0.5 | 0.123 [0.25] | -0.0490 [-0.0513, -0.0466] (-0.0526, -0.0425) | ldaP_4 | False | mlp_b3_d8_r0.5 | -0.0432 [-0.0453, -0.0409] | False | -0.043 |
| b10 | none trained (['mlp_b10_d8', 'mlp_b10_d8_r0.5']) | | | | | | | | |
| b30 | none trained (['mlp_b30_d8', 'mlp_b30_d8_r0.5']) | | | | | | | | |
| b100 | none trained (['mlp_b100_d8', 'mlp_b100_d8_r0.5']) | | | | | | | | |

**Break-even vs the full envelope**: first winning budget None; interpolated inf% (>100), bootstrap percentiles 2.5/50/97.5 = [nan, nan, nan]%; dU_H(f) = {"b3": -0.049, "b10": -1.0, "b30": -1.0, "b100": -1.0}

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b3_d8 | 0.26 | 7.1 | 0.349 | 0.333 | 0.098 | True |
| mlp_b3_d8_r0.5 | 0.26 | 7.1 | 0.352 | 0.333 | 0.098 | True |
| mlp_b10_d8 | 0.31 | 7.2 | 0.335 | 0.343 | 0.101 | False |
| mlp_b10_d8_r0.5 | 0.31 | 7.2 | 0.339 | 0.343 | 0.101 | False |
| mlp_b30_d8 | 0.43 | 7.2 | 0.325 | 0.344 | 0.107 | False |
| mlp_b30_d8_r0.5 | 0.43 | 7.2 | 0.334 | 0.344 | 0.107 | False |
| mlp_b100_d8 | 0.57 | 6.8 | 0.287 | 0.326 | 0.102 | False |
| mlp_b100_d8_r0.5 | 0.57 | 6.8 | 0.300 | 0.326 | 0.102 | False |

## Pabc cross-fold: **INVALID**; valid and powered folds []; f* <= 30% in 0, f* > 30% or censored in 0; block dU_H(b100) = None

## Verdict fields

```
{
 "verdict_original_P0": "INVALID",
 "verdict_tightened_Pabc": "INVALID",
 "P0": {
  "verdict": "INVALID",
  "folds_valid_powered": [],
  "n_valid_powered": 0,
  "n_f_le30": 0,
  "n_f_gt30_or_censored": 0,
  "block_valid_powered": false,
  "block_f_le30": false,
  "block_dU_b100": null,
  "block_ok": false
 },
 "Pabc": {
  "verdict": "INVALID",
  "folds_valid_powered": [],
  "n_valid_powered": 0,
  "n_f_le30": 0,
  "n_f_gt30_or_censored": 0,
  "block_valid_powered": false,
  "block_f_le30": false,
  "block_dU_b100": null,
  "block_ok": false
 },
 "sources_le30_P0": 0,
 "sources_le30_Pabc": 0,
 "sources_total": 7
}
```
