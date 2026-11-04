## E2 (HLCA): seeds [0, 1, 2], folds [0, 1, 2], 97232 cells, 107 donors, B=1000; floor pool 35 arms, learned configs 21 (8 eligible)

| learned config | eligible | in-sample adv excess (lin/mlp, fold-mean) | KL | R2 / pca16 R2 | U_fine [CI] | L [CI] | U_hold |
|---|---|---|---|---|---|---|---|
| e2_cvae_b10 | True | 0.025 / 0.066 | 3.7 | 0.34 / 0.26 | 0.363 [0.338, 0.375] | 0.48 [0.28, 0.53] | 0.429 |
| e2_cvae_b2 | True | 0.057 / 0.184 | 13.0 | 0.41 / 0.26 | 0.673 [0.646, 0.683] | 0.68 [0.52, 0.72] | 0.711 |
| e2_cvae_b50 | False (fold 0: collapsed (KL 0.7, R2 0.21 vs pca16 0.26); fold 1: collapsed (KL 0.6, R2 0.17 vs pca16 0.24)) | 0.025 / 0.027 | 0.7 | 0.19 / 0.26 | 0.055 [0.051, 0.059] | 0.14 [-0.13, 0.18] | 0.040 |
| e2_cvaeadv_b10_l1 | False (fold 0: in-sample adversary 0.371 above memorisation floor 0.288 + 0.05; fold 1: in-sample adversary 0.503 above memorisation floor 0.398 + 0.05) | 0.023 / 0.065 | 3.7 | 0.34 / 0.26 | 0.361 [0.336, 0.372] | 0.48 [0.27, 0.53] | 0.425 |
| e2_cvaeadv_b10_l16 | False (fold 0: in-sample adversary 0.478 above memorisation floor 0.288 + 0.05; fold 1: in-sample adversary 0.506 above memorisation floor 0.398 + 0.05) | 0.011 / 0.064 | 3.7 | 0.34 / 0.26 | 0.350 [0.325, 0.362] | 0.48 [0.26, 0.53] | 0.410 |
| e2_cvaeadv_b10_l4 | False (fold 0: in-sample adversary 0.431 above memorisation floor 0.288 + 0.05; fold 1: in-sample adversary 0.515 above memorisation floor 0.398 + 0.05) | 0.019 / 0.067 | 3.7 | 0.34 / 0.26 | 0.357 [0.332, 0.368] | 0.48 [0.27, 0.53] | 0.417 |
| e2_vae_b10 | True | 0.196 / 0.260 | 4.6 | 0.33 / 0.26 | 0.421 [0.396, 0.432] | 0.49 [0.37, 0.53] | 0.463 |
| e2_vae_b2 | True | 0.347 / 0.429 | 15.0 | 0.41 / 0.26 | 0.701 [0.670, 0.709] | 0.68 [0.58, 0.72] | 0.724 |
| e2_vae_b50 | False (fold 0: collapsed (KL 0.7, R2 0.12 vs pca16 0.26); fold 1: collapsed (KL 0.6, R2 0.11 vs pca16 0.24)) | 0.055 / 0.065 | 0.7 | 0.12 / 0.26 | 0.048 [0.044, 0.051] | 0.10 [-0.18, 0.14] | 0.034 |
| e2_vaeadv_b10_l1 | False (fold 0: adversary not converged (donor-disjoint excess 0.238); fold 0: in-sample adversary 0.517 above memorisation floor 0.288 + 0.05) | 0.183 / 0.258 | 4.6 | 0.33 / 0.26 | 0.418 [0.393, 0.428] | 0.49 [0.37, 0.52] | 0.461 |
| e2_vaeadv_b10_l16 | False (fold 0: adversary not converged (donor-disjoint excess 0.222); fold 0: in-sample adversary 0.509 above memorisation floor 0.288 + 0.05) | 0.072 / 0.251 | 4.6 | 0.33 / 0.26 | 0.413 [0.387, 0.424] | 0.49 [0.35, 0.53] | 0.464 |
| e2_vaeadv_b10_l4 | False (fold 0: adversary not converged (donor-disjoint excess 0.223); fold 0: in-sample adversary 0.528 above memorisation floor 0.288 + 0.05) | 0.136 / 0.249 | 4.5 | 0.33 / 0.26 | 0.418 [0.394, 0.428] | 0.50 [0.36, 0.54] | 0.468 |
| e2_vaeplc_b10_l4 | False (fold 0: adversary not converged (donor-disjoint excess 0.251); fold 0: in-sample adversary 0.526 above memorisation floor 0.288 + 0.05) | 0.196 / 0.264 | 4.6 | 0.33 / 0.26 | 0.421 [0.395, 0.432] | 0.50 [0.38, 0.54] | 0.464 |
| e2x_cvae_b0p5 | True | 0.202 / 0.304 | 29.7 | 0.44 / 0.26 | 0.755 [0.718, 0.763] | 0.76 [0.67, 0.79] | 0.768 |
| e2x_cvae_b1 | True | 0.115 / 0.253 | 21.7 | 0.43 / 0.26 | 0.732 [0.698, 0.741] | 0.73 [0.63, 0.76] | 0.755 |
| e2x_cvaeadv_b1_l16 | False (fold 0: adversary not converged (donor-disjoint excess 0.213); fold 0: in-sample adversary 0.733 above memorisation floor 0.288 + 0.05) | 0.031 / 0.250 | 20.5 | 0.42 / 0.26 | 0.707 [0.678, 0.716] | 0.75 [0.62, 0.79] | 0.738 |
| e2x_cvaeadv_b1_l64 | False (fold 0: adversary not converged (donor-disjoint excess 0.320); fold 0: in-sample adversary 0.717 above memorisation floor 0.288 + 0.05) | 0.252 / 0.386 | 25.6 | 0.37 / 0.26 | 0.673 [0.644, 0.683] | 0.70 [0.59, 0.74] | 0.704 |
| e2x_vae_b0p5 | True | 0.416 / 0.495 | 31.1 | 0.44 / 0.26 | 0.755 [0.717, 0.763] | 0.77 [0.70, 0.80] | 0.767 |
| e2x_vae_b1 | True | 0.400 / 0.475 | 23.4 | 0.43 / 0.26 | 0.741 [0.706, 0.749] | 0.75 [0.66, 0.78] | 0.759 |
| e2x_vaeadv_b1_l16 | False (fold 0: adversary not converged (donor-disjoint excess 0.362); fold 0: in-sample adversary 0.818 above memorisation floor 0.288 + 0.05) | 0.044 / 0.385 | 22.6 | 0.42 / 0.26 | 0.729 [0.693, 0.738] | 0.72 [0.61, 0.76] | 0.747 |
| e2x_vaeadv_b1_l64 | False (fold 0: adversary not converged (donor-disjoint excess 0.399); fold 0: in-sample adversary 0.720 above memorisation floor 0.288 + 0.05) | 0.281 / 0.460 | 28.8 | 0.36 / 0.26 | 0.684 [0.651, 0.694] | 0.70 [0.59, 0.74] | 0.707 |
| (reference) T_scvi | | | | | 0.816 [0.780, 0.826] | 0.80 [0.74, 0.88] | 0.795 |
| (reference) T_harmony | | | | | 0.668 [0.647, 0.678] | 0.37 [0.31, 0.42] | 0.669 |
| (reference) raw50 | | | | | 0.787 [0.745, 0.797] | 1.00 [1.00, 1.00] | 0.792 |

**E2-K1'**: raw50 worst attacker 0.670 vs chance 0.281 (excess 0.389 >= 0.15); attacker controls (excess over chance) {'ctrl_const': 0.0, 'ctrl_noise16': 0.002, 'ctrl_shuffle_batch': -0.003}, probe controls (U minus u0) {'ctrl_const': -0.012, 'ctrl_noise16': 0.001, 'ctrl_shuffle_labels': 0.002}: PASS

### Target scvi (U_T 0.816, L_T 0.8; feasible = pooled L <= 0.85)
E2-K2' (label-only/PCA family at the target leakage): selected {0: 'rp16', 1: 'rp16', 2: 'rp16'} -> U 0.670, L 0.81; gap to target 0.146 (fires if <= 0.05): **False**; retained fraction vs target 0.82 (old 0.70 threshold, sensitivity)
selected floor per fold {0: 'noise_r0.5', 1: 'dith32_r0.5', 2: 'noise_r0.5'}: U 0.738, L 0.81
selected learned per fold {0: 'e2x_vae_b0p5', 1: 'e2x_vae_b0p5', 2: 'e2x_cvae_b0p5'}: U 0.755, L 0.77 (0.6988819942192679, 0.7998134247116171)
**Delta = U_learned - U_floor = 0.016 (0.009, 0.023)** (cluster (0.003, 0.033)); per-seed {0: 0.016, 1: 0.012, 2: 0.022}; gap closed 0.21; MDE80 0.013 (SE 0.004); conditions {'L_ok': True, 'delta_ok': False, 'seeds_ok': False, 'power_ok': True}; resolved null True

### Target harmony (U_T 0.668, L_T 0.37; feasible = pooled L <= 0.42)
E2-K2': no label-only/PCA arm feasible at this leakage in at least one fold
infeasible: learned False, floor True, eligible learned arms 8

### Best configuration per family at each target leakage (test-selected, eligible or not; optimistic, for attribution)
| target | family | config | eligible | U | L |
|---|---|---|---|---|---|
| scvi | vae | e2x_vae_b0p5 | True | 0.755 | 0.77 |
| scvi | cvae | e2x_cvae_b0p5 | True | 0.755 | 0.76 |
| scvi | vaeadv | e2x_vaeadv_b1_l16 | False | 0.729 | 0.72 |
| scvi | cvaeadv | e2x_cvaeadv_b1_l16 | False | 0.707 | 0.75 |
| scvi | vaeplc | e2_vaeplc_b10_l4 | False | 0.421 | 0.50 |
| harmony | vae | e2_vae_b50 | False | 0.048 | 0.10 |
| harmony | cvae | e2_cvae_b50 | False | 0.055 | 0.14 |
| harmony | vaeadv | none feasible | | | |
| harmony | cvaeadv | none feasible | | | |
| harmony | vaeplc | none feasible | | | |

**E2 VERDICT: E2-NO-GO (resolved: Delta 0.016, upper bound 0.023 < 0.05, or leakage constraint failed: {'L_ok': True, 'delta_ok': False, 'seeds_ok': False, 'power_ok': True})**  (verdict_original: SC-0 NO-GO, unchanged)