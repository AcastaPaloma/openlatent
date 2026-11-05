# Engagement report (section 21): candidate x budget; per fold: monitor acc / fresh a_arm vs noise a_noise, ref a_ref / rho / pass

## seed 0, f = 10%  (chance 0.083; a_noise / a_ref per fold: F0: 0.082/0.372, F1: 0.111/0.313, F2: 0.145/0.465)

| candidate | per fold: mon acc, a_arm, rho, ENG1, ENG2, no-collapse (ratio, rank) | passes |
|---|---|---|
| grlr10 | F0: 0.18, 0.476, 1.36, YnY (0.61, 10.8); F1: 0.45, 0.414, 1.50, nnY (0.61, 9.2); F2: 0.42, 0.524, 1.18, nnY (0.60, 11.3) | 0/3 |
| grlr30 | F0: 0.18, 0.484, 1.39, YnY (0.61, 10.4); F1: 0.45, 0.435, 1.61, nnY (0.61, 9.2); F2: 0.42, 0.551, 1.27, nnY (0.63, 10.5) | 0/3 |
| coral3 | F0: 0.46, 0.461, 1.31, nnY (0.55, 4.1); F1: 0.61, 0.400, 1.43, nnn (0.58, 3.7); F2: 0.63, 0.565, 1.31, nnY (0.57, 5.2) | 0/3 |
| coral30 | F0: 0.37, 0.431, 1.20, nnn (0.65, 1.9); F1: 0.54, 0.416, 1.51, nnn (0.66, 1.6); F2: 0.59, 0.566, 1.31, nnn (0.61, 2.1) | 0/3 |

**ENGAGED at f=10% (seed 0): False**

## seed 0, f = 100%  (chance 0.083; a_noise / a_ref per fold: F0: 0.084/0.384, F1: 0.121/0.307, F2: 0.146/0.490)

| candidate | per fold: mon acc, a_arm, rho, ENG1, ENG2, no-collapse (ratio, rank) | passes |
|---|---|---|
| grlr10 | F0: 0.17, 0.520, 1.45, YnY (0.68, 10.2); F1: 0.43, 0.450, 1.77, nnY (0.66, 8.3); F2: 0.42, 0.569, 1.23, nnY (0.70, 10.0) | 0/3 |
| grlr30 | F0: 0.17, 0.527, 1.48, YnY (0.73, 9.9); F1: 0.43, 0.446, 1.75, nnY (0.68, 7.4); F2: 0.42, 0.609, 1.35, nnY (0.76, 9.8) | 0/3 |
| coral3 | F0: 0.46, 0.483, 1.33, nnY (0.63, 4.2); F1: 0.59, 0.426, 1.64, nnn (0.62, 3.8); F2: 0.63, 0.601, 1.32, nnY (0.65, 4.8) | 0/3 |
| coral30 | F0: 0.39, 0.487, 1.34, nnn (0.73, 1.8); F1: 0.53, 0.449, 1.76, nnn (0.67, 1.6); F2: 0.59, 0.580, 1.26, nnn (0.65, 2.0) | 0/3 |

**ENGAGED at f=100% (seed 0): False**

**VERDICT: engagement never achieved (HLCA learned-mask line CLOSED)**