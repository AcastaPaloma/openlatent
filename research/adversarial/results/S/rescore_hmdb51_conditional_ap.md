<!-- provenance: evaluator sha256 eb2d619870c2811430aee56376a306dc71dc930de5bc187e00d9a6cdfb527e0f (frozen match: True, tag s-suite-eval-v1), script rescore.py sha256 76b5b0fca804cb80a5226b656c24ebdacc49403897ca5bb23a40c964b11414bc -->
# Re-score of programme 1's harness predictions (development/burned data, NOT sealed)


## hmdb51_vp/within  (1530 independent units)

| release | P1 printed | recomputed | repro | best-of-2 (A2) | honest cross-fit (H) | A2-H [95% CI] (MDE80) | moves | Bonf. UB K=2 | privacy printed / honest | MDE80 in privacy units |
|---|---:|---:|:-:|---:|---:|---|:-:|---:|---|---:|
| black | 62.6 | 62.6 | y | 62.6 | 64.5 | -1.80 [-2.68, -1.26] (1.06) | MOVES | nan | 0.99 / 0.76 | 0.13 |
| bdq_down64 | 63.4 | 63.4 | y | 63.1 | 64.6 | -1.51 [-2.49, -0.93] (1.17) | MOVES | nan | 0.90 / 0.74 | 0.15 |
| bdq_down32 | 63.9 | 63.9 | y | 63.9 | 65.4 | -1.57 [-2.38, -0.96] (1.12) | MOVES | nan | 0.83 / 0.64 | 0.14 |
| down32 | 65.0 | 65.0 | y | 65.0 | 66.7 | -1.69 [-2.72, -1.07] (1.16) | MOVES | nan | 0.69 / 0.48 | 0.15 |
| personbox_blur16_e24 | 65.8 | 65.8 | y | 65.8 | 67.5 | -1.65 [-2.63, -0.81] (1.27) | MOVES | nan | 0.59 / 0.38 | 0.16 |
| personbox_blur16_s1 | 66.2 | 66.2 | y | 66.2 | 67.5 | -1.34 [-2.42, -0.81] (1.14) | MOVES | nan | 0.54 / 0.37 | 0.14 |
| motion_down16 | 66.3 | 66.3 | y | 66.3 | 67.9 | -1.63 [-2.72, -1.21] (1.12) | MOVES | nan | 0.53 / 0.33 | 0.14 |
| personbox_blur16 | 66.3 | 66.3 | y | 66.2 | 68.3 | -2.05 [-2.76, -1.23] (1.14) | MOVES | nan | 0.52 / 0.28 | 0.14 |
| personbox_black | 66.4 | 66.4 | y | 66.4 | 67.9 | -1.47 [-2.51, -1.01] (1.12) | MOVES | nan | 0.51 / 0.33 | 0.14 |
| personbox_blur16_s2 | 66.5 | 66.5 | y | 66.5 | 68.3 | -1.77 [-2.70, -0.77] (1.39) | MOVES | nan | 0.50 / 0.28 | 0.18 |
| blur4 | 67.2 | 67.2 | y | 67.2 | 69.1 | -1.89 [-2.79, -1.16] (1.20) | MOVES | nan | 0.41 / 0.17 | 0.15 |
| down8 | 67.4 | 67.4 | y | 67.4 | 68.9 | -1.51 [-2.62, -0.72] (1.39) | MOVES | nan | 0.39 / 0.20 | 0.18 |
| gray | 68.8 | 68.8 | y | 68.8 | 70.4 | -1.61 [-2.91, -1.24] (1.23) | MOVES | nan | 0.21 / 0.01 | 0.15 |
| down4 | 68.9 | 68.9 | y | 68.9 | 70.5 | -1.68 [-2.87, -1.11] (1.29) | MOVES | nan | 0.21 / 0.00 | 0.16 |
| raw | 70.5 | 70.5 | y | 70.5 | 72.0 | -1.54 [-2.60, -0.84] (1.29) | MOVES | nan | 0.00 / 0.00 | 0.16 |

Adjacent orderings in programme 1's ranking (paired, same recordings):

| lower leakage | higher leakage | printed gap | best-of-2 gap [CI] (MDE80) resolved | honest gap [CI] (MDE80) resolved |
|---|---|---:|---|---|
| black | bdq_down64 | 0.7 | +0.4 [-0.9, +2.5] (2.5) no | +0.2 [-1.2, +2.2] (2.3) no |
| bdq_down64 | bdq_down32 | 0.5 | +0.8 [-0.5, +2.2] (1.9) no | +0.8 [-0.6, +2.1] (1.9) no |
| bdq_down32 | down32 | 1.2 | +1.2 [-0.4, +2.6] (2.1) no | +1.3 [-0.0, +2.9] (2.0) no |
| down32 | personbox_blur16_e24 | 0.8 | +0.8 [-0.6, +2.3] (2.0) no | +0.8 [-0.8, +2.3] (2.2) no |
| personbox_blur16_e24 | personbox_blur16_s1 | 0.4 | +0.4 [-1.0, +1.3] (1.8) no | +0.1 [-1.0, +1.4] (1.8) no |
| personbox_blur16_s1 | motion_down16 | 0.1 | +0.1 [-1.5, +1.4] (2.1) no | +0.4 [-1.4, +1.9] (2.2) no |
| motion_down16 | personbox_blur16 | 0.1 | -0.1 [-1.1, +1.7] (2.2) no | +0.4 [-1.3, +1.8] (2.1) no |
| personbox_blur16 | personbox_black | 0.1 | +0.2 [-1.3, +1.2] (1.8) no | -0.4 [-1.5, +0.9] (1.7) no |
| personbox_black | personbox_blur16_s2 | 0.1 | +0.1 [-0.9, +1.4] (1.8) no | +0.4 [-0.9, +1.6] (1.8) no |
| personbox_blur16_s2 | blur4 | 0.7 | +0.7 [-0.8, +2.1] (2.0) no | +0.8 [-0.6, +2.2] (2.1) no |
| blur4 | down8 | 0.2 | +0.2 [-1.5, +1.2] (2.0) no | -0.2 [-1.9, +1.1] (2.1) no |
| down8 | gray | 1.4 | +1.4 [+0.3, +3.3] (2.3) yes | +1.5 [+0.2, +3.8] (2.7) yes |
| gray | down4 | 0.1 | +0.1 [-1.6, +1.4] (2.2) no | +0.1 [-2.1, +1.5] (2.5) no |
| down4 | raw | 1.6 | +1.6 [+0.4, +3.1] (1.9) yes | +1.5 [+0.0, +2.9] (2.0) yes |