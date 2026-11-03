# Chest-CT obfuscation: confirmation on the original 169 validation patients ("sealed-3")

Date: 2026-11-03. Plan frozen in `vm_run/FROZEN3.txt` (sha256 83c07040...) before any run on this set. These patients were read once for the very first report with the old arm list; A21 and the comparators below had never been scored on them.

Setup: gallery 169 patients, 240 queries, chance 0.0059, scanner-metadata floor 0.042. Adapters and attackers trained on development patients only (4,083 scans). Linkage = best of 5 fresh attackers.

| Arm (1000 steps unless noted) | Linkage top-1 (seeds) | Linkage removed vs raw | Utility kept (linear) | MLP-probe AUC |
|---|---|---|---|---|
| A0 raw (3 seeds) | 0.474 | 0% | 1.00 | 0.682 |
| B1 PCA-256 (3) | 0.404 | 15% | 0.90 | 0.678 |
| Noise sigma 1.5 (3) | 0.178 | 62% | 0.67 | 0.619 |
| A3 (8) | 0.190 +- 0.014 | 60% | 1.01 | 0.702 |
| A14 (8) | 0.156 +- 0.013 | 67% | 0.92 | 0.681 |
| **A21 (8)** | **0.159 +- 0.020** | **66%** | **1.00** | 0.698 |
| K0 constant (3) | 0.000 | | 0.00 | 0.500 |

Paired patient-level bootstrap on seed-averaged hits (2,000 resamples): A21 - A3 = -0.031 [-0.055, -0.008]; A14 - A3 = -0.034 [-0.062, -0.005]; A21 - A14 = +0.003 [-0.018, +0.024].

Reading, in plain terms: on patients never used for tuning, A21 lowers the chance that an attacker finds the right person by about 3 percentage points compared with A3, at the same disease-probe utility. A14 gets the same linkage but pays about 8% of utility. All of them still sit around 3.7 to 4.5 times above the scanner-only floor, so none is anonymous.

Limits: 169 patients; one hospital; one frozen encoder; the interval covers patient sampling, not a new hospital; A21 was chosen on development data after about 140 configurations; A21 is weaker for nearest-neighbour "similar case" retrieval (kNN utility kept 0.64 vs A3 0.93, see `journal.md` cycle 21); age remains partly decodable (R^2 0.62 raw to 0.41 for A21).
