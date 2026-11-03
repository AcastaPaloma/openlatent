# Chest-CT obfuscation: fresh exam (sealed-2) result

Date: 2026-11-03. Arm list and settings frozen in `vm_run/FROZEN2.txt` before any exam feature was extracted. Reported once.

## Setup (plain terms)
- Exam = 538 CT-RATE patients (1,296 scans, 758 "find the same person's other scan" queries against a gallery of 538) that were never used in any earlier run, tuning or selection.
- Adapters and attackers were trained on 4,083 development scans only. The attacker picks, from 538 people, whose scan is the match. Chance = 0.0019. A scanner-metadata-only attacker scores 0.034 ("floor").
- 13 frozen configurations x 3 seeds = 39 runs, all completed, zero failures. Linkage = best of 5 freshly trained attackers.

## Results (mean of 3 seeds; "kept" = (utility AUC - 0.5) / (raw utility AUC - 0.5))
| Arm | Linkage top-1 | Linkage removed | Utility AUC | Kept (linear probe) | Kept (MLP probe) | Sex AUC |
|---|---|---|---|---|---|---|
| A0 raw features | 0.427 | 0% | 0.719 | 1.00 | 1.00 | 0.996 |
| B1 PCA-256 | 0.325 | 24% | 0.703 | 0.93 | 0.99 | 0.994 |
| B2 random projection | 0.269 | 37% | 0.703 | 0.92 | 1.01 | 0.982 |
| Noise sigma 1 | 0.141 | 67% | 0.682 | 0.83 | 0.80 | 0.979 |
| A2 adversarial, lambda 1 | 0.125 | 71% | 0.678 | 0.81 | 0.91 | 0.963 |
| **A3 adversarial, lambda 3, 1000 steps** | **0.102** | **77%** | **0.707** | **0.94** | **1.03** | 0.929 |
| A8 adversarial, lambda 10 | 0.101 | 77% | 0.692 | 0.88 | 0.94 | 0.926 |
| A1 bottleneck, no privacy term | 0.095 | 78% | 0.652 | 0.70 | 0.75 | 0.863 |
| NoPeek (A5) | 0.082 | 81% | 0.621 | 0.55 | 0.58 | 0.779 |
| Noise sigma 1.5 | 0.080 | 82% | 0.660 | 0.73 | 0.72 | 0.962 |
| LEACE (A7) | 0.075 | 83% | 0.639 | 0.63 | 0.68 | 0.707 |
| **A14 = A3 + light contrastive adversary** | **0.060** | **86%** | 0.690 | 0.87 | 0.95 | 0.910 |
| K0 constant latent | 0.000 | | 0.500 | 0.00 | 0.00 | 0.500 |

Seed spread of linkage is 0.001 to 0.013; bootstrap 95% intervals over patients are about +-0.02 (e.g. A3 [0.092, 0.137] on seed 0).

## Against the frozen decision rule
1. A3 cuts best-attacker linkage by 76% (0.427 -> 0.102): **met** (needs >= 50%).
2. A3 keeps >= 95% of utility: **narrowly missed on the linear probe (0.94)**, met on the nonlinear probe (1.03). Utility CI was not used as the lower-bound test.
3. Constant latent scores chance on linkage and 0.5 AUC: **met**.
4. At or below the metadata floor (0.034): **not met by any arm**. A3 is 3.0x the floor, the best arm A14 is 1.8x. Pixel-space reconstruction attack: **not run**.
5. Beats cheap baselines at matched utility: **met**. At 92-93% kept, PCA leaks 0.325 and random projection 0.269 vs A3 0.102. NoPeek, LEACE and noise sigma 1.5 leak a little less (0.075-0.082) but keep only 55-73% of the utility; A14 leaks less than all of them (0.060) with more utility (87%, 95% on the MLP probe).

## What this does and does not show
- Supports: trained adversarial adapters remove most of the measurable cross-visit linkage from frozen CT-FM features while keeping most disease-probe signal, and they do it far better than noise, PCA or random projection at the same utility. The dev result (0.110 -> 0.102 here) transfers to patients never touched.
- Does not support: anonymity. The best latent still lets a fresh attacker pick the right person 6% of the time among 538 (110x chance), above the scanner-only floor.
- Limits: one hospital (CT-RATE), one frozen encoder, labels are classifier-derived from reports, steps were tuned on dev. Hyperparameters (steps) were not re-tuned on the exam.
