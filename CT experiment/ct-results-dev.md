# Chest-CT obfuscation: development results

Date: 2026-10-27. Written before the sealed run was read. Pre-registration: `ct-preregistration.md`.

Setup: CT-RATE, 800 patients with 2+ scans (1,935 scans), frozen CT-FM features (4,096-d), development
`fit` / `eval` split by patient hash. Eval gallery 406 patients, chance top-1 0.0025. Each arm is 3 seeds.
"Top-1" is the best of five fresh attackers per run, averaged over seeds. Utility is the mean AUC of
a probe over 18 classifier-derived abnormality labels. Scanner floor (metadata-only attacker): 0.028.

| Arm | Top-1 | Utility AUC | Utility kept | Leakage removed | Sex AUC |
|---|---|---|---|---|---|
| B1 PCA-256, the "before" | 0.279 | 0.614 | 0.77 | 0.00 | 0.963 |
| A0 raw 4,096-d | 0.200 | 0.648 | 1.00 | 0.29 | 0.967 |
| B2 random projection | 0.221 | 0.629 | 0.87 | 0.21 | 0.938 |
| A1 bottleneck, no privacy term | 0.134 | 0.605 | 0.71 | 0.52 | 0.898 |
| A2 adversarial, lambda 1 | 0.090 | 0.634 | 0.91 | 0.68 | 0.901 |
| **A3 adversarial, lambda 3** | **0.077** | **0.643** | **0.97** | **0.73** | 0.870 |
| A4 A2 + scanner adversary | 0.069 | 0.610 | 0.74 | 0.76 | 0.808 |
| A5 NoPeek (dCor) | 0.124 | 0.592 | 0.62 | 0.56 | 0.740 |
| A7 LEACE (sex, scanner) | 0.112 | 0.593 | 0.63 | 0.60 | 0.669 |
| A8 adversarial, lambda 10 | 0.063 | 0.600 | 0.68 | 0.78 | 0.856 |
| A9 A3 + scanner adversary | 0.073 | 0.588 | 0.59 | 0.74 | 0.816 |
| A10 A3, wider heads, 2x steps | 0.089 | 0.644 | 0.97 | 0.69 | 0.896 |
| A11 A3, 64-d latent | 0.079 | 0.630 | 0.88 | 0.72 | 0.830 |
| B3 noise sigma 1 | 0.123 | 0.613 | 0.76 | 0.56 | 0.935 |
| B3 noise sigma 2 | 0.064 | 0.581 | 0.55 | 0.78 | 0.881 |
| B3 noise sigma 4 | 0.019 | 0.544 | 0.30 | 0.94 | 0.764 |

`utility kept = (AUC - 0.5) / (raw AUC - 0.5)`; `leakage removed = (before - top1) / (before - chance)` with
`before` = max(A0, B1) = 0.279.

## What the development data support

- Adversarial arms lower linkage monotonically with lambda (A1 0.134, A2 0.090, A3 0.077, A8 0.063) and beat
  noise and a plain bottleneck at matched utility. A3 keeps 97% of utility.
- No arm reaches the scanner floor with usable utility. Only noise sigma 4 does, keeping 30% of utility.
  By the pre-registered rule no arm is a success on development data.
- A4/A9 (scanner adversary) and A8 (lambda 10) buy a little more privacy at a large utility cost.
- A5 and A7 remove sex information best (0.74 and 0.67 AUC) but are weaker on linkage.

## Disclosed deviations and events

- A pilot on partial data (578 to 716 scans) informed Amendment 2 (added B1 to B3 and the stronger "before").
  Pilot files are in `runs/pilot/` on Drive and were not used for any reported number.
- Round 2 (A8 to A11) was declared after round 1. Configurations tried on development data: 15 arms and
  controls (3 of them noise levels).
- A bug in my patch to the training script crashed every run started after it, at the end and before saving.
  Nothing wrong was written; the affected runs were rerun after the fix.
- A6 (contrastive, needs a second reconstruction) did not run: that cache job was not started.
- Pixel-space reconstruction was not run. A feature-space inversion attack (decoder z to features, then the
  raw-trained attacker) is reported instead; a pixel decoder from a 256-d latent would be too coarse to
  be a meaningful attacker.
