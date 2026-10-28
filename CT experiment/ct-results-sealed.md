# Chest-CT obfuscation: sealed result

Date: 2026-10-27. Pre-registration: `ct-preregistration.md`. Development results: `ct-results-dev.md`
(written before this was read). Summary table on Drive: `openlatent_ct/sealed_summary.csv`.

## Verdict

**No arm met the pre-registered decision rule.** The closest, A3 (adversarial lambda = 3), cut best-attacker
cross-visit linkage on held-out patients from 0.353 (PCA-256 features) to 0.126 and kept all of the
abnormality-probe utility. It stays about 3 times above the metadata-only floor (0.042) and about 21 times
above chance (0.0059). The latents are not anonymous.

## Setup

- 169 sealed CT-RATE validation patients with 2+ scans (409 scans), never used for training, tuning or arm
  selection. Gallery = earliest scan per patient (169), queries = later scans (240).
- Adapters and attackers trained on the development `fit` patients only (924 scans); evaluated on the sealed
  patients. Frozen CT-FM features, as in development.
- Every arm 3 seeds. "Top-1" is the best of five fresh attackers per run (kNN, PCA-whitened kNN, logistic and
  MLP pair verifiers, siamese contrastive MLP), averaged over seeds. CI is a patient-level bootstrap on the
  best attacker's hits and does not include seed variation.
- Configurations tried before the sealed run: 15 on the development half (8 arms and 3 noise levels in round 1,
  plus B1 and B2, plus 4 arms in round 2). The sealed run evaluated all of them once.

## Results

| Arm | Top-1 (CI) | Top-5 | Utility AUC | Utility kept | Leakage removed | Sex AUC | Inversion top-1 |
|---|---|---|---|---|---|---|---|
| B1 PCA-256 ("before") | 0.353 (0.300 to 0.420) | 0.625 | 0.619 | 0.76 | 0.00 | 0.972 | 0.183 |
| A0 raw 4,096-d | 0.275 (0.220 to 0.337) | 0.501 | 0.656 | 1.00 | 0.22 | 0.972 | 0.157 |
| B2 random projection | 0.288 (0.227 to 0.354) | 0.500 | 0.632 | 0.85 | 0.19 | 0.952 | 0.146 |
| A1 bottleneck, no privacy term | 0.203 (0.148 to 0.258) | 0.367 | 0.605 | 0.67 | 0.43 | 0.882 | 0.094 |
| A2 adversarial, lambda 1 | 0.135 (0.090 to 0.185) | 0.325 | 0.644 | 0.92 | 0.63 | 0.890 | 0.118 |
| **A3 adversarial, lambda 3** | **0.126** (0.083 to 0.173) | 0.347 | **0.671** | **1.10** | **0.65** | 0.878 | 0.101 |
| A4 A2 + scanner adversary | 0.139 (0.093 to 0.189) | 0.332 | 0.622 | 0.78 | 0.62 | 0.811 | 0.092 |
| A5 NoPeek | 0.179 (0.129 to 0.228) | 0.318 | 0.594 | 0.60 | 0.50 | 0.763 | 0.053 |
| A7 LEACE | 0.186 (0.134 to 0.242) | 0.315 | 0.596 | 0.62 | 0.48 | 0.676 | 0.033 |
| A8 adversarial, lambda 10 | 0.122 (0.078 to 0.168) | 0.346 | 0.628 | 0.82 | 0.67 | 0.854 | 0.093 |
| A9 A3 + scanner adversary | 0.118 (0.074 to 0.162) | 0.301 | 0.599 | 0.64 | 0.68 | 0.825 | 0.090 |
| A10 A3, wider heads, 2x steps | 0.136 (0.091 to 0.185) | 0.353 | 0.671 | 1.09 | 0.63 | 0.880 | 0.122 |
| A11 A3, 64-d latent | 0.149 (0.103 to 0.196) | 0.351 | 0.651 | 0.97 | 0.59 | 0.841 | 0.100 |
| B3 noise sigma 1 | 0.189 (0.142 to 0.242) | 0.422 | 0.634 | 0.86 | 0.47 | 0.941 | 0.121 |
| B3 noise sigma 2 | 0.108 (0.072 to 0.148) | 0.293 | 0.599 | 0.64 | 0.71 | 0.892 | 0.081 |
| B3 noise sigma 4 | 0.043 (0.019 to 0.072) | 0.160 | 0.556 | 0.36 | 0.89 | 0.794 | 0.042 |

Chance 0.0059; metadata-only floor 0.042; raw utility AUC 0.656. `Utility kept = (AUC - 0.5) / (raw AUC - 0.5)`.
`Leakage removed = (0.353 - top-1) / (0.353 - 0.0059)`. A8 to A11 and B3 are single-seed-set means over 3 seeds.

## Against the decision rule

| Criterion | A3 | Any arm |
|---|---|---|
| 1. Utility kept >= 0.95 (CI lower bound >= 0.90) | point estimate 1.10; **CI on utility not computed** | A3, A10, A11 by point estimate |
| 2. Top-1 at or below the metadata floor, and leakage removed >= 0.5 | **fails the floor** (0.126 vs 0.042); leakage removed 0.65 | only noise sigma 4 reaches the floor (0.043), keeping 36% of utility |
| 3. A constant latent scores chance on linkage and utility | **not run** | not run |
| 4. Holds under reconstruction then re-ID | feature-space inversion top-1 0.101, above the floor | no arm at or below the floor except noise sigma 4 |

So criterion 2 fails for every arm that keeps usable utility, and criteria 3 and 4 were not fully tested.

## What the data do support

- Adversarial training lowers measured linkage well beyond a plain bottleneck (A1 0.203), random
  projection and noise at matched utility (B3 sigma 1: 0.189 at 0.86 utility kept), and the effect holds on
  patients never used in development.
- At utility kept >= 0.95 only A3, A10 and A11 qualify, and A3 has the lowest linkage of the three.
- Linkage and utility trade off along a frontier. Heavier noise or a scanner adversary reach lower linkage
  only by losing most of the utility (sigma 4: 0.043 at 0.36).
- A feature-space inversion attack recovers less identity from adapted latents than from raw ones.

## What the data do not support

- Anonymity. A3 still lets a fresh attacker pick the right patient about 1 time in 8 out of 169, and the
  top-5 rate is 0.35.
- Transfer. All data come from one hospital and one frozen encoder; the shortcut floor is a simple cosine
  match on scanner fields, and a trained metadata attacker could score higher.
- Diagnostic quality. The 18 labels are classifier-derived from reports, and "utility" is a linear-probe AUC.
- Statistical strength across seeds. CIs are over eval patients only; the sealed gallery is small (169).

## Deviations and events (full list)

- Pilot runs on partial data informed Amendment 2; they are in `runs/pilot/` and feed no reported number.
- Round 2 and Amendments 1 and 2 were declared in the pre-registration before the runs they govern.
- A bug in my patch to the training script crashed every run started after it, at the end and before saving;
  nothing wrong was written, and the affected runs were rerun after the fix.
- A6 (contrastive, needs a second reconstruction) was dropped: the second-reconstruction cache job was never
  started. Session 2 was never driven through the MCP; it ran nothing in the final grid.
- Pixel-space reconstruction-then-re-ID was not run. A latent of 256 dimensions only supports a coarse volume.
- The degenerate-constant check (criterion 3) and a CI on utility kept were not computed.
- Claims: "reduced measured linkage against the attacks tested", never "anonymous".
