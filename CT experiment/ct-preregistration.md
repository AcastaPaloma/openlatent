# Chest-CT obfuscation: pre-registration

Status: written before any adapter was trained. Date: 2026-10-26. Anything not listed here is an
unplanned change and must be reported as one.

## Question

Can an adapter on frozen CT-FM features keep CT-RATE abnormality information while a fresh attacker
fails at **cross-visit patient linkage** (an attacker holding an identified scan matches it to a shared
latent from a different visit of the same person)?

## Data (fixed)

- CT-RATE train split, 800 patients with 2+ scans (seeded sample, `random.Random(0)`), up to 4 scans per
  patient ordered by `StudyDate`, one reconstruction per scan. 1,935 volumes.
- Volume prep: HU = raw + `RescaleIntercept`, clip [-1000, 1000], resample to 2 mm using the metadata
  spacing, centre-crop or pad to 192^3 with air, fp16.
- Frozen encoder: `project-lighter/ct_fm_feature_extractor` (Apache-2.0, trained on Imaging Data Commons,
  not CT-RATE). Features: the last feature map pooled to 512x2x2x2 = 4,096-d (`s`) and 512-d global (`g`).
  Orientation is not the SPL orientation the CT-FM README asks for. This affects every arm equally.
- Splits by `sha256("ctrate-openlatent-v1:" + patient_id)`: hash < 0.5 is `fit` (adapter training and
  attacker training), the rest is `eval`. No patient appears on both sides.
- **Sealed final exam:** the 169 CT-RATE validation patients with 2+ scans. Not touched until the
  configs below are frozen. Reported once.

## Scores

- **Linkage (privacy, lower is better).** Eval patients with 2+ scans. Gallery = earliest scan, queries =
  later scans. Top-1, top-5, MRR, chance = 1 / gallery size. Ties at expected rank.
- **Shortcut floor.** Same score from scanner and protocol metadata only (no pixels, no dates). An
  obfuscated latent only counts as private if its linkage is not above this floor.
- **Utility (higher is better).** Mean AUC over 18 abnormality labels, logistic probe on standardized `s`,
  C chosen by patient-grouped 5-fold CV on `fit` from {0.001, 0.003, 0.01, 0.03}. Labels are
  classifier-derived from reports, so this is a relative measure, not diagnostic accuracy.
- **Sex AUC and age R^2** as secondary leakage reports.
- `leakage_removed = (raw - z) / (raw - chance)`, `utility_kept = (z - 0.5) / (raw - 0.5)`.

## Arms (fixed list, no additions)

Adapter: MLP 4096 -> 1024 -> 256 with LayerNorm, GELU. Utility head: linear 256 -> 18, BCE. AdamW lr 1e-3,
batch 256, 30 epochs, 3 seeds each. Inputs standardized with `fit` statistics.

| ID | Arm | Privacy term |
|---|---|---|
| A0 | Raw `s` | none (the "before" number) |
| A1 | Utility-only bottleneck | none (lambda = 0 control) |
| A2 | Adversarial, lambda = 1 | gradient-reversal pair-verification head on (|z1 - z2|, z1 * z2); same-patient pairs across scans, negatives include same-scanner patients; head re-initialised every 100 steps; ensemble of 2 widths |
| A3 | Adversarial, lambda = 3 | as A2 |
| A4 | A2 + scanner adversary | extra reversed head predicting scanner model and kernel, weight 1 |
| A5 | NoPeek replication | lambda * distance correlation between `s` and `z`, lambda = 1 |
| A6 | Contrastive (SPLAVU-style) | self-supervised: two reconstructions of the same scan are made hard to match (NT-Xent, negated), no patient labels |
| A7 | LEACE | closed-form erasure of sex and scanner from `s`, then A1 |

A0 and A1 are controls. A2 to A7 are the candidates. Nothing else is tried before the sealed exam.

**Amendment 1 (before any adapter result):** A6 needs a second reconstruction of each scan, and only
reconstruction 1 is cached. A6 runs only if reconstruction 2 is cached for the `fit` patients (about 970
volumes). If that is not done, A6 is dropped and reported as dropped, not replaced.

**Amendment 2 (after the 578-scan pilot of A0 and A1, before any full-size result):** in the pilot the
plain utility-only bottleneck (A1) cut best-attacker linkage from 0.24 to 0.10 at about 99% of raw
utility, with no privacy term. So an adversarial arm only counts as a contribution if it beats the cheap
baselines below. Added controls, no training of a privacy term:

| ID | Control | What it tests |
|---|---|---|
| B1 | PCA to 256 dimensions on `fit`, no learning | does dimension reduction alone remove linkage? |
| B2 | Fixed random projection to 256 dimensions | same, without using the data |
| B3 | A0 plus Gaussian noise, sigma set so utility matches A1 within 0.01 AUC | does noise at matched utility do as well as A1? |

The "before" attacker is also strengthened: A0 is scored both on the raw 4,096 features and on a
256-dimensional PCA of them, and the higher linkage is the reported "before". Attackers on 4,096
dimensions with a few hundred fit scans overfit, which can understate raw leakage. This was not in the
original text and is disclosed here as a change made after the pilot.

## Round 1 result (frozen scan set, 3 seeds, development half only)

No arm met criterion 2. Closest: A3 (adversarial lambda = 3), best-attacker top-1 0.077, utility kept 0.97,
leakage removed 0.73 against the B1 baseline (0.279), scanner floor 0.028. Linkage fell monotonically from
A1 (0.134) to A2 (0.090) to A3 (0.077), and adversarial arms beat noise at matched utility (B3 sigma=1:
0.127 at utility 0.611). The sealed split has not been used.

## Round 2 (declared after round 1, before running; development half only)

Four arms, each a single change motivated by round 1. No further arms are added before the sealed exam.
Total configurations tried by then: 8 arms + 3 controls in round 1, 4 in round 2.

| ID | Change | Why |
|---|---|---|
| A8 | A2 with lambda = 10 | linkage kept falling with lambda; does it keep falling, and at what utility cost? |
| A9 | A3 plus the scanner adversary (weight 1) | A4 lowered linkage but cost utility at lambda = 1; lambda = 3 recovered utility |
| A10 | A3 with 4x wider attacker heads, 2x steps, restart every 100 steps | a stronger training adversary may remove more |
| A11 | A3 with a 64-dimensional latent | less capacity left to carry identity |

Decision rule is unchanged. Round-2 arms are scored on the same development half and are compared to round-1
arms with the number of configurations tried reported, since choosing the best of many is a selection effect.
The sealed run evaluates the full pre-registered list once.

## Fresh attackers (after the adapter is frozen)

At least five, trained from scratch on `fit` latents and scored on `eval`: cosine kNN, PCA-whitened kNN,
logistic pair verifier, 2-layer MLP pair verifier, and a siamese contrastive MLP. Three seeds. The reported
number is the **best** attacker per arm, mean with a patient-level bootstrap 95% CI. A reconstruction
attack (train z -> 64^3 volume decoder, then re-run the best raw attacker on the reconstruction) is run on
the winning arm and the controls only.

## Decision rule

An arm counts as a result only if all of these hold on the sealed split:

1. `utility_kept >= 0.95` (lower CI bound no lower than 0.90).
2. Best-attacker linkage top-1 is at or below the metadata-only floor, and `leakage_removed >= 0.5`.
3. The degenerate check holds: a constant latent scores chance on linkage and utility.
4. The result holds on the reconstruction-then-re-ID attack.

If no arm meets all four, the report says so and states the privacy-utility curve it did get. No claim of
anonymity is made in either case.

## Search budget and logging

- Total trained configurations: 8 arms x 3 seeds, plus nothing else before the sealed exam.
- Every run is logged to `openlatent_ct/runs/ledger.jsonl` with config hash, seed, metrics and wall time.
- The number of configurations tried is reported with the results.

## Known risks

- **Scanner shortcut:** linkage may be partly scanner identity. Controlled by the metadata floor and A4.
- **Single institution:** results may not transfer to other hospitals.
- **Weak utility labels:** classifier-derived, noisy. Utility is relative to the raw probe.
- **Small gallery:** about 400 eval patients, so confidence intervals are wide.
