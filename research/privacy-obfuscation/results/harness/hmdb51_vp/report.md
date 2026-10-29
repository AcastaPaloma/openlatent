# Privacy report: hmdb51_vp

Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.

## Dataset

- 5100 items (3570 train / 1530 test), video, 112x150, 3 channel(s), 12.5 fps, median 33 frames per item
- people visible in 87% of sampled frames (mean 1.5, median area 34.3%), skeletons found in 92%, camera motion 7.4
- targets: action (utility, multiclass, 51 classes); attributes (secret, multilabel, 5 classes)
- warning: R3: the task label alone gives 102% of the raw probe's 'attributes' score above chance; it is scored with the conditional metric (what the footage reveals beyond the task label)

## Where the task and the secrets live (profile)

Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw footage. Probes rank cues, they do not certify privacy.

| view | action | attributes |
|---|---:|---:|
| colour | 0.15 | 0.40 |
| lowres | 0.24 | 0.38 |
| scene | 0.55 | 0.48 |
| person | 0.93 | 0.98 |
| body | 0.36 | 0.54 |
| posture | 0.40 | 0.67 |
| motion | 0.11 | 0.22 |
| label | – | 1.02 |

## Plan

- attributes: strongest cues [('person', 0.98), ('posture', 0.67), ('body', 0.54), ('scene', 0.48)]; task 'action' cues [('person', 0.93), ('scene', 0.55), ('posture', 0.4), ('body', 0.36)]
- train bdq_down64: R2: attributes survives in posture alone (0.67): needs coarsening on top of canonical skeletons
- train bdq_down32: R1: baseline / control

## Results

Headline protocol: within (the dataset's split). Attackers: frame_r50, r3d retrained on the obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait attacker for identities. Frontier hypervolume: 0.553.

| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |
|---|---|---:|---:|---:|---:|---:|---|
| raw | reference | 100.0 [96, 104] | 0.0 | 0.0 | 50.0 | 0.0 | attributes: retrained/frame_r50 70.5 [69.3, 72.7] |
| black | control | 0.0 [-1, 1] | 98.7 | 0.0 | 49.4 | 98.7 | attributes: retrained/r3d 62.6 [59.8, 63.4] |
| personbox_blur16 **(front)** | person blur | 65.0 [61, 69] | 52.3 | 57.9 | 58.6 | 52.3 | attributes: frozen/r3d 66.3 [65.1, 68.5] |
| personbox_black | person removal | 59.7 [56, 64] | 51.4 | 55.2 | 55.6 | 51.4 | attributes: retrained/r3d 66.4 [65.2, 68.5] |
| blur4 **(front)** | blur | 83.2 [79, 87] | 40.9 | 54.9 | 62.1 | 40.9 | attributes: retrained/r3d 67.2 [66.2, 69.5] |
| motion_down16 **(front)** | fixed motion map | 54.8 [51, 59] | 53.0 | 53.9 | 53.9 | 53.0 | attributes: retrained/frame_r50 66.3 [65.0, 68.4] |
| down8 | downsample | 69.4 [66, 73] | 38.8 | 49.8 | 54.1 | 38.8 | attributes: retrained/r3d 67.4 [66.3, 69.6] |
| bdq_down32 **(front)** | learned pool | 29.5 [26, 33] | 83.5 | 43.6 | 56.5 | 83.5 | attributes: retrained/frame_r50 63.9 [62.9, 66.1] |
| down32 | downsample | 26.7 [24, 30] | 68.8 | 38.4 | 47.7 | 68.8 | attributes: retrained/frame_r50 65.0 [63.8, 67.0] |
| gray **(front)** | colour removal | 100.8 [97, 105] | 21.4 | 35.3 | 60.7 | 21.4 | attributes: retrained/frame_r50 68.8 [67.6, 71.2] |
| down4 | downsample | 90.3 [86, 94] | 20.7 | 33.6 | 55.5 | 20.7 | attributes: retrained/r3d 68.9 [67.7, 70.9] |
| bdq_down64 **(front)** | learned pool | 14.6 [12, 17] | 89.7 | 25.1 | 52.2 | 89.7 | attributes: frozen/frame_r50 63.4 [62.1, 65.4] |

### Per secret and protocol

**attributes** (conditional)

| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |
|---|---|---|---:|---:|---:|---:|
| raw | within | retrained/frame_r50 | 70.5 [69.3, 72.7] | 62.5 (61.8 / 62.5) | 70.5 | 0.0 |
| black | within | retrained/r3d | 62.6 [59.8, 63.4] | 62.5 (61.8 / 62.5) | 70.5 | 98.7 |
| personbox_blur16 | within | frozen/r3d | 66.3 [65.1, 68.5] | 62.5 (61.8 / 62.5) | 70.5 | 52.3 |
| personbox_black | within | retrained/r3d | 66.4 [65.2, 68.5] | 62.5 (61.8 / 62.5) | 70.5 | 51.4 |
| blur4 | within | retrained/r3d | 67.2 [66.2, 69.5] | 62.5 (61.8 / 62.5) | 70.5 | 40.9 |
| motion_down16 | within | retrained/frame_r50 | 66.3 [65.0, 68.4] | 62.5 (61.8 / 62.5) | 70.5 | 53.0 |
| down8 | within | retrained/r3d | 67.4 [66.3, 69.6] | 62.5 (61.8 / 62.5) | 70.5 | 38.8 |
| bdq_down32 | within | retrained/frame_r50 | 63.9 [62.9, 66.1] | 62.5 (61.8 / 62.5) | 70.5 | 83.5 |
| down32 | within | retrained/frame_r50 | 65.0 [63.8, 67.0] | 62.5 (61.8 / 62.5) | 70.5 | 68.8 |
| gray | within | retrained/frame_r50 | 68.8 [67.6, 71.2] | 62.5 (61.8 / 62.5) | 70.5 | 21.4 |
| down4 | within | retrained/r3d | 68.9 [67.7, 70.9] | 62.5 (61.8 / 62.5) | 70.5 | 20.7 |
| bdq_down64 | within | frozen/frame_r50 | 63.4 [62.1, 65.4] | 62.5 (61.8 / 62.5) | 70.5 | 89.7 |

## Checks

- PASS measurable (attributes, within): raw 70.5 vs baseline 62.5 (chance 61.8, task label alone 62.5)
- PASS black control (attributes, within): black frames 62.6 [59.8, 63.4] vs baseline 62.5
- frozen raw-trained attackers understate the retrained ones by a median 1.9 points (max 3.3, down8)
- screening vs full evaluation, privacy rank correlation: 0.76

## Recommendation

- **personbox_blur16** (person blur): utility 65.0, privacy 52.3, PUS 57.9
- privacy first (utility >= 50): motion_down16; utility first (privacy >= 50): personbox_blur16
- **certified** (worst over the recommendation and its 4 robustness re-runs): privacy 43.4, utility 63.3, PUS 51.5
- robustness personbox_blur16_dino: privacy 43.4, utility 65.0
- robustness personbox_blur16_e24: privacy 58.9, utility 63.3
- robustness personbox_blur16_s1: privacy 54.3, utility 65.6
- robustness personbox_blur16_s2: privacy 50.0, utility 66.2
- config: `{"type": "person_box", "mode": "blur", "sigma": 16.0, "every": 8, "dilate": 6}`
