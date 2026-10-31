# Privacy report: charades

Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.

## Dataset

- 9072 items (6352 train / 2720 test), video, 112x150, 3 channel(s), 6.25 fps, median 191 frames per item
- people visible in 79% of sampled frames (mean 0.9, median area 30.5%), skeletons found in 85%, camera motion 10.0
- targets: actions (utility, multilabel, 157 classes); identity (secret, multiclass, 90 classes)

## Where the task and the secrets live (profile)

Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw footage. Probes rank cues, they do not certify privacy.

| view | actions | identity |
|---|---:|---:|
| colour | 0.08 | 0.45 |
| lowres | 0.17 | 0.73 |
| scene | 0.48 | 0.90 |
| person | 0.96 | 0.89 |
| body | 0.25 | 0.16 |
| posture | 0.31 | 0.12 |
| motion | 0.05 | 0.08 |
| label | – | 0.01 |

## Plan

- identity: strongest cues [('scene', 0.9), ('person', 0.89), ('lowres', 0.73), ('colour', 0.45)]; task 'actions' cues [('person', 0.96), ('scene', 0.48), ('posture', 0.31), ('body', 0.25)]
- train bdq_down32: R1: baseline / control

## Results

Headline protocol: within (the dataset's split). Attackers: frame_r50, r3d retrained on the obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait attacker for identities. Frontier hypervolume: 0.401.

| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |
|---|---|---:|---:|---:|---:|---:|---|
| raw | reference | 100.0 [98, 109] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/r3d 94.0 [93.1, 94.9] |
| black | control | -0.6 [-1, 2] | 100.0 | 0.0 | 50.0 | 100.0 | identity: retrained/frame_r50 3.6 [2.9, 4.3] |
| pose **(front)** | stick figure | 50.9 [49, 58] | 63.1 | 56.4 | 57.0 | 63.1 | identity: retrained/r3d 37.1 [35.2, 38.8] |
| pose_canon **(front)** | canonical stick figure | 27.0 [26, 32] | 83.1 | 40.8 | 55.1 | 82.4 | identity: retrained/r3d 19.0 [17.5, 20.7] |
| bdq_down32 | learned pool | 24.3 [23, 29] | 38.6 | 29.9 | 31.5 | 38.6 | identity: retrained/r3d 59.2 [57.4, 60.9] |
| motion_down8 **(front)** | fixed motion map | 58.1 [56, 65] | 9.2 | 15.8 | 33.6 | 9.2 | identity: retrained/r3d 85.7 [84.5, 87.1] |
| gray **(front)** | colour removal | 97.3 [95, 107] | 4.9 | 9.2 | 51.1 | 4.9 | identity: retrained/r3d 89.6 [88.4, 90.7] |
| down8 | downsample | 55.6 [53, 63] | 2.9 | 5.6 | 29.2 | 2.9 | identity: retrained/r3d 91.4 [90.3, 92.3] |
| personbox_blur16 | person blur | 65.1 [63, 73] | 2.9 | 5.5 | 34.0 | 2.9 | identity: retrained/r3d 91.4 [90.4, 92.4] |
| blur4 | blur | 62.0 [59, 69] | 2.2 | 4.3 | 32.1 | 2.2 | identity: retrained/r3d 92.0 [91.0, 92.9] |
| down4 | downsample | 77.1 [74, 86] | 1.4 | 2.8 | 39.3 | 1.4 | identity: retrained/r3d 92.7 [91.8, 93.6] |

### Per secret and protocol

**identity** (top1)

| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |
|---|---|---|---:|---:|---:|---:|
| raw | within | retrained/r3d | 94.0 [93.1, 94.9] | 3.8 (3.6 / 3.8) | 94.0 | 0.0 |
| raw | cross_scene | retrained/r3d | 89.7 [88.2, 90.9] | 1.4 (1.4 / 1.4) | 89.7 | 0.0 |
| black | within | retrained/frame_r50 | 3.6 [2.9, 4.3] | 3.8 (3.6 / 3.8) | 94.0 | 100.0 |
| black | cross_scene | retrained/frame_r50 | 1.4 [0.9, 1.8] | 1.4 (1.4 / 1.4) | 89.7 | 100.0 |
| pose | within | retrained/r3d | 37.1 [35.2, 38.8] | 3.8 (3.6 / 3.8) | 94.0 | 63.1 |
| pose | cross_scene | retrained/r3d | 31.5 [29.7, 33.4] | 1.4 (1.4 / 1.4) | 89.7 | 65.9 |
| pose_canon | within | retrained/r3d | 19.0 [17.5, 20.7] | 3.8 (3.6 / 3.8) | 94.0 | 83.1 |
| pose_canon | cross_scene | retrained/r3d | 17.0 [15.4, 18.5] | 1.4 (1.4 / 1.4) | 89.7 | 82.4 |
| bdq_down32 | within | retrained/r3d | 59.2 [57.4, 60.9] | 3.8 (3.6 / 3.8) | 94.0 | 38.6 |
| bdq_down32 | cross_scene | retrained/r3d | 55.6 [53.7, 57.9] | 1.4 (1.4 / 1.4) | 89.7 | 38.6 |
| motion_down8 | within | retrained/r3d | 85.7 [84.5, 87.1] | 3.8 (3.6 / 3.8) | 94.0 | 9.2 |
| motion_down8 | cross_scene | retrained/r3d | 78.1 [76.3, 79.7] | 1.4 (1.4 / 1.4) | 89.7 | 13.2 |
| gray | within | retrained/r3d | 89.6 [88.4, 90.7] | 3.8 (3.6 / 3.8) | 94.0 | 4.9 |
| gray | cross_scene | retrained/r3d | 82.9 [81.4, 84.4] | 1.4 (1.4 / 1.4) | 89.7 | 7.7 |
| down8 | within | retrained/r3d | 91.4 [90.3, 92.3] | 3.8 (3.6 / 3.8) | 94.0 | 2.9 |
| down8 | cross_scene | retrained/r3d | 84.1 [82.5, 85.5] | 1.4 (1.4 / 1.4) | 89.7 | 6.4 |
| personbox_blur16 | within | retrained/r3d | 91.4 [90.4, 92.4] | 3.8 (3.6 / 3.8) | 94.0 | 2.9 |
| personbox_blur16 | cross_scene | retrained/r3d | 85.5 [84.0, 87.0] | 1.4 (1.4 / 1.4) | 89.7 | 4.7 |
| blur4 | within | retrained/r3d | 92.0 [91.0, 92.9] | 3.8 (3.6 / 3.8) | 94.0 | 2.2 |
| blur4 | cross_scene | retrained/r3d | 85.8 [84.4, 87.2] | 1.4 (1.4 / 1.4) | 89.7 | 4.4 |
| down4 | within | retrained/r3d | 92.7 [91.8, 93.6] | 3.8 (3.6 / 3.8) | 94.0 | 1.4 |
| down4 | cross_scene | retrained/r3d | 87.1 [85.6, 88.4] | 1.4 (1.4 / 1.4) | 89.7 | 3.0 |

## Checks

- PASS measurable (identity, within): raw 94.0 vs baseline 3.8 (chance 3.6, task label alone 3.8)
- PASS measurable (identity, cross_scene): raw 89.7 vs baseline 1.4 (chance 1.4, task label alone 1.4)
- PASS black control (identity, within): black frames 3.6 [2.9, 4.3] vs baseline 3.8
- frozen raw-trained attackers understate the retrained ones by a median 57.9 points (max 80.2, motion_down8)
- screening vs full evaluation, privacy rank correlation: 0.74

## Recommendation

- **pose** (stick figure): utility 50.9, privacy 63.1, PUS 56.4
- privacy first (utility >= 50): pose; utility first (privacy >= 50): pose
- **certified** (worst over the recommendation and its 4 robustness re-runs): privacy 63.1, utility 48.3, PUS 54.7
- robustness pose_dino: privacy 64.5, utility 51.6
- robustness pose_e24: privacy 64.3, utility 49.8
- robustness pose_s1: privacy 63.9, utility 50.1
- robustness pose_s2: privacy 63.7, utility 48.3
- config: `{"type": "pose", "every": 4, "width": 2}`
