# Privacy report: kth_ilic

Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.

## Dataset

- 2391 items (1793 train / 598 test), video, 120x160, 1 channel(s), 25.0 fps, median 94 frames per item
- people visible in 78% of sampled frames (mean 0.8, median area 17.2%), skeletons found in 82%, camera motion 3.0
- targets: action (utility, multiclass, 6 classes); identity (secret, multiclass, 25 classes)

## Where the task and the secrets live (profile)

Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw footage. Probes rank cues, they do not certify privacy.

| view | action | identity |
|---|---:|---:|
| colour | 0.50 | 0.63 |
| lowres | 0.48 | 0.89 |
| scene | 0.48 | 0.91 |
| person | 0.99 | 0.97 |
| body | 1.03 | 0.35 |
| posture | 1.02 | 0.24 |
| motion | 0.83 | 0.25 |
| label | – | -0.00 |

## Plan

- identity: strongest cues [('person', 0.97), ('scene', 0.91), ('lowres', 0.89), ('colour', 0.63)]; task 'action' cues [('body', 1.03), ('posture', 1.02), ('person', 0.99), ('motion', 0.83)]
- train bdq_down32: R2: identity is in the people's appearance (0.97); the task carries motion (0.83)
- train bdq_down64: R2: identity is in the people's appearance (0.97); the task carries motion (0.83)

## Results

Headline protocol: within (the dataset's split). Attackers: frame_r50, r3d retrained on the obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait attacker for identities. Frontier hypervolume: 0.886.

| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |
|---|---|---:|---:|---:|---:|---:|---|
| raw | reference | 100.0 [97, 102] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/frame_r50 99.2 [98.0, 100.0] |
| black | control | 0.0 [-7, 8] | 100.0 | 0.0 | 50.0 | 100.0 | identity: retrained/r3d 4.0 [1.3, 7.4] |
| pose_canon_down16 **(front)** | canonical stick figure | 89.0 [85, 93] | 89.6 | 89.3 | 89.3 | 74.4 | identity: retrained/r3d 13.9 [10.2, 17.8] |
| pose_canon_down8 **(front)** | canonical stick figure | 89.9 [86, 94] | 84.9 | 87.3 | 87.4 | 57.9 | identity: retrained/r3d 18.4 [14.2, 22.9] |
| pose_canon_down32 **(front)** | canonical stick figure | 81.8 [76, 87] | 92.6 | 86.9 | 87.2 | 78.9 | identity: retrained/frame_r50 11.0 [7.7, 14.6] |
| bdq_down64 | learned pool | 82.6 [77, 88] | 87.9 | 85.2 | 85.2 | 84.2 | identity: retrained/r3d 15.6 [11.4, 20.2] |
| pose_canon_down4 **(front)** | canonical stick figure | 92.3 [89, 96] | 78.9 | 85.1 | 85.6 | 28.6 | identity: retrained/r3d 24.1 [18.8, 29.8] |
| pose_canon **(front)** | canonical stick figure | 92.5 [89, 96] | 72.2 | 81.1 | 82.4 | 7.5 | identity: retrained/r3d 30.4 [24.6, 36.9] |
| bdq_down32 **(front)** | learned pool | 93.4 [89, 97] | 60.8 | 73.7 | 77.1 | 51.1 | identity: retrained/r3d 41.3 [34.9, 47.9] |
| pose **(front)** | stick figure | 100.2 [98, 102] | 45.7 | 62.7 | 72.8 | 0.0 | identity: retrained/r3d 55.7 [48.9, 61.9] |
| motion_down32 | fixed motion map | 94.6 [91, 98] | 36.7 | 52.9 | 65.7 | 28.0 | identity: retrained/r3d 64.2 [57.6, 70.4] |
| gray | colour removal | 100.0 [97, 102] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/frame_r50 99.2 [98.0, 100.0] |

### Per secret and protocol

**identity** (top1)

| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |
|---|---|---|---:|---:|---:|---:|
| raw | within | retrained/frame_r50 | 99.2 [98.0, 100.0] | 4.0 (4.0 / 4.0) | 99.2 | 0.0 |
| raw | cross_utility | retrained/frame_r50 | 99.9 [99.6, 100.0] | 4.0 (4.0 / 4.0) | 99.9 | 0.0 |
| raw | hardest | retrained/r3d | 100.0 [100.0, 100.0] | 4.0 (4.0 / 4.0) | 100.0 | 0.0 |
| raw | cross_scenario | retrained/skeleton | 26.3 [20.5, 32.4] | 4.0 (4.0 / 4.0) | 26.3 | 0.0 |
| black | within | retrained/r3d | 4.0 [1.3, 7.4] | 4.0 (4.0 / 4.0) | 99.2 | 100.0 |
| black | cross_utility | retrained/frame_r50 | 4.0 [1.5, 7.1] | 4.0 (4.0 / 4.0) | 99.9 | 100.0 |
| black | hardest | retrained/frame_r50 | 4.0 [0.0, 10.0] | 4.0 (4.0 / 4.0) | 100.0 | 100.0 |
| black | cross_scenario | retrained/frame_r50 | 4.0 [1.3, 8.0] | 4.0 (4.0 / 4.0) | 26.3 | 100.0 |
| pose_canon_down16 | within | retrained/r3d | 13.9 [10.2, 17.8] | 4.0 (4.0 / 4.0) | 99.2 | 89.6 |
| pose_canon_down16 | cross_utility | retrained/r3d | 14.2 [11.1, 17.7] | 4.0 (4.0 / 4.0) | 99.9 | 89.4 |
| pose_canon_down16 | hardest | retrained/r3d | 8.5 [3.5, 14.5] | 4.0 (4.0 / 4.0) | 100.0 | 95.3 |
| pose_canon_down16 | cross_scenario | retrained/r3d | 9.7 [6.2, 13.5] | 4.0 (4.0 / 4.0) | 26.3 | 74.4 |
| pose_canon_down8 | within | retrained/r3d | 18.4 [14.2, 22.9] | 4.0 (4.0 / 4.0) | 99.2 | 84.9 |
| pose_canon_down8 | cross_utility | retrained/r3d | 16.2 [12.9, 19.6] | 4.0 (4.0 / 4.0) | 99.9 | 87.3 |
| pose_canon_down8 | hardest | retrained/r3d | 12.6 [6.5, 19.6] | 4.0 (4.0 / 4.0) | 100.0 | 91.1 |
| pose_canon_down8 | cross_scenario | retrained/r3d | 13.4 [9.7, 17.6] | 4.0 (4.0 / 4.0) | 26.3 | 57.9 |
| pose_canon_down32 | within | retrained/frame_r50 | 11.0 [7.7, 14.6] | 4.0 (4.0 / 4.0) | 99.2 | 92.6 |
| pose_canon_down32 | cross_utility | retrained/frame_r50 | 11.9 [8.8, 15.4] | 4.0 (4.0 / 4.0) | 99.9 | 91.8 |
| pose_canon_down32 | hardest | retrained/frame_r50 | 11.1 [5.0, 18.1] | 4.0 (4.0 / 4.0) | 100.0 | 92.7 |
| pose_canon_down32 | cross_scenario | retrained/r3d | 8.7 [5.9, 11.7] | 4.0 (4.0 / 4.0) | 26.3 | 78.9 |
| bdq_down64 | within | retrained/r3d | 15.6 [11.4, 20.2] | 4.0 (4.0 / 4.0) | 99.2 | 87.9 |
| bdq_down64 | cross_utility | retrained/r3d | 18.0 [14.3, 22.1] | 4.0 (4.0 / 4.0) | 99.9 | 85.4 |
| bdq_down64 | hardest | retrained/r3d | 16.6 [8.5, 26.0] | 4.0 (4.0 / 4.0) | 100.0 | 86.9 |
| bdq_down64 | cross_scenario | retrained/r3d | 7.5 [4.5, 10.9] | 4.0 (4.0 / 4.0) | 26.3 | 84.2 |
| pose_canon_down4 | within | retrained/r3d | 24.1 [18.8, 29.8] | 4.0 (4.0 / 4.0) | 99.2 | 78.9 |
| pose_canon_down4 | cross_utility | retrained/r3d | 19.9 [16.5, 23.5] | 4.0 (4.0 / 4.0) | 99.9 | 83.4 |
| pose_canon_down4 | hardest | retrained/r3d | 15.1 [9.0, 21.7] | 4.0 (4.0 / 4.0) | 100.0 | 88.5 |
| pose_canon_down4 | cross_scenario | retrained/r3d | 19.9 [15.1, 25.2] | 4.0 (4.0 / 4.0) | 26.3 | 28.6 |
| pose_canon | within | retrained/r3d | 30.4 [24.6, 36.9] | 4.0 (4.0 / 4.0) | 99.2 | 72.2 |
| pose_canon | cross_utility | retrained/r3d | 21.3 [17.5, 25.0] | 4.0 (4.0 / 4.0) | 99.9 | 82.0 |
| pose_canon | hardest | retrained/r3d | 17.6 [10.6, 25.1] | 4.0 (4.0 / 4.0) | 100.0 | 85.9 |
| pose_canon | cross_scenario | retrained/r3d | 24.6 [19.0, 30.7] | 4.0 (4.0 / 4.0) | 26.3 | 7.5 |
| bdq_down32 | within | retrained/r3d | 41.3 [34.9, 47.9] | 4.0 (4.0 / 4.0) | 99.2 | 60.8 |
| bdq_down32 | cross_utility | retrained/r3d | 40.7 [34.7, 46.4] | 4.0 (4.0 / 4.0) | 99.9 | 61.7 |
| bdq_down32 | hardest | retrained/r3d | 33.2 [21.0, 44.4] | 4.0 (4.0 / 4.0) | 100.0 | 69.6 |
| bdq_down32 | cross_scenario | retrained/r3d | 14.9 [10.5, 20.4] | 4.0 (4.0 / 4.0) | 26.3 | 51.1 |
| pose | within | retrained/r3d | 55.7 [48.9, 61.9] | 4.0 (4.0 / 4.0) | 99.2 | 45.7 |
| pose | cross_utility | retrained/r3d | 41.5 [35.8, 47.4] | 4.0 (4.0 / 4.0) | 99.9 | 60.9 |
| pose | hardest | retrained/r3d | 35.7 [25.2, 46.0] | 4.0 (4.0 / 4.0) | 100.0 | 67.0 |
| pose | cross_scenario | retrained/r3d | 36.6 [30.3, 43.6] | 4.0 (4.0 / 4.0) | 26.3 | 0.0 |
| motion_down32 | within | retrained/r3d | 64.2 [57.6, 70.4] | 4.0 (4.0 / 4.0) | 99.2 | 36.7 |
| motion_down32 | cross_utility | retrained/r3d | 73.1 [68.5, 77.8] | 4.0 (4.0 / 4.0) | 99.9 | 28.0 |
| motion_down32 | hardest | retrained/r3d | 61.8 [50.5, 72.9] | 4.0 (4.0 / 4.0) | 100.0 | 39.8 |
| motion_down32 | cross_scenario | retrained/r3d | 6.4 [3.2, 10.4] | 4.0 (4.0 / 4.0) | 26.3 | 89.5 |
| gray | within | retrained/frame_r50 | 99.2 [98.0, 100.0] | 4.0 (4.0 / 4.0) | 99.2 | 0.0 |
| gray | cross_utility | retrained/frame_r50 | 99.9 [99.6, 100.0] | 4.0 (4.0 / 4.0) | 99.9 | 0.0 |
| gray | hardest | retrained/r3d | 100.0 [100.0, 100.0] | 4.0 (4.0 / 4.0) | 100.0 | 0.0 |
| gray | cross_scenario | retrained/skeleton | 26.3 [20.5, 32.4] | 4.0 (4.0 / 4.0) | 26.3 | 0.0 |

## Checks

- PASS measurable (identity, within): raw 99.2 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, cross_utility): raw 99.9 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, hardest): raw 100.0 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, cross_scenario): raw 26.3 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS black control (identity, within): black frames 4.0 [1.3, 7.4] vs baseline 4.0
- frozen raw-trained attackers understate the retrained ones by a median 17.1 points (max 60.0, motion_down32)
- screening vs full evaluation, privacy rank correlation: 1.00

## Recommendation

- **pose_canon_down16** (canonical stick figure): utility 89.0, privacy 89.6, PUS 89.3
- privacy first (utility >= 50): pose_canon_down32; utility first (privacy >= 50): bdq_down32
- **certified** (worst over the recommendation and its 4 robustness re-runs): privacy 89.5, utility 88.8, PUS 89.1
- robustness pose_canon_down16_dino: privacy 92.3, utility 89.0
- robustness pose_canon_down16_e24: privacy 89.5, utility 89.4
- robustness pose_canon_down16_s1: privacy 90.0, utility 88.8
- robustness pose_canon_down16_s2: privacy 89.5, utility 88.8
- config: `{"type": "chain", "stages": [{"type": "pose", "every": 16, "width": 2, "canonical": 0.5}, {"type": "down", "factor": 16}]}`
