# Privacy report: kth_scenario

Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.

## Dataset

- 2391 items (1791 train / 600 test), video, 120x160, 1 channel(s), 25.0 fps, median 94 frames per item
- people visible in 78% of sampled frames (mean 0.8, median area 17.2%), skeletons found in 82%, camera motion 3.0
- targets: action (utility, multiclass, 6 classes); identity (secret, multiclass, 25 classes)

## Where the task and the secrets live (profile)

Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw footage. Probes rank cues, they do not certify privacy.

| view | action | identity |
|---|---:|---:|
| colour | 0.32 | 0.40 |
| lowres | 0.33 | 0.76 |
| scene | 0.45 | 0.93 |
| person | 1.10 | 0.98 |
| body | 1.09 | 0.32 |
| posture | 1.02 | 0.23 |
| motion | 0.33 | 0.19 |
| label | – | 0.00 |

## Plan

- identity: strongest cues [('person', 0.98), ('scene', 0.93), ('lowres', 0.76), ('colour', 0.4)]; task 'action' cues [('person', 1.1), ('body', 1.09), ('posture', 1.02), ('scene', 0.45)]
- train bdq_down32: R1: baseline / control

## Results

Headline protocol: within (the dataset's split). Attackers: frame_r50, r3d retrained on the obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait attacker for identities. Frontier hypervolume: 0.848.

| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |
|---|---|---:|---:|---:|---:|---:|---|
| raw | reference | 100.0 [94, 105] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/frame_r50 96.2 [93.3, 98.5] |
| black | control | 0.0 [-8, 8] | 100.0 | 0.0 | 50.0 | 100.0 | identity: retrained/frame_r50 4.0 [1.3, 7.3] |
| pose_canon_down16 **(front)** | canonical stick figure | 78.6 [71, 87] | 85.4 | 81.9 | 82.0 | 74.4 | identity: retrained/r3d 17.5 [13.5, 22.2] |
| pose_canon_down4 **(front)** | canonical stick figure | 88.4 [81, 95] | 73.2 | 80.1 | 80.8 | 28.6 | identity: retrained/r3d 28.7 [23.3, 34.5] |
| pose_canon_down8 **(front)** | canonical stick figure | 80.9 [73, 89] | 78.7 | 79.8 | 79.8 | 57.9 | identity: retrained/r3d 23.7 [18.8, 28.7] |
| pose_canon **(front)** | canonical stick figure | 90.5 [84, 97] | 71.1 | 79.6 | 80.8 | 4.5 | identity: retrained/r3d 30.7 [25.0, 37.0] |
| pose_canon_down32 **(front)** | canonical stick figure | 69.3 [62, 77] | 90.1 | 78.3 | 79.7 | 78.9 | identity: retrained/frame_r50 13.2 [9.3, 17.3] |
| pose **(front)** | stick figure | 101.1 [96, 106] | 59.9 | 74.9 | 79.9 | 0.0 | identity: retrained/r3d 41.0 [34.2, 47.7] |
| bdq_down32 | learned pool | 74.5 [67, 82] | 70.3 | 72.4 | 72.4 | 53.4 | identity: retrained/r3d 31.3 [25.2, 37.8] |
| motion_down16 | fixed motion map | 87.3 [80, 94] | 46.1 | 60.3 | 66.7 | 5.9 | identity: retrained/r3d 53.7 [46.7, 59.7] |
| motion_down32 | fixed motion map | 61.4 [52, 70] | 55.0 | 58.0 | 58.2 | 28.1 | identity: retrained/r3d 45.5 [38.8, 52.2] |

### Per secret and protocol

**identity** (top1)

| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |
|---|---|---|---:|---:|---:|---:|
| raw | within | retrained/frame_r50 | 96.2 [93.3, 98.5] | 4.0 (4.0 / 4.0) | 96.2 | 0.0 |
| raw | cross_utility | retrained/frame_r50 | 100.0 [100.0, 100.0] | 4.0 (4.0 / 4.0) | 100.0 | 0.0 |
| raw | hardest | retrained/frame_r50 | 99.5 [98.5, 100.0] | 4.0 (4.0 / 4.0) | 99.5 | 0.0 |
| raw | cross_scenario | retrained/skeleton | 26.3 [20.5, 32.4] | 4.0 (4.0 / 4.0) | 26.3 | 0.0 |
| black | within | retrained/frame_r50 | 4.0 [1.3, 7.3] | 4.0 (4.0 / 4.0) | 96.2 | 100.0 |
| black | cross_utility | retrained/frame_r50 | 4.0 [1.5, 7.1] | 4.0 (4.0 / 4.0) | 100.0 | 100.0 |
| black | hardest | retrained/frame_r50 | 4.0 [0.0, 10.0] | 4.0 (4.0 / 4.0) | 99.5 | 100.0 |
| black | cross_scenario | retrained/frame_r50 | 4.0 [1.3, 8.0] | 4.0 (4.0 / 4.0) | 26.3 | 100.0 |
| pose_canon_down16 | within | retrained/r3d | 17.5 [13.5, 22.2] | 4.0 (4.0 / 4.0) | 96.2 | 85.4 |
| pose_canon_down16 | cross_utility | retrained/r3d | 14.2 [11.1, 17.7] | 4.0 (4.0 / 4.0) | 100.0 | 89.4 |
| pose_canon_down16 | hardest | retrained/r3d | 10.5 [4.5, 17.0] | 4.0 (4.0 / 4.0) | 99.5 | 93.2 |
| pose_canon_down16 | cross_scenario | retrained/r3d | 9.7 [6.2, 13.5] | 4.0 (4.0 / 4.0) | 26.3 | 74.4 |
| pose_canon_down4 | within | retrained/r3d | 28.7 [23.3, 34.5] | 4.0 (4.0 / 4.0) | 96.2 | 73.2 |
| pose_canon_down4 | cross_utility | retrained/r3d | 19.9 [16.5, 23.5] | 4.0 (4.0 / 4.0) | 100.0 | 83.4 |
| pose_canon_down4 | hardest | retrained/r3d | 17.5 [10.5, 23.5] | 4.0 (4.0 / 4.0) | 99.5 | 85.9 |
| pose_canon_down4 | cross_scenario | retrained/r3d | 19.9 [15.1, 25.2] | 4.0 (4.0 / 4.0) | 26.3 | 28.6 |
| pose_canon_down8 | within | retrained/r3d | 23.7 [18.8, 28.7] | 4.0 (4.0 / 4.0) | 96.2 | 78.7 |
| pose_canon_down8 | cross_utility | retrained/r3d | 16.2 [12.9, 19.6] | 4.0 (4.0 / 4.0) | 100.0 | 87.3 |
| pose_canon_down8 | hardest | retrained/r3d | 10.5 [5.0, 17.0] | 4.0 (4.0 / 4.0) | 99.5 | 93.2 |
| pose_canon_down8 | cross_scenario | retrained/r3d | 13.4 [9.7, 17.6] | 4.0 (4.0 / 4.0) | 26.3 | 57.9 |
| pose_canon | within | retrained/r3d | 30.7 [25.0, 37.0] | 4.0 (4.0 / 4.0) | 96.2 | 71.1 |
| pose_canon | cross_utility | retrained/r3d | 23.9 [20.0, 28.2] | 4.0 (4.0 / 4.0) | 100.0 | 79.2 |
| pose_canon | hardest | retrained/r3d | 15.0 [9.0, 21.0] | 4.0 (4.0 / 4.0) | 99.5 | 88.5 |
| pose_canon | cross_scenario | retrained/r3d | 25.3 [19.6, 31.2] | 4.0 (4.0 / 4.0) | 26.3 | 4.5 |
| pose_canon_down32 | within | retrained/frame_r50 | 13.2 [9.3, 17.3] | 4.0 (4.0 / 4.0) | 96.2 | 90.1 |
| pose_canon_down32 | cross_utility | retrained/frame_r50 | 11.5 [8.3, 14.8] | 4.0 (4.0 / 4.0) | 100.0 | 92.2 |
| pose_canon_down32 | hardest | retrained/frame_r50 | 8.0 [3.0, 13.5] | 4.0 (4.0 / 4.0) | 99.5 | 95.8 |
| pose_canon_down32 | cross_scenario | retrained/r3d | 8.7 [5.9, 11.7] | 4.0 (4.0 / 4.0) | 26.3 | 78.9 |
| pose | within | retrained/r3d | 41.0 [34.2, 47.7] | 4.0 (4.0 / 4.0) | 96.2 | 59.9 |
| pose | cross_utility | retrained/r3d | 38.5 [33.2, 44.3] | 4.0 (4.0 / 4.0) | 100.0 | 64.1 |
| pose | hardest | retrained/frame_r50 | 15.0 [7.0, 24.0] | 4.0 (4.0 / 4.0) | 99.5 | 88.5 |
| pose | cross_scenario | retrained/r3d | 35.8 [29.6, 42.6] | 4.0 (4.0 / 4.0) | 26.3 | 0.0 |
| bdq_down32 | within | retrained/r3d | 31.3 [25.2, 37.8] | 4.0 (4.0 / 4.0) | 96.2 | 70.3 |
| bdq_down32 | cross_utility | retrained/r3d | 45.1 [39.2, 51.1] | 4.0 (4.0 / 4.0) | 100.0 | 57.2 |
| bdq_down32 | hardest | retrained/r3d | 18.0 [10.5, 26.5] | 4.0 (4.0 / 4.0) | 99.5 | 85.3 |
| bdq_down32 | cross_scenario | retrained/r3d | 14.4 [10.0, 19.4] | 4.0 (4.0 / 4.0) | 26.3 | 53.4 |
| motion_down16 | within | retrained/r3d | 53.7 [46.7, 59.7] | 4.0 (4.0 / 4.0) | 96.2 | 46.1 |
| motion_down16 | cross_utility | retrained/r3d | 94.4 [91.2, 96.9] | 4.0 (4.0 / 4.0) | 100.0 | 5.9 |
| motion_down16 | hardest | retrained/r3d | 50.0 [37.0, 63.5] | 4.0 (4.0 / 4.0) | 99.5 | 51.8 |
| motion_down16 | cross_scenario | frozen/frame_r50 | 4.0 [1.3, 8.0] | 4.0 (4.0 / 4.0) | 26.3 | 100.0 |
| motion_down32 | within | retrained/r3d | 45.5 [38.8, 52.2] | 4.0 (4.0 / 4.0) | 96.2 | 55.0 |
| motion_down32 | cross_utility | retrained/r3d | 73.1 [68.5, 77.8] | 4.0 (4.0 / 4.0) | 100.0 | 28.1 |
| motion_down32 | hardest | retrained/r3d | 39.0 [27.5, 51.0] | 4.0 (4.0 / 4.0) | 99.5 | 63.4 |
| motion_down32 | cross_scenario | retrained/r3d | 6.4 [3.2, 10.4] | 4.0 (4.0 / 4.0) | 26.3 | 89.5 |

## Checks

- PASS measurable (identity, within): raw 96.2 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, cross_utility): raw 100.0 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, hardest): raw 99.5 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS measurable (identity, cross_scenario): raw 26.3 vs baseline 4.0 (chance 4.0, task label alone 4.0)
- PASS black control (identity, within): black frames 4.0 [1.3, 7.3] vs baseline 4.0
- frozen raw-trained attackers understate the retrained ones by a median 26.7 points (max 49.5, motion_down16)
- screening vs full evaluation, privacy rank correlation: 0.98

## Recommendation

- **pose_canon_down16** (canonical stick figure): utility 78.6, privacy 85.4, PUS 81.9
- privacy first (utility >= 50): pose_canon_down32; utility first (privacy >= 50): pose
- **certified** (worst over the recommendation and its 4 robustness re-runs): privacy 84.8, utility 75.5, PUS 79.9
- robustness pose_canon_down16_dino: privacy 90.1, utility 78.6
- robustness pose_canon_down16_e24: privacy 86.6, utility 78.0
- robustness pose_canon_down16_s1: privacy 84.8, utility 75.5
- robustness pose_canon_down16_s2: privacy 84.8, utility 77.5
- config: `{"type": "chain", "stages": [{"type": "pose", "every": 16, "width": 2, "canonical": 0.5}, {"type": "down", "factor": 16}]}`
