# Privacy report: utdmhad

Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.

## Dataset

- 861 items (648 train / 213 test), video, 112x150, 3 channel(s), 15.0 fps, median 51 frames per item
- people visible in 100% of sampled frames (mean 1.0, median area 10.1%), skeletons found in 100%, camera motion 1.0
- targets: action (utility, multiclass, 27 classes); identity (secret, multiclass, 8 classes)
- warning: identity: only 8 classes; identity among few people is easy to recover from body shape and motion alone, so expect low privacy scores for every obfuscator
- warning: R4: identity has only 8 classes; body and motion tell so few people apart, so expect low privacy scores for every obfuscator

## Where the task and the secrets live (profile)

Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw footage. Probes rank cues, they do not certify privacy.

| view | action | identity |
|---|---:|---:|
| colour | 0.77 | 1.00 |
| lowres | 0.77 | 1.00 |
| scene | 0.56 | 1.00 |
| person | 0.95 | 1.00 |
| body | 0.84 | 0.97 |
| posture | 0.80 | 0.76 |
| motion | 0.76 | 0.89 |
| label | – | 0.01 |

## Plan

- identity: strongest cues [('lowres', 1.0), ('scene', 1.0), ('person', 1.0), ('colour', 1.0)]; task 'action' cues [('person', 0.95), ('body', 0.84), ('posture', 0.8), ('colour', 0.77)]
- train bdq_down32: R2: identity is in the people's appearance (1.00); the task carries motion (0.76)
- train bdq_down64: R2: identity is in the people's appearance (1.00); the task carries motion (0.76)

## Results

Headline protocol: within (the dataset's split). Attackers: frame_r50, r3d retrained on the obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait attacker for identities. Frontier hypervolume: 0.412.

| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |
|---|---|---:|---:|---:|---:|---:|---|
| raw | reference | 100.0 [100, 100] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/frame_r50 100.0 [100.0, 100.0] |
| black | control | 0.0 [-2, 3] | 99.5 | 0.0 | 49.7 | 99.5 | identity: retrained/frame_r50 12.7 [8.5, 17.4] |
| motion_down64 **(front)** | fixed motion map | 60.5 [53, 67] | 51.9 | 55.8 | 56.2 | 50.3 | identity: retrained/r3d 54.5 [47.9, 61.0] |
| bdq_down64 **(front)** | learned pool | 39.5 [33, 46] | 72.7 | 51.2 | 56.1 | 66.8 | identity: retrained/r3d 36.2 [29.6, 42.7] |
| pose_canon_down32 | canonical stick figure | 40.0 [34, 47] | 36.9 | 38.4 | 38.4 | 34.8 | identity: retrained/frame_r50 67.6 [61.5, 74.2] |
| pose_canon_down16 **(front)** | canonical stick figure | 63.4 [57, 70] | 18.7 | 28.9 | 41.1 | 18.7 | identity: retrained/r3d 83.6 [78.4, 88.3] |
| pose_canon_down8 **(front)** | canonical stick figure | 71.7 [66, 78] | 7.0 | 12.7 | 39.3 | 7.0 | identity: retrained/r3d 93.9 [90.6, 96.7] |
| pose_canon **(front)** | canonical stick figure | 79.0 [74, 84] | 6.4 | 11.9 | 42.7 | 4.3 | identity: retrained/r3d 94.4 [91.5, 97.7] |
| bdq_down32 | learned pool | 77.1 [72, 83] | 6.4 | 11.8 | 41.7 | 6.4 | identity: retrained/r3d 94.4 [91.1, 97.2] |
| pose_canon_down4 | canonical stick figure | 76.6 [71, 82] | 4.8 | 9.1 | 40.7 | 3.2 | identity: retrained/r3d 95.8 [93.0, 98.1] |
| motion_down16 **(front)** | fixed motion map | 100.0 [100, 100] | 0.0 | 0.0 | 50.0 | 0.0 | identity: retrained/r3d 100.0 [100.0, 100.0] |
| motion_down32 | fixed motion map | 97.6 [95, 100] | 0.0 | 0.0 | 48.8 | 0.0 | identity: retrained/r3d 100.0 [100.0, 100.0] |

### Per secret and protocol

**identity** (top1)

| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |
|---|---|---|---:|---:|---:|---:|
| raw | within | retrained/frame_r50 | 100.0 [100.0, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 0.0 |
| raw | cross_utility | retrained/frame_r50 | 100.0 [100.0, 100.0] | 12.5 (12.5 / 12.5) | 100.0 | 0.0 |
| raw | hardest | retrained/frame_r50 | 100.0 [100.0, 100.0] | 12.7 (12.7 / 12.7) | 100.0 | 0.0 |
| raw | cross_trial | retrained/frame_r50 | 100.0 [100.0, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 0.0 |
| black | within | retrained/frame_r50 | 12.7 [8.5, 17.4] | 12.2 (12.2 / 12.2) | 100.0 | 99.5 |
| black | cross_utility | retrained/frame_r50 | 12.5 [8.7, 16.4] | 12.5 (12.5 / 12.5) | 100.0 | 100.0 |
| black | hardest | retrained/frame_r50 | 12.7 [5.6, 19.7] | 12.7 (12.7 / 12.7) | 100.0 | 100.0 |
| black | cross_trial | retrained/frame_r50 | 12.7 [8.5, 17.4] | 12.2 (12.2 / 12.2) | 100.0 | 99.5 |
| motion_down64 | within | retrained/r3d | 54.5 [47.9, 61.0] | 12.2 (12.2 / 12.2) | 100.0 | 51.9 |
| motion_down64 | cross_utility | retrained/r3d | 41.1 [35.9, 46.7] | 12.5 (12.5 / 12.5) | 100.0 | 67.3 |
| motion_down64 | hardest | retrained/r3d | 40.8 [29.6, 52.1] | 12.7 (12.7 / 12.7) | 100.0 | 67.7 |
| motion_down64 | cross_trial | retrained/r3d | 55.9 [49.3, 62.4] | 12.2 (12.2 / 12.2) | 100.0 | 50.3 |
| bdq_down64 | within | retrained/r3d | 36.2 [29.6, 42.7] | 12.2 (12.2 / 12.2) | 100.0 | 72.7 |
| bdq_down64 | cross_utility | retrained/r3d | 38.0 [32.4, 43.6] | 12.5 (12.5 / 12.5) | 100.0 | 70.9 |
| bdq_down64 | hardest | retrained/r3d | 36.6 [26.8, 47.9] | 12.7 (12.7 / 12.7) | 100.0 | 72.6 |
| bdq_down64 | cross_trial | retrained/r3d | 41.3 [34.7, 48.4] | 12.2 (12.2 / 12.2) | 100.0 | 66.8 |
| pose_canon_down32 | within | retrained/frame_r50 | 67.6 [61.5, 74.2] | 12.2 (12.2 / 12.2) | 100.0 | 36.9 |
| pose_canon_down32 | cross_utility | retrained/frame_r50 | 56.1 [50.5, 62.4] | 12.5 (12.5 / 12.5) | 100.0 | 50.2 |
| pose_canon_down32 | hardest | retrained/r3d | 53.5 [42.3, 66.2] | 12.7 (12.7 / 12.7) | 100.0 | 53.2 |
| pose_canon_down32 | cross_trial | retrained/r3d | 69.5 [62.9, 75.1] | 12.2 (12.2 / 12.2) | 100.0 | 34.8 |
| pose_canon_down16 | within | retrained/r3d | 83.6 [78.4, 88.3] | 12.2 (12.2 / 12.2) | 100.0 | 18.7 |
| pose_canon_down16 | cross_utility | retrained/frame_r50 | 64.5 [58.5, 70.4] | 12.5 (12.5 / 12.5) | 100.0 | 40.6 |
| pose_canon_down16 | hardest | retrained/r3d | 60.6 [47.9, 71.8] | 12.7 (12.7 / 12.7) | 100.0 | 45.2 |
| pose_canon_down16 | cross_trial | retrained/frame_r50 | 81.7 [76.1, 86.9] | 12.2 (12.2 / 12.2) | 100.0 | 20.9 |
| pose_canon_down8 | within | retrained/r3d | 93.9 [90.6, 96.7] | 12.2 (12.2 / 12.2) | 100.0 | 7.0 |
| pose_canon_down8 | cross_utility | retrained/r3d | 67.2 [61.7, 72.5] | 12.5 (12.5 / 12.5) | 100.0 | 37.5 |
| pose_canon_down8 | hardest | retrained/frame_r50 | 71.8 [61.9, 81.7] | 12.7 (12.7 / 12.7) | 100.0 | 32.3 |
| pose_canon_down8 | cross_trial | retrained/r3d | 93.4 [90.1, 96.7] | 12.2 (12.2 / 12.2) | 100.0 | 7.5 |
| pose_canon | within | retrained/r3d | 94.4 [91.5, 97.7] | 12.2 (12.2 / 12.2) | 100.0 | 6.4 |
| pose_canon | cross_utility | retrained/r3d | 73.9 [68.6, 79.1] | 12.5 (12.5 / 12.5) | 100.0 | 29.9 |
| pose_canon | hardest | retrained/r3d | 73.2 [62.0, 83.1] | 12.7 (12.7 / 12.7) | 100.0 | 30.6 |
| pose_canon | cross_trial | retrained/r3d | 96.2 [93.4, 98.6] | 12.2 (12.2 / 12.2) | 100.0 | 4.3 |
| bdq_down32 | within | retrained/r3d | 94.4 [91.1, 97.2] | 12.2 (12.2 / 12.2) | 100.0 | 6.4 |
| bdq_down32 | cross_utility | retrained/r3d | 81.5 [76.7, 85.7] | 12.5 (12.5 / 12.5) | 100.0 | 21.1 |
| bdq_down32 | hardest | retrained/r3d | 80.3 [70.4, 88.7] | 12.7 (12.7 / 12.7) | 100.0 | 22.6 |
| bdq_down32 | cross_trial | retrained/r3d | 91.5 [87.8, 94.8] | 12.2 (12.2 / 12.2) | 100.0 | 9.6 |
| pose_canon_down4 | within | retrained/r3d | 95.8 [93.0, 98.1] | 12.2 (12.2 / 12.2) | 100.0 | 4.8 |
| pose_canon_down4 | cross_utility | retrained/frame_r50 | 73.9 [69.0, 79.1] | 12.5 (12.5 / 12.5) | 100.0 | 29.9 |
| pose_canon_down4 | hardest | retrained/frame_r50 | 70.4 [59.2, 80.3] | 12.7 (12.7 / 12.7) | 100.0 | 33.9 |
| pose_canon_down4 | cross_trial | retrained/r3d | 97.2 [94.8, 99.1] | 12.2 (12.2 / 12.2) | 100.0 | 3.2 |
| motion_down16 | within | retrained/r3d | 100.0 [100.0, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 0.0 |
| motion_down16 | cross_utility | retrained/r3d | 99.0 [97.6, 100.0] | 12.5 (12.5 / 12.5) | 100.0 | 1.2 |
| motion_down16 | hardest | retrained/r3d | 100.0 [100.0, 100.0] | 12.7 (12.7 / 12.7) | 100.0 | 0.0 |
| motion_down16 | cross_trial | retrained/r3d | 100.0 [100.0, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 0.0 |
| motion_down32 | within | retrained/r3d | 100.0 [100.0, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 0.0 |
| motion_down32 | cross_utility | retrained/r3d | 89.9 [86.4, 93.4] | 12.5 (12.5 / 12.5) | 100.0 | 11.6 |
| motion_down32 | hardest | retrained/r3d | 85.9 [77.5, 93.0] | 12.7 (12.7 / 12.7) | 100.0 | 16.1 |
| motion_down32 | cross_trial | retrained/r3d | 99.1 [97.7, 100.0] | 12.2 (12.2 / 12.2) | 100.0 | 1.1 |

## Checks

- PASS measurable (identity, within): raw 100.0 vs baseline 12.2 (chance 12.2, task label alone 12.2)
- PASS measurable (identity, cross_utility): raw 100.0 vs baseline 12.5 (chance 12.5, task label alone 12.5)
- PASS measurable (identity, hardest): raw 100.0 vs baseline 12.7 (chance 12.7, task label alone 12.7)
- PASS measurable (identity, cross_trial): raw 100.0 vs baseline 12.2 (chance 12.2, task label alone 12.2)
- PASS black control (identity, within): black frames 12.7 [8.5, 17.4] vs baseline 12.2
- frozen raw-trained attackers understate the retrained ones by a median 81.7 points (max 87.8, motion_down16)
- screening vs full evaluation, privacy rank correlation: 0.98

## Recommendation

- **motion_down64** (fixed motion map): utility 60.5, privacy 51.9, PUS 55.8
- privacy first (utility >= 50): motion_down64; utility first (privacy >= 50): motion_down64
- robustness motion_down64_dino: privacy 78.1, utility 60.5
- robustness motion_down64_e24: privacy 51.9, utility 62.9
- robustness motion_down64_s1: privacy 51.9, utility 59.5
- robustness motion_down64_s2: privacy 54.0, utility 53.7
- config: `{"type": "chain", "stages": [{"type": "diff", "gain": 4.0}, {"type": "down", "factor": 64}]}`
