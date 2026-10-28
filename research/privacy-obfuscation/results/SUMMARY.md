# Summary of results (generated from results/runs/*/result.json)

All privacy numbers are from attackers **retrained on the obfuscated training set** (the best of a frame ResNet-50 and a video R3D-18; DINOv2 where run). Means ± population s.d. over seeds/variants listed. Frozen (raw-trained) attackers, the protocol of the paper we started from, are in `frozen_attackers.md`; they understate leakage by a median 47 identity points on KTH.

## KTH (25 people, 6 actions), Ilic split

| method | action top-1 | identity top-1 | identity, unseen actions | runs |
|---|---:|---:|---:|---:|
| raw video | 97.5 | 99.5 | 99.9 | 1 |
| blur sigma 8 | 94.5 | 95.0 | – | 1 |
| downsample x16 | 91.8 | 92.6 | 99.0 | 1 |
| frame difference | 96.5 | 97.5 | – | 1 |
| fixed motion map x16 (diff->down16) | 94.8 | 84.6 | – | 1 |
| fixed motion map x32 | 93.6 ± 0.1 | 63.4 ± 0.5 | 65.9 | 2 |
| fixed motion map x48 | 83.0 ± 0.6 | 36.6 ± 0.3 | – | 2 |
| fixed motion map x64 (1x2 px) | 70.6 ± 0.6 | 11.5 ± 0.8 | 13.9 | 3 |
| fixed motion map x120 (1 px) | 54.2 | 6.9 | – | 1 |
| plain BDQ, a5 ens + adversary warm-up | 70.2 ± 8.8 | 26.9 ± 3.3 | 14.4 | 2 |
| plain BDQ, a5 ens (4 seeds) | 89.0 ± 3.7 | 50.0 ± 8.7 | – | 4 |
| VAE latent edit (best) | 96.2 | 75.6 | – | 1 |
| **learned pool: BDQ -> x16** | 95.4 ± 0.9 | 57.5 ± 2.6 | – | 4 |
| **learned pool: BDQ -> x24** | 93.2 ± 0.3 | 51.8 ± 3.0 | – | 2 |
| **learned pool: BDQ -> x32 (a3/a5)** | 91.9 ± 1.0 | 41.2 ± 6.5 | 26.9 | 6 |
| **learned pool: BDQ -> x32 (a1)** | 92.0 ± 0.7 | 37.8 ± 3.9 | – | 3 |
| **learned pool: BDQ -> x48 (a3)** | 87.4 ± 0.1 | 25.0 ± 0.3 | 17.0 | 2 |
| **learned pool: BDQ -> x64** | 80.9 ± 2.0 | 11.4 ± 1.5 | 6.0 | 6 |
| learned pool x64, no adversary (a0) | 81.4 ± 0.4 | 11.4 ± 0.7 | – | 2 |
| **learned pool: BDQ -> x120 (1 px)** | 76.3 ± 0.6 | 8.4 ± 1.0 | 5.0 | 3 |
| learned pool x120, no adversary (a0) | 76.5 ± 1.6 | 7.9 ± 0.7 | – | 2 |
| 3D U-Net -> x64 (a0) | 66.1 | 9.9 | – | 1 |

Chance: action 16.7, identity 4.0. Brightness-only control (one number per frame): identity 22.4. All-black control: 16.7 / 4.0. Shuffled train labels: identity at chance for every attacker.

## KTH, scenario split (test recording conditions unseen in training)

| method | action | identity | identity, unseen actions | runs |
|---|---:|---:|---:|---:|
| raw video | 88.7 | 97.7 | – | 1 |
| fixed motion map x16 | 81.0 | 50.7 | – | 1 |
| fixed motion map x32 | 58.0 | 42.7 | – | 1 |
| fixed motion map x64 | 23.3 | 9.5 | – | 1 |
| plain BDQ a3 | 67.0 ± 7.5 | 41.1 ± 11.2 | – | 4 |
| learned pool x16 | 74.9 ± 1.2 | 37.2 ± 0.2 | – | 2 |
| learned pool x32 (a3) | 64.2 ± 2.0 | 31.7 ± 0.5 | – | 2 |
| learned pool x64 | 51.6 ± 0.7 | 12.8 ± 0.9 | – | 3 |

## Charades (90 people filmed in their own homes; action = 157-way multilabel mAP)

| method | action mAP | identity top-1 | runs |
|---|---:|---:|---:|
| raw | 15.8 | 93.5 | 1 |
| blur sigma 8 | 9.0 | 86.9 | 1 |
| downsample x16 | 8.4 | 85.0 | 1 |
| downsample x64 (1x2 RGB px) | 5.5 | 38.8 | 1 |
| person boxes blacked out | 10.6 | 90.6 | 1 |
| frame difference | 12.9 | 89.4 | 1 |
| BDQ, no adversary | 12.6 | 88.8 | 1 |
| BDQ a3 ens | 5.8 ± 0.1 | 22.4 ± 1.3 | 3 |

Floors: action mAP 4.4 (blind predictor), identity 3.6 (most frequent person). Three adversarial runs collapsed exactly to these floors (content-free output); they are excluded and rerun with a fixed collapse guard.

## Privacy attributes (VP-UCF101, VP-HMDB51; 5 attributes, cMAP)

| dataset | raw action / cMAP | blind cMAP floor | best obfuscation (action / cMAP) |
|---|---:|---:|---|
| VP-UCF101 | 91.8 / 76.6 | 58.7 | nothing tried gets below ~66 cMAP while keeping action (e.g. down16 53.7 / 70.3) |
| VP-HMDB51 | 64.9 / 75.2 | 58.8 | down16 29.7 / 66.6; BDQ 63.1 / 72.7 |

## Findings

1. **The frozen-attacker protocol is the main reason published obfuscations look private.** Retrained attackers recover 95-99% KTH identity through blur, pixelation and downsampling up to x16.
2. **On KTH, a learned temporal-difference stage (BDQ) in front of a tiny spatial pool is the best obfuscator found**: it sits below the fixed motion-map curve at every pool size (x32: 41 vs 63 identity at ~92-94 action; x48: 25 vs 37 identity with 4 more action points; x64: ~11 identity for both but +10 action), and below plain BDQ, consistently across pool sizes, seeds, privacy weights, 24-epoch attackers, 2.5 s views and DINOv2. At 1-2 numbers per frame it needs no adversary at all; at ~15 numbers per frame the adversary buys ~15 points.
3. **Most identity that survives a coarse motion map is person-specific action style.** With the attacker tested on actions it never trained on, the learned pools leak 27% (x32), 17% (x48), 6% (x64), 5% (1 px), against 66% for the fixed x32 map and ~99% for raw or x16 video.
4. **Where identity is the scene, obfuscating the person does not help.** Charades: removing people leaves 91% identity and two RGB pixels per frame leave 39%; only adversarial BDQ removes most of it, at the cost of almost all action. UTD-MHAD: each subject has their own recording session, and mean colour alone identifies 73%.
5. **Appearance attributes (VP-UCF101, VP-HMDB51) were not removed by any method tried** while keeping action.

Caveats and bugs found along the way are logged in `docs/journal.md` (release-path edge frames, the collapse guard for many-class and multilabel tasks, a UTD-MHAD premise that turned out wrong).
