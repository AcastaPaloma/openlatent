## KTH: this replication vs paper Table 2 (averaged over models)

| Setting | Action top-1 | Privacy top-1 | f_0.5 | Models (action / privacy) |
|---|---|---|---|---|
| Paper, original frames | 90.15 | 91.87 | 0.49 | 13 / 5 |
| Paper, ours | 88.67 | 5.46 | 0.92 | 13 / 5 |
| This run, original frames | 88.52 | 88.67 | 0.50 | 9 / 5 |
| This run, obfuscated, default matcher | 88.22 | 78.27 | 0.55 | 9 / 5 |
| This run, obfuscated, use_targeted=true | 87.63 | 45.47 | 0.71 | 9 / 5 |

### Action models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `c2d_r50` | 83.33 | 85.33 | 82.67 |
| `i3d_r50` | 83.33 | 80.00 | 82.00 |
| `mvit_base_16x4` | 86.00 | 84.67 | 88.00 |
| `r2plus1d_r50` | 89.33 | 88.00 | 86.67 |
| `slow_r50` | 88.67 | 90.00 | 88.67 |
| `slowfast_r50` | 90.67 | 92.00 | 90.67 |
| `x3d_l` | 93.33 | 93.33 | 92.67 |
| `x3d_m` | 90.67 | 90.67 | 90.00 |
| `x3d_s` | 91.33 | 90.00 | 87.33 |

Still training: `csn_r101`. Not run: `e2s_x3d_{s,m,l}` (need training-split
optical flow and upstream's separate E2S trainer).

### Privacy models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `resnet101` | 97.33 | 90.67 | 55.33 |
| `resnet18` | 98.00 | 85.33 | 52.67 |
| `resnet50` | 98.00 | 86.67 | 54.67 |
| `vit` | 67.33 | 56.67 | 30.67 |
| `vit_b_32` | 82.67 | 72.00 | 34.00 |

The two ViTs train less well with upstream's privacy settings (best test top-1 66.7 and 82.0
after 500 epochs, against 96-98 for the ResNets), which pulls the privacy averages down on
every row. ResNet-only privacy averages: 97.78 / 87.56 / 54.22. Relative to each model's own
original-frame accuracy, default obfuscation keeps 84-93% of it for every privacy model, and
targeted keeps 41-57%. The test split has 150 clips, so one clip is 0.67 points.

Obfuscation lerp weight (mean of template similarity maps, 981 test frames), default matcher:
mean 0.117, median 0.113, p90 0.173, p99 0.267, max 0.422; pixels >= 0.25: 1.48%, >= 0.5: 0.00%.
Targeted matcher: mean 0.34; 28.8% of pixels >= 0.5.
