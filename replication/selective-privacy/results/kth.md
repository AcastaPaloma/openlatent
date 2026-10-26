## KTH: this replication vs paper Table 2 (averaged over models)

| Setting | Action top-1 | Privacy top-1 | f_0.5 | Models (action / privacy) |
|---|---|---|---|---|
| Paper, original frames | 90.15 | 91.87 | 0.49 | 13 / 5 |
| Paper, ours | 88.67 | 5.46 | 0.92 | 13 / 5 |
| This run, original frames | 87.78 | 90.17 | 0.49 | 3 / 4 |
| This run, obfuscated, default matcher | 86.67 | 79.83 | 0.53 | 3 / 4 |
| This run, obfuscated, use_targeted=true | 86.00 | 48.33 | 0.69 | 3 / 4 |

### Action models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `i3d_r50` | 83.33 | 80.00 | 82.00 |
| `slow_r50` | 88.67 | 90.00 | 88.67 |
| `x3d_s` | 91.33 | 90.00 | 87.33 |

### Privacy models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `resnet101` | 97.33 | 90.67 | 55.33 |
| `resnet18` | 98.00 | 85.33 | 52.67 |
| `resnet50` | 98.00 | 86.67 | 54.67 |
| `vit` | 67.33 | 56.67 | 30.67 |

`vit` trains poorly with upstream's privacy settings (best test top-1 66.7 after 500 epochs,
against 96-98 for the ResNets), so it lowers every privacy average; without it the privacy
averages are 97.78 / 87.56 / 54.22. The test split has 150 clips, so one clip is 0.67 points.

Obfuscation lerp weight (mean of template similarity maps, 981 test frames), default matcher:
mean 0.117, median 0.113, p90 0.173, p99 0.267, max 0.422; pixels >= 0.25: 1.48%, >= 0.5: 0.00%.
Targeted matcher: mean 0.34; 28.8% of pixels >= 0.5.
