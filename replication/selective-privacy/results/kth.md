## KTH: this replication vs paper Table 2 (averaged over models)

| Setting | Action top-1 | Privacy top-1 | f_0.5 | Models (action / privacy) |
|---|---|---|---|---|
| Paper, original frames | 90.15 | 91.87 | 0.49 | 13 / 5 |
| Paper, ours | 88.67 | 5.46 | 0.92 | 13 / 5 |
| This run, original frames | 91.33 | 97.78 | 0.47 | 1 / 3 |
| This run, obfuscated, default matcher | 90.00 | 87.56 | 0.51 | 1 / 3 |
| This run, obfuscated, use_targeted=true | 87.33 | 54.22 | 0.67 | 1 / 3 |

### Action models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `x3d_s` | 91.33 | 90.00 | 87.33 |

### Privacy models

| Model | Original frames | Obfuscated, default matcher | Obfuscated, use_targeted=True |
|---|---|---|---|
| `resnet101` | 97.33 | 90.67 | 55.33 |
| `resnet18` | 98.00 | 85.33 | 52.67 |
| `resnet50` | 98.00 | 86.67 | 54.67 |

Obfuscation lerp weight (mean of template similarity maps, 981 test frames), default matcher:
mean 0.117, median 0.113, p90 0.173, p99 0.267, max 0.422; pixels >= 0.25: 1.48%, >= 0.5: 0.00%.
Targeted matcher: mean 0.34; 28.8% of pixels >= 0.5.
