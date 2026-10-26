## KTH: this replication vs paper Table 2

| Setting | Action top-1 | Privacy top-1 | f_0.5 |
|---|---|---|---|
| Paper, original (13 action / 5 privacy models, averaged) | 90.15 | 91.87 | - |
| Paper, ours (same averaging) | 88.67 | 5.46 | 0.92 |
| This run, original, action x3d_s | 91.33 | 98.00 | 0.47 |
| This run, ours, action x3d_s + privacy resnet18 | 90.00 | 85.33 | 0.52 |

Models evaluated:

- action `x3d_s`: obf_arm_hair_leg_torso=90.00, original=91.33
- privacy `resnet18`: obf_arm_hair_leg_torso=85.33, original=98.00

Obfuscation lerp weight (mean of template similarity maps, 981 test frames):
mean 0.117, median 0.113, p90 0.173, p99 0.267, max 0.422; pixels >= 0.25: 1.48%, >= 0.5: 0.00%.
