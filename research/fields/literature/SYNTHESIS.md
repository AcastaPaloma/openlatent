# Phase-1 synthesis (2026-11-04, after 6 scouts; scores are scouts' own, /30, abstracts-heavy so treat as priors)

| # | Candidate field | Scouts that surfaced it | Score | Benchmark (deterministic?) | Main risk |
|---|---|---|---|---|---|
| 1 | **Source-invariant embeddings**: keep label, hide centre/batch/source. Pathology (PathoROB), Cell Painting (BatchBench), single-cell | medical, other-vision, other-modalities (3 of 6, independently) | 26 / 26 / 26 | PathoROB: 99,392 patches, 34 centres, fixed metrics; BatchBench: sealed test sources | headroom (PathoROB RI already 0.89-0.93); batch confounded with biology; DANN only +0.23 pp so the gap is real |
| 2 | **Robotics: distractor/embodiment-free latent actions** (LAPA-style), MetaWorld/Distracting Control | robotics | 27 | Distracting Meta-World, MT10 with paired with/without-distractor renders = exact leakage ground truth | crowded, toy scale; masks help only if nuisance and task occupy different pixels |
| 3 | **Identity-free expression / speech emotion** on CREMA-D (+RAVDESS) | affect, methods (VoicePrivacy protocol) | 25 | CREMA-D open (ODbL), same sentences in all emotions; VoicePrivacy-style retrained attacker | acted data; frozen-attacker flaw in the only prior (WACV 2025) is our gap |
| 4 | Egocentric / AR wearer privacy | other-vision, robotics | 26 / 24 | EgoPrivacy (no defence track) | gated data, scene = identity |
| 5 | EEG subject identity, gaze privacy, CARE-PD gait, XR motion | medical, modalities | 23-26 | partial | tiny cohorts, licences, benchmark to build |

Dropped by scouts with reasons: MRI/CT defacing, rPPG, AV/dataset-release masking, fall detection, VoicePrivacy as a research
field (saturated; kept as protocol template), clinical/engagement affect, wearable IMU/text/tabular/LiDAR/WiFi.

Method-side consensus (methods_transfer.md): rate bottleneck with no adversary is the mandatory control; adversarial removal only
protects against the attacker family it saw; closed-form linear erasure (LEACE/ISP) is the deterministic floor.

## Provisional focus (not yet locked)
A. Source-invariant embeddings (pathology + Cell Painting, one codebase)  B. Robotics latent actions free of distractor/embodiment
C. Identity-free affect (CREMA-D) -- reuses the existing identity harness.
Egocentric stays the fallback because the user's earlier privacy work carries straight over.

## Contrarian round (must fail to beat A-C before lock-in)
1. Missed-fields scout: search for fields none of the six scouts touched.
2. Stress-test scout: try to break A-C (saturation in 2026, benchmark determinism, licence, downloadability from the VM).
