# Replication: Selective, Interpretable and Motion Consistent Privacy Attribute Obfuscation (CVPR 2024)

Upstream: [f-ilic/SelectivePrivacyPreservation](https://github.com/f-ilic/SelectivePrivacyPreservation) at commit `7e8b98d`.
Paper: [arXiv 2403.12710](https://arxiv.org/abs/2403.12710).

Status: **plan, not yet run.** Last updated 2026-10-26.

## Goal

Re-run the upstream pipeline in Google Colab on a GPU runtime, compare our numbers to the paper's
Table 2, and say plainly whether the method reproduces. Survive Colab VM disconnects without losing
finished work.

## Paper numbers to match (Table 2, "Ours" row)

| Dataset | Original action | Original privacy | Ours action | Ours privacy | Ours f0.5 |
|---|---|---|---|---|---|
| IPN  | 88.05 | 91.92 | 87.11 | 51.93 | 0.68 |
| KTH  | 90.15 | 91.87 | 88.67 | 5.46  | 0.92 |
| SBU  | 86.31 | 85.96 | 86.74 | 13.19 | 0.87 |

Metrics (from the paper, Sec. 4.1, as summarised in secondary sources; verify against the PDF):

- Action and privacy are top-1 accuracy, in percent. Lower privacy is better.
- f_lambda(a, p) = (1 - lambda) * a + lambda * (1 - p), with a and p as fractions. Table 2 uses lambda = 0.5.

## Protocol the upstream code implements

- **Action.** Action models are trained on original frames (Kinetics-400 pretrained), then evaluated with
  obfuscation applied at test time only. The paper states it never trains on obfuscated video. The table
  averages 13 action architectures.
- **Privacy.** Five classifiers (ResNet18, ResNet50, ResNet101, ViT-B/16, ViT-B/32), trained on original frames
  (ImageNet pretrained), evaluated under obfuscation. The table averages the five.
- **Templates.** KTH and SBU use torso, arm, leg, hair. IPN uses cheek, eyes, forehead, hair.
- **Model selection.** `create_table_action.py` picks each run's best checkpoint by *test* top-1. This is the
  upstream protocol. It is optimistic, so we report it as such.

## Pipeline order (per dataset)

1. **Data.** Frames under `data/<ds>/<video>/frame%04d.png` (KTH layout from `data/splits/kth/extract_kthframes.sh`).
   KTH zips come from csc.kth.se (reachable). SBU and IPN are not in the repo. Their hosts need checking from Colab.
2. **AFD motion-consistent noise.** `afd/generate_<ds>.py`. Needed for the selective "Ours" path.
3. **Similarity maps.** `matcher/similarity_from_template_for_dataset.py`, one run per template, using the DINO
   descriptors in `output/descriptors/`. Writes `data/<ds>_<template>_sim/`.
4. **Train.** `action_train.py` and `privacy_train.py` on original frames.
5. **Select.** `create_table_action.py` copies each run's best checkpoint to `best.pt`.
6. **Evaluate with obfuscation.** `action_eval.py` / `privacy_eval.py` with `-selectively_mask --obfuscate <templates>`.
7. **Compare.** Compute f_0.5 and the deltas from the reported accuracies.

Masks (YOLO) are only needed for the naive baselines, so we skip them in the first pass.

## Scope of the first pass

- **KTH first.** It is the smallest dataset and has the clearest effect (privacy 91.87 to 5.46).
- Action: a small subset of the 13 architectures. Full coverage only if the Colab budget allows.
- Privacy: ResNet18 and ResNet50 first. The other three follow if time allows.
- SBU and IPN after KTH, if their data can be obtained.

## Known gaps in upstream (to verify, not assume)

- Training has no resume. It saves only the best checkpoint, with no optimizer or RNG state. A disconnect
  loses the run. We add per-epoch checkpoints and resume (see Colab runtime section below).
- `Simulation` writes to a dated, numbered directory, so a restart would not find its own run. We pin the
  output directory.
- The paper's model-selection rule uses the test set (see above).

## Colab runtime and durability

- **Runtime.** A GPU runtime is required. `.cuda()` is called unconditionally in the training code. The first
  probe found a CPU-only VM (2 vCPU, 12 GB RAM, no CUDA), so the runtime must be switched first.
- **Persistence.** Everything that is expensive to produce (extracted frames, AFD maps, similarity maps,
  checkpoints, results) lives on Google Drive. Completion markers make each stage skippable on restart.
- **Commits.** Each logical change is committed to this repo as it lands.

## Layout (planned)

```
replication/selective-privacy/
  README.md          this file: plan, status, results
  notebooks/         exported Colab notebook(s) that drive the run
  scripts/           stage scripts and the resume helper
  patches/           changes applied to the upstream checkout, pinned to 7e8b98d
  results/           per-run metrics and the comparison table
```

## Status

- [x] Read upstream code, identify protocol and data layout.
- [x] Confirm KTH download host is reachable from the sandbox (HTTP 200).
- [ ] Switch the Colab runtime to GPU.
- [ ] Mount Drive, clone upstream at 7e8b98d, install dependencies.
- [ ] KTH frames, AFD, similarity maps.
- [ ] KTH action and privacy runs with resume.
- [ ] Comparison table and verdict.
