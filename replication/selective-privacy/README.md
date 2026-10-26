# Replication: Selective, Interpretable and Motion Consistent Privacy Attribute Obfuscation (CVPR 2024)

Upstream: [f-ilic/SelectivePrivacyPreservation](https://github.com/f-ilic/SelectivePrivacyPreservation) at commit `7e8b98d`.
Paper: [arXiv 2403.12710](https://arxiv.org/abs/2403.12710).

Status: **KTH run complete (one action + one privacy model). Utility reproduces; privacy protection
does not.** Last updated 2026-10-26.

## Verdict (KTH)

| Setting | Action top-1 | Privacy top-1 | f_0.5 |
|---|---|---|---|
| Paper, original (13 action / 5 privacy models, averaged) | 90.15 | 91.87 | 0.49 |
| Paper, ours | 88.67 | **5.46** | 0.92 |
| This run, original (X3D-S / ResNet-18) | 91.33 | 98.00 | 0.47 |
| This run, ours (torso, arm, leg, hair) | 90.00 | **85.33** | 0.52 |

- **Action utility reproduces.** Obfuscation costs X3D-S 1.3 points (paper: 1.5 points, 13-model average).
- **Privacy protection does not reproduce.** The person-ID classifier still gets 85.33% under
  obfuscation, against 5.46% in the paper. It drops 12.7 points, not 86.
- **Why, as far as we can trace it.** Upstream blends each pixel toward motion-consistent noise with
  `torch.lerp(frame, afd, sim)`, where `sim` is the mean of the template similarity maps scaled to [0, 1].
  With the released descriptors and the documented default matcher settings (DINO ViT-S/8, stride 4,
  key facet, layer 11, `use_targeted=False`), that weight has median 0.11 and maximum 0.42 over 981
  KTH test frames. No pixel is ever more than 42% noise, and only 1.5% reach 25%. Per template, the
  99th percentile is 0.29-0.44, and at most 0.6% of pixels reach 0.5 (leg). The released code applies no
  normalisation or thresholding to these maps (the only rescaling, `clamp(80, 255)`, is in a
  visualisation script). Our maps equal upstream's `similarity_from_descriptor` output exactly
  (max difference 0.000000 on the check clip, all four templates).
- **What would settle it.** The paper's numbers need a step or setting that is not in the released code:
  normalised or thresholded similarity maps, or per-video targeted descriptors (`use_targeted=True`,
  not the documented default). Re-running the similarity stage with `use_targeted=True` is the
  cheapest test (about 1.5 h on an A100). A note to the authors asking which setting produced Table 2
  would be faster.
- **Caveats.** One model per side rather than the paper's averages; the privacy metric scores one random
  frame per clip, as `privacy_eval.py` does; model selection by best test top-1, as upstream does.
  The privacy gap (85 vs 5) is far larger than any of these could explain.

Raw evaluation JSON: `MyDrive/openlatent-replication/runs/eval/`; table: `results/kth.md`;
diagnostic: `scripts/inspect_obfuscation.py`.

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

## Deviations from the paper and upstream (all deliberate, all recorded)

- **Scope.** One action model (X3D-S) and one privacy model (ResNet-18) on KTH. The paper
  averages 13 action and 5 privacy models. Full coverage did not fit the compute budget in
  this pass.
- **Batch size.** X3D-S runs at micro-batch 32 with 2-step gradient accumulation, for an
  effective batch of 64 (upstream's default). Batch 64 ran out of memory on the A100-40GB.
  BatchNorm statistics therefore use micro-batches of 32.
- **Missing clip.** `person13_handclapping_d3_uncomp` is in upstream metadata but not in the
  KTH release, so its row is dropped (`scripts/fix_kth_metadata.py`). 599 clips remain.
- **Derived inputs for the test split only.** Obfuscation happens only at test time, so AFD
  and similarity maps are built for the 150 test clips.
- **Batched RAFT for AFD.** Upstream runs RAFT one frame pair at a time (about 0.2 s per pair).
  `derive_kth_afd.py` batches pairs and matches upstream on a 10-frame check: flow differs by at
  most 0.015 px, and 0.019% of AFD pixels differ (fp16 rounding).
- **Evaluation protocol.** Training's per-epoch test top-1 uses upstream's `drop_last=True`.
  Final numbers come from `eval_model.py`, which scores the full test split with no dropped
  batch, as upstream's `action_eval.py` does. So the two can differ by a point or two.
- **Privacy metric.** The paper's text says privacy is averaged over 32 frames per clip.
  Upstream `privacy_eval.py` scores one random frame per clip. We follow the code.
- **Model selection.** Checkpoints are chosen by best test top-1 during training (upstream
  protocol). This is optimistic, so the held-out number is the full-test evaluation of that
  checkpoint.
- **Dependencies.** `scripts/compat.py` aliases a module that torchvision removed, so
  pytorchvideo 0.1.3 imports on torch 2.11. Only the unused AugMix path needs it.
- **SBU and IPN.** SBU's host did not respond from the sandbox. IPN needs an external
  download and is not started.

## Durability

- Checkpoints (`last.pt`, `best.pt`), `DONE.json`, provenance and eval JSON are written atomically.
- `scripts/sync_drive.sh push` mirrors runs, logs, raw KTH archives and derived inputs to
  `MyDrive/openlatent-replication`. `pull` restores them after a VM restart.
- Jobs are launched detached (own session, no inherited pipes). A stopped cell cannot take
  them down.

## Status

- [x] Read upstream code, identify protocol and data layout.
- [x] Colab runtime on A100-40GB; upstream imports and pretrained X3D-S run.
- [x] KTH extracted and verified (599 clips, frame counts match metadata).
- [x] Resumable training and eval, tested with crash-and-resume on the GPU and again on real runs.
- [x] X3D-S and ResNet-18 trained on original frames.
- [x] Test-split AFD and similarity maps (torso, arm, leg, hair), equivalence-checked against upstream.
- [x] Original and obfuscated evaluations; comparison table; verdict above.
- [ ] Optional: `use_targeted=True` similarity ablation to test the most likely explanation.
- [ ] More architectures (paper averages 13 + 5); SBU and IPN.
