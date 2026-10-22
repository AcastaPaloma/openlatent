# Video harness: dataset contract and pipeline design

Status: design, not implemented. Date: 2026-10-22.

## Goal

Take a labelled video dataset that meets the contract below, train a SPLAVU-style
anonymizing adapter on embeddings from a frozen video foundation model, and report
how much privacy leakage it removes and how much task utility it keeps. The result
should be comparable across datasets and reproducible from a run directory.

The method follows SPLAVU, "Privacy Beyond Pixels: Latent Anonymization for
Privacy-Preserving Video Understanding" (Fioresi, Dave, Shah, ICLR 2026;
[arXiv 2511.08666](https://arxiv.org/abs/2511.08666), MIT code at
[UCF-CRCV/PrivacyBeyondPixels](https://github.com/UCF-CRCV/PrivacyBeyondPixels)).
It is the latent-space, transformer successor to SPAct (CVPR 2022) from the same lab.

Non-goals for v1: dense tasks (segmentation, captioning, detection boxes), producing
anonymized *videos*, formal guarantees (DP, dFIL), and anything that runs in a browser.

## Method in one paragraph

A frozen encoder `f_E` (VideoMAE-B in v1) turns each clip into one embedding `x`.
An Anonymizing Adapter Module `f_A` (3-layer self-attention encoder, 8 heads, about
14.6M parameters for VideoMAE-B) maps `x` to `z` of the same shape. It starts as the
identity. Training minimizes

    L = w_LC * ||f_A(x) - x||^2  +  w_T * L_task(f_T(z))  -  w_B * L_NT-Xent(static clips)

with SPLAVU's defaults `w_LC = 100, w_T = w_B = 1`. The privacy term is self-supervised:
two random frames of a video are each tiled into a "static clip", encoded, passed through
`f_A`, and the adapter is trained to make them *hard* to match contrastively. No private
labels are needed for training. SPLAVU reports a 35% relative drop in VISPR cMAP for I3D
(63.64 to 41.07) and about 29% for VideoMAE-B, with utility near baseline
(I3D Kinetics-400 top-1 62.67 to 62.11).

Two properties shape this design:

- **Cheap after extraction.** On HMDB51 with VideoMAE-B, SPLAVU reports 185.3 min to train
  on raw video against 4.0 min of feature extraction plus 1.4 min of adapter training on
  cached features. Everything after stage 3 below works on cached vectors.
- **Leaks under white-box inversion.** With the adapter's weights, their white-box inversion
  pushes VP-HMDB51 privacy cMAP to 73.7, above the raw level. The adapter therefore stays
  private to the data owner, and the harness always reports this attack.

## Threat model

| Party | Has | Does not have |
|---|---|---|
| Data owner | raw videos, labels, encoder, adapter | — |
| Data consumer (and attacker) | `z` for the shared split, the public encoder `f_E` and its weights, any public data (VISPR, Kinetics, faces) | raw videos, adapter weights |

Headline privacy numbers assume this (black-box) attacker. White-box inversion, where
the attacker also has `f_A`, is reported as the worst case. Claims are worded as
"reduced measured leakage against the attacks tested", never "anonymized".

## Dataset contract

A dataset is a directory with a `dataset.yaml` and a manifest CSV. The ChestX-ray14
module becomes one adapter that emits this format (see "Reuse of the existing harness").

### `dataset.yaml`

```yaml
name: ucf101
contract_version: 1
media_root: videos/            # relative to the dataset directory
manifest: manifest.csv
media_kind: video              # video | frames (a folder of numbered JPG/PNG per sample)

group_column: group_id         # required; no group may cross splits
identity_column: null          # optional person ID; enables re-ID scoring

splits:
  mode: hash                   # hash | column
  column: null                 # with mode: column, values train | val | eval
  eval_frac: 0.3
  val_frac: 0.1
  salt: openlatent-v1

targets:                       # at least one role: utility target with train: true
  - name: action
    column: action
    kind: multiclass           # binary | multiclass | multilabel | continuous
    role: utility              # utility | privacy
    train: true                # used in L_task when training the adapter
  - name: gender
    column: perceived_gender
    kind: binary
    role: privacy

reference_privacy: [vispr]     # public attribute sets scored through the same encoder+adapter
```

### `manifest.csv`

One row per sample (a video, or a trimmed segment of one).

| Column | Required | Meaning |
|---|---|---|
| `sample_id` | yes | unique, stable string; used as the row key everywhere |
| `path` | yes | file (or frame folder) under `media_root` |
| `start_sec`, `end_sec` | no | trim window; whole file if absent |
| *group column* | yes | the unit that must not cross splits: person, recording session, or source video. For UCF101 this is the `gXX` group. |
| *identity column* | no | person ID if known; may equal the group column |
| *target columns* | per `targets` | labels; multilabel values are pipe-separated (`a|b`), as in ChestX-ray14 |

### Constraints (v1)

The validator (`openlatent video validate`) rejects a dataset unless:

1. `sample_id` is unique, every `path` exists and decodes, and every sample has at least
   16 frames at the encoder's sampling stride inside its trim window.
2. Every `train: true` target has no missing values and at least 2 classes in each split.
   Privacy targets may have missing values; those rows are skipped for that target only.
3. There are at least 20 groups in the eval split (fewer gives meaningless confidence
   intervals) and, if `identity_column` is set, at least 2 identities with 2+ eval samples.
4. With `mode: column`, no group appears in two splits.
5. Tasks are clip-level: one label per sample per target.

It warns, but does not reject, when content is not human-centric (VISPR reference
attributes are about people) or when a privacy target is more than 95% one class.

## Run directory

Every stage reads and writes inside one run directory, so a run is reproducible and
shareable without the raw videos.

```
runs/<run_id>/
  run.yaml                  # dataset path, encoder, adapter config, seeds, sweep
  splits.csv                # sample_id, group_id, split, probe_role, reid_role
  features/<encoder>/
    clips.npy  clips.ids.txt        # one row per (sample_id, clip_index)
    static.npy static.ids.txt       # K tiled-frame static clips per train sample
    eval.npy   eval.ids.txt         # one row per eval sample: mean of 5 evenly spaced clips
    reference/vispr.npy ...         # reference privacy sets, tiled to static clips
  adapters/w_B=<value>/seed=<s>/adapter.pt
  latents/w_B=<value>/seed=<s>/eval.npy  eval.ids.txt
  reports/
    tier1/raw.json  tier1/w_B=<value>_seed=<s>.json
    tier2/...  inversion/...
    summary.json  summary.md  curve.png
  stages/<stage>.json        # input hashes, versions, timing; used to skip finished stages
```

Feature files use the format `openlatent.features` already reads: an `.npy` matrix plus a
sibling `.ids.txt`.

## Pipeline

| # | Stage | Command | Deterministic? | Hardware |
|---|---|---|---|---|
| 0 | Hardware check | `openlatent doctor` | yes | any |
| 1 | Validate dataset | `openlatent video validate` | yes | CPU |
| 2 | Group-disjoint splits | `openlatent video split` | yes | CPU |
| 3 | Extract embeddings | `openlatent video extract` | yes* | GPU recommended |
| 4 | Score raw embeddings (tier 1, "before") | `openlatent video score --raw` | yes | CPU |
| 5 | Train adapters over a `w_B` sweep and seeds | `openlatent video train` | seeded | GPU or Apple MPS |
| 6 | Encode eval split with each adapter | `openlatent video encode` | yes | CPU/GPU |
| 7 | Score each latent (tier 1, "after") | `openlatent video score` | yes | CPU |
| 8 | Fresh attackers (tier 2) | `openlatent video attack` | seeded | CPU/GPU |
| 9 | Inversion attacks | `openlatent video invert` | seeded | GPU |
| 10 | Summary and privacy-utility curve | `openlatent video report` | yes | CPU |

`openlatent video run --dataset <dir> --run <dir>` runs every stage and skips any stage
whose `stages/<stage>.json` input hashes match.

\* Extraction uses validation transforms only (short side 256, 224 center crop, ImageNet
normalization), `torch.use_deterministic_algorithms(True)`, and frame choices from
`stable_unit` hashes, so it is repeatable on one machine. It is the only stage that touches
raw video; after it, raw data is not needed.

### Stage details

**0. `doctor`.** Reports the torch device (CUDA, MPS or CPU), GPU memory, free disk, and
an estimate for stages 3 and 5 from the dataset's frame count. It refuses stage 3 on CPU
for datasets above a configurable size rather than running for days.

**2. Splits.** Generalizes `data/chestxray14.make_splits` from `patient_id` to the group
column. `split` is train / val / eval by group hash (or the given column). Within eval,
`probe_role` is fit / test by group hash, and `reid_role` is gallery / query by identity
(the first sample per identity by `sample_id` is the gallery item).

**3. Extract.** For every sample: all non-overlapping 16-frame clips (stride 2, SPLAVU's
setting for VideoMAE) into `clips.npy`; for train samples, `K = 8` static clips (a hashed
random frame tiled 16 times) into `static.npy`; for eval samples, the mean of 5 evenly
spaced clip embeddings into `eval.npy`. Reference privacy images (VISPR) are tiled into
static clips the same way.

**5. Train.** Follows SPLAVU's recipe:
- warm-start `f_A` as the identity: L1 reconstruction, AdamW, lr 2e-5, 100 epochs
- pretrain the task head `f_T` on raw embeddings
- joint training: lr 1e-4, batch 512 (linear scaling rule), 100 epochs, loss as above;
  each step samples one cached clip per train sample and two of its static clips
- sweep `w_B` over `{0, 0.25, 0.5, 1, 2, 4}` with 3 seeds each; `w_B = 0` is the
  "adapter without privacy" control

Two details are not in the paper text: the NT-Xent projection head and temperature, and
how the 3-layer self-attention adapter tokenizes a single clip embedding. Take both from
the released code before implementing.

**4 and 7. Tier 1 (deterministic).** Reuses the existing scorers unchanged, on `eval.npy`:

| Score | Scorer | Needs |
|---|---|---|
| re-ID top-k, MRR, verification AUC | `scorers.reid.reid_scores` | `identity_column` |
| privacy target probes | `binary_probe`, `ridge_probe`, `multilabel_probe`, and a new `multiclass_probe` | privacy targets |
| utility target probes | same probes | utility targets |
| identity alignment | `scorers.dependence.identity_alignment` | identity or group column |
| x vs z | `linear_cka`, `distance_correlation` | raw and latent eval features |
| reference privacy | logistic probes on VISPR static clips, fit on its train split, scored on its test split | stage 3 reference features |

**8. Tier 2 (headline).** On each frozen adapter's latents, train fresh attackers from
scratch, at least 5 architectures: logistic, kNN (k=10), SPLAVU's 2-layer MLP
(Linear(d, d) → ReLU → Linear), a 3-layer MLP with dropout, and gradient-boosted trees on
a 256-dim PCA. Train each on every privacy target and on VISPR (cMAP over its attributes),
over 3 seeds. Report the **best** attacker per target as mean ± 95% CI. Retrain the utility
heads the same way.

**9. Inversion.** Black-box: train an MLP `g: z → x` on pairs from the owner's train split
(the strongest case of a consumer who obtained some pairs), then run the tier-1 and tier-2
attackers that were trained on *raw* embeddings on `g(z)`. White-box: with `f_A`, optimize
an embedding `x̂` so that `f_A(x̂) ≈ z`, then score `x̂` the same way. Pixel reconstruction
through the VideoMAE decoder is optional (v2) and only for the demo.

**10. Report.** `summary.json` holds, per `w_B`: utility kept and leakage removed for every
target (normalized against chance, as `harness.compare_reports` does today), the tier-2
best-attacker numbers, and the inversion numbers. `curve.png` plots leakage removed (x)
against utility kept (y) for the adapter sweep plus the baselines: raw, Gaussian noise
at matched utility, random projection, PCA to matched dimension, LEACE erasure of the
privacy targets (`EleutherAI/concept-erasure`), and face blurring before extraction with
EgoBlur (`facebookresearch/EgoBlur`, Apache-2.0), the "what people do today" baseline.

## Reuse of the existing harness

| Existing | Change |
|---|---|
| `scorers/reid.py`, `scorers/dependence.py` | none |
| `scorers/probes.py` | add `multiclass_probe` (top-1 and balanced accuracy, chance = majority-class rate) with the same grouped-CV selection |
| `features.py` | none; video features use the same `.npy` + `.ids.txt` format |
| `data/chestxray14.py` | split logic moves to a generic `data/splits.py` keyed on group and identity columns; ChestX-ray14 becomes an adapter that writes a `dataset.yaml` (group = identity = patient, privacy = sex and age, utility = 14 findings) |
| `harness.py` | `run_report` loops over the targets in the dataset spec instead of hard-coding sex, age and findings. Each report section records its own `role`, `chance` and direction, so `compare_reports` no longer needs `PRIVACY_METRICS`. Report schema goes to 2; schema 1 reports are refused, not migrated. |
| `cli.py` | existing commands stay; video commands live under `openlatent video ...` |

New modules: `video/contract.py` (load and validate), `video/extract.py` (decoding,
clip sampling, encoder registry), `video/adapter.py` (AAM and training loop),
`video/attacks.py` (tier 2 and inversion), `video/report.py`.

New dependencies (an optional `video` extra, so the ChestX path stays light): `torch`,
`torchvision`, `transformers` (VideoMAE and V-JEPA 2 encoders), `PyAV` for decoding,
`pyyaml`. Choices are by maintenance as of 2026-10: PyAV is active, while `decord` has had
no commits since 2022; the original VideoMAE repo has had none since 2023, so encoders load
through `transformers` (or `facebookresearch/vjepa2`, MIT, active).

None of the method repos are maintained (SPAct: last commit 2022-06, with open issues about
missing files; PrivacyBeyondPixels: a single initial commit, 2026-04; the token-pruning
paper's repo has no license, so its code cannot be reused). The SPLAVU repo is a reference
for reading, not a dependency: the adapter is small enough (3 layers, about 14.6M
parameters) to implement here against maintained libraries.

## Milestones

1. **Reproduce SPLAVU** on UCF101 (action) with VISPR as the privacy reference, using
   VideoMAE-B. Pass if the relative VISPR cMAP reduction is within 5 points of the paper's
   ~29% and UCF101 top-1 is within 2 points of the raw-embedding baseline. If this fails,
   stop and diagnose before generalizing.
2. **Contract and stages 1 to 4** on UCF101 and HMDB51, with the generic splits and the
   refactored `run_report`. ChestX-ray14 reports must still reproduce exactly through the
   new code path.
3. **Stages 5 to 10** end to end on UCF101, then cross-dataset (train the adapter on
   UCF101, evaluate on HMDB51 and PA-HMDB51).
4. **A dataset with identities.** NTU RGB+D 60 RGB videos (40 people) or Toyota
   Smarthome, so re-ID is scored on real person IDs and not only VISPR attributes.

## Open questions

1. **What consumers receive.** One `z` per sample (mean of clips), or one per clip?
   Per clip supports more downstream tasks and gives an attacker more material.
2. **Encoder choice.** VideoMAE-B first for cost; V-JEPA-H is stronger but much heavier
   to extract. Is one encoder per release acceptable?
3. **Who holds the adapter.** The design assumes the data owner keeps it private. Publishing
   it for others to reuse on their data turns every consumer into a white-box attacker.
4. **Group definition per dataset.** UCF101's `gXX` groups share actors and backgrounds,
   but they are not person IDs. Should UCF101 runs report re-ID at all, or only VISPR and
   attribute targets?
