# privharness: from any labelled visual dataset to a certified obfuscator

**Goal:** a person brings a dataset of videos or images. It is labelled with:
- a **task** we want to keep, such as an action or activity;
- one or more **secrets** we want to hide, such as who is in it or appearance attributes.

The harness then:
- prepares the data;
- measures where the task and each secret live in the footage;
- searches and trains obfuscators suited to that structure;
- attacks each one with a suite of retrained adversaries;
- returns scores with fixed definitions, sanity checks and a recommendation.

**Why it's built this way:** everything here is grounded in our experiments (`docs/journal.md`, `results/SUMMARY.md`) and in the literature reviewed on 2026-10-29 (references at the end). The main lessons:

1. **Frozen-attacker numbers are not privacy.** Attackers trained on raw footage understate leakage by about 47 points on KTH (median). Elazar & Goldberg (2018) and Pittaluga et al. (2019) show the same: a post-hoc classifier trained to saturation recovers what the training adversary missed. Every headline number here comes from attackers retrained on the obfuscated data.
2. **Where a secret lives decides what can hide it.**
   - **Charades:** identity is the home, so removing the person does nothing; canonical stick figures work.
   - **KTH:** identity is body shape and motion. Canonical stick figures with a downsample beat every learned obfuscator.
   - **UTD-MHAD:** 8 people with their own sessions are identifiable from almost anything.
   - **VP-UCF101 / VP-HMDB51:** the task label implies most attributes, and the remainder is structural.

   So the harness profiles before it searches.
3. **A secret the task label implies is not a leak of the footage.** Scores are measured from a baseline that includes a task-label attacker (Wang et al. 2019; Jayaraman & Evans 2022). Multilabel secrets the task implies use the conditional metric.

## Pipeline

```
ingest / adapt  ->  profile  ->  plan  ->  screen  ->  evaluate (shortlist)  ->  robustness  ->  report
   (CPU)            (GPU)        (CPU)     (GPU)        (GPU, parallel)          (GPU)          (CPU)
```

| stage | module | output |
|---|---|---|
| ingest | `ingest.py` (raw media and a JSON description) or `adapters/prepared.py` (datasets already decoded by ppv) | `dataset.json`, `frames/*.npy`, `ingest_report.json` |
| profile | `profile.py`, `views.py`, `probes.py` | `profile.json` |
| plan | `planner.py`, `candidates.py` | `plan.json` |
| screen | `screen.py` | `screen.json`, `shortlist.json` |
| evaluate | `evaluate.py`, `attackers.py`, `protocols.py` | `eval/<candidate>/{result.json, preds.npz, anonymizer.pt}` |
| report | `report.py`, `scoring.py` | `report.md`, `report.json`, `frontier.svg` |

`python -m privharness search <root>` runs profile through report on one GPU host. Each stage resumes from its outputs.

## 1. Ingestion

**Description file:** a JSON file declares, for each column:
- utility targets and secrets, each multiclass or multilabel;
- group keys such as `source` (the recording an item was cut from), scene, session and camera;
- optional segments;
- how to split.

**Rows** come from a CSV or JSONL table, or from a folder walk with a regex whose named groups become columns.

**Decoding:**
- Videos go through ffprobe/ffmpeg; images and frame folders through OpenCV.
- Frames are resized so the short side is 112 by default, with the long side capped and both sides even, then optionally resampled in time.
- Media without colour are stored with one channel.

**Splits:**
- `column`: given by the data.
- `stratified`: whole recordings are assigned per class of a target. By default that is the first multiclass secret, so every identity sits on both sides, as identity classification needs.
- `group`: hold out values of a group.

A recording is never on both sides; validation refuses it.

**Audit** (`ingest_report.json`):
- decode failures;
- label balance per split;
- secret classes missing from one side;
- warnings for small identity sets;
- **near-duplicate footage across the split** (Barz & Denzler 2019).

For near duplicates, an 8×8 average hash of three frames is only a gate. The confirmation is the correlation of per-frame motion-energy traces (≥ 0.99 within ±4 frames of lag), because static-camera recordings hash alike. On KTH, planted copies score 1.00 (5 of 5 found) and other recordings at most 0.98.

## 2. Profiling: where does each target live?

Up to 2,500 training and 1,500 test items are sampled, with recordings kept whole. Each item gets one view per cue, and each view keeps one kind of information:

| view | keeps | features |
|---|---|---|
| raw | everything | DINOv2 ViT-S/14, mean and std over 8 frames |
| video | appearance and motion | Kinetics-400 R(2+1)D-18 penultimate layer, 16-frame clip |
| colour | lighting, session, dominant colours | per-frame mean/std and a 4×4 grid |
| lowres | coarse layout | ×16 down-up, DINOv2 |
| scene | background, room, objects not on people | people blacked out (Faster R-CNN), DINOv2 (human-masked probe, Choi et al. 2019) |
| person | appearance of the people | union person crop, DINOv2 |
| body | body proportions, posture, motion | hip-centred, torso-scaled keypoints and bone ratios (Keypoint R-CNN) |
| posture | posture and motion without proportions | limb unit directions |
| motion | where and how much things move | pooled |frame difference| |
| label (secrets only) | what the task label implies | one-hot task label |

**Probes:** each target gets a linear probe per view (weight decay chosen on a validation split by recording) under the dataset split, with grouped bootstrap intervals. The profile reports:
- **predictability**, (view − chance) / (raw − chance);
- **implication** of each secret by the task label;
- **transfer** of multiclass secrets to held-out task classes, which separates "how this person does this task" from identity that carries across activities;
- dataset statistics: person presence, count and area, skeleton rate, camera motion and class balance.

Probes rank cues; they never certify privacy.

## 3. Planning

**R1 feasibility:**
- Temporal families need video.
- Stick figures need skeletons on at least 50% of frames.
- Person boxes need people on at least 50% of frames.

**R2 cue match:**
- Secret in scene or colour, task in body → stick figures and person removal.
- Secret in body proportions (body ≫ posture) → canonical stick figures.
- Secret in person appearance, task in motion → motion maps, learned pools and coarse canonical figures.
- Secret surviving in posture → coarsening on top of canonical figures.
- Task needing the scene → warn, and add blur, downsampling and learned BDQ.

**R3 implied labels:** if the task label alone gives at least 75% of the raw probe's score above chance on a multilabel secret, that secret uses the conditional metric.

**R4 small identity sets:** fewer than 10 classes triggers a warning.

**R5 attackers:**
- Video: frame ResNet-50, R3D-18 (motion and gait), and a skeleton/gait attacker for identities.
- Images: frame ResNet-50 and a DINOv2 probe.
- The obfuscator trains against ResNet-18 and R3D. The ResNet-50 frame attacker and DINOv2 are families it never trained against (MPPAR's held-out-attacker principle).

**R6 protocols:**
- `within` (headline);
- `cross_utility` and `hardest` for multiclass secrets when the task has 3 or more classes;
- `cross_<group>` for each extra group key.

**Candidate catalogue** (`candidates.py`):
- **Fixed:** raw, black, gray, blur (4/8/16), downsample (×4 to ×64), person box (black or blur), fixed motion maps (×8 to ×64), stick figures, and canonical stick figures with downsampling (×1 to ×32).
- **Learned (adversarial):** BDQ (α 1 and 3) and BDQ → pool (×16/32/64).

**Multiple secrets:** learned candidates face a composite adversary with one head per secret, and the summed entropy is rescaled per segment (`ppv.train.segments`), so every secret is defended at once.

## 4. Screening

Each feasible fixed candidate obfuscates one clip per item of a 2,300-item sample. Two frozen-feature probes attack its output: DINOv2 on frames and R(2+1)D on the clip. The same probes measure the task.

**Shortlist:** the proxy Pareto front, then the best proxy trade-off scores, then the planner's priorities, plus raw, black and the planned learned candidates.

The report gives the rank correlation between screening and full evaluation, so screening's reliability is measured rather than assumed.

## 5. Full evaluation

Per candidate:
1. Learned candidates are trained adversarially, with restarts and collapse guards.
2. The whole dataset is obfuscated to 8-bit output, as it would be released.
3. Fresh utility models are trained on the obfuscated training items (SPAct's protocol).
4. For each secret and protocol, three kinds of attacker run:
   - **retrained:** fresh attackers on obfuscated training items (headline);
   - **frozen:** the same architectures trained on raw footage, cached per protocol (the protocol most published numbers use);
   - **skeleton:** a keypoint-sequence TCN for multiclass secrets.

Per-item probabilities are saved for every model.

## 6. Scores (`scoring.py`)

All scores run 0–100; higher is better for privacy, utility and the trade-offs.

| quantity | definition |
|---|---|
| metric | top-1 (multiclass); mean AP over classes (multilabel, cMAP); conditional cMAP = AP within each task class where the attribute varies, averaged (implied multilabel secrets) |
| chance | majority training class, measured on test (top-1); expected AP of random scores on the test labels, Monte Carlo (AP metrics) |
| baseline | max(chance, task-label attacker), where the task-label attacker is P(secret \| task label) with add-one smoothing |
| worst-case score A | the highest score over every attacker (retrained, frozen, skeleton) under the protocol |
| leakage L | (A − baseline) / (A_raw − baseline); undefined, reported as *not measurable*, when raw is within 2 points of the baseline |
| privacy P | 1 − clip(L, 0, 1); dataset privacy is the minimum over secrets under `within`; the worst over all protocols is also reported |
| utility U | (u − chance_u) / (u_raw − chance_u) for the primary task, best utility model |
| PUS | 2PU / (P + U), the headline trade-off (high only when both are high) |
| f_0.5 | 0.5 U + 0.5 P (Ilic et al. 2024), for comparability |
| frontier | hypervolume of the (U, P) Pareto front in [0, 1]² |
| uncertainty | 95% bootstrap over test recordings (1,000 draws); the recommendation is re-run with attacker seeds 1 and 2, 24-epoch attackers and DINOv2 (probe and fine-tune) |

**Checks:**
- **measurable:** raw is at least 2 points above the baseline.
- **black control:** all-black frames must score at the baseline.
- **frozen gap:** how much frozen attackers understate retrained ones (reported, not used).
- **screening reliability.**

**Recommendation:** the highest PUS. The privacy-first pick (highest P with U ≥ 0.5) and utility-first pick (highest U with P ≥ 0.5) are also given.

## Known limits of this first iteration

- **Learned obfuscators are video-only** (BDQ uses frame differences). Image datasets get fixed families only.
- **No reconstruction attacker.** An attacker that inverts the obfuscation (E2VID-style for events, super-resolution for low resolution) is not yet in the suite. Retrained classifiers cover recognition, not reconstruction.
- **Profiles are linear probes** on frozen features. They rank cues and guide the search; they do not bound what a trained attacker recovers.
- **No cross-dataset protocol.** Training attackers on one dataset and testing on another (Wu et al.; GenPriv) needs two datasets with a shared secret.
- **Licences** are recorded in `dataset.json` meta. The harness never redistributes data, and derived frames stay on the machines that hold the source.

## References

- Wu et al., ECCV 2018 / TPAMI 2020, arXiv 1807.08379, 1906.05675: adversarial framework, restarted and ensembled budget models, unseen attackers.
- Dave et al., SPAct, CVPR 2022, arXiv 2203.15205: retrain-from-scratch evaluation protocol; novel-action and novel-privacy protocols.
- Kumawat & Nagahara, BDQ, ECCV 2022, arXiv 2208.02459.
- Li et al., STPrivacy, ICCV 2023, arXiv 2301.03046: video-level attackers; VP-UCF101 / VP-HMDB51 attributes.
- Ilic et al., CVPR 2024, arXiv 2403.12710: KTH re-split, linear trade-off f_λ; non-adapted attackers.
- Peng et al., MPPAR, NeurIPS 2023: held-out attacker and attribute splits.
- Elazar & Goldberg, EMNLP 2018, arXiv 1808.06640; Pittaluga, Koppal & Chakrabarti, WACV 2019, arXiv 1802.05214; McPherson, Shokri & Shmatikov, arXiv 1609.00408: training adversaries understate leakage, and retrained CNNs defeat blur and pixelation.
- Wang et al., ICCV 2019, arXiv 1811.08489 (dataset vs model leakage); Jayaraman & Evans, CCS 2022, arXiv 2209.01292 (imputation baseline).
- Xu et al., ICLR 2020, arXiv 2002.10689 (V-information): leakage is defined relative to an attacker family, as here.
- Choi et al., NeurIPS 2019, arXiv 1912.05534 (scene bias, human-masked probes); Li et al., RESOUND, ECCV 2018; Mimetics, arXiv 1912.07249.
- Barz & Denzler 2019, arXiv 1902.00423 (near-duplicates across splits).
- Moon et al., AAAI 2023, arXiv 2111.15129 (skeletons re-identify about 80%); GaitGraph, arXiv 2101.11228.
- Bouthillier et al., MLSys 2021, arXiv 2103.03098 (variance sources; seeds and intervals).
- Zitzler & Thiele, IEEE TEC 1999 (hypervolume).
