# Human re-identification study (KTH; four obfuscators, every query masked)

Can people re-identify someone from obfuscated video, and does that track what the machine
attackers manage? Every mystery clip goes through one of four filters from the repo's KTH runs
(strong protocol, ilic split; identity top-1 of the best retrained attacker, chance 4%, raw 99.5%):

| filter | run | what people see | machine identity | action |
|---|---|---|---|---|
| blur | `p1_blur8` | Gaussian blur, sigma 8 | 95.0 | 94.5 |
| unet | `p2_unet_a1_ens` | learned residual 3D U-Net: a black silhouette on white | 89.1 | 91.8 |
| bdq | `p2_bdq_a3_ens` | learned blur-difference-quantise: light motion patches on dark | 68.6 | 96.7 |
| pose | `p12_kth_pose_canon_e16` | canonical stick figure (template limb lengths), keyframe every 16 | 29.1 | 91.5 |

Picked from `preview/filters_d1.gif` and `preview/filters_d2.gif` (`preview/make_preview.py`): every
candidate on all six actions. These four keep every action visible. Not used:
- `p2_bdq_a3_lr3e-3` (46.2): blanks boxing, handwaving and handclapping.
- `p11_kth_mask_k25_bdq_a1` (82.9): shows the DINOv2 mask's patch grid, not the person.
- `p11_kth_mask_k25_a1` (96.7): pastes the real person into the mask.
- `p2_vae_a1_ens` (82.6): releases a flat grey frame; the attacker reads identity from a residual
  no viewer can see. Worth reporting on its own as a human/machine gap.
- `p1_bdqfixed_s2_l8` (87.8): even quantisation levels put a boundary at zero difference, so
  sensor noise makes the frame flicker like pure noise.
- `bdq_down*` chains: output ends at 5x3 to 20x15 pixels.

## Task

Each trial shows a "mystery person" clip and a lineup of four people (A-D). The participant
picks who it is (forced choice, chance 25%) and rates confidence 1-5.

- Query: one filter, 32 frames at stride 2 (the attackers' clip stride), any of the six actions,
  scenario d1 or d2 (outdoors; same clothes as d4, unlike d3).
- Lineup: four people walking in d4 (indoors), each shown raw and, directly beneath, the same
  window through the trial's filter. The background never matches the query but the person's
  clothes do, so only the filter stands between the viewer and the identity.

### Why the lineup shows each candidate raw and filtered

The human plays the identity adversary. On KTH the adversaries predict one thing: which of the 25
people a clip shows (`priv_task="person"` in `ppv/run.py`); the lineup asks the same question,
4-way. The machine numbers above are the strong protocol: the attacker is trained on labelled
*obfuscated* clips. A raw-only lineup would instead match the weak protocol (attacker trained on
raw, tested on obfuscated), where every filter is near chance for machines:

| filter | strong (trained on obfuscated) | weak (trained on raw) |
|---|---|---|
| blur | 95.0 | 8.7 |
| unet | 89.1 | 4.3 |
| bdq | 68.6 | 3.8 |
| pose | 29.1 | 4.0 |

Showing each candidate through the filter gives people labelled obfuscated examples, so human
accuracy is comparable to the strong column. The attribute adversaries (face, skin colour,
gender, nudity, relationship) belong to the VP-UCF101 / HMDB51 runs; KTH has no such labels, so
the study asks identity only.

## Design (20 trials)

- 10 target people, each seen under two different filters. The 10 pairs are all 6 pairs of the 4
  filters plus a 4-cycle, so each filter has 5 trials and every pair of filters is compared
  within at least one person.
- Trials 1-10 show every target once, trials 11-20 the other filter (5+ trials later); each filter
  has 2-3 trials per half, and the first half is a carryover-free comparison on its own.
- No filter twice in a row. A target's two trials differ in action, scenario, lineup subsequence,
  distractors and answer letter. If the filter shows nothing in a candidate's lineup window (a
  pose-detector miss), that candidate's next walking subsequence is used (`lineup_seqs` in the key). Each action appears 3-4 times; each filter gets >= 4 actions.
- Every person appears in 3-4 lineups (targets included), so "I've seen this one before" carries
  no information. Answers: 5 per letter.
- Query windows are ranked by BDQ activity (one motion measure for every filter) and each trial
  takes the top-ranked window whose output shows something in at least half its frames: the
  attacker's best case, so a "people can't" result is conservative.
- The intro shows the original and all four filters on person 25, who appears in no trial.
- `design(seed=0)` in `make_stimuli.py` builds all of this deterministically.

There is no raw query trial: the "before" reference is the machine identity column above (and
raw KTH is near-trivial for people with matching clothes).

## Files

- `make_stimuli.py`: design, KTH clip download (HTTP range requests on the official zips),
  anonymizers built from the privacy-research branch's `ppv` with checkpoints from the HF repo,
  GIF rendering. Writes `stimuli/*.gif` and `answer_key.csv`.
- `answer_key.csv`: per trial: target, filter, run, query clip and window, BDQ activity, lineup,
  lineup subsequence per candidate, correct letter.
- `verify.py`: asserts every lineup holds its target exactly once at the keyed letter, and draws
  `verify_1.png` / `verify_2.png` (raw query next to the lineup, target outlined) for checking by eye.

```bash
PYTHONPATH=<privacy-research>/research/privacy-obfuscation \
uv run --no-project --with torch --with torchvision --with huggingface_hub \
  --with safetensors --with pillow --with numpy python human_study/make_stimuli.py --cache <dir>
```

## Analysis

Score each trial against `answer_key.csv`. Per filter: accuracy and mean confidence, next to the
machine identity column above. Per target: better / equal / worse between its two filters.
Use a mixed logistic model (filter fixed; participant and target random) for the filter effects,
and report the first-half-only comparison as a check for carryover.
