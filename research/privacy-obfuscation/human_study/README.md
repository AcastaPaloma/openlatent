# Human re-identification study (KTH; raw vs four obfuscators)

Can people re-identify someone from obfuscated video, and does that track what the machine
attackers manage? Five conditions, from the repo's KTH runs (strong protocol, ilic split;
identity top-1 of the best retrained attacker, chance 4%):

| condition | run | what it is | machine identity | action |
|---|---|---|---|---|
| raw | `p1_raw` | original video | 99.5 | 97.5 |
| blur | `p1_blur8` | Gaussian blur, sigma 8 | 95.0 | 94.5 |
| unet | `p2_unet_a1_ens` | learned residual 3D U-Net (frame + video adversaries); outputs a silhouette | 89.1 | 91.8 |
| bdq | `p2_bdq_a3_lr3e-3` | learned blur-difference-quantise | 46.2 | 89.1 |
| pose | `p12_kth_pose_canon_e16` | canonical stick figure (template limb lengths), keyframe every 16 | 29.1 | 91.5 |

Not used, because a person would see nothing:
- `p2_vae_a1_ens` (82.6): releases a flat grey frame with border artifacts (its gallery GIF too);
  the attacker reads identity from a residual signal no viewer can see. Worth reporting on its
  own: a large human/machine gap, but a human trial would only measure chance.
- `bdq_down*` chains: the output ends at 5x3 to 20x15 pixels.
- BDQ on d2 or upper-body clips: a blank grey screen.

## Task

Each trial shows a "mystery person" clip and a lineup of four people (A-D). The participant
picks who it is (forced choice, chance 25%) and rates confidence 1-5.

- Query: one condition, 32 frames at stride 2 (the attackers' clip stride), outdoors (d1/d3),
  walking, jogging or running.
- Lineup: always raw, walking, d4 (indoors), so the background never matches the query and the
  match must come from the person. This is the attacker's view: a raw gallery, an obfuscated probe.

## Design (20 trials)

- 10 target people, each seen in two conditions. The 10 targets cover the 10 condition pairs
  exactly once, so every pair of conditions has a within-person comparison and each condition
  has 4 trials.
- Trials 1-10 show every target once, trials 11-20 the other condition (5-13 trials later).
  Which condition of a pair comes first follows a regular tournament, so every condition has
  2 trials in each half: practice and memory effects hit all conditions equally, and the first
  half is a carryover-free comparison on its own.
- No condition twice in a row. A target's two trials differ in action, scenario, lineup
  subsequence, distractors and answer letter.
- Every person appears in 3-4 lineups (targets included), so "I've seen this one before" carries
  no information. Answers: 5 per letter.
- Query windows: windows are ranked by BDQ activity (the same motion measure in every condition)
  and each trial takes the top-ranked one whose output shows something in at least half its
  frames. This is the attacker's best case, so a "people can't" result is conservative. The pose
  detector misses some people on every keyframe of a window; `window_rank` in the key records
  how far down the ranking a trial went.
- The intro shows all five conditions on person 25, who appears in no trial.
- `design(seed=0)` in `make_stimuli.py` builds all of this deterministically.

## Files

- `make_stimuli.py`: design, KTH clip download (HTTP range requests on the official zips),
  anonymizers built from the privacy-research branch's `ppv` with checkpoints from the HF repo,
  GIF rendering. Writes `stimuli/*.gif` and `answer_key.csv`.
- `answer_key.csv`: per trial: target, condition, run, query clip and window, BDQ activity, lineup,
  correct letter.

```bash
PYTHONPATH=<privacy-research>/research/privacy-obfuscation \
uv run --no-project --with torch --with torchvision --with huggingface_hub \
  --with safetensors --with pillow --with numpy python human_study/make_stimuli.py --cache <dir>
```

## Notes on what people see

- BDQ (learned thresholds 6 below -1, 2 at +0.47, 7 at +0.89 on 4x the blurred frame difference):
  the static background is one constant grey; a pixel lights up only when it brightens by more
  than ~30 grey levels between frames, so a moving person leaves a sparse trailing-edge silhouette.
- Fixed BDQ (`p1_bdqfixed_s2_l8`, 87.8% machine identity) was dropped: with an even number of
  levels its quantisation boundary sits at zero difference, so sensor noise turns the frame into
  flickering blotches that look like pure noise to a person.

## Analysis

Score each trial against `answer_key.csv`. Per condition: accuracy and mean confidence, next to the
machine identity column above. Per target: better / equal / worse between its two conditions.
Use a mixed logistic model (condition fixed; participant and target random) for the condition
effects, and report the first-half-only comparison as a check for carryover.
