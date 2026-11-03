# Human re-identification study (KTH, raw vs learned BDQ)

Can people re-identify someone from our obfuscated video as well as from raw video?
Obfuscator: the learned BDQ of run `p2_bdq_a3_lr3e-3` (privacy-research branch; alpha 3,
frame adversary, learned sigma 2.69). Machine reference, strong protocol: identity 46.2%
(R3D video attacker) / 18.7% (frame attacker), action 89.1%; raw 99.5%, chance 4%.

## Task

Each trial shows a "mystery person" clip and a lineup of four people (A-D). The
participant picks who it is (forced choice, chance 25%) and rates confidence 1-5.

- Query: raw or BDQ, 32 frames at stride 2 (the attackers' clip stride), outdoors (d1-d3).
- Lineup: always raw, walking, d4 (indoors), so the background never matches the query and
  the match must come from the person. This is the attacker's view: a raw gallery, an obfuscated probe.

## Design choices (and why)

- 5 cases (target people 3, 8, 12, 17, 22), each seen once raw and once obfuscated: 10 trials.
- The two trials of a case are never adjacent (3-6 trials apart) and differ in action or
  scenario, query window, lineup subsequence, distractors and answer letter, so the
  second trial is not a memory test of the first.
- Trials 1-5 show each case once (3 obfuscated, 2 raw); trials 6-10 show the other condition.
  The first half is a carryover-free comparison; the full set gives the within-case comparison.
  Obfuscated-first in 3 cases and raw-first in 2 keeps practice effects from favouring one condition.
- Targets also appear as distractors in other trials, so "I've seen this person before" is not
  a usable cue. Answers are spread A3 B3 C2 D2.
- Queries are walking, jogging and running only, in the same mix for both conditions
  (raw J W J R R, obfuscated R R W J J). The learned BDQ keeps only strong brightening motion:
  boxing, handwaving and handclapping come out at 0.0-0.4% active pixels and the far d2 clips
  at 0.01-0.25%, i.e. a blank grey screen, which would test nothing.
- Query windows are chosen by one rule in both conditions: the 32-frame window with the most
  BDQ activity in that video. This is the attacker's best case, so a "humans can't" result is
  conservative. `answer_key.csv` records the active-pixel fraction of every query.
- The intro example (person 25 jogging) appears in no trial.

## Files

- `make_stimuli.py`: reads only the needed AVIs from the official KTH zips (HTTP range
  requests), applies the learned BDQ at full frame rate (as `train.materialize` does),
  writes `stimuli/*.gif` and `answer_key.csv`.
- `answer_key.csv`: per trial: case, condition, target, query clip, BDQ activity, lineup, correct letter.

```bash
# anonymizer.pt: runs/p2_bdq_a3_lr3e-3/anonymizer.pt in the (private) HF checkpoint repo
uv run --no-project --with torch --with pillow --with numpy python human_study/make_stimuli.py \
  --cache <dir> --checkpoint <anonymizer.pt>
```

## What the obfuscator outputs

Learned thresholds: 6 below -1, 2 at +0.47, 7 at +0.89 (on 4x the blurred frame difference).
So the static background is one constant grey (6/15), and a pixel lights up only when it
brightens by more than ~30 grey levels between frames: the trailing edge of a dark figure
moving over a bright background. A frame shows a few slivers; the moving sequence shows a
sparse silhouette.

The earlier draft used fixed BDQ (`p1_bdqfixed_s2_l8`, 87.8% machine identity). With an
even number of levels its quantisation boundary sits at zero difference, so sensor noise
turns the whole frame into flickering blotches; that output looked like pure noise to people.

## Analysis

Score each trial against `answer_key.csv`. Report accuracy and mean confidence per condition
(raw vs obfuscated), per case (better / equal / worse after obfuscation), and the first-half-only
comparison. With 5 cases, use a per-participant paired test (e.g. a mixed logistic model
with participant and case effects) rather than per-case significance.
