# Human re-identification study (KTH, raw vs fixed BDQ)

Can people re-identify someone from our obfuscated video as well as from raw video?
Machine reference: fixed BDQ (`configs/p1/p1_bdqfixed_s2_l8.json`) keeps 87.8% identity
(video attacker, strong protocol; raw 99.5%, chance 4%).

## Task

Each trial shows a "mystery person" clip and a lineup of four people (A-D). The
participant picks who it is (forced choice, chance 25%) and rates confidence 1-5.

- Query: raw or BDQ, 32 frames at stride 2 (the attackers' clip stride), from scenario d1-d3 (outdoors).
- Lineup: always raw, walking, d4 (indoors), so the background never matches the query and
  the match must come from the person. This is the attacker's view: a raw gallery, an obfuscated probe.

## Design choices (and why)

- 5 cases (target people 3, 8, 12, 17, 22), each seen once raw and once obfuscated: 10 trials.
- The two trials of a case are never adjacent (3-6 trials apart) and differ in action,
  scenario, query subsequence, lineup subsequence, distractors and answer letter, so the
  second trial is not a memory test of the first.
- Trials 1-5 show each case once (3 obfuscated, 2 raw); trials 6-10 show the other condition.
  The first half is a carryover-free comparison; the full set gives the within-case comparison.
  Obfuscated-first in 3 cases and raw-first in 2 keeps practice effects from favouring one condition.
- Targets also appear as distractors in other trials, so "I've seen this person before" is not
  a usable cue. Answers are spread A3 B3 C2 D2.
- The intro example uses person 25, who appears in no trial.

## Files

- `make_stimuli.py`: reads only the needed AVIs from the official KTH zips (HTTP range
  requests), applies the repo's own `Fixed("bdq_fixed", sigma=2, levels=8)` at full frame rate,
  writes `stimuli/*.gif` and `answer_key.csv`.
- `answer_key.csv`: per trial: case, condition, target, query clip, lineup, correct letter.

```bash
uv run --no-project --with torch --with pillow --with numpy python human_study/make_stimuli.py --cache <dir>
```

## Note on the obfuscator

With an even number of levels, `bdq_fixed` puts a quantisation boundary at zero difference,
so 99.97% of output pixels sit on two grey levels: the output is close to a 1-bit
sign-of-motion map, and sensor noise makes the static background flicker. A single frame
shows no person; only the coherent motion across frames does.

## Analysis

Score each trial against `answer_key.csv`. Report accuracy and mean confidence per condition
(raw vs obfuscated), per case (better / equal / worse after obfuscation), and the first-half-only
comparison. With 5 cases, use a per-participant paired test (e.g. a mixed logistic model
with participant and case effects) rather than per-case significance.
