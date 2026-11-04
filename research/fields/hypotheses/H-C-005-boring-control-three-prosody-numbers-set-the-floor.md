# H-C-005 (BORING CONTROL): Three or six hand-made prosody numbers give emotion >= 40% UAR and speaker >= 4x chance, so a learned z must beat that pair to be called anything

- Focus: C | Kind: boring control that can kill an exciting result (analogue of the KTH brightness-only control, 22.4% identity from one number per frame) | Cost: 0 GPU-h (CPU, ~1 h) | Prior: 0.60
- Depends on: Builder-C splits and the raw WAVs only.

## Statement
Features per clip: (3-number version) duration, mean RMS energy, median F0; (6-number version) add F0 standard deviation, energy standard deviation, voiced fraction. With attackers trained as in the benchmark: speaker accuracy on T1 is >= 4x chance (>= 0.17 for ~23 enrolled actors) and emotion UAR (actor-disjoint) is >= 0.40 with the 3-number set and >= 0.45 with the 6-number set. A learned adapter whose emotion UAR is not at least 15 points above the 6-number set at lower-or-equal speaker leakage has not learned anything beyond a prosody summary.

## Mechanism
Arousal is carried by F0, energy and rate, so a handful of numbers separates (anger, fear, happy) from (sad, neutral); median F0 separates sex and partly individuals; duration/tempo identifies speakers even after timbre is removed (2507.15214, abstract opened by the scout). This is the speech counterpart of our KTH finding that one brightness number per frame identifies 22% of people and that hand-built canonical stick figures beat learned masks. If a 6-number hand-built z sits on or past the learned frontier, the programme has a hand-built best (a result, but not a learned-method win).

## Experiment (exact)
- Data: CREMA-D, 4 folds. Features computed with librosa pYIN (or torchaudio pitch) at 16 kHz, 25 ms / 10 ms frames; unvoiced frames excluded from F0.
- Attackers: kNN-5, LR, MLP, LightGBM (C-SUITE minus sequence nets). Speaker: enrolled test-fold actors on S_TR x E_TR, scored on T1/T2/T3. Emotion: trained on encoder-train actors, scored on test-fold actors.
- Variants: raw numbers; per-clip normalised (subtract utterance median F0: "speaker-normalised F0", the canonical-pose analogue); per-actor-oracle normalised (X-008 style).
- Seeds: 3 attacker seeds x 4 folds; bootstrap over actors. Chance from class priors.
- Metric: absolute accuracies and the L, U normalisation against the WavLM raw reference.

## Prediction and falsifier
- Prediction: 3-number: speaker T1 0.15-0.30, emotion UAR 0.38-0.50; 6-number: speaker 0.20-0.40, emotion 0.45-0.58; median-F0-normalised variant lowers speaker by >= 0.05 at emotion -0.03.
- Falsified (important direction) if the 6-number set gets speaker <= 2x chance with emotion UAR >= 0.80 x the best learned adapter's: then a hand-built canonical-prosody z is the winner at that rate and all learned wins must be reported against it. Falsified (other direction) if emotion from 6 numbers < 0.30: prosody summary is not informative on acted data and the control is weak.

## Cost
0 GPU-h; ~1 CPU-h of F0 tracking (pYIN is slow, ~0.2 s per clip).

## Dependencies
None beyond the WAVs and the splits; run first as a calibration.

## Prior and why
0.60. Acted speech with exaggerated prosody should be easy on arousal, and 6-way UAR from 6 numbers around 0.45-0.55 is plausible; speaker accuracy with 23 enrolled and 82 clips each is the uncertain part.

## Sources
[S] 2507.15214 (literature/other_modalities.md #5); our results/SUMMARY.md (brightness-only control, canonical skeleton); [S] 2409.03655.
