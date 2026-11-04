# H-C-006 (BORING CONTROL): The non-speech part of a CREMA-D clip (leading/trailing silence, room and microphone) identifies actors >= 5x chance

- Focus: C | Kind: boring control / benchmark-validity check | Cost: 0 GPU-h | Prior: 0.40
- Depends on: Builder-C WAVs and splits only.

## Statement
Using only the first 150 ms and last 150 ms of each clip (verified non-speech by an energy VAD; clips with no such segment excluded), a log-mel (64-band) mean+std descriptor (128 numbers) lets an MLP/kNN identify the enrolled test-fold actors on T1 at >= 5x chance (>= 0.21 for ~23 actors; >= 0.05 for the full 91-way). If true, CREMA-D "identity" contains a recording-session / channel component that no speech-content mask can remove, and all leakage numbers must be reported both on raw clips and on silence-trimmed clips.

## Mechanism
In the KTH replication, the scene alone identified "people" because each person was filmed in their own sessions (journal 2026-10-27 08:35, 10:20; SUMMARY finding 5: UTD-MHAD sessions identify 73% by mean colour). CREMA-D actors were recorded in separate sessions, so the room noise floor and microphone frequency response can act as an actor fingerprint. This is the audio version of "the scene is the identity".

## Experiment (exact)
- Data: CREMA-D WAVs (16 kHz mono).
- Segmenting: energy VAD (threshold at noise-floor + 6 dB), take up to 150 ms before first speech frame and after last; require >= 80 ms available, otherwise drop the clip (report the kept fraction per actor).
- Features: log-mel mean+std; long-term average spectrum in 1/3-octave bands; optionally 50/60 Hz hum amplitude and its harmonics.
- Attackers: kNN-5 cosine, LR, MLP, LightGBM. Splits as in the benchmark: train on S_TR x E_TR clips of the enrolled actors, test T1/T2/T3; chance from the number of kept actors.
- Controls: shuffled actor labels (must be at chance); same pipeline on speech-only segments (positive control, should be high).
- Seeds: 3 attacker seeds x 4 folds; bootstrap over actors.
- Metric: absolute accuracy and L relative to the raw-WavLM attacker.

## Prediction and falsifier
- Prediction: non-speech-only speaker accuracy on T1 in 0.20-0.60 for ~23 actors (5-14x chance); T2/T3 similar (it does not depend on sentence or emotion, which is the signature of a channel cue); shuffled-label control at chance.
- Falsified if accuracy <= 2x chance on all folds: the channel confound is negligible and the full-clip benchmark needs no trimming variant. If accuracy >= 0.5 on average, escalate: the identity benchmark is partly a channel benchmark and the primary numbers must be on trimmed audio.

## Cost
0 GPU-h; ~30 CPU-min.

## Dependencies
None (run in the first hour; informs the benchmark definition itself).

## Prior and why
0.40. CREMA-D was recorded in a controlled studio setting by one team, so session effects may be small; but I have seen no CREMA-D statement either way. The test is cheap and decisive for validity.

## Sources
Our journal and SUMMARY (scene and session identity on KTH/UTD-MHAD); [S] 2412.00277 (CREMA-D protocol) via literature/contrarian_stress_test.md 3.1.
