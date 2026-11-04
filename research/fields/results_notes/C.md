# Focus C: identity-free emotion on CREMA-D (audio track) - results log

Builder-C. Code: `research/fields/src/c_cremad/`. Job specs: `research/fields/jobs/specs/c_*.yaml`. Raw per-actor counts: HF
`KuanKuanKuan/openlatent-fields` under `runs/c_*` (uploaded by the worker). Nothing below is a result unless it carries an interval
and a seed count; single-seed numbers are marked "not yet replicated".

## 1. Benchmark as built (frozen before any learned number was looked at)

Data: CREMA-D (ODbL), downloaded from the official GitHub LFS media endpoint, 7,442 clips / 91 actors / 6 emotions / 12 sentences
verified (`check_data.py`, `results/c_data_check.json`). 16 kHz mono wav, mean 2.53 s (1.26-5.0). Emotion prior max 0.171 (NEU is
the minority class, 1,087 clips), so emotion chance for UAR is 1/6. Three byte-identical wav pairs carry two different emotion
labels (1006_TIE HAP/NEU, 1013_WSI DIS/SAD, 1017_IWW ANG/FEA): label noise, same actor and sentence so never split across a boundary.

Frozen features (`extract_feats.py`): WavLM-Base+ (primary), HuBERT-Base, wav2vec2-Base; frame level for layers 3/6/9/12 and per-clip
mean/std for every hidden state. Derived features stay on the VM disk, never uploaded.

Splits (`splits.json`, deterministic):
- Actors: 4 folds (23/23/23/22 actors, stratified by sex x race). For fold f the sealed TEST actors are fold f, 12 further actors
  are VALIDATION, the remaining 56-57 are encoder TRAIN actors. Each actor is a test actor in exactly one fold.
- Identity attacker: trained on S_TR x E_TR clips of ALL 91 actors (its own enrolment data); S_TR = 8 sentences (2 of them used only
  to pick attacker hyper-parameters), E_TR = ANG, HAP, NEU, SAD. Scored on T1 = S_TE x E_TR (4 unseen sentences, PRIMARY),
  T2 = S_TR x E_TE (unseen emotions DIS, FEA), T3 = S_TE x E_TE (both unseen). No overlap (verified). Chance 1/91 = 1.10%.
- Identity is reported separately for actors the encoder never saw ("unseen" = fold test actors) and encoder-train actors ("seen").
- Emotion: probes retrained on the released representation using encoder-train actors, hyper-parameters on validation actors,
  refit on train+val actors, scored on the sealed test actors only (UAR, chance 1/6).
- Secondary secrets: sex, race (3 classes) and age attackers, actor-disjoint (train+val actors, scored on test actors).

Attacker suite (all retrained from scratch on the released representation; hyper-parameters and epochs picked on held-out
sentences): linear, MLP, cosine kNN, LightGBM (held-out family), small transformer and x-vector TDNN (for sequence releases).
Worst case = max over the suite per resample.

Leakage: L = (ID - base) / (ID_raw - base), with ID the worst-case T1 top-1 on unseen actors, base the emotion-label-only identity
guess (argmax_a P(a | emotion) fitted on the attacker training set; = chance up to rounding), ID_raw the same statistic on the raw
frozen features. Utility: emotion UAR of the best retrained probe; retention = (U - 1/6)/(U_raw - 1/6).
Intervals: 95% percentile bootstrap over the 91 actors (B=1000, identical resamples for every method so differences are paired),
mean over seeds inside each resample; per-seed spread reported separately. All thresholds below are multiples of chance (1/91).

## 2. Controls and gates (status, 2026-11-04 ~09:45Z)

All numbers: WavLM-Base+ audio track, 4 actor-disjoint folds pooled over the 91 test actors, 95% bootstrap interval over actors,
identity = worst attacker (LDA unless noted) on T1 (unseen sentences). Hand-crafted descriptors are fold-independent, so their identity
attack is run once over all 91 actors. "x chance" = top-1 / (1/91). Attacker seeds: 1 for hand-crafted sets (not yet replicated), 2 of 3 for raw.

**Gate 1, measurability (H-C-001a): PASS.** Raw frozen features (uniform mix of layers 3/6/9/12, mean+std): identity T1 **0.804
[0.775,0.835] = 73x chance** (LDA; linear 0.717, MLP 0.685, TDNN 0.465, kNN 0.269, LightGBM 0.325), emotion-label-only baseline 0.011 (1.0x).
T3 (unseen sentences and emotions) 0.752. Emotion UAR 0.756 [0.736,0.776] (actor-disjoint). The strongest attacker is the cheapest one
(shrinkage LDA), which is also the one no adversary in the literature trains against. 2 of 3 seeds; third pending.

**Gate 2, silence / channel (H-C-006): FAIL for the full-clip benchmark. The 300 ms of non-speech around the speech identifies actors.**

| descriptor (what the attacker sees) | identity T1 [95% CI] | x chance | share of raw (L vs raw) | emotion UAR |
|---|---|---:|---:|---:|
| clip length only (1 number) | 0.030 [0.019,0.057] (MLP) | 2.7x | 0.02 | 0.225 |
| length + leading/trailing silence duration (3 numbers) | 0.060 [0.036,0.087] | 5.5x | 0.06 | 0.299 |
| 9 channel numbers (noise floor, DC, peak, clipping, edge RMS, durations) | 0.147 [0.121,0.185] | 13.4x | 0.17 | 0.423 (contains peak level, so speech-contaminated) |
| 1/3-octave spectrum + hum of the non-speech frames (24 numbers) | 0.262 [0.225,0.303] | 23.8x | 0.32 | 0.229 |
| log-mel mean+std, VAD head only (150 ms before speech) | 0.376 [0.337,0.420] | 34.2x | 0.46 | 0.210 |
| log-mel mean+std, VAD tail only | 0.357 [0.324,0.392] | 32.5x | 0.43 | 0.268 |
| **log-mel mean+std, VAD head+tail (128 numbers, no speech)** | **0.618 [0.579,0.656]** | **56.2x** | **0.76** | 0.261 |
| first 150 ms of the FILE (clips with >=300 ms leading silence, 77% of clips) | 0.456 [0.408,0.508] | 41.5x | 0.56 | 0.180 (= chance) |
| last 150 ms of the FILE (>=300 ms trailing silence, 87% of clips) | 0.346 [0.300,0.391] | 31.5x | 0.42 | 0.184 (= chance) |
| positive control: log-mel of the speech frames | 0.839 [0.811,0.868] | 76.4x | 1.04 | 0.511 |
| shuffled actor labels, head+tail log-mel | 0.013 [0.010,0.021] | 1.2x | 0.00 | - |

The T2 and T3 columns are within a few points of T1 for every non-speech set (not shown), as expected for a session/microphone cue that
is independent of sentence and emotion; the first and last 150 ms of the file carry no emotion (UAR at chance), so these are room/mic
signatures and not leaked speech. Residue caveat: the head/tail segments are defined by an energy VAD (noise floor + 6 dB, 20 ms guard); the
file-edge variants avoid the VAD boundary and give the same picture. Consequence (coordinator rule): any identity number on full clips
is partly a channel number. Every learned z is reported against the 0.62 (56x, L = 0.76) channel floor and, where run, on speech-trimmed input (`--trim`).

**Gate 3, prosody floor (H-C-005): learned z must beat these.** Actor-disjoint emotion UAR (chance 0.167) and identity (T1, worst attacker):

| hand-built descriptor | emotion UAR [95% CI] | identity T1 [95% CI] | x chance | L vs raw |
|---|---|---|---:|---:|
| 3 numbers: duration, mean RMS, median F0 | 0.407 [0.391,0.422] | 0.088 [0.065,0.117] | 8.0x | 0.10 |
| 6 numbers: + F0 std, energy std, voiced fraction | 0.454 [0.439,0.469] | 0.127 [0.100,0.159] | 11.6x | 0.15 |
| 6 numbers, F0 in semitones (no absolute pitch) | 0.435 [0.419,0.451] | 0.077 [0.056,0.102] | 7.0x | 0.08 |
| 26-number rich set (percentiles, formants, jitter, shimmer, HNR, tilt, rate) | 0.516 [0.498,0.533] | 0.256 [0.222,0.292] | 23.3x | 0.31 |
| 6 numbers, ORACLE per-actor z-scoring (uses the test actor's own statistics) | 0.535 [0.520,0.551] | 0.022 [0.017,0.032] | 2.0x | 0.01 |

Predictions in H-C-005 vs observed: emotion from 3 / 6 numbers 0.38-0.50 / 0.45-0.58 (observed 0.41 / 0.45: inside, low end); speaker
accuracy 0.15-0.30 / 0.20-0.40 for ~23 actors (observed 8x / 12x chance in the 91-way setting, i.e. below the predicted range); semitone
normalisation lowered identity by 0.05 at an emotion cost of 0.02 (as predicted). New: per-actor standardisation of six prosody numbers
raises emotion to 0.535 while identity drops to 2x chance. It needs the actor's own statistics (oracle), so it is a different threat
model, but it is the floor to beat for any per-actor-calibrated release (H-X-008 on C).

## 3. Log

(newest last)
