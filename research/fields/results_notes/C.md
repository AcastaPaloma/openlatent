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

## 2b. Pre-registration: held-out-emotion protocol (written 2026-11-04 ~11:35Z, before any c_he_* or c_lda4_* number was read)

Why: on the full-emotion protocol the label-supervised LDA projection to 5 dims (UAR 0.743 [0.725,0.763], identity 7.0x chance) is
at least as good as every learned encoder run so far (paired UAR difference -0.015 to -0.022, significant; identity difference
-0.006 to -0.031, significant only for VQ). A learned encoder can only justify itself where a 5-dim linear projection of the labels
throws away information the task still needs: emotions the projection never saw.

Protocol: encoder / projection fitted with the labels ANG, HAP, NEU, SAD only (train-fold actors). The released z is then probed with
six-way probes trained on train+val actors (all six labels visible to the PROBE, never to the encoder) and scored on the sealed test actors.
Primary utility U_HE = mean recall of DIS and FEA on test actors (chance 1/6 = 0.167; a z that kept nothing about the unseen emotions
sits near the recall that the four known-class directions happen to give them, which `c_lda4_*` measures). Secondary: recall of the four
trained emotions; identity T1 with the full attacker suite (unseen test actors), unchanged.

Comparators (same inputs, same attackers, same folds): `c_lda4_k3` (C-1 = 3 LDA dims), `c_lda4_k5` (3 LDA + 2 residual-PCA dims, width
matched to d=4/16 encoders as closely as possible) and `c_lda4_k16` (width matched to the d=16 encoder); raw frozen features as the
upper reference.

Success (all must hold, learned encoder vs the width-matched LDA comparator, paired bootstrap over the 91 actors, seed 0 first):
1. U_HE(learned) - U_HE(LDA) >= +0.05 with the 95% interval of the difference above 0;
2. identity T1 (worst attacker) of the learned z not above the LDA comparator by more than 0.02 absolute (upper end of the 95% interval of
   the difference <= 0.02);
3. recall of the four trained emotions within 0.03 of the comparator.
A winner is then replicated with seeds 1 and 2 (same sign, same thresholds on the 3-seed mean) before any claim. If no learned encoder
(rate d4/d16/d64, adversarial d16, VQ g2k64) meets 1-3, the learned-encoder arm of Focus C is closed: on CREMA-D audio the supervised
5-dim projection (or a rate bottleneck) is the method, the contribution is the benchmark and its controls (channel floor, prosody floor,
retrained-attacker suite), and the next question moves to where a projection cannot be fitted (no labels, new emotion sets, RAVDESS transfer).

## 2c. Pre-registration: task-agnostic adversarial autoencoder (written 2026-11-04 ~11:58Z by the VM clock, before any AE number exists)

Why: the label-supervised family (rate, adversary, VQ) loses to the LDA floor and fails 2b. Builder A found one exploratory lead for the family not yet
tested here: an autoencoder trained WITHOUT class labels through a rate-limited Gaussian channel, with identity adversaries retrained from scratch
(A.md PROTOCOL 2, `ae`, beta 0.001, dl 32, lam 10; midnight only, post hoc, one seed). This is the one targeted round for it on CREMA-D audio.

Model (`ae.py`, method `ae`): input = per-clip mean+std of the uniform layer mix of the frozen WavLM-Base+ frames (1,536 numbers, standardised with
train-actor statistics); encoder Dropout(0.2) -> Linear(1536,512) -> GELU -> Linear(512, 2 d_z) (mean, log-variance); decoder d_z -> 512 -> 1536;
loss = MSE reconstruction + beta KL + lam x confusion, where the confusion term is the mean KL(p || uniform) of two adversaries (a linear and an MLP
classifier of the 56-57 encoder-train actors, re-initialised from scratch and re-trained on the current z in each of 12 rounds, 150 steps, then
the encoder takes 100 steps). NO emotion labels are used anywhere in encoder training. Released z = the mean (no sampling noise: the
conservative choice for privacy). Trained on encoder-train actors only, fold by fold, exactly like the other encoders.
Grid (all run, none dropped after seeing results): d_z in {16, 32}, beta = 0.001, lam in {0, 3, 10} -> 6 configurations x 3 seeds x 4 folds.
Self-supervised variant (contrastive or masked-frame): NOT run in this round (not cheap on this budget); the reconstruction objective is the
task-agnostic arm.

Selection (validation actors only, never test): per configuration pool the 12 validation actors of each fold across seeds; U_HE_val = recall of DIS+FEA
with the best six-way probe fitted on the train actors (probe sees all six labels, the encoder none); ID_val = worst-attacker identity T1 on the
validation actors. Eligible if ID_val <= 0.10; choose the eligible configuration with the highest U_HE_val (ties: lower ID_val); if none is eligible,
the one with the lowest ID_val. The chosen configuration is then also run on speech-trimmed input (`--trim`), same seeds.

Comparators (same folds, same attackers, fitted on the same four emotions, `c_lda4_k{16,32}` = 3 LDA dims + residual-PCA dims padded to the AE width;
trimmed input for the trimmed variant) and the prosody floors (`c_pros_*`, probes on the six hand-built descriptors; their DIS+FEA recall is read from
the existing runs). The AE is trained without the four-emotion subset, so the 4-class recall criterion of 2b does not apply and is only reported.

Success (the 2b criterion, unchanged, on the 3-seed mean, paired bootstrap over the 91 actors): (1) U_HE(AE) - U_HE(LDA-4 padded to the same d_z) >= +0.05
with the 95% interval above 0; (2) identity T1 of the AE not above the comparator by more than 0.02 (upper end of the 95% interval of the difference
<= 0.02); (3) the same holds on the speech-trimmed variant (otherwise the gain leans on the channel and is reported as such). Failing any of 1-3 closes
Focus C as a negative result with the by-products already measured (channel floor 56x chance, LDA beating the neural attackers, oracle per-actor
prosody floor 0.535 UAR at 2x chance, LEACE fitted on train actors leaving 53x on unseen actors).

## 2d. Result of the pre-registered AE round (3 seeds x 4 folds, 2026-11-04 ~12:45Z): FAIL

Selection on validation actors only (`select_ae.py`; U_HE_val / ID_val T1, worst attacker): d16 lam0 0.470 / 0.110 (not eligible, ID > 0.10),
d16 lam3 0.241 / 0.046, d16 lam10 0.235 / 0.053, d32 lam0 0.570 / 0.331 (not eligible), **d32 lam3 0.317 / 0.055 (selected)**, d32 lam10 0.297 / 0.068.
Test results, 3 seeds, 95% bootstrap over actors, comparator = LDA fitted on ANG/HAP/NEU/SAD padded with residual-PCA dims to the same width (`ae_report.py`;
comparator is one seed, LDA being deterministic apart from the attackers):

| AE config | seeds | U_HE = recall DIS+FEA [95% CI] | identity T1 [95% CI] | x chance | comparator | dU_HE vs comparator [95% CI] | dID vs comparator [95% CI] | seeds with dU_HE >= +0.05 |
|---|---:|---|---|---:|---|---|---|---|
| d16 lam0 (no adversary) | 3 | 0.468 [0.432,0.505] | 0.152 [0.123,0.189] | 13.8x | LDA4 k16 (0.450 / 6.5x) | +0.018 [-0.007,+0.043] | +0.081 [+0.055,+0.111] | 0/3 |
| d16 lam3 | 3 | 0.261 [0.244,0.278] | 0.042 [0.035,0.056] | 3.9x | LDA4 k16 | -0.189 [-0.224,-0.155] | -0.028 [-0.048,-0.008] | 0/3 |
| d16 lam10 | 3 | 0.246 [0.232,0.262] | 0.059 [0.046,0.073] | 5.3x | LDA4 k16 | -0.204 [-0.240,-0.168] | -0.012 [-0.035,+0.008] | 0/3 |
| d32 lam0 (no adversary) | 3 | 0.576 [0.538,0.614] | 0.382 [0.337,0.430] | 34.8x | LDA4 k32 (0.534 / 36.8x) | +0.042 [+0.027,+0.057] | -0.022 [-0.049,+0.004] | 1/3 |
| **d32 lam3 (validation-selected)** | 3 | **0.326 [0.307,0.345]** | 0.073 [0.061,0.089] | 6.6x | LDA4 k32 | **-0.208 [-0.240,-0.179]** | -0.331 [-0.376,-0.289] | **0/3** |
| d32 lam10 | 3 | 0.294 [0.275,0.316] | 0.060 [0.052,0.077] | 5.5x | LDA4 k32 | -0.240 [-0.273,-0.208] | -0.344 [-0.387,-0.301] | 0/3 |

Per-seed point estimates of the selected configuration: U_HE 0.322 / 0.338 / 0.317, identity 0.082 / 0.072 / 0.065 (7.5x / 6.5x / 5.9x).

**Verdict: FAIL, 0 of 3 seeds, for the validation-selected configuration (d32, lam 3) and for every adversarial configuration.** Criterion 1 (dU_HE >= +0.05,
interval above 0) is not met by any configuration: the adversarial AEs lose 0.19-0.24 of held-out-emotion recall to the width-matched LDA comparator. The
"identity within 0.02" criterion is met only trivially, because the 32-dim LDA+residual-PCA comparator itself leaks 36.8x chance (the residual PCA
dimensions carry the speaker/channel). The fair view is therefore equal leakage: the selected AE (identity 6.6x chance) against the leakage-matched LDA-4 comparators
(k16 at 6.5x, U_HE 0.450; k5 at 7.2x, 0.369; k3 at 4.4x, 0.363) gives dU_HE = -0.132 [-0.164,-0.100], -0.051 [-0.078,-0.022] and -0.044 [-0.072,-0.017]
with identity differences within +-0.02 for the first two. The adversary-free AE (lam 0) is the only variant with a positive difference (d32: +0.042 [+0.027,+0.057], 1 of 3 seeds
above +0.05; d16: +0.018 [-0.007,+0.043]), but it is not eligible under the selection rule (validation identity 0.331 and 0.110) and it leaks 34.8x / 13.8x chance
at test; its gain over the matched LDA is a reconstruction effect, not an adversary effect.
What the adversary buys: at d32, lam 0 -> 3 cuts identity 34.8x -> 6.6x chance and costs U_HE 0.576 -> 0.326; the resulting point sits below the label-supervised
LDA-4 frontier (0.450 at 6.5x).
Against the hand-built floors in the same metric (DIS+FEA recall at 91-way identity x chance): six prosody numbers 0.268 at 11.6x, 26 numbers 0.346 at 23x, 26 numbers with
oracle per-actor standardisation 0.459 [0.433,0.485] at 6.0x, six numbers with oracle standardisation 0.373 at 2.0x, label-free PCA-16 of the pooled features 0.431 at 12.8x. The selected AE (0.326
at 6.6x) is below every one of the three oracle/label-free floors with comparable leakage.
Trimmed variant and dithered-code control: see 2e (pending at the time of writing this block).

## 3. Results so far (2026-11-04 ~11:55Z)

Everything here is WavLM-Base+, audio only, 4 actor-disjoint folds pooled over the 91 test actors, 95% bootstrap over actors, identity =
worst attacker on T1 (unseen sentences) for actors the encoder never saw. **Seeds: raw reference 3; every other row is a single seed (seed
0 / one deterministic fit) and is NOT yet replicated.** Attackers: rows run on the GPU queue use the full suite (linear, MLP, kNN, shrinkage
LDA, LightGBM, plus transformer and x-vector TDNN when a frame sequence is released); `c_fixcpu_*` and `c_lda*` ran on the VM CPU with the
pooled-feature families only (linear, MLP, kNN, LDA, LightGBM), which is the full suite for utterance-level releases and the strongest
single family (LDA) for the sequence-level fixed baselines. Raw per-actor counts: HF `runs/c_*`; tables: `results/c_summary.{json,md}`.

**Reference, controls and fixed floors**

| run | seeds | emotion UAR [95% CI] | identity T1, worst attacker [95% CI] | x chance | L vs raw | id T1 seen actors | id T3 |
|---|---:|---|---|---:|---:|---:|---:|
| c_ref_raw | 3 | 0.756 [0.736,0.776] | 0.804 [0.775,0.835] | 73.2x | 1.00 | nan | 0.752 |
| c_ref_raw_trim | 3 | 0.744 [0.725,0.764] | 0.638 [0.593,0.682] | 58.1x | 1.00 | nan | 0.552 |
| c_ctrl_constant | 3 | 0.167 [0.167,0.167] | 0.011 [0.011,0.038] | 1.0x | 0.00 | nan | 0.011 |
| c_ctrl_length | 1 | 0.228 [0.222,0.237] | 0.028 [0.022,0.055] | 2.5x | 0.02 | nan | 0.026 |
| c_ctrl_shuf_emo | 1 | 0.177 [0.170,0.185] | 0.804 [0.775,0.835] | 73.2x | 1.00 | nan | 0.752 |
| c_fixcpu_mean | 1 | 0.751 [0.730,0.772] | 0.816 [0.787,0.848] | 74.3x | 1.01 | nan | 0.758 |
| c_fixcpu_mean_trim | 1 | 0.747 [0.728,0.770] | 0.676 [0.635,0.719] | 61.5x | 1.06 | nan | 0.596 |
| c_fix_leace_tr | 1 | 0.753 [0.732,0.773] | 0.587 [0.539,0.634] | 53.4x | 0.73 | 0.400 | 0.563 |
| c_fixcpu_leace_oracle | 1 | 0.753 [0.736,0.772] | 0.480 [0.432,0.533] | 43.7x | 0.59 | nan | 0.446 |
| c_fixcpu_pca16 | 1 | 0.557 [0.537,0.575] | 0.141 [0.111,0.177] | 12.8x | 0.16 | 0.141 | 0.163 |

**Label-supervised linear projection (shrinkage LDA on emotion), pooled attackers**

| run | seeds | emotion UAR [95% CI] | identity T1, worst attacker [95% CI] | x chance | L vs raw | id T1 seen actors | id T3 |
|---|---:|---|---|---:|---:|---:|---:|
| c_lda_k5 | 1 | 0.743 [0.725,0.763] | 0.076 [0.060,0.102] | 7.0x | 0.08 | 0.040 | 0.067 |
| c_lda_k5_trim | 1 | 0.732 [0.711,0.752] | 0.081 [0.058,0.107] | 7.4x | 0.11 | 0.035 | 0.063 |
| c_lda_k8 | 1 | 0.741 [0.722,0.762] | 0.088 [0.065,0.112] | 8.0x | 0.10 | 0.050 | 0.072 |
| c_lda_k16 | 1 | 0.744 [0.724,0.764] | 0.092 [0.073,0.115] | 8.4x | 0.10 | 0.097 | 0.105 |

**Learned encoders (full-emotion supervision)**

| run | seeds | emotion UAR [95% CI] | identity T1, worst attacker [95% CI] | x chance | L vs raw | id T1 seen actors | id T3 |
|---|---:|---|---|---:|---:|---:|---:|
| c_rate_d4 | 1 | 0.721 [0.700,0.741] | 0.052 [0.035,0.078] | 4.8x | 0.05 | 0.021 | 0.061 |
| c_rate_d16 | 1 | 0.728 [0.707,0.747] | 0.071 [0.056,0.097] | 6.5x | 0.08 | 0.043 | 0.061 |
| c_rate_d64 | 1 | 0.731 [0.710,0.750] | 0.090 [0.066,0.118] | 8.2x | 0.10 | 0.060 | 0.076 |
| c_adv_d16_l1 | 1 | 0.723 [0.705,0.744] | 0.058 [0.045,0.081] | 5.3x | 0.06 | 0.033 | 0.056 |
| c_vq_g2k64 | 1 | 0.725 [0.703,0.746] | 0.045 [0.034,0.069] | 4.1x | 0.04 | 0.022 | 0.043 |

**Held-out-emotion protocol (encoder/projection fitted with ANG, HAP, NEU, SAD only; probes on z trained with all six labels)**

| run | seeds | 6-way UAR | recall DIS+FEA (never seen by encoder) | recall ANG,HAP,NEU,SAD | identity T1 | x chance | L |
|---|---:|---|---|---|---|---:|---:|
| c_lda4_k3 | 1 | 0.634 [0.619,0.652] | 0.363 [0.332,0.396] | 0.807 [0.788,0.826] | 0.049 [0.035,0.066] | 4.4x | 0.05 |
| c_lda4_k5 | 1 | 0.641 [0.624,0.661] | 0.369 [0.340,0.401] | 0.799 [0.777,0.821] | 0.079 [0.055,0.108] | 7.2x | 0.09 |
| c_lda4_k16 | 1 | 0.669 [0.649,0.689] | 0.450 [0.411,0.489] | 0.795 [0.775,0.814] | 0.071 [0.053,0.093] | 6.5x | 0.08 |
| c_he_rate_d4 | 1 | 0.600 [0.584,0.616] | 0.279 [0.254,0.304] | 0.836 [0.817,0.855] | 0.047 [0.032,0.073] | 4.3x | 0.05 |
| c_he_rate_d16 | 1 | 0.621 [0.607,0.638] | 0.354 [0.328,0.380] | 0.820 [0.802,0.839] | 0.067 [0.047,0.092] | 6.1x | 0.07 |
| c_he_rate_d64 | 1 | 0.635 [0.619,0.652] | 0.376 [0.350,0.404] | 0.810 [0.789,0.831] | 0.081 [0.061,0.102] | 7.3x | 0.09 |
| c_he_adv_d16_l1 | 1 | 0.607 [0.590,0.623] | 0.301 [0.276,0.326] | 0.823 [0.803,0.842] | 0.061 [0.044,0.081] | 5.5x | 0.06 |
| c_he_vq_g2k64 | 1 | 0.594 [0.581,0.609] | 0.309 [0.283,0.336] | 0.847 [0.830,0.865] | 0.052 [0.036,0.079] | 4.7x | 0.05 |


Paired differences to the label-supervised LDA-5 floor (seed 0, bootstrap over actors, learned minus LDA): rate d16 emotion UAR -0.015
[-0.026,-0.007], identity -0.006 [-0.027,+0.017]; rate d4 -0.022 [-0.034,-0.012] / -0.024 [-0.048,-0.003]; adversarial d16 (lambda 1)
-0.020 [-0.030,-0.010] / -0.019 [-0.037,+0.003]; VQ g2k64 (12 bits) -0.018 [-0.029,-0.009] / -0.031 [-0.053,-0.008]. Adversarial d16 vs rate-only d16:
identity -0.013 [-0.027,+0.003], UAR -0.005 [-0.013,+0.006] (no measurable adversary effect at this width).

### Held-out-emotion protocol against the pre-registered criterion (section 2b)

Comparator = LDA fitted on the same four emotions, width-matched: `c_lda4_k5` for the d4/d16/VQ encoders, `c_lda4_k16` for d16/d64 (paired
bootstrap, seed 0). U_HE = recall of the never-seen DIS and FEA with the best six-way probe on z.

| learned encoder | U_HE | vs LDA comparator (dU_HE [95% CI]) | identity T1 (x chance) | dID vs comparator | criteria 1/2/3 |
|---|---|---|---|---|---|
| rate d4 | 0.279 | vs k5: -0.090 [-0.116,-0.066] | 0.047 (4.3x) | -0.032 | fail / pass / fail |
| rate d16 | 0.354 | vs k16: -0.096 [-0.128,-0.068] | 0.067 (6.1x) | -0.004 [-0.032,+0.025] | fail / fail / fail |
| rate d64 | 0.376 | vs k16: -0.074 [-0.102,-0.048] | 0.081 (7.3x) | +0.010 [-0.016,+0.031] | fail / fail / fail |
| adversarial d16 (lambda 1) | 0.301 | vs k5: -0.068 [-0.096,-0.043] | 0.061 (5.5x) | -0.019 | fail / pass / fail |
| VQ g2k64 (12 bits) | 0.309 | vs k5: -0.060 [-0.088,-0.033] | 0.052 (4.7x) | -0.028 | fail / pass / fail |

Against the weaker 3- and 5-dim comparators the learned encoders are statistically tied rather than worse (d16: -0.009 [-0.035,+0.016] vs k3, -0.015 [-0.041,+0.011] vs k5; d64: +0.014 [-0.010,+0.036] vs k3, +0.007 [-0.015,+0.030] vs k5), but never reach +0.05, and the 16-dim comparator is the width-matched one for d16.

**No learned encoder meets criterion 1 (needs +0.05 and an interval above 0); the best case is d64 vs the 3-dim comparator, +0.014 [-0.010,+0.036].**
An LDA projection padded to 16 dims with residual PCA directions keeps the most about the unseen emotions (U_HE 0.450 [0.411,0.489] at identity
6.5x chance); raw frozen features give 0.692. Per the pre-registration the learned-encoder arm of Focus C is closed on this evidence (single seed,
but every difference has the wrong sign or an interval containing zero, so replication seeds were not spent). Unqueried: encoders with a
reconstruction or self-supervised term (they keep unseen-emotion information by construction and would be the next thing to try, at a leakage cost
that must be measured), and a larger held-out set than two emotions.

## 4. Log

- 08:25Z Builder-C started. CREMA-D downloaded from the official GitHub LFS media endpoint (7,442 clips / 91 actors / 6 emotions / 12 sentences
  verified; 3 byte-identical wav pairs carry two emotion labels). WavLM-Base+, HuBERT-Base and wav2vec2-Base features cached on the VM disk only.
- 08:55Z Mistake of mine: six job specs had an unquoted `cmd` containing `: ` (YAML error), which crashed workers 0 and 3 for ~15 min. Fixed in
  `make_jobs.py` (cmd is now a JSON-quoted YAML scalar) and every spec is validated with `yaml.safe_load` in a staging dir before entering `pending`.
- 09:05Z First real raw numbers: identity T1 0.71-0.80 across attackers, with shrinkage LDA strongest (0.804); LDA added to the suite (it beats the
  neural attackers on pooled embeddings, as in few-shot speaker ID) and the raw reference rerun with the full suite.
- 09:20Z H-C-006: the non-speech head/tail (300 ms) identifies actors at 56x chance. Escalation rule met (>=10x chance, >0.5 of the speech
  accuracy 0.839): every full-clip identity number is partly a channel number; speech-trimmed input (`--trim`) is run next to it.
- 09:40Z H-C-005: six prosody numbers give emotion UAR 0.454 at 11.6x chance identity; the learned encoders are far above that pair
  (UAR 0.72, identity 5-8x), so the prosody floor does not bind. Oracle per-actor standardisation of the same six numbers: 0.535 / 2.0x.
- 10:00Z First learned results: a plain 16-dim emotion-supervised bottleneck already cuts identity from 73x to 6.5x chance at 96% of the raw
  emotion UAR; the adversary adds nothing measurable (paired, seed 0). LEACE fitted on the 56 training actors leaves 53x on unseen actors
  (L = 0.73), as H-C-001(c) predicted; oracle LEACE (fitted on the attacked actors) still leaves 44x.
- 11:15Z Coordinator's lesson from Focus A applied: LDA on emotion to 5 dims reaches UAR 0.743 at 7.0x chance (trimmed 0.732 / 7.4x), at least as
  good as every learned encoder (paired UAR difference -0.015 to -0.022). Held-out-emotion criterion pre-registered at 11:35Z before any
  c_he_* or c_lda4_* number was read; result above: no learned encoder passes.
- Open and not run by 11:55Z (queued, will upload to HF as they finish): rate ladder d1/d2/d8/d32/d128/d256, posterior-6 and hard-label
  releases, adversarial d16 lambda 3, d64 lambda 1, strong-adversary arms, VQ g4k64, seeds 1-2 for rate d16 and adversarial d16, trimmed learned
  encoders. Re-run `summarize_all.py` + `notes_tables.py` on the VM to refresh `results/c_summary.*`.
