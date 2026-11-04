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

## 2. Controls and gates (status)

(filled in as results arrive)

## 3. Log

(newest last)
