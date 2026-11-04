# H-C-001: Speaker is decodable at every layer that carries emotion; LEACE fitted on training actors does not erase unseen actors

- Focus: C (CREMA-D audio) | Kind: premise check + deterministic floor | Cost: ~3 GPU-h | Prior: 0.70
- Depends on: Builder-C frozen features (`src/c_cremad/extract_feats.py`: frame-level layers 3/6/9/12 and per-clip mean/std for every hidden state), `splits.json` (4 actor folds; attacker sets T1/T2/T3).

## Statement
(a) Premise (FOCUS kill rule): with raw frozen WavLM features the retrained speaker attacker on the sealed test-fold actors (~22-23 actors, trained on S_TR x E_TR) beats the task-label-only baseline (= chance, since emotion labels are independent of actor in the balanced design) by far: T1 accuracy >= 0.85, T3 (unseen sentences and emotions) >= 0.60.
(b) No hidden-state layer has both emotion utility U >= 0.95 x U_best and L_worst <= 0.5: the minimum of L_worst over layers with U >= 0.95 U_best is >= 0.7. Layer selection is not a free lunch.
(c) LEACE(speaker) fitted on the 56 training actors' pooled features drives a retrained linear speaker probe to chance on those training actors (L_LR <= 0.10) but leaves the *unseen* test-fold actors with L_LR >= 0.25, and an MLP/kNN probe with L_worst >= 0.50. Fitting LEACE on the test-fold actors themselves (oracle) gives L_LR ~ 0, which bounds the seen/unseen gap.

## Mechanism
Layer-wise probing of 11 speech SSL models finds early layers hold acoustics, middle layers abstract speaker traits, and larger models keep speaker information in deep layers (2501.05310, abstract opened); emotion lives in middle layers (2410.03037, search result only). HuBERT speaker information is "hard to contain in a compact subspace" and LEACE "has difficulty generalizing to unseen data" (2605.01381, abstract opened). Our protocol enrols *unseen* actors for the attacker by construction, so a subspace learnt from 56 actors misses part of the speaker variation of the other 23. Speaker verification trained on emotion embeddings alone still works (2409.03655, scout-opened), which is the reason (b) should hold.

## Experiment (exact)
- Data: CREMA-D (91 actors, 12 sentences, 6 emotions, 7,442 clips), fold f=0..3 of `splits.json`; encoder-train actors fit any adapter/eraser, val actors for hyperparameters, test-fold actors are the attacker's enrolled population (chance ~ 1/23 = 4.3%).
- Encoders: microsoft/wavlm-base-plus (default, layers 3/6/9/12 frame-level; stats for all hidden states), wavlm-large, hubert-large-ll60k, emotion2vec (2312.15185) as an emotion-specialised comparator; features: mean+std pooled per layer (frame-level for later hypotheses).
- Probes. Emotion: logistic regression and MLP trained on train actors, scored on test-fold actors (UAR, chance 1/6). Speaker, A-style C-SUITE: LR, MLP (2x512), kNN-5 cosine, LightGBM, and (when a sequence is released) a small x-vector TDNN; trained on test-fold actors' S_TR x E_TR clips, tested on T1 (S_TE x E_TR, primary), T2, T3.
- Ladder: raw per layer; LEACE(speaker labels of train actors) rank 55; LEACE fitted on the test-fold actors (oracle); INLP x5; CMVN-style per-clip mean/std standardisation of frames before pooling.
- Seeds: 4 folds x 3 attacker seeds; bootstrap over actors (resample test-fold actors with their clips).
- Metric: L = (acc_att - chance)/(acc_raw - chance) per attacker, L_worst = max; U = (UAR - 1/6)/(UAR_raw - 1/6); also the raw-vs-label-only gap.

## Prediction and falsifier
- Prediction: (a) T1 raw speaker accuracy 0.85-1.00 at the best layer, T3 0.6-0.9; emotion UAR raw (actor-disjoint) 0.50-0.70 for WavLM-base+ best layer (my estimate; label-noise-limited, human audio-only is reported at 40.9% in Cao 2014 per a secondary source, not opened); (b) as stated, min L_worst >= 0.7; (c) LEACE-train: L_LR(train actors) <= 0.10, L_LR(unseen) 0.25-0.6, L_worst(unseen) >= 0.5, oracle ~0; U drop <= 0.03.
- Falsified if (a) T1 raw speaker < 0.5 (the identity attacker is weak; C's premise fails), or (b) some layer has U >= 0.95 U_best and L_worst <= 0.5 (layer choice is a trivial baseline all later methods must beat), or (c) L_LR(unseen) <= 0.10 (LEACE generalises across actors; the unseen-actor angle in X-005 is moot).

## Cost
~3 GPU-h: three extra encoders at ~0.5 GPU-h each + probes.

## Dependencies
Builder-C: feature cache and `attack_masks`, probe suite; LEACE/INLP code shared with A.

## Prior and why
0.70. (a) is near certain; (b) and (c) are the uncertain ones (~0.7 and ~0.6). The unseen-actor LEACE result is a real unknown because x-vector style speaker spaces are low dimensional, so 56 actors may span most of the relevant subspace.

## Sources
[O] 2501.05310; [O] 2605.01381; [O] 2312.15185 emotion2vec; [S] 2409.03655 (literature/affect_behaviour.md #11); [S] 2412.00277 WACV baselines (literature/contrarian_stress_test.md 3.1).
