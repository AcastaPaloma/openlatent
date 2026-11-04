# H2-008: On CREMA-D, a label-supervised emotion projection destroys sentence content while an identity-adversarial label-free AE keeps it (multi-task probes, the sharpest "task does not determine the content" test already available)

- Focus: C | Kind: learned method vs floor on a multi-task utility | Cost: ~1.5 GPU-h | Prior: 0.20
- Depends on: `c_ae_*` jobs and their `ae.py`; an added sentence-ID probe in `src/c_cremad/evaluate.py` (small change; released z must be stored or re-derived); gates G1/G2 below.

## Statement
Use three actor-disjoint downstream tasks on the released z: emotion (6-way UAR), sentence identity (12-way accuracy, chance 0.083; every actor reads all 12 sentences), and, if clip labels allow, emotion-level (low/medium/high on the sentence that has levels). Gates first, from floors only (CPU minutes):
- G1 (measurability): raw WavLM-Base+ mean+std features give sentence accuracy >= 0.30 (>= 3.6x chance) on unseen actors, else this task cannot be resolved on pooled features and the hypothesis is dropped (frame-level features would be the alternative).
- G2 (collapse): the label-supervised LDA-emotion-5 projection (c_lda_k5) keeps <= 50% of raw's above-chance sentence accuracy.
If G1 and G2 pass, then for the V-selected AE (dz 16 or 32, lam in {3, 10}, no emotion labels in training) with identity T1 <= 10x chance (L <= 0.13):
(a) sentence accuracy above chance >= 0.6 x raw's and >= LDA-5 + 0.15;
(b) the multi-task utility U_MT = mean of the normalised (acc - chance)/(raw - chance) of emotion and sentence is >= LDA comparator (width-matched, 3 LDA dims + residual PCA padded to dz) + 0.08, 95% paired bootstrap over 91 actors above 0, 3-seed mean;
(c) the same holds on speech-trimmed input (otherwise the gain leans on the channel, Gate 2 of C).

## Mechanism
Emotion labels define 5 LDA dimensions, and sentence/phonetic content is orthogonal to emotion in a balanced design, so the label-only projection keeps almost none of it (the C pre-registration expects this collapse only for held-out emotions, where it was small: LDA-16 padded keeps 0.450 U_HE vs raw 0.692). A task whose label is not emotion makes "the task does not determine the representation" literally true. The AE sees no labels, so it keeps content if content survives the speaker adversary; sentence content is shared across all actors and does not help identify them, whereas channel and timbre do, which is why it is the most plausible survivor.

## Experiment (exact)
- Data: CREMA-D (existing splits.json, 4 actor-disjoint folds, encoder-train / validation / sealed-test actors, 91 actors), WavLM-Base+ pooled mean+std (1,536-d), also `--trim` variant.
- Arms: raw; LDA-emotion (k = 5, 8, 16 padded); rate/adv/VQ label-supervised encoders (c_rate_d16, c_adv_d16_l1, c_vq_g2k64); AE grid (dz 16/32 x lam 0/3/10, 3 seeds); prosody six and 26 numbers; per-actor-normalised features (H2-017) as the transductive reference.
- Probes: best of linear / MLP / LDA retrained on z of encoder-train actors, hyper-parameters on validation actors, scored on sealed test actors. Identity attackers as in C (LDA strongest; linear, MLP, kNN, LightGBM, TDNN where applicable); T1 primary.
- Selection: validation actors only, rule of C pre-registration 2c (identity_val <= 0.10, then highest U_MT_val).
- Controls: constant at chance on all tasks; sentence probe on prosody-six must be near chance (otherwise sentence id is leaking through length/rate); shuffled sentence labels at chance.

## Prediction and falsifier
- Prediction: G1 passes (raw sentence acc 0.35-0.6; 0.6 prior), G2 passes (LDA-5 sentence acc 0.10-0.18; 0.85); AE sentence acc 0.25-0.5 at identity 6-10x chance; DeltaU_MT vs LDA +0.08 to +0.20.
- Falsified (for the AE) if DeltaU_MT < +0.05 or its lower CI <= 0, or if the trimmed variant loses > 0.04 of the gain; if G1 fails the hypothesis is untestable on pooled features, not refuted.

## Cost
Sentence probes on cached features are cheap; 18 AE jobs x 3 min ~0.9 GPU-h plus probe/attacker evaluation; trimmed variant +0.5 GPU-h.

## Dependencies
Builder C (store z or recompute); no new data.

## Prior and why
0.20 (G1 0.6 x G2 0.85 x pass given gates ~0.4). Mean+std pooling averages over time, which may leave little lexical content; speaker identity is carried by exactly the spectral envelope that also encodes phonemes, so the speaker adversary may remove content too.

## Sources
results_notes/C.md sections 2b, 2c, 3 (LDA floors, held-out emotion table, AE pre-registration); no external paper relied on.
