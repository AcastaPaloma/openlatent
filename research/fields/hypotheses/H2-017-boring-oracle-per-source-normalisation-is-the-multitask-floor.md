# H2-017: BORING CONTROL. Per-actor standardisation of the full WavLM feature vector (oracle actor ids) is the multi-task floor on CREMA-D: identity <= 10x chance with emotion and sentence content intact; a learned latent matters only where the source id is unavailable at release

- Focus: C (A has the matching `centre_whiten` rows) | Kind: boring control, redefines the threat model | Cost: ~0.3 GPU-h | Prior: 0.45 that the control wins (identity <= 10x chance)
- Depends on: cached WavLM-Base+ mean+std features and splits (exist); attackers (`evaluate.py`); the sentence probe of H2-008.

## Statement
Standardise each test actor's 1,536-d pooled feature vector with that actor's own mean (arm a), mean and std (arm b), or PCA-64 plus whitening (arm c, as `centre_whiten` in A), using the actor's clips across all emotions and sentences (a design in which each actor has every emotion and sentence, so the actor mean is content-balanced). Arms: oracle (all of the actor's clips), few-shot (10 random clips per actor). On the sealed test actors with the full C attacker suite (LDA, linear, MLP, kNN, LightGBM, TDNN not applicable on pooled features), arm (b) satisfies: identity T1 <= 10x chance (<= 0.11); emotion UAR >= 0.70 (raw 0.756); sentence accuracy above chance >= 0.8 x raw's above-chance (H2-008 gate G1 must have passed); few-shot (10 clips) identity <= 15x chance.

## Mechanism
The six-number prosody floor with ORACLE per-actor z-scoring reached emotion 0.535 at 2.0x chance identity (c_pros_p6_actornorm), against 12x for the same numbers without normalisation; the actor mean is exactly the session/channel/timbre offset that carries 76% of the identity signal (CREMA-D Gate 2 channel floor 0.62 = 56x). The A side already shows the same thing: per-centre whitening gives L 0.05-0.58 at a 2-9 point label cost and was the strongest floor in the gate table. If the same holds for full 1,536-d features, the interesting question for learned latents shifts from "can it hide the source" to "can it do so without being told the source", i.e. the inductive threat model, which is also the only one under which a released foundation-model embedding operates.

## Experiment (exact)
- Data/splits/attackers as C.md sections 1-3 (4 actor-disjoint folds, 91 actors, T1 primary, bootstrap over actors, 3 attacker seeds; speech-trimmed variant as well).
- Arms: a/b/c above, oracle and few-shot (10 clips, 5 random draws); comparators: raw (0.804, 73x), LDA-5 (0.743 UAR, 7.0x), AE (H2-008), rate d16 (0.728, 6.5x).
- Report the (emotion UAR, sentence acc, identity x chance) triple and the number of clips per actor needed to reach 10x.
- Controls: shuffled actor labels at chance; per-actor standardisation of a constant input equals constant (no leakage through the statistics themselves).

## Prediction and falsifier
- Prediction (control wins): identity 5-10x chance for arm (b) with oracle ids; emotion UAR 0.70-0.74; sentence content kept; 10 clips: 8-15x.
- Falsified if identity > 20x chance for arm (b) or (c) (LDA exploits second-order or nonlinear actor structure the standardisation leaves; then inductive learned latents and the LDA-5 floor remain the reference), or if emotion UAR < 0.65.

## Cost
~0.3 GPU-h (feature standardisation is CPU; the TDNN-free attacker suite is cheap on pooled features).

## Dependencies
Builder C code; none else.

## Prior and why
0.45. Prosody-six went to 2x, but there the features are six scalars; a 1,536-d vector leaves more room for the LDA attacker (the strongest in C) to find residual structure, as in A where whitened centres still leaked L 0.2-0.6 on midnight.

## Sources
results_notes/C.md Gate 2, Gate 3 (c_pros_p6_actornorm), section 3; results_notes/A.md gate item 2 (`centre_whiten`); H-X-008 in the round-1 file.
