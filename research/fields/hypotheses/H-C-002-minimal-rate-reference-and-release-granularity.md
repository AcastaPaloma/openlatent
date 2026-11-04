# H-C-002: Rate does most of the work in audio too: the 6-number emotion posterior leaks <= 0.2, and at matched numbers a time-resolved release leaks >= 0.1 more than a pooled one

- Focus: C | Kind: learned method (rate bottleneck, no adversary; mandatory control) | Cost: ~3 GPU-h | Prior: 0.45 (compound; first half 0.7, second half 0.45)
- Depends on: Builder-C frame-level features and splits; C-SUITE.

## Statement
(a) A minimal-rate reference, z = softmax posteriors of a 6-way emotion classifier trained on train actors (6 numbers per clip), keeps U >= 0.97 (it *is* the task output) and has L_worst <= 0.20 on T1 (unseen sentences, enrolled test-fold actors).
(b) With emotion-CE-only bottlenecks (no adversary) of pooled dimension d in {2, 3, 6, 12, 24, 48, 96, 192}, L_worst increases monotonically with d, and L(48) - L(6) >= 0.30 while U gains <= 0.02 beyond d = 12.
(c) At matched numbers per clip (~20), releasing 5 windows (0.5 s each) x 4 numbers has L_worst >= pooled-d=20 + 0.10 at |dU| <= 0.02, because the release exposes temporal dynamics; shuffling the window order lowers L by < 0.05 (order is not the leak, window content is).

## Mechanism
KTH: 1-2 numbers per frame needed no adversary, adversary buys ~15 identity points only at ~15 numbers per frame (journal 2026-10-28 05:20; SUMMARY finding 3); temporal order was not the leak (journal 20:35, H25 rejected). A 6-number posterior contains the label (independent of actor in CREMA-D's balanced design, so label-only leakage = chance) plus a small idiosyncratic "how confidently does this actor express it" residue. What you release decides what leaks: a 128-way gaze token leaked 38% identity single-frame but continuous gaze output 72.6% (2609.09188, scout-opened); context-dependent duration/timing features carry speaker information that survives timbre removal (2507.15214, scout-opened).

## Experiment (exact)
- Data: CREMA-D, 4 folds, T1 primary (T2, T3 secondary).
- Encoder: wavlm-base-plus best layer from H-C-001 (frame-level, 50 Hz). Bottleneck: mean+std pooling -> MLP (768*2 -> 512 -> d), emotion CE on encoder-train actors, early stop on val actors; z frozen after training. Sequence release: windows of 25 frames, per-window mean pool, shared projection to 4 numbers, released as a (T_w x 4) matrix; variants with per-frame d_f = 2.
- References: emotion-posterior z (d = 6); raw pooled; LEACE from H-C-001; PCA-d and random projection at equal d.
- Attackers: C-SUITE (LR, MLP, kNN, LightGBM on pooled/flattened z; x-vector TDNN and BiGRU-attention on sequence z).
- Seeds: 3 encoder seeds x 3 attacker seeds x 4 folds; bootstrap over test-fold actors.
- Metric: L_worst, U (UAR); frontier (U, L) with area; label-only baseline = chance.

## Prediction and falsifier
- Prediction: (a) L_worst(posterior-6) 0.05-0.20; (b) L(48) - L(6) in 0.3-0.6, U(d=12) >= 0.97 U(d=192); (c) sequence minus pooled L_worst in 0.10-0.30; order-shuffle effect < 0.05.
- Falsified if L_worst(posterior-6) > 0.35 (speaker information rides on the posterior scale; then the "rate is enough" story fails in audio and the adversary matters from the start), or if (b) L is non-monotone/flat in d (rate is not the dial), or if the sequence release leaks <= pooled + 0.03 (granularity is irrelevant).

## Cost
~3 GPU-h: 8 dims x 3 seeds x 4 folds x ~2 min plus sequence variants, probes on cached features.

## Dependencies
Builder-C: frame-level features (already cached for layers 3/6/9/12), fold handling, sequence attacker.

## Prior and why
0.45 compound. The 6-number result is very likely small (91 actors do not fit in 6 numbers) but with only ~22 enrolled actors, kNN on 6-d posteriors can reach 3x chance cheaply; the granularity result is the genuinely uncertain part.

## Sources
[S] 2609.09188, 2507.15214 (literature/affect_behaviour.md, other_modalities.md); our journal and SUMMARY; [O] 2501.05310.
