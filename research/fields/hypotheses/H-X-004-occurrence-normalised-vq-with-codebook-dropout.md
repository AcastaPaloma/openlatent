# H-X-004 (A and C): Occurrence-normalised VQ bottlenecks leak >= 0.10 less than plain VQ at equal codebook size; codebook dropout adds >= 0.05 and removes the usage-histogram attack

- Focus: A + C | Kind: learned method (discrete bottleneck, rate-limited) | Cost: ~6 GPU-h | Prior: 0.30
- Depends on: Builder-A / Builder-C adapter loops; a product-VQ module (EMA codebooks, Sinkhorn-balanced assignment).

## Statement
A product-VQ adapter (M sub-codebooks x K = 64 codewords, 6M bits per sample) on frozen embeddings, trained for the task label, leaks L_worst(V0, plain EMA-VQ) - L_worst(V1, occurrence-normalised / balanced-assignment VQ) >= 0.10 at equal (M, K) in both A and C; V2 = V1 + codebook dropout (each step a random subset of codewords or whole sub-codebooks, p ~ U(0, 0.5), is masked) lowers L_worst a further >= 0.05, and reduces a *usage-histogram-only* attacker (sees only the multiset of codeword counts per patch/clip) from >= 3x chance to <= 2x chance. In C, discrete tokens lose emotion relative to a continuous bottleneck of equal dimensionality (U(VQ) <= U(continuous) - 0.03).

## Mechanism
Privacy PORCUPINE shows that uneven codebook occupancy gives an uneven information rate that leaks speaker identity, and fixes it with space-filling VQ and occurrence normalisation (ISCA 2024, scout-opened). Quantisation in SSL speech units "often discards prosodic and paralinguistic information" such as emotion (2505.15667, abstract opened), which is the cost side; segment-level codebooks recover some of it. LAPA uses VQ latent actions but measures no leakage (2410.11758, scout-opened). Codebook dropout is my own design (inspired by variable-bitrate residual VQ; I did not open a source), intended to prevent a few rare codewords from becoming per-source fingerprints.

## Experiment (exact)
- A: PathoROB phikonv2_clsmean -> adapter -> product VQ with M in {1, 2, 4, 8}, K = 64 (6-48 bits/patch); secret = centre; tasks = tissue. C: wavlm-base-plus frames -> 0.5 s window pooling -> product VQ per window, M in {1, 2, 4}, K = 64; clip-level pooled VQ and the sequence of window tokens as two release formats; secret = actor; task = emotion.
- Arms: V0 plain EMA-VQ; V1 Sinkhorn-balanced assignments (target uniform usage) / usage-entropy regulariser; V2 V1 + codebook dropout; V3 V2 + single adversary on the straight-through embedding; continuous bottleneck of equal dimension (d = 4M) as reference; LEACE-then-VQ.
- Attackers: A-SUITE / C-SUITE on the one-hot / embedded tokens (bag-of-tokens histogram + MLP; token-sequence Transformer for C), plus the usage-histogram-only attacker.
- Seeds: 3 x 3 (A: 3 datasets; C: 4 folds).
- Metric: L_worst per release format, U, codebook perplexity (target >= 0.9 K).

## Prediction and falsifier
- Prediction: perplexity of V0 <= 0.5 K; V1 >= 0.9 K; L_worst(V0) - L_worst(V1) in 0.10-0.25; V2 adds 0.05-0.10; usage-histogram attacker drops to <= 2x chance for V2; C: U(VQ, M=2) 0.04-0.10 below the continuous reference.
- Falsified if L_worst(V1) is within 0.03 of V0 on both foci (usage skew is not the leak at these sizes), or V2 raises L (dropout adds noise the attacker averages away), or U(VQ) loses > 0.15 (discrete is unusable for emotion).

## Cost
~6 GPU-h (A 2, C 4: per-window VQ on frames is heavier).

## Dependencies
Builder-A / Builder-C; new VQ module (Sinkhorn, EMA, dropout), small.

## Prior and why
0.30. The PORCUPINE effect was shown for speaker identification at SSL-feature level so C is plausible; for A, patches are one token per sample and the codeword is already a class-ish cluster, so V0 vs V1 may not matter. Codebook dropout is a speculative addition (0.35 conditional on V1 working).

## Sources
[S] Privacy PORCUPINE (Interspeech 2024); [O] 2505.15667 Segmentation-Variant Codebooks; [S] 2410.11758 LAPA; [snip] 2501.13497, 2508.08399 (k-means/VQ speaker disentanglement; not relied on).
