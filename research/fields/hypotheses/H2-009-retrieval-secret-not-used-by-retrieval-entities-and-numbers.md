# H2-009: Reframing D: when the secret is numeric/entity tokens that retrieval does not use, the oracle headroom exists and the vector-space floors give nothing at 3% nDCG loss

- Focus: D (reframe) | Kind: headroom check first, learned arm only if it passes | Cost: ~0.5 GPU-h for the check, ~6 GPU-h for the learned stage | Prior: 0.20 (headroom check passes), 0.08 for the full claim
- Depends on: `src/d_embedpriv/` (data, splits, attackers, floors, aggregator, all exist); GTR-base embeddings cache; no PII-Masking-300K (licence flag stays).

## Statement
Redefine the secret as the bag of numeric tokens of the document (tokens containing a digit: years, doses, sample sizes, p-values, percentages) with df >= 3 over the defender+attacker pools (primary tier N1), and capitalised mid-sentence tokens as tier N2 (secondary). Tier sizes are to be measured BEFORE the check from text only. On GTR-base x NFCorpus (existing splits D/A/V/C, sealed C = 894 docs, attackers ridge/MLP/kNN retrained on the released vectors):
- H1 measurability: R_raw - R_prior (R-precision on N1, worst attacker) >= 0.05 with CI excluding 0;
- H2 room: the oracle (digits deleted from the text before embedding) keeps test nDCG@10 retention >= 0.97 AND headroom = L(F*, tau = 0.97) - L(oracle) >= 0.30, where F* is the strongest vector-space floor chosen on V (PCA-d, random projection, noise, LEACE of the N1 bag subspace, retrieval-trained linear map);
- only then the learned `adv` / `bott_mlp` kill test of D pre-registration section 2.5 (K1-K4, margin m = min(0.10, 0.5 x headroom)).
Secondary clause (c): the retrieval-trained linear map (`sup_lin`, the label-only floor of D) loses >= 0.05 relative nDCG@10 on SciFact queries compared with raw (it is fitted on NFCorpus qrels) while a task-agnostic AE retains >= 0.97.

## Mechanism
D failed because rare words are retrieval-critical: deleting them costs 26% nDCG@10 and the attacker re-infers them from context (L_oracle 0.59), so the secret and the utility overlap. Numbers are the textbook case of a secret the retrieval task does not need: topical queries ("does coffee cause cancer") rarely depend on a sample size or a dose. Then deleting them is nearly free (retention ~ 1.0) and the oracle is a real target, but the released object is the vector alone, so deletion is not available to the defender and a vector-space map has to do it. Floors that are blind to the secret (PCA, noise, random projection) cannot target digits; LEACE of the N1 subspace is linear and D showed it is the weakest floor for rare words. Whether a vector map can approach the oracle is the open question; this hypothesis tests only that the headroom exists, per the programme lesson.

## Experiment (exact)
- Stage 0 (CPU, no GPU): tier sizes, tokens per doc, df distribution of N1/N2 on D+A+SciFact; written down before any model number.
- Stage 1 (check): reuse D's embedding job with a new oracle (`oracle_mask_digits`, `oracle_mask_N2`), floors from D `d_hr2`, controls const/shuffle; retrained ridge/MLP/kNN; selection on V; numbers on C; bootstrap over C documents; 3 seeds for stochastic arms; strongest floor at tau in {0.97, 0.90}.
- Stage 2 (only if H1 and H2 pass): `bott_mlp` control and `adv` (lambda 0.3, 3) at d in 16..512, K1-K4 as pre-registered.
- Secondary (c): SciFact queries with the NFCorpus-trained maps, no refit.

## Prediction and falsifier
- Prediction: H1 R_raw - R_prior 0.05-0.12 (digit tokens are less lexically visible than rare words); oracle retention 0.97-1.0; L(oracle) 0.0-0.4; floors at 0.97 leave L >= 0.8, giving headroom 0.4-0.8.
- Falsified (NO-GO, same pattern as the rare-word tier) if H1 fails (GTR-base does not leak numbers measurably), or oracle retention < 0.97, or headroom < 0.10 at tau = 0.97 or 0.90 with the oracle at its own retention.

## Cost
Check: re-embedding ~3 GPU-min per oracle variant + CPU attackers ~1 h. Learned stage ~6 GPU-h if reached.

## Dependencies
Builder D harness (finished its role; code remains), PII-Masking-300K stays unused until the owner decides the licence.

## Prior and why
0.20 for the check passing: H1 is the uncertain part (a dense T5 embedding preserves topics better than digit strings, so R_raw may be indistinguishable from the prior; 0.5), and given H1 the oracle needs to be nearly free and the floors need to be poor (0.4). The claim is deliberately confined to recoverable text, not PII.

## Sources
results_notes/D.md sections 2, 5.1-5.2 (oracle retention 0.738, L_oracle 0.592, H2 headroom 0.041); FOCUS.md Focus D status (reframe candidate: a secret retrieval does not use).
