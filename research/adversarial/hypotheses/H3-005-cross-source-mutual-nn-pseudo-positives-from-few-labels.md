# H3-005: Cross-source mutual-nearest-neighbour pseudo-positives, bootstrapped from the 1% (or Target2-only) adapter, recover at least half of the gap to the 10% label budget (NEW idea; the open "label-free learning on JUMP" item)

- Focus: JUMP label budget; semi-supervised use of the unlabelled production wells | Kind: new method, with label-free linear floor and oracle-pair controls | GPU-h: 3 | Prior: 0.20
- Depends on: F3 adapters b1 and a (warm starts, F.md section 9), the F2/F3 SupCon code, `ldaM`; independent of F4 and H3-001. Idea source: mutual-nearest-neighbour batch correction (MNN, fastMNN, Scanorama in the JUMP benchmark of Arevalo et al.) turned into positive pairs for SupCon.

## Statement
Start from the adapter trained on regime b1 (Target2 classes plus 1% of production compounds). In each of R = 3 rounds: (1) embed all Fit production wells (labels hidden for all but the labelled 1%); (2) form candidate positives by MUTUAL nearest neighbours between wells of DIFFERENT sources under cosine similarity (k = 1, mutual), keeping pairs whose similarity exceeds tau, with tau set on the labelled part so that pair precision is >= 0.80 there; (3) merge the pairs into pseudo-classes (connected components, size capped at the number of sources); (4) retrain the adapter from scratch with SupCon on true labels plus pseudo-classes. Then, per split, over 3 splits:
(a) U_H(b1 + MNN-3) - U_H(b1) >= +0.025 (b1 is 0.124-0.127 in F3; b10 is 0.148-0.200);
(b) the recovered fraction (U_H(b1+MNN) - U_H(b1)) / (U_H(b10) - U_H(b1)) >= 0.5;
(c) true precision of the retained pseudo-pairs (computed on the hidden labels, a diagnostic only, never used for selection) is >= 0.70 at round 3;
(d) the same scheme starting from regime `a` (Target2 ids only, no production labels) gains >= +0.02 over `a`.

## Mechanism
The only fully label-free arm in F3 (a reconstruction AE) lost to the label-free floors, and the metadata-pair arm `c` gained +0.004 to +0.006 over same-label LDA. What F3 never used is the redundancy in the unlabelled wells: every production compound is present in several sources, so a decent embedding makes the cross-source mutual nearest neighbour of a well its replicate more often than chance. Those are exactly the positives SupCon needs. Confirmation bias is the threat: pairs the adapter already ranks high are pairs it already handles, so gains may be small.

## Experiment (exact)
- Data and splits: JUMP S5, view A, splits s0 / s1 / s2 (F3 construction). Fit wells: production T' wells (unlabelled except the regime's labelled subset) plus Target2 wells.
- Warm starts: F3 b1 and a adapters (d = 8 and 64; r in {0, 0.5} for the final V-selection only). Rounds R = 3, tau calibrated on the labelled subset (b1: ~300 production compounds; a: the 301 Target2 classes) at precision 0.80; at most 2 pseudo-positives per well per round; pseudo-classes of size <= number of sources.
- Controls: (i) RANDOM-PAIR control: same number of pairs sampled uniformly across sources (must give no gain, tests that extra training steps or the SupCon batch composition are not the source of the gain); (ii) ORACLE-PAIR control: the same number of pairs drawn from the TRUE hidden labels (upper bound for an N-pair budget; if the oracle gain is < 0.01 pair count is not the limitation); (iii) PERMUTED-LABEL memorisation arm as F3 8.5; (iv) the label-free linear floor `ldaMNN`: PCA-256 then LDA fitted on the pseudo-classes obtained by MNN pairing in the RAW PCA-256 space (no adapter, no labels), d = 8, as the cheapest alternative use of the same pairs.
- Comparators and matching: F3 rules (E_match(R) = label-free floors + ldaM_R; E_full = every floor incl. 100%-label LDA); V-selection; H read once; dU in mean AP with compound-bootstrap and plate-cluster 95% intervals; margin delta_U = 0.010.
- Seeds vs test units: units H compounds / plates; 3 compound splits; one adapter seed per round (not yet replicated at the seed level); 2 label draws for b1 on the 3 splits only if the first draw passes the gate.
- Pre-declared stop: if round-1 precision at tau for 0.8 on the labelled part keeps fewer than 1% of wells, the scheme is "not enough clean pairs" and the hypothesis is FALSIFIED on (a) without further rounds.

## Prediction and falsifier
- Predicted: round-1 retained-pair fraction 2-6% of wells; precision on hidden labels 0.65-0.85; (a) +0.010 to +0.040 (point +0.020); (b) 0.2-0.6; (d) +0.005 to +0.020; random-pair control +0.000 +- 0.003; oracle-pair control +0.03 to +0.08.
- FALSIFIED if (a) is < +0.010 in >= 2 of 3 splits (below the margin), OR hidden-label precision at round 3 is < 0.50 (confirmation bias compounds), OR `ldaMNN` matches the adapter's gain.
- A pass on (a)+(b) still says nothing about a truly label-free adapter: the 1% labels are the seed. A pass on (d) alone is the label-free-ish result.

## Cost / dependencies
3 GPU-h: 3 splits x 2 regimes x 3 rounds x (adapter 8k steps ~1.5 min at d = 8, ~3 min with pseudo-pairs) plus mutual-NN search on ~130k x d (GPU, seconds) and attackers (CPU). Controls add ~1 h; included.

## Prior
0.20. For: cross-source replicates are exactly the structure MNN exploits, and the oracle-pair control will show how much room pairs give. Against: raw cross-source retrieval is weak (U_raw 0.051), the b1 adapter at U 0.12 has a low top-1 replicate hit-rate, so a precision-0.8 threshold keeps few pairs, and F3 `c` (clean pairs without ids) only reached +0.005 over same-label LDA.

## Cheapest floor that could kill it
`ldaMNN` (linear LDA on MNN pseudo-classes in raw PCA space) and the random-pair control; at the other end, simply labelling 3% instead of 1% of compounds (the b3 point of F4) — if MNN does not beat b3 it adds nothing a larger label budget does not.
