# H3-007: Boring control that could kill the only positive result: the F3 gain is not carried by well position, chemical near-duplicates, or the same-plate negative design

- Focus: JUMP label budget | Kind: boring control (shortcut audit of the retrieval protocol) | GPU-h: 1.5 | Prior: 0.75 that the F3 result survives every sub-control (0.25 that at least one of them removes more than half of the gain)
- Depends on: F2/F3 evaluation code and S5 wells; compound metadata (`compound.csv.gz` with InChIKey / SMILES, 8.6 MB, BSD-3 metadata repo, not yet on the VM) and `well.csv.gz`; no new profile download. Zero-GPU metadata checks first.

## Statement
F3 measures U_H as cross-source replicate retrieval where positives are the same compound in other sources and negatives are other compounds on the SAME plate. Three shortcuts could inflate the supervised adapter's gain over the linear floors without any compound biology:
(A) WELL POSITION. If the same compound sits at the same well position (row, column, edge flag) in different sources (the library plates were replicated across partners [UNVERIFIED for production plates; F recorded that production layouts differ across plates but did not check across sources]), a representation that encodes position retrieves replicates. Position-only releases and well-position decodability from the adapter output are the tests.
(B) NEAR-DUPLICATE COMPOUNDS. Held-out compounds that share an InChIKey connectivity block (stereoisomers, salts, tautomers) or have Morgan Tanimoto >= 0.8 with a labelled training compound can be retrieved by chemistry similarity learned from labels.
(C) SAME-PLATE NEGATIVE BIAS. F found that raw sits below chance because source-carrying embeddings rank other-source positives below same-plate negatives (the reason JK1 fired).
Claim: after removing (A) by position-matched negatives, (B) by pruning H, and (C) by a negative set drawn from other plates of the query's source, the F3 headline survives: dU_H(b10 vs E_full) >= +0.015 in 3/3 splits and dU_H(b100 vs E_full) >= 0.7 x its original +0.135 (>= +0.095); position-only code U_H < ctrl_noise16 + 0.01; a position decoder on the adapter output reads <= 2x chance.

## Mechanism
Retrieval AP rewards any cue shared by replicates and absent from same-plate distractors; position and chemical neighbourhood are cues of that kind, and a supervised map trained with compound labels is exactly the model that would learn them if they are predictive. The linear LDA floors can use them less.

## Experiment (exact)
Stage 0 (CPU, metadata only, before any GPU): (A0) for H compounds with cross-source positives, the fraction of positive well pairs that share Metadata_Well; chance is about 1/384. (B0) the number and share of H compounds with an InChIKey connectivity-block (first 14 characters) match or Tanimoto >= 0.8 (Morgan r2, 2048 bits) to a T' compound. If A0 < 5% and B0 < 2%, the sub-control is reported as "no exposure" and the corresponding re-score is skipped.
Stage 1 (re-score; adapters retrained deterministically with the F3 seeds, b10 and b100, d = 8 and 64, r in {0, 0.5}; floors as F3; 3 splits):
1. POSITION-ONLY RELEASE: a 3-number code (row, column, edge flag) from metadata, plus its 8-d one-hot-bucket variant, scored with the F protocol: U must be <= U(ctrl_noise16) + 0.01.
2. POSITION DECODABILITY: ridge / LightGBM attackers on the adapter output predicting row, column and edge flag (cross-fit by plate); report skill vs the best constant.
3. POSITION-MATCHED NEGATIVES: negatives = wells of other compounds in the query's source at the same well position on other plates (neg_sameby = source and well, neg_diffby = compound), same positives; recompute U_H for all arms.
4. SOURCE-LEVEL NEGATIVES: negatives = random wells of other compounds in the query's source (drawn across plates, size-matched to the same-plate set).
5. NEAR-DUPLICATE PRUNING: drop from H (and from the positives' compound pool) every compound with a connectivity-block match or Tanimoto >= 0.8 to any T' compound; recompute U_H and dU_H.
6. SHUFFLED-POSITION CONTROL: permute well positions among wells of the same plate (a within-plate shuffle); a position-driven gain must vanish.
All with F's repaired control (shuffled-compound AP on ctrl_noise16 within +-0.01).
- Seeds vs test units: units H compounds / plates; 3 compound splits replicate H; one adapter seed per split (not yet replicated at seed level); metadata checks are deterministic. Paired differences as F with compound-bootstrap and plate-cluster intervals, B = 1000; margin 0.010.

## Prediction and falsifier
- Predicted: A0 between 0.3% and 4% (no sharing) with probability 0.6; position-only U within 0.005 of noise16; position decoder skill <= 0.05 above constant; B0 < 1.5% of H; dU_H(b10 vs E_full) under controls 3, 4, 5: +0.020 to +0.034; b100: +0.11 to +0.14; shuffled-position control does not change dU.
- FALSIFIED (the F3 positive is a protocol artefact) if EITHER (i) position-only U_H >= U(ctrl_noise16) + 0.02 AND the adapter's dU_H(b10) under position-matched negatives falls below +0.010 in >= 2 of 3 splits, OR (ii) near-duplicate pruning removes >= 50% of dU_H(b100) (to < +0.068), OR (iii) source-level negatives reverse the sign of dU_H(b10). Any one is enough: the label-budget statement must then be reported as protocol-dependent.

## Cost / dependencies
1.5 GPU-h: retrain 3 splits x 8 adapters (~1.5 min each) and rescore (AP on GPU, ~1 min per arm). CPU: Morgan fingerprints for ~40k H and ~56k T' compounds (RDKit, minutes). Needs the compound metadata copied to the VM; the licence is BSD-3 (metadata), CC0 (profiles).

## Prior
0.75 survive. JUMP compounds are unique JCP2022 ids with a diverse library; position artifacts are real in plate-based imaging but F's production layouts were already observed to differ between plates. The risk mass is concentrated in (A), where the check is a single metadata query.

## Cheapest floor that could kill it
A position-only 3-number release. If it retrieves replicates above the noise embedding, the position channel exists and every retrieval claim of the line (F2, F3, F4, H3-001) is re-stated against position-matched negatives.
