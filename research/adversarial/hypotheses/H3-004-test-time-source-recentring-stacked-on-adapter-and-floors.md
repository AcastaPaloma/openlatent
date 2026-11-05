# H3-004: Test-time (transductive) per-source recentring of the RELEASED low-dimensional vector lowers source leakage of the supervised adapter and of the LDA floors alike, and is the cheapest remaining way to move the F3 operating point (NEW idea)

- Focus: JUMP label budget; test-time calibration as a defence | Kind: new method on top of the only positive result, with a same-treatment floor | GPU-h: 2 | Prior: 0.45 that adapter leakage drops by >= 0.10 at |dU| <= 0.01; 0.30 that the break-even budget moves to <= 3%
- Depends on: F3/F4 adapters (re-trainable deterministically from the F2/F3 code and seeds); H3-001 (unseen-source arm, optional). Literature: test-time adaptation helps most on JUMP-like batch shift (BatchBench scout-read: ARM-BN 0.260 vs DANN 0.170 vs ResNet18 0.175, `fields/literature/other_vision.md`; AdaBN 1603.04779; Tent 2006.10726); the programme-1 A hypothesis H-A-005 proposed it but it was never run.

## Statement
Let z be the released vector (F2/F3 adapter output, d = 8 or 64, or an LDA / PCA floor of the same d). A TTA(z) step subtracts the per-source mean (and optionally divides by the per-source per-dimension SD, or applies a ridge-regularised d x d whitening) computed from the unlabelled released vectors of the TEST wells of that source (transductive: no labels, no compound id, no training).
(a) On the b10 and b100 adapters, TTA lowers the retrained-attacker source leakage L_H by >= 0.10 absolute (for d = 64, from about 0.43 to <= 0.33) at |dU_H| <= 0.010.
(b) The same TTA applied to the best same-label floor (`ldaM_f_8`, `ldaP_8`, `ldaP_16`) and to `dpca16_r1` lowers their L_H by a similar amount (>= 0.05), so the comparison that matters is adapter+TTA versus floor+TTA, and the adapter keeps dU_H >= +0.10 at b100 and >= +0.02 at b10.
(c) The break-even budget against the TTA-treated envelope is <= 3% in the 3 splits (vs 10% without TTA), because TTA lowers the adapter's leakage so that its V-selected matched floor is a lower-utility point.
(d) TTA applied to the RAW 737-d profiles stays at L >= 0.99 for the retrained nonlinear attackers (F's T_centre / T_swhiten64 reproduce), so the benefit is specific to low-d supervised outputs.

## Mechanism
After supervised projection, most source signal in a d = 8 vector is a first- and second-moment offset per instrument (the F2 attacker reads 0.40-0.54 at d = 8-64, 2-2.7x chance). Removing the offset is cheap and exact in low dimension, where per-source statistics are well estimated from ~10^4 wells per source. In 737-d the per-source mean/covariance is poorly estimated and nonlinear attackers read higher moments (E: per-batch centring raised nonlinear leakage in HLCA; F: T_centre / T_combat leave L at 0.99-1.00). The programme's floors were never allowed this move (references excluded), and nobody checked it on the adapter's output.

## Experiment (exact)
- Data: JUMP S5 (Arevalo-5), view A, splits s0 / s1 / s2 (F3 construction: Fit, V, H; H about 40k compounds). Adapters: F3 regimes b10 and b100 (and b1, a for the break-even question), d in {8, 64}, r in {0, 0.5}, retrained with the F3 seeds (one seed per split; deterministic code path). Floors: ldaM_f_8, ldaP_{4,8,16}, ldaT2_{8,16}, dpca16_{r1,r2}, pca16.
- TTA operators (applied to released vectors of the evaluation wells only; statistics from all H and V wells of the same source): T1 mean-centre per source; T2 mean + per-dimension SD per source; T3 ridge-whitening (lambda = 1e-2 trace/d) per source; T4 mean-centre per PLATE (plates with >= 30 evaluated wells); T5 INDUCTIVE control: per-source mean and SD from the Fit wells of that source (no test-time statistic; needs the source id, valid only for seen sources). If T5 matches T1/T2, transduction is not needed (a deployment fact, not a negative).
- Metrics: U_H as F (cross-source mean AP, same-plate negatives; TTA recomputed before AP), L_H as F (retrained linear / MLP / kNN / LightGBM, plate-disjoint halves, cross-fit; attackers see the TTA'ed vectors, so a nonlinear attacker may exploit any residual). V-selection: operator and dither level chosen on V.
- Comparators: envelope arms with and without TTA; (b)-(c) are tested by running F3's matching rule with TTA applied identically to adapter and floors (the pre-registered same-treatment comparison) and, separately, against the untreated E_full.
- Unseen-source arm (optional, needs H3-001's code): leave-source-out for sources 8 and 10 with T1-T4 on the unseen source's wells.
- Controls: TTA on `ctrl_noise16` (U unchanged, L at chance); TTA on a source-shuffled assignment (statistics from random wells of the same plate pool, not the right source) must give no leakage reduction; a placebo that adds the same mean shift to all sources.
- Seeds vs test units: units H compounds (U), plates (L); 3 compound splits replicate the H set; one adapter seed per split (not yet replicated at the seed level); TTA is deterministic. Margin delta_U = 0.010 (F3), leakage tolerance 0.05, L-axis margin 0.15 as F.

## Prediction and falsifier
- Predicted: T2 on mlp_d64 b100: L_H 0.43 -> 0.30 (0.25-0.36), dU_H(self) within +-0.01; on ldaP_8: 0.22 -> 0.15; (b) adapter+TTA minus floor+TTA at b100 +0.10 to +0.14 (F2 was +0.135); (c) break-even <= 3% in 2 of 3 splits with probability 0.30; (d) raw+TTA L >= 0.97; T5 within 0.03 of T2 in L (so deployment does not need the transductive pool).
- FALSIFIED if TTA lowers the adapter's L_H by < 0.05 at |dU| <= 0.01 in >= 2 of 3 splits (nonlinear attackers read what recentring leaves), OR if floor+TTA lowers L by more than adapter+TTA so that dU_H(b100) falls below +0.05 (TTA is a floor technique and erases the learned advantage). The shuffled-source control showing any reduction invalidates the run.

## Cost / dependencies
2 GPU-h: retrain 3 splits x (b1, b10, b100, a) x 2 d x 2 r = 48 adapters (~1.5 min each), TTA and attackers (CPU, threads <= 8). No download.

## Prior
0.45 for (a), 0.30 for (c). Reasons for (a): the largest source signal in a low-d supervised output is an offset and the estimator is well-conditioned at 10^4 wells per source; against: E and F both found per-source normalisation raised or left nonlinear leakage in high d, and the attackers are retrained on the TTA'ed vectors.

## Cheapest floor that could kill it
`ldaM` and `ldaP_8` with the same TTA (the same-treatment floor in (b)), and the inductive T5 control; the transductive Harmony reference run on the 8-d outputs is a context row.
