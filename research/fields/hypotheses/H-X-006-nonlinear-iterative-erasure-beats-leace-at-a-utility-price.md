# H-X-006 (A and C): Iterative nonlinear erasure (Obliviator-type) halves LEACE's worst-case leakage but costs >= 2 points of task accuracy; kernel erasure is attacker-specific

- Focus: A + C | Kind: closed-form-ish / post-hoc nonlinear eraser as a floor and as a rival to learned adapters | Cost: ~3 GPU-h | Prior: 0.35
- Depends on: H-A-001 and H-C-001 (LEACE references); Obliviator code (arXiv 2603.07529) or a reimplementation (+2 days builder time).

## Statement
On frozen embeddings with centre (A) or actor (C) as the secret, (1) iterated-linear erasure (INLP/LEACE cascade, 5 rounds) lowers L_worst by < 0.15 vs a single LEACE; (2) a kernel erasure fitted with an RBF kernel of bandwidth gamma_0 protects against an RBF attacker of the same gamma_0 (L <= 0.15) but a mismatched-gamma attacker (gamma_0 x 4 or / 4) and an MLP recover L >= 0.30; (3) an Obliviator-type iterative nonlinear eraser achieves L_worst <= 0.5 x L_worst(LEACE) with biology/emotion U loss of 2-6 points in at least one focus; and (4) the best learned adapter from X-002 matches (3) within 0.05 L at lower U loss.

## Mechanism
LEACE is optimal against linear probes only (2306.03819, abstract opened). Kernelised minimax erasure protects against nonlinear adversaries "of the same kernel type" and "the protection does not transfer to different nonlinear adversaries" (2201.12191, abstract opened). Obliviator models nonlinear dependencies iteratively with composed kernels, claims protection against nonlinear adversaries, and studies "the cost of nonlinear guardedness", i.e. utility falls as erasure proceeds (2603.07529, abstract opened; no datasets or numbers on the page). So the ordering I expect is LEACE > INLP-cascade > kernel (family-specific) > Obliviator > learned population-minimax, with a utility tax that grows along it. The straightforward meaning: a nonlinear closed-form floor exists and is the right thing to beat; a learned adapter that cannot beat it is not a method.

## Experiment (exact)
- Data/encoders: A: PathoROB, phikonv2_clsmean, secret centre, ID-fit and OOD-evaluate; C: CREMA-D wavlm-base-plus pooled best layer, secret actor, fit on 56 train actors, evaluate on test-fold actors (unseen).
- Erasers: LEACE; INLP/LEACE x5; kernel erasure (RFF-approximated RBF, gamma_0 from the median heuristic, rank sweep); Obliviator (if code is released; otherwise a minimal kernel-composition reimplementation, three erasure strengths); learned adapter (X-002 population minimax) at d = 64.
- Attackers: A-SUITE / C-SUITE plus RBF-SVM at gamma_0/4, gamma_0, 4 x gamma_0 (to expose the kernel-family effect).
- Seeds: 3 attacker seeds; erasers deterministic or 3 seeds; bootstrap over slides/actors.
- Metric: L_worst, U, and the trade-off curve (U vs L) over erasure strength; area under the curve.

## Prediction and falsifier
- Prediction: (1) true; (2) matched-gamma RBF attacker L <= 0.15, mismatched/MLP L >= 0.30; (3) Obliviator L_worst(A) 0.2-0.35 vs LEACE ~0.5-0.7, U loss 0.02-0.06; (4) learned adapter within 0.05 L and U loss <= 0.03 in at least one focus.
- Falsified if Obliviator-type erasure is no better than LEACE at equal U on both foci (nonlinear erasure does not help on real embeddings), or if kernel erasure with mismatched attackers keeps L <= 0.15 (kernel family generalises more than claimed), or the learned adapter loses to (3) by > 0.1 L.

## Cost
~3 GPU-h (Obliviator is iterative and kernel-based; N = 30k-60k points requires Nystrom/RFF).

## Dependencies
H-A-001, H-C-001; Obliviator availability.

## Prior and why
0.35. Direction is plausible and published (its abstract), but I have not seen numbers, scale (kernels on 60k x 2048) is a risk, and the NeurIPS-2025 abstract may not transfer to pathology/speech.

## Sources
[O] 2306.03819; [O] 2201.12191; [O] 2603.07529; [O] 2605.01381 (LEACE generalisation weakness).
