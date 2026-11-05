# H3-018: What would make us STOP FOR REAL: a pre-registered decision table over Round 3 that ends the learned-defence lines (not merely a pause) and states what would NOT be a reason to stop

- Focus: programme-level | Kind: stop hypothesis / decision table | GPU-h: 0 (reads the outcomes of the other Round-3 files) | Prior: 0.25 that the full stop condition is met; 0.50 that the one positive result generalises enough to continue as a single line
- Depends on: F4, T3, H3-001 to H3-017; the lesson recorded on 5 Nov 03:40Z (an idle machine and an empty queue are not stop conditions; a stop should require the controls that could overturn it) and Auditor-1 on the programme-1 stop over-statement. Supersedes H-X-002 for the purpose of the next decision; H-X-002's original table is kept.

## Statement
Four legs, each evaluated with BOTH original and tightened criteria reported side by side; none may be loosened after data.
- Leg 1 (the positive generalises?). GENERALISES iff F4's verdict is GENERALISES or PARTIAL AND H3-001 reaches break-even <= 30% in >= 7 of 9 held-out sources AND H3-007 does not remove >= 50% of the b100 gain AND at least one second dataset or encoder (H3-003, or the H3-002 K-sweep read at its stated slope) supports the K-rule without a contradicting win. FAILS iff F4 is DOES-NOT-GENERALISE OR H3-007 kills the protocol OR H3-001 fails in >= 3 of 9 sources. Otherwise AMBIGUOUS.
- Leg 2 (do floors dominate defenders?). HOLDS iff no learned defender in T3, H3-009, H3-011 (chest X-ray) and H3-014 / H3-015 (wearable) beats the floor envelope by the stated margins (CXR 0.02 U / 0.03 L, IMU 0.0285 U / 0.150 L) at 99% (CXR) / 95% (IMU) bounds, AND H3-010 / H3-016 show the floors' leakage rises by < 0.04 (CXR) and the best dither-free floor is unchanged (IMU) under stronger attackers.
- Leg 3 (new ideas). EXHAUSTED iff none of H3-004, H3-005, H3-006, H3-012 produces a win replicated in >= 2 of 3 splits (or 2 of 3 label draws) at its margin.
- Leg 4 (power). OK iff at most 50% of the pre-registered Round-3 comparisons end INCONCLUSIVE (realised MDE80 above the margin); otherwise the data cannot answer and more GPU on the same units is wasted.

STOP FOR REAL iff: Leg 1 FAILS, Leg 2 HOLDS, Leg 3 EXHAUSTED, Leg 4 OK, AND every boring control that could overturn the stop has run (H3-007, H3-013, H3-016, H3-017). Consequence: no further GPU on learned privacy defences or adapters in this repository's current form; write the final summary (scoped claims, floors tables, the label-budget curve as a by-product, MN-v2, protocol findings), publish the benchmark and code artefacts, and tell the owner the VM can be released. CONTINUE (not stop) if ANY of: Leg 1 GENERALISES (then the line is "supervised adapters on rich label spaces", with a stage-3 pre-registration: seeds >= 3 per cell, a third dataset, deployment-side TTA); any new-idea win replicates; a boring control has not run; Leg 4 shows the units, not the methods, are the limit AND a cheap unit-increasing step exists (the 10-source table, more WISDM segments).
NOT reasons to stop, by pre-commitment: idle GPUs; an empty queue; a single null; a literal gate failure that a labelled sensitivity analysis qualifies; fatigue or budget in agent-hours (only the owner's budget counts); a rival paper. Not reasons to continue: sunk cost; an un-replicated single seed; an underpowered positive.

## Mechanism
After two programmes, every learned-versus-floor comparison that passed its validity gates ended with a cheap floor matching or beating the learned arm, with one exception: the supervised nonlinear adapter on JUMP once about 10% of production labels exist (F3). That exception is label-hungry and found in one dataset with one adapter family, and its generalisation is exactly what Round 3 tests. The only way a stop is credible is if it is conditioned on the controls that could change it, and the only way it avoids the 5 Nov mistake is if "no work left in the queue" is explicitly excluded as a reason.

## Experiment (exact)
No new experiment. At the end of the Round-3 wave the main agent fills the table from `result.json` fields (`verdict_original`, `verdict_tightened`, `inconclusive`, MDE80 printed) and writes the stop decision to FOCUS.md and the journal, with the counts of looks (forking paths, F18). Needed fields per hypothesis: realised MDE80 vs margin; number of looks at sealed material; replication status (seeds vs units). A pre-declared audit: an independent replication auditor re-derives each leg's headline from raw files before the decision is accepted (as Auditor-1 for programme 1).

## Prediction and falsifier
- Predicted probabilities: Leg 1 GENERALISES 0.50, AMBIGUOUS 0.15, FAILS 0.35; Leg 2 HOLDS 0.80; Leg 3 EXHAUSTED 0.55; Leg 4 OK 0.70; the joint stop 0.25 (positively correlated through a shared mechanism: if the K-rule is right, no new idea changes much). P(some learned line continues) 0.60.
- FALSIFIED (we should not stop) if any single learned result beats its floor envelope by the margin with replication and passes the four boring controls: that result is the programme's one continuing line. Falsified in the other direction (we should have stopped earlier) if two or more leg-1 hypotheses are INCONCLUSIVE for power reasons and the remaining GPU would have been better spent on the unit-increasing steps (10-source table, more WISDM segments) without a pre-registered question.

## Cost / dependencies
0 GPU-h; about half a day of the main agent's reading. Depends on the others finishing in the order that satisfies the controls clause.

## Prior
0.25 (full stop), 0.50 (Leg 1 generalises and the line continues). Reasons: the F3 effect is 13x the margin at 100% labels and 3x at 10%, and has a mechanistic account (K >> d), which is why its generalisation is a real question and not a formality.

## Cheapest floor that could kill it
A stop is "killed" by an un-run cheap control. The four controls cost under 5 GPU-h in total; running them first is the cheapest guard against a wrong stop.
