# H2-019: STOP RULE. The programme's central claim ("a learned task-agnostic latent beats the strongest leakage-matched trivial floor on held-out tasks") is false, and the stop condition is met when the next pre-registered tests fail

- Focus: programme-level | Kind: pre-registered stopping hypothesis | Cost: none of its own (it reads the others) | Prior: 0.35 that the stop condition is met
- Depends on: verdicts of H2-001, H2-014, H2-015, H2-016, C's AE round (C.md 2c), H2-008, and the stage-1 headroom checks of H2-009 to H2-013.

## Statement
Decision point DP2 is reached when the verdicts above are in. The learned-obfuscating-latent research line STOPS (the output becomes benchmark + controls + negative results) if ALL of the following hold, with the margins and comparators fixed before the numbers arrive:
- S1 (A): H2-001 is falsified (the 3-seed V-selected Delta_held over the inductive floors is < +0.02 in midnight/Tolkach and midnight/TCGA, or <= 1 of 6 cells passes), OR H2-014 shows a dithered label-free code within 0.03 of the AE in >= 3 of 4 cells, OR H2-015 shows the observed gain below the 95th percentile of the null maximum in both midnight cells;
- S2 (C): the AE round fails criteria 1-3 of C.md 2c on the 3-seed mean, and H2-008 either fails its gate (G1 or G2) or fails (a)-(c);
- S3 (new fields): of the five candidates (H2-009 to H2-013) at most one passes its stage-1 headroom check, or none of the learned arms that pass beats its pre-declared inductive floor by the pre-declared margin with CI above 0 in >= 2 of 3 seeds.
Otherwise the programme continues, restricted to the cells that passed (no widening of the scope, no new margins).
Independent triggers: (S0) if H2-016 shows MDE80 > 0.08 for the claim being tested in every available cell and the cross-fit remedy is not affordable, the claims are suspended (not the line) until the instrument can resolve them; (E0) if H2-014 and H2-015 both kill the AE lead, stop all AE work in A at once and go directly to the stage-1 checks of H2-009 to H2-013.

## Mechanism and arithmetic
Four pre-registered attempts (B', A in-distribution, C label-supervised, D) have failed for a shared structural reason and the sole lead is one seed on one encoder. S1-S3 would add at least eight consecutive failures with the same cause. If the per-test success probability under the optimistic prior that justified the programme is p = 0.25, the chance of zero successes in eight independent tests is 0.75^8 = 0.10 (p = 0.30: 0.058); the tests are not independent (same pattern recurring), so this is a lower bound on how surprising "no win" would be and an upper bound on how much repeated failure should count as new evidence. The rule stops where continuing would be a bet that all eight were bad luck. Successes at the headroom stage alone do not count against stopping; only a learned arm beating its floor at the pre-declared margin does.

## What stopping would produce
By-products already measured: CREMA-D full-clip identity is 76% recording-channel signal (non-speech 0.618 = 56x chance); LDA beats neural identity attackers; label-supervised bottlenecks reach 4-8x chance identity at ~96% of raw emotion UAR; LEACE fitted on train units leaks (53x on unseen actors; 0.63-0.89 L on unseen slides); PathoROB RI has headroom for extractable encoders and a retrained centre probe is the honest metric; D's oracle headroom (0.041) and the "secret overlaps utility" mechanism; B' "clutter, not leakage". The recommended deployable method in every field so far is a transductive per-source normalisation (H2-017) plus a rate cap, not a learned adapter.

## Prediction and falsifier
- Prediction (programme claim false): S1 and S2 and S3 hold (prior 0.35; S1 ~0.7, S2 ~0.8 correlated with S1, S3 ~0.5 over five candidates).
- Falsified (programme continues) if any single field yields a learned arm that beats the strongest leakage-matched floor class by its pre-declared margin with CI above 0 on 3 seeds AND survives H2-014/H2-015 AND H2-005's second secret. One such cell in two independent fields would be the first real result.

## Cost
None of its own. If every H2 hypothesis ran in full the total is ~50 GPU-h; the stage-1 headroom checks of H2-009 to H2-013 are ~7 GPU-h of that, and the rule exists to avoid spending the ~40 GPU-h of learned stages on cells that cannot change the verdict.

## Dependencies
The preregistrations of the listed hypotheses must be committed before their numbers; the margins here are not to be re-opened after DP2.

## Prior and why
0.35. The pattern is strong and consistent, but the single lead is real until H2-001/014/015 say otherwise, and two candidate fields (JUMP, astronomy) have independent reasons to pass the headroom stage.

## Sources
journal 2026-11-04 ~12:30Z; results_notes/A.md-D.md; FOCUS.md (status lines).
