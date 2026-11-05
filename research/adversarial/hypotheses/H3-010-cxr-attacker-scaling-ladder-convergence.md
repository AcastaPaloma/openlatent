# H3-010: Chest X-ray attackers have not converged: a training-set scaling ladder (and visit pooling) raises the linkage of the downsample floor by more than the paired MDE, shrinking, but not erasing, its advantage over the defender

- Focus: R-CXR; adversary convergence | Kind: attacker-side robustness of the floors (point (iv): adversaries mostly did not converge) | GPU-h: 4.5 | Prior: 0.50 that L(down_f4) rises by >= 0.03 over the ladder; 0.80 that the defender-versus-floor ordering survives
- Depends on: T2 / T3 evaluator and the attacker-scaling ladder pre-registered in T.md 3.5 and NEVER RUN (stage 1 stopped it; T.md 6 "Not run: scaling ladder"); no new data. The ladder uses S-role patients as extra attacker-training data: valid for floors and `raw` (no defender ever touched them as attackers), defender-SEEN for `pcxr` (sensitivity arm only, as pre-registered).

## Statement
Programme-2 attackers were trained on about 4,060 matched triples (A role, about 2,470 patients) for at most 30 epochs with early stopping; the T.md rule "a defended release does not count as holding unless its worst case does not rise across the ladder by more than the paired MDE" was never applied. Ladder rungs: R1 = A (about 4.1k triples), R2 = A + 2,500 S patients (about 9k), R3 = A + 5,000 S (about 14k), R4 = A + all S multi-visit patients not in F or T (about 21k triples; if H3-008's bucket F exists it is excluded). Claims, on the T3 test units (rotations 0 and 1, paired across rungs):
(a) For `down_f4` the family-F1 (Siamese) and best-F3 worst-case L rise from R1 to R4 by >= 0.03 (0.805 -> >= 0.835), while `raw` rises by <= 0.02 (it is nearer saturation, 0.887);
(b) the advantage of `down_f4` over `raw` therefore shrinks from 0.082 to <= 0.06 at R4 but its advantage over `pcxr_mu0.01` (0.055 at R1) stays >= 0.03;
(c) a learning-curve fit L(N) = L_inf - a N^(-b) on rungs R1-R3 predicts R4 within 0.01 for the floors (the BoN-style extrapolation check from the leakage-measurement scout); if it fails, no extrapolation is claimed;
(d) pooling P = 2 images per side at test time (patients with >= 3 images; positives use one image on one side and the mean pair score over 2 images on the other) raises L(down_f4) by >= 0.03, versus <= 0.015 for `labels14` is not applicable (code floor).

## Mechanism
Heavier downsampling destroys fine pixel structure that a Siamese network needs a lot of data to compensate for with coarse shape cues; the F1 verifier trained on 4k triples is data-starved on a 64 x 64 effective image, while raw images carry texture a small network already uses. If true, part of the floors' advantage is an attacker-compute artefact. Evidence in the programme: F1 matched = mismatch = anon2 for raw (no signal in condition choice), the frozen-embedding head beat the Siamese by 0.046 on raw (so the Siamese is not near the ceiling of the view), and every programme-1 adversary result was limited by convergence.

## Experiment (exact)
- Releases: `raw`, `down_f4`, `down_f2`, `rw3_mu0.01`, `pcxr_mu0.01` (C1, S-trained; seen-patient sensitivity at R2-R4), `blur_s2`, `jpeg_q30`; `labels14` and `const` as code / null anchors (code floor attacked by the F3 heads only).
- Attackers: F1 Siamese ResNet-50 (match condition, 30 epochs, patience 5 as T2; and a long-schedule variant 90 epochs, patience 15, on R4 only), F3 (frozen DINOv2-B CLS + mean patch, logistic and 2x256 MLP pair heads, best on V), F5a (demographic-feature attacker, to see whether the added data moves it). Each rung trains one F1 per release (3 seeds at R1 and R4 for `raw`, `down_f4` and `pcxr`).
- Test: T3 rotations 0 and 1 pooled (about 3,060 patients), N2 matched negatives, patient-cluster bootstrap B = 2000 with SHARED resamples across rungs and releases so that rung differences are paired (the unit noise cancels); worst case L over F1 / F3 / F5 inside each resample (as T). Sealed-query ledger: one look per (release, config, rung) digest; the rung id is in the config, so each rung is a distinct sealed query (the ledger is not circumvented: the test pairs are the same; the entries are legitimately new attackers). State this to the custodian before use.
- Visit pooling (d): patients with >= 3 images in T; scores averaged over 2 images; MDE larger (stated).
- Seeds vs test units: units fixed (T3 rotations), seeds replicate attacker training noise (3 at R1 and R4), the ladder replicates data volume; rotations 0 and 1 replicate the attacker-train / validation roles; S contributes only training rows.
- Margins: a rung change is "real" if its paired CI excludes 0 and it exceeds the printed paired MDE80 (T2: 0.008-0.010).

## Prediction and falsifier
- Predicted: L_F1(down_f4): 0.760 (R1), 0.775, 0.790, 0.805 (R4); L_F3 saturates earlier (+0.01); L(raw) +0.01; pcxr (seen) +0.02; extrapolation check passes for 4 of 5 floors; ordering of releases by L unchanged at R4; P = 2 pooling +0.04 for `down_f4`, +0.03 for `raw`.
- FALSIFIED (attackers had converged) if every image release rises by < 0.01 from R1 to R4 (paired MDE) and the extrapolation is flat. FALSIFIED on (b) if at R4 `pcxr` is within 0.02 of `down_f4` or below it (the floor's edge was an attacker-data artefact).

## Cost / dependencies
4.5 GPU-h: F1 on 3 releases x 4 rungs (6, 12, 22, 30 min = 70 min each, 3.5 h) + F3 heads on 7 releases x 4 rungs (~0.7 h) + P = 2 pooling (0.3 h). Embeddings for S patients are cached by the DINOv2 pipeline. The long-schedule variant is the first cut if over budget.

## Prior
0.50 for (a). For: small-data starvation of Siamese on low-resolution inputs is a standard finding; the F3 head's lead over F1 on raw suggests F1 is not at the ceiling. Against: the frozen-embedding head already reads most of what a downsampled image holds, and DINOv2 features of a 64-px image saturate quickly.

## Cheapest floor that could kill it
A single frozen-embedding head with all S data (R4) already on `down_f4`: if its L equals the R1 value, attackers had converged on that view, and only the F1 rows would move.
