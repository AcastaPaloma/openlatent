# H-G-002: CXR gate: on frozen-FM latents with matched negatives and ~1,400-4,700 patients, a population-trained anonymiser can be tested for a held-out-family gap at MDE ~0.02 AUC

- Focus: G (conditional) | Kind: gate / learned method vs controls | GPU-h: ~3 (latent-only: DINOv2-L feature extraction for 112k images ~1; adapter trainings 5 folds x 4 rotations x 4 methods, minutes each) | Prior: 0.15 that G opens; 0.65 that the gap exists
- Depends on: H-CXR-001 (demographic attacker, MN-v1), H-CXR-002 (stage-0 passed), H-CXR-004 (family definitions), download gate. Shares the 6 GPU-h gate budget with H-G-001. Preference note: CXR has the power (paired MDE80 ~0.02 AUC at 1,424 patients, ~0.011 at 4,707) that the IMU data lack, so this is the better-powered place to ask the charter's second question.

## Statement
Latent release z = A(f(x)) where f = frozen DINOv2-L (Apache-2.0; RAD-DINO optional, but its pretraining includes NIH-CXR) CLS + mean-patch embedding and A a learned residual MLP adapter (768 -> 256). Families (fixed in advance): G1 learned pair heads on z (MLP / Siamese head, InfoNCE); G2 metric kNN (cosine, L2, Canberra) with a differentiable surrogate for in-loop use; G3 gradient-boosted pair attacker on |z_a - z_b| and z_a * z_b (non-differentiable, alternating retrain); G4 demographic + device regressors on z (H-CXR-001 estimators on latents). Rotate each family out. Pre-registered claims: (1) Gamma = AUC_HO(D_pop) - AUC_in(D_pop) >= 0.04 for G3 or G4 held out (MN-v1 negatives); (2) D_pop minus D_single (strongest attacker, 5 restarts, compute-matched) within +-0.02; (3) D_pop minus the lower envelope of {D_dither/PCA-k rate bottleneck, label-only LDA on the 14 findings, random projection} within +-0.02 at matched held-out-finding utility (+-0.01 AUC; 4 findings withheld from adapter training, linear probe on z). G OPENS only if D_pop beats the envelope by >= 0.03 AUC on the held-out family in >= 3 of 4 rotations over 3 seeds and 5 patient folds; P = 0.15.

## Mechanism
Same as H-G-001, plus: label-only LDA on the 14 findings already holds the 14-label utility with a 14-d code, so utility must be measured on findings the LDA was not fitted to (the CXR scout's D2 and Muthyala et al.: CLS embeddings lose small-lesion signal that patch-local probes keep). A demographic attacker held out tests whether the adapter learned "hide identity" or just "hide a fixed set of cues".

## Experiment (exact)
- Data / units: ChestX-ray14, patient unit, 13,302 multi-visit patients. No image-space defender is trained, so full rotation is allowed: 5 patient-hash folds (adapter and attackers trained on 4, tested on 1; pooled out-of-fold n = 13,302 for the primary number; official-test fold 1,424 reported), 3 seeds inside each fold, 4 family rotations.
- Defenders: D_pop, D_single, D_dither (PCA-k, noise on z, no adversary), D_LDA14 (closed form), random projection to the same width. Utility: 14-finding linear-probe AUC and held-out-finding linear-probe AUC on z.
- Evaluation: MN-v1 AUC and Rank-1; worst over the four families; held-out-family numbers for the verdict; patient-cluster bootstrap B = 2000; paired MDE80 computed from the real fold-pooled SE.
- Controls: constant / shuffled at 0.5; raw latent AUC well above label-only (LDA14) AUC.

## Prediction and falsifier
- Predicted: Gamma(G3 held out) 0.04-0.08; D_pop within +-0.02 of D_single and of D_dither; held-out-finding utility of D_LDA14 drops by >= 0.05 AUC relative to raw, but D_dither loses a similar amount.
- FALSIFIED (G stays closed) if Gamma < 0.02 in all rotations or D_pop does not beat the envelope by >= 0.03 AUC on the held-out family in >= 3 of 4 rotations. Falsified the other way if it does, with CI excluding 0 and agreement across seeds.

## Cost / dependencies
~3 GPU-h (H-G-001 + H-G-002 <= 6 GPU-h in total, the FOCUS.md gate).

## Prior
0.15 (open). Reasons: programme 1 (adversary adds nothing beyond bottleneck / noise); the better power makes a real, small positive resolvable, which is why the prior is slightly above the IMU gate.

## Cheapest floor that could kill it
PCA-k or random projection of z at matched utility (a rate bottleneck without an adversary) and the single-strongest-attacker-with-restarts defender.
