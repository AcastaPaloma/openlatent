# H-CXR-004: A retrained, multi-family suite with a held-out non-learned family raises PriCheXy-Net's measured linkage by >= 0.08 AUC over its own attacker and flips defence rankings; a defence-agnostic pretrained attacker works as a fixed-code member

- Focus: R-CXR | Kind: attacker-side re-audit / held-out family | GPU-h: ~12 (4 learned families x 7 releases x ~0.4 at 256 px, plus the defence-agnostic pretraining ~2) | Prior: 0.75
- Depends on: H-CXR-001 (demographic attacker), H-CXR-003 releases (PriCheXy mu levels + floors), download gate. Literature anchor (scout-read, to be verified against the PDF): LAPIS 2610.40722 reports PriCheXy-Net worst-of-3 retrained cross-study AUC 0.826 vs 0.577 in its own paper, protocols differ; scoop note in FOCUS.md.

## Statement
On the releases of H-CXR-003 (PriCheXy mu = 0.01, equal-strength random flow, blur sigma 4, downsample x4, PCA-64, raw), with the family split fixed in advance:
- P1 (primary): worst-over-suite AUC minus the AUC of PriCheXy's own attacker (retrained Siamese-R50) is >= +0.08 on PriCheXy mu = 0.01, under both random negatives (predicted +0.15 to +0.25) and MN-v1 (predicted +0.05 to +0.12).
- P2: the HELD-OUT family alone (registered NCC / SSIM, not used for validation selection or any defender training) reaches >= 0.80 on random negatives and >= 0.70 on MN-v1 on PriCheXy mu = 0.01.
- P3: ranking of {PriCheXy mu = 0.01, random flow, blur, downsample, PCA} flips between the Siamese-R50 ranking and the NCC / SSIM ranking (Kendall tau <= 0.5).
- P4: a defence-agnostic Siamese pretrained only on random obfuscations of the TRAIN patients (random flow, blur, downsample, JPEG, noise at random strengths; never sees PriCheXy or the evaluated floors) is within 0.05 AUC of the defence-retrained Siamese on >= 4 of 6 releases zero-shot, and within 0.03 with 10% fine-tuning labels, i.e. it is usable as a fixed-code suite member without per-defence retraining.

## Mechanism
In-training verifiers at chance do not predict a fresh attacker (Elazar and Goldberg), and a non-learned matcher uses the matching principle the defence never trained against: a smooth warp bounded by mu leaves landmark topology and global shape that registration can undo, while a CNN verifier trained on deformed images may lose to it or beat it depending on the ranking. LAPIS's structural attacker flips the ranking of PriCheXy vs LAPIS (22.4% vs 34.7% Rank-1 against AUC_cross).

## Experiment (exact)
- Families (fixed before any number; selection of the reported "worst" on validation patients only): F1 Siamese-R50 (own attacker); F2 ConvNeXt-T verifier; F3 ViT-S verifier; F4 frozen DINOv2-L (Apache-2.0) kNN with cosine / L2 / Canberra + InfoNCE head (RAD-DINO optional but its pretraining set includes NIH-CXR, so its unsupervised exposure to our test images is a leak to disclose); F5 registered NCC / SSIM with affine registration, non-learned (HELD-OUT); F6 demographic + device GBM (H-CXR-001); F7 defence-agnostic pretrained Siamese (P4); F8 acquisition fingerprint (H-CXR-005).
- Metrics: AUC and Rank-1 / Rank-5 (1,424-patient gallery); worst-over-suite on the sealed test via validation-selected member; naive max for comparison; leave-one-family-out table; patient-cluster bootstrap B = 2000; normalisation L = (AUC - 0.5) / (AUC_raw - 0.5).
- Test-unit design: PriCheXy arm = fixed 1,424 official-test patients, floors rotate over 5 patient folds; seeds (3 per learned member) replicate training noise only for the PriCheXy arm (stated); paired MDE80 ~0.02 AUC, Rank-1 ~3.2 points (ANALYTIC).
- Controls: constant and shuffled -> 0.5; raw above MN-v1 metadata; F5 on a pair of identical images -> AUC ~1 (sanity of the registration).

## Prediction and falsifier
- Predicted: P1 +0.15 to +0.25 (random negs) and +0.05 to +0.12 (MN-v1); P2 NCC / SSIM 0.80-0.90 random, 0.65-0.78 MN-v1 for mu = 0.01; P3 tau in [-0.2, 0.5]; P4 zero-shot gap 0.03-0.08.
- FALSIFIED if: P1 < +0.03 on both protocols (the single-attacker evaluation was adequate for CXR: LAPIS's gap is a protocol artefact), or the held-out family never exceeds the in-suite worst case by more than MDE80 and the ranking is identical (tau >= 0.8). P4 falsified if zero-shot gap > 0.10 on >= 4 of 6 releases (defence-specific retraining is necessary; the pretrained member is not a fixed-code option).

## Cost / dependencies
~12 GPU-h; shares fits with H-S-002 / H-S-004.

## Prior
0.75 for P1 (one paper from another group already moved 0.577 to 0.826, but under a different split); P2-P4 are lower (0.5-0.6) and carry the novelty.

## Cheapest floor that could kill it
The demographic + device line (F6): on random negatives it already exceeds every image attacker; if the "worst case" is always F6 the suite adds nothing beyond metadata and all differences vanish under MN-v1.
