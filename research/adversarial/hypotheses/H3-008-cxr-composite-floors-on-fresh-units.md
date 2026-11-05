# H3-008: Chest X-ray: composite and fine-grid floors built on the 4x downsample, scored on patients nobody has ever scored, move the floor envelope by at least 0.02 and keep every learned defender below it

- Focus: R-CXR | Kind: floors vs learned, with FRESH test units (the T2/T3 limit "same 4,588 patients" removed for floors) | GPU-h: 6 | Prior: 0.40 (envelope moves by >= 0.02), 0.70 (no image floor reaches L <= 0.75 at raw utility)
- Depends on: T3 (in flight; T.md section 8: full 37-floor pool, two seeds / two operating points / two checkpoints, rotations 0-1 of the t3 salt, results are "roles re-drawn over the same patients"); the frozen floor list (sha256 41c6b059...c01d38) stays untouched, this adds a SECOND list that must be hashed and committed before a custodian salt for the new bucket is committed (README order: floors first, salt second); the custodian salt `t4_cxr_salt.txt` (to be committed).

## Statement
T2/T3 compare one retrained PriCheXy-style defender with floors on 4,588 generator-unseen multi-visit patients that have already been sealed test units twice. Floors need no defender training, so the 8,714 multi-visit patients of the generator-SEEN role S are untouched test material for floors: no attacker has scored them, no floor was tuned on them. Carve a sealed bucket F (one hash bucket, about 2,900 multi-visit patients) out of S with a new salt. On F:
(a) at least one NEW floor (composite or fine-grid, list below) lowers worst-case linkage L below the T2 best floor `down_f4` (0.805 at utility 0.786 on T2 units) by >= 0.02 at equal utility (>= raw - 0.005), the comparison being made ON F against `down_f4` re-scored on F (not against the T2 number): L_new(F) <= L_down_f4(F) - 0.02;
(b) no image floor reaches L <= 0.75 at utility >= raw - 0.01 (the labels14 code floor at 0.69 stays more than 0.05 below every image floor);
(c) the rank order of the families by L at equal utility on F (down < jpeg < blur < quant) equals the rank order on T3's rotations (Spearman >= 0.9);
(d) T2's retrained defender `pcxr_mu0.01` (defender-SEEN on F, which can only flatter it) is above the new envelope by >= 0.03 in L.

## Mechanism
The six nominated T2 floors were chosen by closeness to the defender's validation leakage, so the envelope is a sparse set of points; composites of two cheap operations (smooth then subsample then quantise) are the obvious next floors and have never been screened. Downsampling removes the fine structure that linkage models use; JPEG at moderate quality does similar damage at a different rate; whether the damages add is an empirical question. Separately, an envelope found on units that were also used to nominate and select (T2: validation-only nomination, but the same patients across rotations and two attempts) is open to unit-reuse optimism; F removes that for the floors.

## Experiment (exact)
- Floor list v2 (frozen and hashed BEFORE the salt): (i) `down_fX` for X in {3, 4, 5, 6, 8, 12} with area down / bilinear up and, for X in {4, 6}, a Lanczos up variant; (ii) composites `down_fX__jpeg_qY` (X in {2, 3, 4}, Y in {10, 30, 50}), `down_fX__quantd_l{8,16}`, `blur_sS__down_f2` (S in {1, 2}); (iii) `down_f4__noise_sN` (N in {0.01, 0.02, 0.05}); (iv) ideal low-pass at cutoffs {16, 24, 32, 48} cycles per image (FFT mask, no resampling kernel); (v) re-listed from the frozen list as anchors: `raw`, `const` (analytic), `down_f2`, `down_f4`, `jpeg_q30`, `blur_s2`, `quant_l32`, `rw3_mu0.01`; code floor `labels14`. At most 45 releases.
- Roles: S2 = S \ F (for the utility classifier and any floor fitted on images, e.g. PCA), A = attacker-train role of the T3 partition (rotation 0), V = validation, test = F. N2 matched negatives built within F by T's `pairs.py` (same sex, view, age, dims, spacing flags); G0 (metadata-only) re-run on F before any image number.
- Screen (validation only): every floor on A and V pairs with F3 (frozen DINOv2 head + retrieval scorers) and the utility classifier. Nomination (committed before any F-look): choose floors by a target-leakage grid L_val in {0.70, 0.74, 0.78, 0.82} (two per target, at most two per family), so that the frontier is mapped and not just the neighbourhood of one defender.
- Sealed evaluation on F: nominated floors, `raw`, `const`, `labels14`, `pcxr_mu0.01` C1 (defender-seen sensitivity): F1 (match), F2, F3, F4, F5a / F5b, F6; worst case L inside each bootstrap resample; utility = DenseNet-121 fine-tuned 2 epochs on DEFENDED images of S2 only and scored on F (T4 must not reuse T2's utility classifier, which saw F).
- Selection bias of a designed floor: the best composite is selected on V from ~45; the sealed F estimate is the unbiased number; report the V-to-F shrinkage.
- Envelope rule: A2 rule (linear interpolation of the lower envelope in both axes, inside every patient resample), margins 0.02 (U) and 0.03 (L), 99% bounds when more than one defender or floor family is compared.
- Seeds vs test units: units = F patients (about 2,900); no rotation (one sealed bucket; stated), so seeds (attacker initialisations, 3 for F1 on `raw` and the two best floors, 1 otherwise) replicate training noise only; the patient-cluster bootstrap (B = 2000) covers test sampling. Paired MDE80 of an AUC difference is ESTIMATED 0.013-0.02 at n about 2,900 (T.md 3.9).

## Prediction and falsifier
- Predicted: best new floor L_F about 0.775-0.800 at U within 0.005 of raw; the composites `down_f3__jpeg_q50`, `down_f4__quantd_l16` and the ideal low-pass at 32 cycles are within 0.01 of one another; improvement over `down_f4` on F: +0.00 to +0.04 (point +0.015), so (a) has probability 0.40; (b) 0.70; (c) 0.80; (d) 0.85.
- FALSIFIED on (a) if no new floor beats `down_f4` by >= 0.02 with 99% lower bound > 0 (the T2 envelope was already tight). FALSIFIED on (b) if any image floor reaches L <= 0.75 at U >= raw - 0.01 (the envelope moves down 0.05: T2/T3 defender verdicts become weaker statements). FALSIFIED on (d) if `pcxr` is within 0.03 of the new envelope or below it even though it is advantaged by having seen F.

## Cost / dependencies
6 GPU-h: screen 45 floors x ~1.2 min (F3 only) = 1 h; sealed 10 floors x ~0.4 h = 4 h; utility 10 x 0.1 h = 1 h; no retraining. Needs the committed list-v2 hash then the salt; the 42 GB bank and the T2/T3 code are on the VM.

## Prior
0.40 for (a). Downsample-then-JPEG likely saturates near 0.78-0.80 at raw utility because T2's validation screen already showed jpeg_q30 (0.828), quant_l32 (0.843), blur_s2 (0.856) in one band above down_f4 (0.800); the question is whether adding them stacks. 0.70 for (b): the image-versus-code gap to labels14 (0.69) is 0.11 and no image release has crossed it.

## Cheapest floor that could kill it
For (a), `down_f4` itself. For (b), the labels14 code floor is the lower bound that shows (b) is about images only. The hypothesis is killed on (d) by nothing cheaper than the defender-seen advantage test itself.
