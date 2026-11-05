# H3-013: Boring control that could kill the headline "identity linkage beyond demographics is large": persistent diagnoses (not anatomy) may carry the linkage and the labels14 floor; test inside No-Finding pairs and with label-matched negatives

- Focus: R-CXR | Kind: boring control / confound audit | GPU-h: 1.5 (arm A is zero-GPU) | Prior: 0.80 that raw linkage survives (>= 0.85) in the No-Finding stratum and under label-matched negatives
- Depends on: T2 / T3 sealed score files (`scores_*.json` keyed by hashed patient id; the builder holds the id-to-image map), the ChestX-ray14 CSV (`Finding Labels`), the T3 partition. Arm A needs no new training; arm B retrains attackers on a new pair design for four releases.

## Statement
T's matched negatives (N2) control sex, view, age, dimensions and spacing flags, but not the 14-label vector. Patients carry chronic findings (cardiomegaly, pacemaker leads, surgical clips, emphysema, nodules, fibrosis) that persist across visits. The labels14 CODE floor (14 logits) links at 0.690-0.770 versus raw 0.887-0.891, which T reads as "the label-only floor for any 14-finding utility claim". If persistent pathology drives that floor and part of the image-level linkage, then (i) the floor is not an anatomy-free baseline and (ii) the defender-versus-floor verdicts concern a mixture of anatomical identity and diagnosis persistence.
Claims:
(A) Stratifying existing T2 / T3 sealed pairs by pair type (both images No Finding, one No Finding, neither), within-stratum worst-case AUC: raw L_NF-NF >= 0.85 and the labels14 code floor L_NF-NF <= 0.58; in the neither-NF stratum the labels14 floor is >= 0.78 and raw >= 0.90; `down_f4` lowers L in NF-NF by at least as much as overall (>= 0.06).
(B) With negatives matched to the positive partner on the full 14-label vector (N3: exact label-vector match, else Hamming <= 1, else dropped and counted), retrained attackers give raw L(N3) >= 0.85 (a drop of <= 0.04 from N2), labels14 L(N3) <= 0.58, `down_f4` still below raw by >= 0.06 and `pcxr_mu0.01` still above `down_f4` by >= 0.03.

## Mechanism
Same-patient visits share slowly varying anatomy (thoracic cage, heart shape, clavicles, spine curvature) and persistent pathology. Metadata matching removes demographic correlates but a predicted-label vector is a compressed pathology fingerprint. The control separates the two channels: NF-NF pairs and label-matched negatives close the pathology channel, leaving anatomy. If raw linkage barely moves, the programme's "identity beyond demographics" is anatomical and the labels14 floor is a pathology-persistence artefact (still a valid floor for a 14-finding utility claim, but an upper bound on privacy only for pathology-preserving releases). If raw linkage collapses toward the labels14 value, the T programme measured diagnosis persistence.

## Experiment (exact)
- Arm A (zero GPU, ~0.5 CPU-h): from T2 sealed rotations 0-2 and T3 rotations 0-1 score files, attach for each pair the NIH labels of the two images (mapping inside the VM). Pair types: NF-NF, NF-F, F-F, where "F" = at least one of the 14 findings. For positives and negatives separately by their own labels; AUC within each pair type; pooled across rotations with patient-cluster bootstrap. Releases: `raw`, `labels14`, `down_f4`, `down_f2`, `rw3_mu0.01`, `pcxr_mu0.01`, plus the demographic-feature attacker F5. Report n_pos per stratum and the paired MDE80 (ESTIMATED 0.03 for NF-NF at about 1,100-1,400 positives).
- Arm B (1.5 GPU-h): new pair design N3 on the T3 partition, rotation 0 and 1: for each positive pair the negative's partner is drawn from the same role with the SAME 14-label vector as the positive's partner, same sex / view / age (+-1) / dimension flags (N2 rules otherwise); positives with no label-vector match within Hamming 1 are dropped and counted (expected loss: rare multi-label vectors). Retrain F1 (match, 1 seed), F3 best, F5 for `raw`, `down_f4`, `pcxr_mu0.01` (T3 C1), `labels14` (F3 heads only), on A-role triples built with the same N3 rule (so the attacker does not learn a label-mismatch shortcut), test on T pairs.
- Controls: G0 metadata GBM re-run on N3 (<= 0.56); `const` at chance; the labels14 code on N3 must fall (if not, N3 failed to close the pathology channel).
- Seeds vs test units: units = the patients of T3's rotations (3,060) and T2's 4,585 (arm A pools whatever exists, labelled by source); arm B: 1 seed for F1, 3 for F3 heads; bootstrap B = 2000 patients; the units overlap T2/T3 (stated; this is a stratification of existing units, not fresh data).

## Prediction and falsifier
- Predicted: A: raw L_NF-NF 0.84-0.88, labels14 L_NF-NF 0.52-0.60, NF-NF share of positives 25-40%; B: raw 0.84-0.88, labels14 0.52-0.58, G0 on N3 <= 0.55, 10-25% of positives dropped.
- FALSIFIED (the headline needs rewording) if raw L_NF-NF < 0.80 or raw L(N3) < 0.80 (more than a third of the excess over chance was diagnosis); or if labels14 L_NF-NF >= 0.65 (the logits carry anatomical / habitus information beyond the label) which would break the reading "labels14 = pathology persistence". Either outcome changes how the labels14 floor and the `down_f4` verdict are worded.

## Cost / dependencies
Arm A 0.5 CPU-h. Arm B 1.5 GPU-h (4 releases x 2 rotations x ~12 min with F1 + F3 + F5 at 1 seed). No new download.

## Prior
0.80 for survival. Reasons: raw `core_w24` (frame removed) links at 0.878 and the different-detector stratum at 0.860, so the signal is not an acquisition artefact; labels14 at 0.69 is well below raw, so pathology persistence explains at most the part that raw and labels14 share. The risk is that NF-NF pairs are younger / healthier and thus differ in distribution, which the stratum-wise AUC does not fully remove; reported.

## Cheapest floor that could kill it
The labels14 code floor and the TRUE-label vector itself (a 14-bit release: if the true labels link at 0.68, T's floor is the labels, not the logits).
