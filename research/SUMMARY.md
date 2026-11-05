# OpenLatent research summary, 2026-11-04 to 2026-11-05 (programmes 1 and 2)

Scope: what was measured on one rented 4x RTX A6000 VM in about 20 hours, by a main agent plus about 15 builder/scout/auditor subagents. Every figure below is traceable to
`research/fields/FOCUS.md`, `research/adversarial/FOCUS.md` (outcome blocks) and the per-focus notes under `results_notes/`. Two independent audits
were run and their corrections are applied: Auditor 1 (`research/fields/AUDIT_findings.md`) re-derived programme 1's six headline numbers from raw result files; Auditor 2 (`research/adversarial/AUDIT_findings.md`) checked programme 2's literature claims, licences and design before any programme-2 number existed. Programme-2 result numbers were NOT independently re-derived. Single seeds and
development splits are labelled in the source blocks and are repeated here only where they matter. Nothing here is a claim about all methods or all datasets.

## What was asked, in one line each
- Programme 1: find other fields where a learned mask/latent that keeps a task signal and removes another property is useful and verifiably benchmarkable, then test learned methods there.
- Programme 2: build a robust suite of adversarial models (attackers) and use it to re-audit privacy defences for human activity recognition (wearable IMU) and CT-adjacent medical imaging (chest X-ray).

## Programme 1 (learned adapters that keep a label and hide a source, five fields)
| field | result | strength |
|---|---|---|
| Pathology centre invariance (A) | no learned adapter beats the lower envelope of cheap floors at matched leakage on held-class label accuracy; cross-fit midnight/TCGA -0.028 [-0.063, +0.001], midnight/Tolkach +0.003 [-0.007, +0.012]; kNN-retention positive on midnight/Tolkach does not clear the frontier selection null | resolved on Tolkach (ceiling-limited), bounded on TCGA |
| Identity-free emotion on CREMA-D (C) | no label-supervised or task-agnostic learned encoder beats a label-only linear projection or hand-built prosody floors at matched leakage (held-out-emotion criterion fails, 3 seeds, trimmed variant too); full-clip identity is largely a recording-channel signal (non-speech alone 56x chance) | strong |
| Retrieval privacy (D) | pre-registered headroom check fails at the oracle's own retention; the repaired adversarial arm trains and fails the learned kill test at 0.90/0.97 retention (2 seeds) | gate + 2-seed learned null |
| Single-cell batch integration (E) | inductive encoders sit -0.037 [-0.045, -0.026] below the inductive floor envelope at the scVI point (MDE80 0.020); a post-hoc fill config closes about a fifth of the gap (+0.016) | resolved null for the encoder family tried |
| JUMP Cell Painting (F) | literal verdicts NO-GO (gate clauses judged design flaws). F3 reported a supervised nonlinear adapter beating the best same-label floor by +0.057 to +0.137, BUT THE H3-007 STAGE-0 CONTROL (CPU, metadata only) FOUND THE RETRIEVAL PROTOCOL REWARDS SHARED PLATE LAYOUT: 28% of cross-source positive well pairs share the well position (chance 0.31%), and a 3-number position-only code with no profile information scores U_H 0.324, equal to the best adapter (0.317-0.321) and +0.13 over the best linear floor; 3.7% of held-out compounds have near-duplicate training neighbours. STATUS: F3/F4 positive is NOT INTERPRETABLE until stage 1 (position-restricted positives, position-matched negatives, near-duplicate pruning, source-level negatives) reports | OPEN; the earlier 'positive' is withdrawn as a headline |
| Robotics latent actions (B') | the learned arm is no better than zero-shot SAM masks across three correlated-distractor kinds (SAM minus learned -0.003 to -0.051 on two, about equal on the third; rule needs +0.05); the harm is input clutter, not leakage | bounded, one learned arm, synthetic |

By-products that held up: closed-form linear erasure fitted on seen units fails on unseen ones (0 of 8 pathology cells); cheap supervised linear codes (LDA/PCA) are unexpectedly strong floors for "keep the label, hide the source";
the claim "the adversary adds nothing beyond a bottleneck" was RETRACTED where adversaries had not trained (pathology, early retrieval arm) and is supported only for the repaired retrieval arm.

## Programme 2 (robust attacker suites and re-audit of published defences)
- Wearable IMU (burned split of the shipped checkpoints, not sealed): the anonymising autoencoder's published "<7% identity at >92% activity" reproduces under its own frozen attacker (4.07% vs 6.98%) but a retrained suite re-identifies 83% of held-out trials (msda 0.826 [.733, .906]; rep 0.878); a label-supervised 3-dimension code beats it on both leakage (0.365) and utility. NO-GO-MATCHED.
- Chest X-ray (sealed rotations for the comparison; literal verdicts NO-GO via gate clauses): linkage beyond demographics is large (raw AUC 0.885 vs metadata floor 0.528); the released defence does not replicate its published 57.7 under its own protocol (67.0 +- 3.9); a retrained defender (one operating point, mostly the pretrained generator) lowers worst-case linkage by 0.027 but plain 4x downsampling is better (0.805 vs 0.860 at equal utility). Post-hoc sensitivity, one seed.
- Instrument (sealed attacker suite + certificate): trustworthy as a leak detector for leak types it was built or retrained to see (0 false alarms in 224 and 106 nulls); misses unseen types (window-order dependence 0/20, phase-amplitude coupling 7/20); therefore a clean certificate is not evidence of no leak. Programme 1's HAR method rankings are mostly unresolved at 150-recording test sets (resolves differences of >= 5-8 points).

## Cross-cutting findings
1. A published privacy figure depended on how the attacker was trained in every re-measurable case (frozen/mismatched vs retrained/matched).
2. Cheap floors (label-supervised linear codes, downsampling, noise/dither, per-source normalisation where valid) matched or beat most learned adapters at equal utility; the one candidate exception (a supervised nonlinear adapter with enough labels on JUMP) is under a position-confound cloud (see the JUMP row) and is not claimed.
3. Power, not effort, limited most conclusions: several comparisons could only resolve differences of 0.05 to 0.14.
4. Dataset confounds are large (CREMA-D channel signal, chest X-ray acquisition cues, MotionSense non-gait identity); every benchmark needed its own floor and shortcut controls.

## Process record (what to trust)
Pre-registration before numbers was followed with disclosed exceptions: Focus A criterion tightened after the gate; JUMP and chest X-ray gate clauses relaxed only as labelled post-hoc deviations (literal verdicts kept); one sealed-evaluation start before the nomination table (killed after ~5 s, no score computed, ledger entry consumed under an amendment);
several clock/label errors and a swapped-agent message of mine corrected in the journals; an unsupported "adversary adds nothing" claim retracted. Weights/derived artefacts are on HuggingFace `KuanKuanKuan/openlatent-fields` (all successful runs verified uploaded, chest X-ray jobs limited to JSON/log/md); code, logs and result JSON are on GitHub branch `fields-research`.

## Limits
Frozen-feature adapters and a few encoders; small unit counts on several test sets; one synthetic robotics set; one retrained chest X-ray defender; the chest X-ray bank cannot be checked against an official copy; programme-2 sealed folds were used only for the T2 comparison; EEG and astronomy candidates were never run; the label-free adversary question is untested where adversaries did not converge.

## Open items
Label-free learning on JUMP beyond the metadata-only arm; a converged adversary scored on held-out units; more chest X-ray defenders and floors (blur, quantisation, JPEG, PCA); a sealed retrained wearable defender and WISDM replication; PII-Masking-300K licence decision (not used).
