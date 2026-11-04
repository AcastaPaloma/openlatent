# FOCUS, programme 2 (locked 2026-11-04 ~17:35Z VM clock)

Basis: six scouts, a missed-directions contrarian (36 searches, ~25 directions, nothing beats the shortlist, ~88-90% confidence) and a stress test that measured power and data access
on the real artefacts (`literature/contrarian_stress_test.md`). Evidence in `literature/`. Change this file only with a journal entry saying why.
Numbers below come from scout reads (mostly abstracts) unless marked MEASURED by the stress test; anything quoted in a write-up must first be verified against the primary source.

## Focus R-CXR: re-audit chest X-ray linkage defences under a retrained attacker suite   [GO-WITH-CHANGES]
- Keep: diagnostic utility beyond the 14 labels (a label-only projection already holds those). Hide: patient linkage across visits.
- Data: ChestX-ray14 via the HF mirror (42 GB; MEASURED: the NIH Box link 404s and GCS 403s). 13,302 multi-visit patients, 1,424 in the official test split. PriCheXy-Net code and weights retrievable.
- Why it survives the power gate: n is large enough for a certificate to resolve programme-1-sized effects (MEASURED: power only at n >= 1,000).
- MANDATORY protocol changes, both from the stress test: (1) matched negatives, because metadata alone (age + sex) reaches AUC 0.983 on PriCheXy's own test pairs; (2) a demographic attacker
  in the suite, because an image-estimated demographic attacker simulates to ~0.885. Without both, any "linkage" number is a demographics number.
- Floors first (a learned defence counts only against the lower envelope): random smooth warp of equal strength, blur, downsample, PCA, noise/dither, rate bottleneck, label-only projection.
- First benchmark: replicate PriCheXy-Net (81.8 -> 57.7 AUC) under its own protocol and under ours; report both. ~35 GPU-h.
- Scoop note: LAPIS (arXiv 2610.40722, 29 Oct 2026, unreviewed, no code) partially scooped the rescoring, without floors and without a held-out family; that gap is ours.
- Kill: if the floor envelope matches every learned defence at matched linkage AUC over 3 seeds with cross-fit, or if the demographic attacker explains the whole signal.

## Focus R-IMU: re-audit wearable IMU obfuscators under a retrained attacker suite   [GO-WITH-CHANGES]
- Keep: activity recognition. Hide: user identity (NOT gender: MEASURED, PrivDiffuser's gender split is window-level over 24 users and cannot carry a claim).
- Data: MotionSense (MEASURED: only 24 subjects, 144 test trials) plus a second CC-BY/MIT set with user IDs for replication. PrivDiffuser ships a checkpoint.
- MEASURED correction to the scouts: de-meaning, magnitude-only and 16x downsampling are NOT floors here (a random forest keeps 55-78% identity). The floors that bite are activity-LDA codes.
- First benchmark: PrivDiffuser on MotionSense under its own protocol and under a retrained suite (linear, MLP, kNN, LightGBM, random forest, a sequence model). ~10 GPU-h.
- Kill: if the activity-LDA floor matches the learned obfuscators at matched identity accuracy, or if 24 subjects cannot resolve the margin (power audit first, zero GPU).

## Instrument S (inside R, not a focus): sealed suite + certificate   [GO as a 2-day, zero-GPU instrument]
- MEASURED: as a standalone focus it has no power on our HAR data (KTH has 150 independent recordings; paired MDE80 5-8 points against programme-1 margins of 2-3 points; the certificate at zero
  leak reads chance +3.7 points at K=1 and +7.1 at K=30). LLM-Attack Suite, SynthGuard-ReleaseBench and BI already hold the sealed-audit framing.
- So S is built as the evaluator for R-CXR and R-IMU: pre-declared attacker list, validation-selected attacker scored on a sealed split, cluster bootstrap over patients/subjects, a Bonferroni-style
  upper bound over the suite, the smallest resolvable effect printed next to every difference, planted-leak calibration (a dialled known leak) and a shuffled-label null for the whole search loop.
- Includes the "evidence budget" module from the contrarian (26/30, judged a module not a focus): pool K clips/windows/images per person and report the K needed to reach a set identification rate.
  MEASURED input: the HAR harness `preds.npz` files DO hold per-clip probabilities (78 files), so the pooling study needs no reruns (this corrects an earlier note of mine that said predictions were absent).

## Focus G: defender trained on an attacker population, held-out family   [CONDITIONAL third, gate <= 6 GPU-h]
- The charter's second falsifiable clause, and the learned-method core. MEASURED: its MDE (7-8 points) currently exceeds the expected effect, so it does not start.
- Gate before any population training: on R-IMU or R-CXR, show that a leave-one-attacker-family-out gap exists at all (train a defender against families 1..n-1, test on family n, against a
  single-strongest-attacker-with-restarts control and a rate/noise floor at matched leakage). If the gap is inside the MDE, G stays closed.

## Standing rules (inherited, plus the audit additions in CHARTER.md)
Pre-register before any number; floors first with the strongest trivial floor named; headroom/oracle check and power audit before fixing a margin; attackers retrained on the released
representation with a held-out family and worst case reported; cross-fit or rotating test units so seeds replicate test noise; select on validation, never the sealed split; controls at chance;
no identity attack on data whose terms forbid identifying individuals; kill criteria tightened only with the original verdict alongside, never loosened.
