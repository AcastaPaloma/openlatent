# N: negative-control calibration of Instrument S on real defended IMU releases. Pre-registration (v1) and results

Builder-N (negative-control builder, FOCUS amendments A3/A6, audit F10), programme 2, branch `fields-research`. Status: DONE (pre-registration 8f11607; manifest hash 67b6678; run 2026-11-04; verdict MIXED, section 8). Code: `research/adversarial/src/n_control/` (CPU only).
Independent of Builder S: I did not write the instrument, I do not edit it, and I grade it on leaks it was not built to find.
This file is committed BEFORE any certificate exists. Section 5 (criteria) is never edited except by a dated amendment that tightens a criterion; both verdicts are then reported.
Results are appended under section 8 in later commits. Single-seed statements are labelled "not yet replicated".

## 0. Amendments log
- (none yet)

## 1. Question, instrument under test, what is NOT tested
Question: does the frozen S certificate detect identity leaks planted into REAL defended releases when the leak is of a type S did not build for, and what does it print
about leaks it cannot read? Instrument under test, unchanged (git tag `s-suite-eval-v1` -> `6cfc65e`; evaluator sha256 `eb2d619870c2811430aee56376a306dc71dc930de5bc187e00d9a6cdfb527e0f`, printed in every certificate):
- attackers: `s_suite.attackers.SUITE12` (lr_raw_c0.1, lda_raw_auto, knn5_raw, rf_raw_100, lr_feat_c1, lda_feat_auto, knn5_feat, rf_feat_200, et_feat_200, lgbm_feat, mlp_feat_64, knn1_raw), retrained on the
  release's train split via `fit_predict_pool`; views `raw` (flattened window) and `feat` (S's `planted.features`: per-channel mean, std, mean |diff|, log energy of 8 two-bin bands over FFT bins 1..16).
- certificate: `s_suite.certificate.certify` with `SuiteDecl("n-motionsense-suite12", SUITE12, accuracy, alpha 0.05, n_boot 2000, seed 0, margin 0.02, n_classes 24)`; this is S's planted-engine declaration with
  n_classes = 24 (declaration digest printed by the runner). `val` is used only for the choice of the reported attacker, `test` is scored once. Units = sessions. Query through S's sealed-query ledger (one query per release digest).
- NOT tested: S's programme-1 re-score (E1), selection-bias studies, the evidence-budget module, or Builder T's CXR evaluator (a different suite built by view). A result here says nothing about them.
- Chosen NOT to give S any extra views: S declared its suite and features; the suite is graded as declared. An amended suite (new tag) needs a NEW manifest (section 7).

## 2. Design
**Base releases (real defended outputs).** Builder H's MotionSense releases of the Anonymizing AutoEncoder (`msda`, IoTDI'19; published identity claim < 7%) and the replacement AE (`rep`, W-P2DS'18; published 15.9%):
`~/work/h_imu/out/aae_rel/{msda,rep}_s0.npz` on the VM, sha256 `5d0fd79436782b9052cf96f77122af483cfe7cf9a6f474ea2a3c5ffc9979ce10` (msda) and `4ec1fda04b7210a1dccfd02ccb1bbcf7ecb1ceae72b8b6d2347a265c44c519bc` (rep); H's window builder
`h_imu/common.py` sha256 `79d8721dc7a85cde1e12b4dfb613327bd4806c6d126565018763b9e344e75394`. Windows: H's analysis-set windows with stride 40, at most 60 evenly spaced windows per session -> 12,621 windows of shape (2, 128)
(channel 0 = |rotationRate|, channel 1 = |userAcceleration + gravity|, standardised by H), 264 sessions = 24 real subjects x 11 (activity, trial) codes. The PrivDiffuser release and programme-1's video releases are NOT used:
the former was not available at build time (its leak target is gender anyway, FOCUS A1), the latter exist here only as predictions, and a signal-level leak cannot be planted into predictions.
**Pseudo-identity.** The secret of a release is a PSEUDO-identity in 0..23 assigned at random to SESSIONS, so the unplanted base carries no identity signal and the planted leak has an exactly known ground truth. This is the price of exact truth:
the labels are synthetic, the windows are real. Real subjects are partitioned over splits (12 train, 4 val, 8 test subjects), so no real-identity cue crosses a split; within a split pseudo-identities are dealt out in a random cycle by activity
(every pseudo-identity has the same number of sessions +-1; test units = 88 sessions, val 44, train 132).
**Seeds versus test units.** Every release draws its own subject partition, pseudo-identity assignment and leak key, so replicates re-draw the test units and the labelling; they share the two base releases (one defence configuration, one generation seed each),
so replicate variance is NOT base-window variance. Stated as a limitation, not hidden.

**Leak families** (all are additive and keyed; identity-dependent parameters come from a 128-bit key). Classification into SEEN / UNSEEN is relative to Builder S's declared views and planted types (S planted bias, gain and an in-band spectral line):
| family | carries the identity in | class | why |
|---|---|---|---|
| `bias` | constant offset on the two channel means | SEEN mechanism | S's mean feature reads it (S's own `bias`) |
| `specin` | a tone at an identity-specific (bin 3..14, channel) with random phase | SEEN mechanism | S's band-energy features read it at 2-bin resolution (S's own `spectral`) |
| `xphase` | the phase lag between the two channels at two fixed bins; energy identical for all identities | UNSEEN | S's features are per channel; the cross term is not a feature, and linear/kNN/RF on raw windows cannot form it |
| `ampfreq` | tone position as a Latin function of the window level (duodecile of the channel-0 mean): a nonlinear amplitude-frequency interaction; marginals over windows nearly identical across identities (a marginal-only reader is capped at 1/12) | UNSEEN, readable in principle by the nonlinear learners | needs mean x band-energy interaction |
| `stealth` | a tone at an identity-specific bin in 40..63 (S reads bins 1..16), first-difference power equalised across identities so std and mean|diff| carry no identity to first order, random phase | UNSEEN | declared hand features do not read the band; random phase defeats raw linear/kNN |
| `pooled` | the `specin` mechanism at doses calibrated on the SESSION-pooled rate | regime UNSEEN | weak per window, strong once a session's independent-phase windows are pooled; the certificate scores per-window top-1 |
Dropped from the first draft: a per-session constant (`sessoff`): a bias offset does not accumulate over a session's windows because the window means share a large session nuisance, so pooling does not amplify it (measured with the oracle); it would only have duplicated `bias`.

**Ground truth: key-aware matched filter ("oracle").** For each release the oracle uses the key, the dose and the base release's nuisance statistics (per-activity mean power per bin and channel; per-activity window-mean covariance for `bias`), and scores every window with the
log-likelihood of each of the 24 identities under a complex-Gaussian noise model (noncentral tone likelihood with random phase; the phase-lag likelihood for `xphase`; Gaussian window mean for `bias`). It never trains. Omega_w = its window-level top-1 on the release's test windows,
Omega_s = its session-pooled top-1 (sum of window log-likelihoods; session-mean Gaussian for `bias`). It is a legitimate classifier, so Omega is a LOWER bound on the Bayes rate: the planted level is a floor, and S may legitimately exceed it. Chance is 1/24 = 0.0417.
**Dose ladder.** Per (base, family) the tone/offset amplitude is calibrated (oracle only, own keys and seeds) so that mean Omega_w = 0.08, 0.15, 0.30, 0.60, 0.90 (levels 1..5); for `pooled`, mean Omega_s = 0.20, 0.40, 0.70, 0.90, 0.99. Realised Omega of every release (it varies with the key) is the x-axis of the analysis, not the target.
**Counts.** plant: 2 bases x 6 families x 5 levels x 8 replicates = 480; null (unplanted, random pseudo-identities): 2 x 40 = 80; shuffled twins (planted releases whose labels are unit-permuted within each split, levels 2, 4, 5 of every family): 2 x 6 x 3 x 4 = 144; total 704 releases.
The null and the shuffled twins are the shuffled-label null for the WHOLE search loop (12 retrained attackers, validation selection, bootstrap bound): every one of them goes through the identical pipeline.
**Sealing.** (1) the manifest (rid, kind, family, level, base, amp, replicate, assignment seed, key, shuffle seed; plus the calibration table and the base-file hashes) is written as canonical JSON on the VM at `~/sealed_manifest/manifest.json`, mode 0400, outside the repo;
its SHA-256 is committed in section 9 BEFORE any release is generated. (2) releases are generated deterministically from the manifest and named by their digest (sha256 of windows, labels, unit ids and split); the sha256 of the sorted digest list is committed BEFORE any certificate is run.
The digest -> manifest map and the oracle report are written at generation time to `~/sealed_manifest/` (mode 0400) and not read until unsealing. (3) the certificate runs in a clean clone at a commit descending from the tag, the ledger directory `~/sealed_ledger_N/` is outside the checkout.
Honest limit: every agent on the VM runs as the same unix user, so mode 0400 and the ledger location make tampering DETECTABLE (hash commitment, `s_suite.ledger` check), not impossible. The generator code and the family definitions in this file are public by necessity; what is sealed is which release is which family at which dose.
At unsealing the manifest is published (`research/adversarial/results/N/manifest.json`), and anyone can check its sha256 against section 9.

## 3. Reference attackers (readability evidence, not part of the verdict)
`n_control/reference.py` trains key-free logistic regressions on the train split with four views (log power bins 1..64 of both channels; cos/sin of the inter-channel phase lag; one-hot window level x log power of bins 3..14; channel means).
Purpose: show that a leak S misses is learnable from data by SOME attacker with the right view, so a miss belongs to the suite's views and not to the data. They are my own attackers, not S's, and are reported beside S's results.

## 4. Disclosures: what I saw before this pre-registration
1. I read CHARTER, FOCUS (A1-A6), AUDIT F8-F10, S.md, `s_suite/` (including `planted.py`, `attackers.py`, `certificate.py`, `ledger.py`) and `literature/leakage_measurement.md`. I know S's views and planted types; that is what "unseen" is defined against.
2. Design iterations on the ORACLE only (no S code was run on any planted or real-base data): four dose-calibration sweeps on the two bases. Changes they caused: (a) `ampfreq` redesigned (first version used the window std as the level, which a large tone corrupts; the oracle collapsed at high dose);
   (b) `sessoff` dropped and `pooled` introduced (see section 2); (c) the oracle's noise model changed from the window's own neighbouring bins to the base release's per-activity reference spectrum with a per-window gain (the neighbourhood model mis-handled gait harmonics: specin at amplitude 0.1 reached Omega_w 0.09 vs 0.33);
   (d) a label-free per-hypothesis recentring of the log-likelihoods was tried and rejected (it lowered the saturation level). `stealth` bins moved to 40..63 with equalised first-difference power so that std and mean|diff| carry no identity to first order.
3. One pass of the reference attackers on 18 development releases (msda base, calibration-namespace keys, levels 2/4/5; the run was killed at 18 of 36 jobs after a CPU-cap violation of mine). Seen: each family is read by its matching view and by no other (xphase 0.33-0.59 window top-1 by the phase-lag view only; ampfreq 0.28-0.56 by the level x spectrum view only, 0.04-0.05 by the marginal log-spectrum;
   stealth 0.39-0.77 and specin 0.46-0.83 by the log-spectrum; bias 0.39-0.57 by the means; null views at chance 0.03-0.05).
4. A timing run of `fit_predict_pool(SUITE12)` on iid Gaussian windows with random labels (6,500/2,500/3,500 windows): 67 s on one core, test accuracies 0.035-0.047 (chance). No certificate was computed on it.
5. CPU incident (mine): that reference run set its thread caps after importing numpy, so they had no effect (6 processes at 56 threads); the main agent confined it with taskset and I killed it. All later runs set the caps on the command line, at most 8 worker processes, and the certificate run is a queued `r_` job.

## 5. Pre-registered criteria (coded in `analyse.py`; thresholds are constants at its top)
Definitions. A release is detected iff the certificate verdict is `LEAK`; it is assured iff `BELOW_MARGIN`; else `NOT_RESOLVED`. B = the certificate's `baseline_used`. Omega_w, Omega_s as in section 2 (planted labels).
Level 4 and 5 pooled over both bases and all replicates (n = 32 per family) is the PRIMARY level (planted window top-1 about 0.60 and 0.90; for `pooled` session-pooled 0.90 and 0.99). Intervals are two-sided 95% Clopper-Pearson.
- **G1 (false alarm of the whole search loop).** Pool the 80 null releases and the 144 shuffled twins. G1 passes iff the LEAK rate <= 0.05 and the one-sided 95% Clopper-Pearson upper bound <= 0.10. Null and twins are also reported separately.
- **G2 (power on mechanisms S planted itself).** `bias` and `specin` are each DETECTED at the primary level. If G1 or G2 fails the verdict is INCONCLUSIVE (the design cannot grade the unseen families), never NEGATIVE.
- **Per-family status at the primary level.** DETECTED iff detection rate >= 0.80 and the CP lower bound >= 0.60 (with n = 32: at least 26 of 32); MISSED iff detection rate < 0.50; otherwise PARTIAL.
- **Verdict on "detects leaks it was not built to find"** (unseen = xphase, ampfreq, stealth): YES iff at least 2 of 3 are DETECTED and, for all three, the assured (BELOW_MARGIN) rate at the primary level is <= 0.10. NO iff at least 2 of 3 are MISSED. MIXED otherwise.
  `pooled` is reported as a secondary regime endpoint with the same status rule evaluated on Omega_s.
- **Minimum detectable dose (MDD) per family.** The lowest level L such that every level >= L has a detection rate >= 0.80 (point estimates; n = 16 per cell); reported with the mean Omega_w and Omega_s of that level, and "not reached" if level 5 fails.
- **False assurance.** Rate of BELOW_MARGIN at the primary level, per family. A family with rate > 0.10 means the printed BELOW_MARGIN cannot be read as "no leak" for that mechanism.
- **Miss rate for unseen families** = 1 - detection rate at the primary level, with CP interval. Also by level (all five) and by base (msda, rep).
- **Reported versus planted.** Per family and level: mean selected-attacker accuracy, B, mean Omega_w, and the attainment (selected - B) / (Omega_w - B). Printed MDE honesty (secondary): among releases with Omega_w - B >= the printed `excess_mde80`, the detection rate per family; S's printed MDE is honest for a family iff it is >= 0.80.
- Shuffled twins: detection per family and level must collapse to the G1 level; the released-label oracle on twins must sit at chance (checked, reported).
Primary endpoints: G1, G2, the three unseen-family statuses, the verdict. Everything else (levels below 4, per-base cells, attainment, which S attacker carries the signal, reference attackers) is exploratory.

## 6. Predictions written before any certificate (so that surprises are visible)
bias DETECTED (0.90), specin DETECTED (0.80; S's band resolution caps it near 50% window top-1, which is still far above chance), xphase MISSED (0.85), stealth MISSED (0.80), ampfreq MISSED or PARTIAL (P(MISSED) 0.55; the tree/MLP attackers can in principle read mean x band energy but need many windows per cell),
pooled: not detected at levels 1-2, DETECTED at 4-5 (0.70). G1 passes (0.80; the cluster bootstrap on 88 units is the risk). Verdict: NO 0.50, MIXED 0.25, INCONCLUSIVE 0.20, YES 0.05.

## 7. Run plan, stop rules, what counts as a deviation
1. Commit this file and the code (this commit). 2. On the VM: base build, final dose calibration (oracle only), `manifest build` (random keys from os.urandom), commit its sha256 (section 9). 3. `manifest generate`; commit the digest-list sha256.
4. Plumbing smoke of the runner in the clean clone on a SYNTHETIC release not in the manifest, with a throw-away ledger. 5. Queue one `r_` job (`r_n_cert`, 8 worker processes) that runs all 704 certificates once. 6. `python -m s_suite.ledger` must print "ledger clean".
7. Unseal (`analyse.py`), append section 8, publish the manifest, per-release table and certificates.
- One query per digest. A release that crashes BEFORE its ledger entry is retried; a ledger entry without a certificate is reported as lost and not re-queried. No tuning, no re-run with another seed, no optional stopping: all 704 are run before any certificate is read.
- If S's evaluator differs from the frozen sha256, the run is invalid. If S changes (new tag), the old manifest is burned for that new version; a new manifest N2 is required, and results on N2 are a separate pre-registration.
- A criterion may only be tightened after data, with both verdicts reported (programme-1 lesson); loosening is a labelled deviation and does not change the verdict.

## 8. Results (unsealed 2026-11-04 ~20:15Z VM clock; one pass of `analyse.py`; all numbers from `results/N/`)
Provenance checks: manifest sha256 equals the committed `e5a511e1...f912`; digest list equals `fb40e7cc...2a23`; ledger `9c659d20...71ca` clean, 704 entries; every certificate prints `frozen_match = true` and `split_status = "sealed"`; 704/704 certificates, 0 errors.
Files: `results/N/{manifest.json, release_map.json, oracle.json, INDEX.txt, ledger.jsonl, certificates.tgz, per_release.csv, results.json, tables.md, reference.json, base_record.json}`.

### 8.1 Validity gates (both pass; the design could grade the unseen families)
- **G1 false alarm of the whole loop: 0 LEAK in 224 nulls** (null 0/80, shuffled twins 0/144; two-sided 95% CP upper bound 1.6%, one-sided 1.3%). The certificate is conservative here: its baseline is the majority-class share (mean 0.056, uniform chance is 0.0417) and the null selected-attacker accuracy is 0.043 (twins 0.041).
  The price is resolution: on true nulls it prints BELOW_MARGIN in only 45/224 (20%) and NOT_RESOLVED in 179/224 (88 test units). The shuffled twins carry the planted structure with decoupled labels: released-label oracle 0.044 (chance) vs planted-label oracle 0.569.
- **G2 power on the mechanisms S planted itself: PASS.** bias 32/32 and specin 32/32 detected at the primary levels (CP lower 0.89). Detection starts at Omega_w 0.146 for bias (level 2) and 0.307 for specin (level 3); bias level 1 (Omega_w 0.085) is 8/16.

### 8.2 Detection by family and level (n = 16 per cell; Omega = key-aware matched filter on the release's test split; chance = 0.0417)
| family | level | n | Omega_w | Omega_s | S selected acc | baseline | detect (CP95) | BELOW_MARGIN | NOT_RESOLVED | printed MDE80 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias | 1 | 16 | 0.085 | 0.097 | 0.156 | 0.056 | 8/16 (0.25-0.75) | 0 | 8 | 0.053 |
| bias | 2 | 16 | 0.146 | 0.184 | 0.415 | 0.057 | 16/16 (0.79-1.00) | 0 | 0 | 0.073 |
| bias | 3 | 16 | 0.312 | 0.388 | 0.627 | 0.055 | 16/16 (0.79-1.00) | 0 | 0 | 0.060 |
| bias | 4 | 16 | 0.618 | 0.711 | 0.774 | 0.057 | 16/16 (0.79-1.00) | 0 | 0 | 0.054 |
| bias | 5 | 16 | 0.890 | 0.922 | 0.971 | 0.057 | 16/16 (0.79-1.00) | 0 | 0 | 0.020 |
| specin | 1 | 16 | 0.084 | 0.131 | 0.045 | 0.056 | 0/16 (0.00-0.21) | 3 | 13 | 0.026 |
| specin | 2 | 16 | 0.149 | 0.302 | 0.056 | 0.056 | 0/16 (0.00-0.21) | 0 | 16 | 0.029 |
| specin | 3 | 16 | 0.307 | 0.595 | 0.124 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.053 |
| specin | 4 | 16 | 0.617 | 0.844 | 0.274 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.084 |
| specin | 5 | 16 | 0.908 | 0.980 | 0.695 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.081 |
| xphase | 1 | 16 | 0.088 | 0.209 | 0.045 | 0.056 | 0/16 (0.00-0.21) | 6 | 10 | 0.024 |
| xphase | 2 | 16 | 0.167 | 0.531 | 0.041 | 0.056 | 0/16 (0.00-0.21) | 4 | 12 | 0.020 |
| xphase | 3 | 16 | 0.372 | 0.853 | 0.058 | 0.056 | 2/16 (0.02-0.38) | 1 | 13 | 0.024 |
| xphase | 4 | 16 | 0.652 | 0.959 | 0.156 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.047 |
| xphase | 5 | 16 | 0.912 | 0.996 | 0.411 | 0.057 | 16/16 (0.79-1.00) | 0 | 0 | 0.074 |
| ampfreq | 1 | 16 | 0.083 | 0.224 | 0.039 | 0.055 | 0/16 (0.00-0.21) | 5 | 11 | 0.021 |
| ampfreq | 2 | 16 | 0.150 | 0.577 | 0.044 | 0.056 | 0/16 (0.00-0.21) | 3 | 13 | 0.024 |
| ampfreq | 3 | 16 | 0.299 | 0.864 | 0.057 | 0.056 | 1/16 (0.00-0.30) | 3 | 12 | 0.031 |
| ampfreq | 4 | 16 | 0.621 | 0.968 | 0.123 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.053 |
| ampfreq | 5 | 16 | 0.913 | 0.995 | 0.235 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.073 |
| stealth | 1 | 16 | 0.083 | 0.148 | 0.044 | 0.056 | 0/16 (0.00-0.21) | 2 | 14 | 0.024 |
| stealth | 2 | 16 | 0.158 | 0.435 | 0.042 | 0.056 | 0/16 (0.00-0.21) | 4 | 12 | 0.024 |
| stealth | 3 | 16 | 0.325 | 0.751 | 0.047 | 0.056 | 0/16 (0.00-0.21) | 1 | 15 | 0.026 |
| stealth | 4 | 16 | 0.634 | 0.898 | 0.051 | 0.056 | 0/16 (0.00-0.21) | 1 | 15 | 0.029 |
| stealth | 5 | 16 | 0.916 | 0.991 | 0.114 | 0.055 | 8/16 (0.25-0.75) | 3 | 5 | 0.045 |
| pooled | 1 | 16 | 0.103 | 0.192 | 0.045 | 0.056 | 0/16 (0.00-0.21) | 3 | 13 | 0.026 |
| pooled | 2 | 16 | 0.184 | 0.415 | 0.072 | 0.056 | 3/16 (0.04-0.46) | 0 | 13 | 0.038 |
| pooled | 3 | 16 | 0.415 | 0.710 | 0.174 | 0.057 | 16/16 (0.79-1.00) | 0 | 0 | 0.062 |
| pooled | 4 | 16 | 0.676 | 0.891 | 0.327 | 0.056 | 16/16 (0.79-1.00) | 0 | 0 | 0.087 |
| pooled | 5 | 16 | 0.955 | 0.993 | 0.846 | 0.055 | 16/16 (0.79-1.00) | 0 | 0 | 0.075 |

Primary level (levels 4+5 pooled, n = 32), pre-registered statuses: **bias DETECTED 32/32, specin DETECTED 32/32, xphase DETECTED 32/32 [0.89, 1.00], ampfreq DETECTED 32/32 [0.89, 1.00], stealth MISSED 8/32 [0.11, 0.43], pooled DETECTED 32/32.**
**Verdict (pre-registered rule): MIXED.** Two of the three unseen families are DETECTED, so the YES rule fails only through its assurance clause: stealth prints BELOW_MARGIN in 4/32 primary releases (12.5%, CP [0.04, 0.29]) against a limit of 0.10. NO is not met (one family MISSED, not two).

### 8.3 What the table says that the verdict hides
- **Unseen families are detected only at much higher doses than seen mechanisms.** Minimum detectable dose (lowest level from which every higher level has detection >= 0.80): bias level 2 (Omega_w 0.15, Omega_s 0.18), specin level 3 (0.31, 0.60), pooled level 3 (0.42, 0.71), **xphase level 4 (Omega_w 0.65, Omega_s 0.96), ampfreq level 4 (0.62, 0.97), stealth not reached** (level 5, Omega_w 0.92, is 8/16 on msda and 0/16 on rep).
  At level 3 a key-aware filter identifies 85-86% of test SESSIONS for xphase and ampfreq (75% for stealth) while S detects 2/16, 1/16 and 0/16. Unseen-family miss rate at level 3: xphase 0.88, ampfreq 0.94, stealth 1.00; at the primary level: xphase 0 [0, 0.11], ampfreq 0 [0, 0.11], stealth 0.75 [0.57, 0.89].
- **Reported leakage versus planted level** (primary levels, ratio of means (selected - baseline) / (Omega_w - baseline)): bias 1.17 (S exceeds a lower-bound oracle), pooled 0.70, specin 0.61, xphase 0.31 (selected 0.283 vs Omega_w 0.78), ampfreq 0.17 (0.179 vs 0.77), stealth 0.04 (0.083 vs 0.78). The certificate is correct that "some leak exists" for xphase and ampfreq, but its reported magnitude is a fraction of the planted level.
- **Which S attackers carry it (mean test accuracy at level 5, exploratory).** xphase: only the raw-view nonlinear attackers (knn1_raw 0.40, knn5_raw 0.41, rf_raw 0.39); every `feat` attacker, LR and LDA sit at chance 0.04-0.05. ampfreq: tree/MLP ensembles on the feature view (rf_feat 0.24, et_feat 0.19, lgbm_feat 0.19) with kNN raw 0.10-0.12; LR/LDA at chance. stealth: rf_raw 0.11, knn1_raw 0.12 only; all `feat` attackers 0.04-0.05.
  The stealth detections are all msda level 5 (tone amplitude 0.2-0.3 above that release's high-band floor); on the smooth `rep` release the oracle identifies 89-92% of windows and S is at chance (selected 0.034-0.057, verdicts BELOW_MARGIN 3/8, NOT_RESOLVED 5/8).
- **False assurance.** BELOW_MARGIN while the session-pooled oracle is >= 0.75: bias 0/21, specin 0/32, pooled 0/36, **xphase 1/45, ampfreq 4/49, stealth 5/43**. For stealth, 5 of 45 releases with Omega_w >= 0.30 carry an explicit "below margin" statement. These counts are small; the claim is that the statement is possible, not that it is typical (most misses read NOT_RESOLVED).
- **Printed MDE80 honesty (secondary).** Among releases with Omega_w - B >= the printed MDE80 of the selected attacker's excess, the detection rate is bias 59/67 (0.88), specin 48/70 (0.69), pooled 51/79 (0.65), xphase 34/73 (0.47), ampfreq 33/76 (0.43), stealth 8/75 (0.11). Only `bias` meets the pre-registered 0.80. This is a statement about what the MDE means: it is the resolution for the attacker S selected, not a promise to detect a leak of that size by any attacker.
- **Shuffled twins:** 0/144, every family and level 0/8.

### 8.4 Reference attackers (exploratory, key-free, same train/test split; not part of the verdict)
Window top-1 at levels 4 / 5 (both bases pooled; session-pooled values are in `reference.json`): **stealth 0.43 / 0.82 by the log-spectrum view (S's best: 0.05 / 0.11)**; xphase 0.43 / 0.66 by the phase-lag view (S: 0.16 / 0.41); ampfreq 0.30 / 0.55 by the level x spectrum view (S: 0.12 / 0.24; the marginal log-spectrum view is at 0.06 / 0.07, so the leak is an interaction); specin 0.45 / 0.81 and pooled 0.53 / 0.89 by the log-spectrum; bias 0.35 / 0.52 by the means; null 0.04.
So every family S under-reports is recoverable from data by an attacker with the matching view: the misses belong to S's declared views, not to the data. These reference attackers were written with knowledge of the families, so this is an existence proof, not a claim that such a suite would have been declared in advance.

### 8.5 Prediction scorecard (section 6)
Correct: bias DETECTED, specin DETECTED, stealth MISSED, pooled detected at 4-5 and not at 1-2 (3/16 at level 2), G1 pass. **Wrong: xphase (predicted MISSED 0.85, observed DETECTED 32/32) and ampfreq (predicted MISSED 0.55, observed DETECTED 32/32); verdict (predicted NO 0.50, observed MIXED).**
Why I was wrong on xphase: I asserted that kNN and RF on raw windows cannot form the cross term. They partly do (0.40 at level 5). My untested explanation is that the planted phase space is only a 2-torus (two bins), so with about 6,500 training windows a nearest neighbour often shares both phases and then the channel-1 lag identifies the identity; that would predict the kNN read to collapse as the number of tones grows. I did not test this; treat it as a hypothesis. It also means the xphase detection is partly an accident of memorisation geometry, which the attainment of 0.31 is consistent with.

### 8.6 Deviations, limits, and what the result does not say
- Deviations: none from the criteria of section 5. The runner was started directly instead of through the queue (section 9); the unused queue spec was removed.
- Replicates share the two base releases (one defence configuration and one generation seed each): the intervals count replicates as independent although they re-draw only the subject partition, labelling and key. Per cell n = 16, primary n = 32: a 32/32 only bounds the rate below by 0.89.
- Labels are pseudo-identities; leaks are additive and keyed; the oracle is a lower bound on the Bayes rate; real defences leak differently. Two defences (AAE, replacement AE) on one dataset with 88 test units; the result is about this declaration of S (SUITE12, these two views, 128-sample two-channel windows), not about Builder T's CXR evaluator, S's re-score of programme 1, or its evidence-budget module.
- The primary level (Omega_w about 0.6 and 0.9) was fixed before data; at level 3 (Omega_w about 0.3) the unseen families are mostly missed. A reader who wants "detects at 30% identification" gets NO for xphase and ampfreq; one who accepts 60% gets YES for them. Both are in the table; the pre-registered one is level 4+5.

### 8.7 Does S earn its place?
- **As a calibrated detector for leaks its views can read: yes.** 0/224 false alarms, seen mechanisms detected from Omega_w 0.15-0.31, correct-sign magnitudes (attainment 0.6-1.2).
- **As evidence that a release does NOT leak: no.** A `LEAK` is trustworthy; an absence of `LEAK` is not. It read two unseen mechanisms only at 60%+ planted window identification (0.62-0.65, against 0.15-0.31 for the seen mechanisms: 2-4x) and at 17-31% of their planted level, and it missed a high-band leak entirely on the smooth release (oracle 0.9, S at chance), with explicit BELOW_MARGIN statements on 4/32 (primary) stealth releases.
  The certificate's own wording is suite-conditional; the measured consequence is that BELOW_MARGIN must not be quoted as "no leakage" unless the suite has a view for the mechanism. Practical rule: quote a certificate only next to a view-complete suite and the printed MDE, never alone.
- **Recommended amendment (a new tag, a NEW manifest N2; this manifest is burned for any suite that adds these views):** declare a high-band log-spectrum view, a cross-channel phase-lag view and a level x spectrum interaction learner, keep the zero-false-alarm calibration, and re-grade on fresh families (amplitude modulation, phase-amplitude coupling, band-limited noise with identity-specific envelope) that this paper trail has not shown S.

## 9. Record
- 2026-11-04 19:09Z (VM clock): MANIFEST COMMITMENT. Pre-registration commit `8f11607` (code + this file) was pushed first; the clean clone `~/n_run/openlatent` on the VM is at that commit
  (tag `s-suite-eval-v1` is an ancestor, `s_suite.freeze --check` passes). The manifest was then built from the oracle-only calibration (`n_control.calibrate`) with random keys from `os.urandom`:
  `~/sealed_manifest/manifest.json`, canonical JSON, 143,094 bytes, mode 0400, 704 entries (plant 480, shuf 144, null 80),
  **SHA-256 `e5a511e188eb492afe6a4682459008804a86eb8d1bfa098b45c08ee35f82f912`**. No release has been generated and no S code has touched any base or planted data at the time of this commit.
  Reproducibility: the file is `canon(manifest)`; `sha256sum` of the unsealed copy published at the end must equal the hash above.
- 2026-11-04 19:10Z (VM clock): RELEASES GENERATED from the committed manifest (`n_control.manifest generate`, clean clone at `8f11607`): 704 files in `~/n_releases/<digest>.npz` (8.9 GB; arrays X, y, unit, split only), 704 distinct digests,
  **SHA-256 of the sorted digest list (`INDEX.txt`) `fb40e7cce95d0c6cc64c9a8cc1a55d8f9e08b0d2f3c2ec1c2e45753358e22a23`**. The sealed digest -> manifest map and the sealed oracle report were written to `~/sealed_manifest/` (mode 0400) and have not been read.
  No certificate exists at the time of this commit.
- 2026-11-04 19:12Z (VM clock): runner smoke on two SYNTHETIC releases (not in the manifest, throw-away ledger): pipeline, ledger and `frozen_match` work. The runner was first queued as `r_n_cert` (priority 10) but sat behind nine priority-4/5 GPU jobs (hours), so I WITHDREW the spec (deleted from `~/queue/pending`, removed from git)
  and started the same command directly: 8 single-thread worker processes, `nice -n 10`, thread caps in the environment (CPU etiquette: at most 8 worker processes). Run start 19:13Z in the clean clone at `62eb352`; ledger `~/sealed_ledger_N/`; ~55 s per release per worker (about 80 minutes for 704).
  Nothing in the certificates has been read; `analyse.py` is run once, after all 704 exist.
- 2026-11-04 ~20:10Z (VM clock): CERTIFICATE RUN COMPLETE, before unsealing. 704 of 704 certificates, 0 errors, 704 ledger entries (one per digest, no repeats), `python -m s_suite.ledger ~/sealed_ledger_N` prints "ledger clean"
  (sha256 of `ledger.jsonl` `9c659d2053540810b9d45687a9f724c0919cdeb212f440939bd5923ad83771ca`); every entry records evaluator sha256 `eb2d6198...27e0f`, tag `s-suite-eval-v1`, git HEAD `62eb352c7c1277ad532581887cd6fb5911dda180`, clean tree.
  No certificate has been opened by me. Next: `analyse.py` once (unseal), reference attackers, append section 8.
