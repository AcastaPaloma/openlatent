# N: negative-control calibration of Instrument S on real defended IMU releases. Pre-registration (v1) and results

Builder-N (negative-control builder, FOCUS amendments A3/A6, audit F10), programme 2, branch `fields-research`. Code: `research/adversarial/src/n_control/` (CPU only).
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

## 8. Results
(to be appended)

## 9. Record
(commit hashes of the manifest and of the digest list will be appended before the corresponding runs)
