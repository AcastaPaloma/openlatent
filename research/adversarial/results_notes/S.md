# S: sealed suite, certificate, evidence budget. Pre-registration (v1, 2026-11-04 ~18:50Z VM clock) and results

Builder-S, programme 2, branch `fields-research`. Code: `research/adversarial/src/s_suite/` (CPU only, 44 unit tests pass on numpy 2.5 / py3.12 and, for the library, on numpy 1.21 / py3.10).
This file is written and committed BEFORE any number is computed from programme 1's stored predictions. Results are appended under "Results" in later commits; the text above
"Results" is never edited except by a dated amendment that tightens (never loosens) a kill.

## 0. Amendments log
- 2026-11-04 ~18:50Z, v1: re-scoped after Auditor-2 F8-F10 and FOCUS A3 (main-agent message). Changes against my first draft: (a) the programme-1 re-score is the FIRST deliverable;
  (b) evaluator frozen by tag + sha256, printed in every certificate; sealed-query ledger and ancestor-commit check implemented; (c) `planted.py` is a unit test of the statistical engine only,
  no detection-power claim is made from it; real calibration belongs to the negative-control builder; (d) no split is called sealed unless it passed the ledger: every programme-1 test split,
  MotionSense trials 11-16, the official CXR test list and PriCheXy's 5,000 pairs are DEVELOPMENT/BURNED data; (e) primary endpoints declared (section 4), everything else exploratory.
  None of these loosens a kill condition: the kills in section 5 are the ones I was given, made checkable.

## 1. What S is, after the audit
1. `certificate.py`: a pure, deterministic function. Input: a declared attacker list (`SuiteDecl`, digest printed), saved predictions on a selection split and a scored split with independent-unit ids.
   Output (JSON): per-attacker value, cluster-bootstrap SE and 95% CI, MDE80 = 2.8 x SE beside every number, the attacker chosen on the selection split and its score on the scored split, naive max
   (comparison only), a Bonferroni (Clopper-Pearson on effective counts at alpha/K) and a studentised max-statistic bootstrap upper bound over the whole declared suite, the resolution floor (the
   smallest suite bound the design can ever return), chance and label-only baseline, normalised leakage, verdict in {LEAK, BELOW_MARGIN, NOT_RESOLVED}. Printed: evaluator version, sha256 of the
   evaluator, expected sha256 from `FREEZE.json`, git tag, `split_status`.
2. `freeze.py`, `ledger.py`: evaluator freeze, sealed-query ledger, `python -m s_suite.ledger LEDGER_DIR` (exit 1 on violation) = the script check that the pre-registration tag is an ancestor of the
   commit that produced each sealed number, that the tree was clean and the evaluator frozen, and that no (release digest, split) was queried twice without a declared amendment.
3. `selection_bias.py`, `evidence_budget.py`, `locked_eval.py` (labels held by a separate process, the agent sees one scalar; Ladder rule; single final query bound to an artifact digest; shuffled-label
   null of the search loop tested), `harness_io.py`, `rescore.py` (deliverable), `run_real.py`, `run_planted.py`, `planted.py`, `attackers.py`, `oracle.py`.
4. `split_status`: a split is "sealed" only if its single query is on the ledger (release digest = hash of the released representation, not of predictions). Everything in this file is
   `development_burned` or `synthetic`. The ledger directory must be custodian-owned and outside the checkout (the module refuses a directory inside the repo); a builder-owned directory
   cannot prevent a builder re-run, it only makes one detectable. No ledger exists yet because no sealed number exists yet.

## 2. Freeze record
- git tag `s-suite-eval-v1` on the commit that adds this file and `s_suite/FREEZE.json`. The prereg commit is the commit that tag points to (hash recorded in section 9 by a later commit).
- evaluator sha256 (stats, certificate, selection_bias, evidence_budget, harness_io, locked_eval, ledger, freeze):
  `eb2d619870c2811430aee56376a306dc71dc930de5bc187e00d9a6cdfb527e0f`. Analysis scripts hashed per file in `FREEZE.json` (rescore, run_real, run_planted, planted, attackers, oracle).
- `python -m s_suite.freeze --check` must pass before every analysis run and its output is logged; any later edit of a hashed file needs a new tag and an entry in section 0.
- planted-engine suite declaration (SUITE12, K = 12): `lr_raw_c0.1, lda_raw_auto, knn5_raw, rf_raw_100, lr_feat_c1, lda_feat_auto, knn5_feat, rf_feat_200, et_feat_200, lgbm_feat, mlp_feat_64, knn1_raw`;
  alpha 0.05, 2000 bootstrap resamples, seed 0, margin 0.02, 25 classes; declaration digest `ef9814e1b67b53348648087e60d0f6cfe2b80a35f5a63b6f63a13fd48c9b4902`.
- Fixture check (determinism, E6): the unit-test fixture certificate (`tests/test_certificate.py::build(0, [0.3,0.2,0.1])`, 3 attackers) has digest
  `f860e80038f2ae040e43ac0b0f6c7ad4399df9b00673a2337e839231ea01b98c` on the laptop (numpy 2.5, py3.12) and on the VM (numpy 1.21, py3.10).

## 3. Data, burned-set ledger, seeds versus test units
- Data: programme 1's `preds.npz` + `result.json` (HF `KuanKuanKuan/openlatent-privacy`, `harness/<dataset>/eval/<release>/`, 78 files: kth_ilic 16, kth_scenario 15, utdmhad 16, charades 15, hmdb51_vp 16;
  ucf101_vp has no evaluation) and each `report.json` in `research/privacy-obfuscation/results/harness/` for programme 1's printed verdicts. Public datasets are not re-uploaded.
- BURNED (development) sets: every programme-1 test split above (programme 1 selected releases, attackers and epochs on them); MotionSense trials 11-16 and the stress test's 144-trial numbers; the
  official ChestX-ray14 test list and PriCheXy's 5,000 test pairs. No number from S on these is sealed, none may be quoted as a sealed result.
- Independent unit: the RECORDING (`groups.source`: KTH `person01_boxing_d3`, UTD-MHAD one clip, Charades one video, HMDB51 one video). KTH ilic: 598 clips = 150 recordings; scenario: 600 clips = 150.
  Session (evidence budget): KTH person x scenario, Charades identity x scene; UTD-MHAD and HMDB51 have none (independence test unavailable there, stated per cell).
- Seeds versus test units: the test units are fixed by programme 1, so seeds replicate only selection noise. Test-unit noise is carried by the recording-cluster bootstrap (1000 resamples; 300 for
  the slow multilabel path), and the honest value uses 20 random half-splits of the recordings (6 for HMDB51) with the selection redone inside every bootstrap resample. On synthetic data every
  replicate draws new identities and new units, so replicates are independent datasets.

## 4. Primary endpoints (at most six; everything else is exploratory)
Primary protocol is `within` (the dataset's own split). Other protocols (cross_utility, hardest, cross_scenario/scene/trial) are exploratory.
Declared suite for the programme-1 re-score: retrained `frame_r50` and `r3d` (K = 2).

**E1. Re-score of programme 1 (the charter's first falsifiable clause; zero GPU).** For each release with a printed `within` score in `report.json`:
- A1 = printed headline (worst attacker over the attackers listed in report.json, top-1; HMDB51: mean AP over the 5 attributes). Recomputed from `preds.npz`; the cell is REPRODUCED iff |recomputed - printed| <= 0.1 point. Non-reproduced cells are excluded from E1 and listed.
- A2 = best of the 2 declared retrained attackers on the full test (programme 1's selection rule, K = 2). H = honest cross-fit value: 20 random half-splits of the recordings; the attacker is selected on one half, scores the other half, both directions; H = mean over splits.
- D = A2 - H with a paired recording-cluster bootstrap (selection redone inside each resample). MDE80(D) = 2.8 x SE(D).
- **Verdict moves** iff |D| > MDE80(D) (and |D| > 0 if SE(D) = 0). Also reported in programme 1's privacy units (divide by raw - base; undefined when raw - base < 2 points).
- Reported as E1: the number of reproduced non-saturated releases (A2 in (0.06, 0.95)) whose verdict moves, per dataset, with each release's A1, A2, H, D, MDE80(D), Bonferroni UB (K = 2); and, for programme 1's ranking
  (releases sorted by A1), the number of adjacent orderings that are resolved (95% bootstrap interval of the paired gap excludes 0) under A2 and under H, with the MDE80 of each gap.
- Headline pairs from the stress test and SUMMARY.md are included in the adjacent-pair table by construction when adjacent; non-adjacent headline pairs (canon x16 vs learned bdq x64; canon x16 vs canon x8 / x32; seed and 24-epoch variants) are computed from the same bootstrap draws and listed separately.

**E2. Selection bias on the harness.** For every identity release with >= 2 retrained attackers (frame_r50, r3d, +skeleton if present): `split_inflation` (400 random half-splits of recordings; 200 outer bootstrap resamples). Quantities:
optimism = best-of-K on the selection half minus the same attacker scored on the other half; inflation = best-of-K on the scoring half minus the validation-selected attacker on the same half. Endpoint: median optimism at K = 2
over non-saturated cells with its bootstrap range; "selection correction matters" iff the median optimism at K = 2 is >= 1 point, or the largest family-pool optimism (pose_canon_down16 / motion_down64 / pose release plus its seed, epoch and DINO variants, K up to ~10) is >= 2 points.
Reported with intervals. The harness cannot support K = 1..30 (at most ~11 retrained attackers per family): K up to 30 is covered only on synthetic data (E5).

**E3. Evidence budget (identity datasets).** Item-level posteriors are pooled by summing clipped log-probabilities. Grid K = 1,2,3,4,6,8,10,15,20; 100 draws per person, person-level bootstrap (1000).
Schemes: random, same_session, same_record, diverse (one item per session first). Endpoint: for each non-saturated cell (K = 1 accuracy in (0.06, 0.90)), gain = acc(K=10) - acc(K=1) under `diverse` pooling and its MDE80 (2.8 x person-bootstrap SE);
the contrarian's kill (i) is "gain inside MDE80 in >= 90% of cells" (then effective K ~ 1). Independence test: diverse - same_session accuracy at the largest K both can form, paired over persons, with MDE80; dependence is detected iff the 95% interval excludes 0.
N* for targets 0.5 and 0.9 with person-bootstrap intervals, and N*_pred = ln(C) / I_item from the temperature-calibrated per-item cross-entropy; kill (ii) is "N*_pred within [0.5, 2] x N* in every cell with N* available" (then N* is a derived certificate field, not a finding).
Controls: black release at chance for every K and every scheme (accuracy upper CI < chance + MDE80), raw release above black. A cell that fails the control invalidates the dataset's E3 table. Module validation on the toy model (rho in {0, 0.5, 0.9}, 20 seeds each; shuffled-person control) runs first; E3 on real data is
reported only if the toy passes: dependence detected at K = 10 in >= 18/20 seeds for rho = 0.5 and 0.9 and in <= 2/20 for rho = 0, shuffled control below chance + 0.08 at every K.

**E4. Engine validity on synthetic data (`run_planted`, unit-test status).** Planted windows (null, bias, gain, spectral channels; 25 identities; session offsets; units = sessions of 4 windows), 3 doses per leak channel (oracle accuracy about 8%, 14%, 35%), 100 replicates per leak cell, 300 null replicates,
n_test in {150, 600, 1425, 3000} units. Pass iff (a) the Bonferroni suite bound covers the true accuracy of the best declared attacker (population sample of 2,500 sessions) in >= 93% of replicates, pooled over leak cells at n <= 1425, and the one-sided CI of the selected attacker covers its true value in >= 90%;
(b) the LEAK verdict fires in <= 8% of null replicates at each n and in <= 8% of unit-label-shuffled controls (n = 150, 1425); (c) the reported (val-selected) value is within 0.5 point of its true accuracy on average.
Fail on (a) or (b) = engine invalid; no certificate may be used until fixed (new tag). The detection-vs-dose curve, the oracle table, dose recovery and the one-line-statistic comparison (kNN-5 and LDA on raw windows vs the suite) are recorded as EXPLORATORY CIRCULAR outputs: the planted channels map onto the declared features by construction, so they are not evidence of power for real leaks (F10).

**E5. Truth-based selection bias (synthetic, K = 1..30).** Same replicates, 30 fitted attackers (pool in `attackers.py`, first 12 = SUITE12), true accuracies from the population sample, 40 random K-subsets per replicate, K in {1,2,3,5,10,20,30}, n_test in {150, 1425}.
Reported with replicate-bootstrap 95% intervals: winner's-curse bias of the best-of-K on the scored split, bias of the validation-selected value (pass: |bias| <= 0.5 point for every K), selection loss, and inflation (best-of-K minus validation-selected). "Selection correction matters" iff inflation at K = 10 >= 1 point at n = 150.

**E6. Reproducibility and freeze.** (a) fixture digest identical on both environments after the runs; (b) `freeze --check` passes at every run; (c) every real certificate/summary prints the evaluator sha256 equal to section 2. Fail = results not releasable.

Multiplicity: E1-E3 are descriptive counts with printed intervals, E4-E6 are pass/fail engine checks with one pre-set threshold each; no p-value gating among them. Everything not listed above, including all non-`within` protocols, the family pools beyond K = 2, the planted detection curves and the HMDB51 conditional-AP variant, is exploratory.
Looks: the number of analysis passes over each real data file is stated in the Results section; the pre-registration is one pass of each script, and every rerun is logged as an amendment.

## 5. Kill conditions (as given; made checkable)
- **K1: the certificate has no resolving power at the available n.** Checked on REAL data and by arithmetic, not on planted data: (a) HAR scale: median paired MDE80 of the adjacent-pair gaps and of D across non-saturated programme-1 cells on the 150-recording KTH splits; K1 holds at HAR scale if that median exceeds 5 points
  (the stress test already measured 5-8 points; expectation: K1 holds on KTH, MotionSense (144 trials, about 96 clean for released checkpoints) and UTD-MHAD (8 persons)); (b) CXR scale: the resolution floor (`zero_leak_floor`) at n = 1,424 and n = 4,707 units and K = 12 must be <= chance + 5 points,
  which the stress test table already gives analytically (+1.8 points at K = 30, n = 1,424); K1 for CXR is not triggered unless this arithmetic fails. K1 at HAR scale does NOT kill S; it fixes S's role there as the reporter of what is NOT resolved (the printed MDE), and it is the reason S is evaluated for R-CXR first.
  Detection power for real leaks is not asserted by S; it is owned by the negative-control builder.
- **K2: a one-line statistic matches the suite at every dose.** The original wording needs planted doses; F10 shows planted doses cannot carry it. So: (a) on planted data it is recorded as exploratory and no verdict is drawn; (b) K2 is carried unchanged to the negative-control builder's sealed manifest (real defended releases, unseen leak families); (c) on programme 1's data I report K2r: for each non-saturated release, the honest value H of the K = 2 suite against the honest value of frame_r50 alone and of r3d alone (paired, with MDE80);
  K2r triggers if one single declared attacker is within its MDE80 of H in >= 90% of non-saturated releases, meaning the K = 2 suite adds nothing over that attacker on programme 1's data (reported, not a kill of S for CXR).
- **K3 (new, engine): E4(a) or E4(b) fails.** Engine invalid.
- Original verdicts are always reported beside any tightening. Nothing here is loosened.

## 6. Disclosures: what was seen before this pre-registration (looks so far)
1. From the stress test (read before building): KTH has 150 independent recordings; recording-cluster SE and design effects of best-of-2 attackers for 9 releases; paired MDE80 5-8 points for the headline comparisons; K = 2 winner's-curse optimism 0.5-0.6 points on average (2.8 worst release); the certificate floor chance + 3.7 (K = 1) to + 7.1 (K = 30) at 150 units.
2. The contrarian's evidence-budget toy numbers (13.5% per window -> 55% at K = 10 if independent, ~20% with a shared session offset at rho = 0.5).
3. Programme 1's published `report.md` for kth_ilic (headline top-1 per release and CIs, the identity table for four releases) and the structure of `report.json` and SUMMARY.md excerpts. I recomputed NOTHING from any `preds.npz`; I inspected keys, array shapes and id formats only.
4. Synthetic only: an oracle pilot used to set the dose grid (accuracy of the best attacker with large training data per channel and dose), one planted replicate's attacker accuracies, and a 4-job smoke run of `run_planted` (2 cells x 2 replicates; plumbing check; its outputs were read and are discarded).
5. Unit tests ran on fabricated predictions only (44 tests).
Everything after the freeze is one pass of each script unless an amendment says otherwise.

## 7. Run plan (order is binding)
1. `python -m s_suite.freeze --check`, then the tests.
2. `rescore` on kth_ilic, kth_scenario, utdmhad, charades, hmdb51_vp (primary protocol `within`; the other protocols as exploratory) -> E1, K1(a), K2r.
3. `run_real select`, `run_real budget` after `run_real toy` passes -> E2, E3.
4. `run_planted` on the VM CPU (not the queue, no GPU) -> E4, E5; fixture digest on both environments -> E6.
5. Only if E1 shows headroom (a release whose verdict moves AND whose honest value leaves room above the floor): propose 1-2 new attacker families on KTH (needs the programme-1 pipeline and a GPU pose pass; not started here). No new attacker is trained in this pass.

## 8. Results
(to be appended)

## 9. Record
- 2026-11-04 ~19:05Z: tag `s-suite-eval-v1` (annotated, pushed) points to commit `6cfc65e` (a main-agent commit that swept S.md, `s_suite/FREEZE.json` and the code in; their content at that commit is byte-identical to this pre-registration; `python -m s_suite.freeze --check` passes against it). That commit is the prereg_commit for every later ledger check. No real-data number existed at that time.
