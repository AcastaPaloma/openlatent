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
- 2026-11-04 ~19:35Z, v1.1 (analysis-script amendments after the first pass of `rescore`; the evaluator sha256 is UNCHANGED, `freeze --check` passes; tag `s-suite-analysis-v1.2`; the earlier tag `s-suite-analysis-v1.1` points at a tree in which another agent's commit had swept in an interim edit of `freeze.py` (a one-line change to the analysis list, since reverted), so v1.1 is superseded by v1.2):
  (1) HMDB51 metric. The pre-registered plain mean AP did not reproduce programme 1's printed numbers (1 of 15 releases reproduced; outputs kept as `rescore_hmdb_plainmap_not_reproduced`). Programme 1's `plan.json` says metrics.attributes = "conditional"
  (mean over the 5 attributes of the mean AP within each action class where the attribute varies). `rescore.py` now uses that metric for HMDB51 only (new `rescore.py` sha256 `881134326759da8c26de99f92480e635c044981ff79f1e127f5fc2af3956246d`). By the pre-registered rule, non-reproduced cells are excluded from E1; the conditional re-run is the HMDB51 E1 result if it reproduces.
  (2) E3 control rule. Applied literally, the black-release control fails in `same_session` pooling at large K for kth_ilic and charades. Cause (diagnosed, not used to relax the rule): a constant predictor scores exactly 1/(number of eligible persons) and the eligible set shrinks as K grows. Tightening: an evidence-budget point is reported only if at least max(10, 40% of persons)
  (all persons if fewer than 10) can form the pool, and the control is applied per scheme. Both verdicts are reported side by side in Results. The chance reference stays 1/C (changing it to 1/n_eligible would loosen the control and is not done).
  (3) `summarize.py` (new, turns the result JSONs into the endpoint tables; sha256 `48b5b85282c94d170e988f8dc78818aba7c65a7b7902bb47289fb328e034882d`) is an analysis script that is NOT listed in `FREEZE.json` because adding it to `freeze.ANALYSIS_FILES` would change the evaluator hash; its hash is recorded here instead.
  (4) Compute etiquette (CHARTER, main-agent message): `run_planted` was restarted with 8 processes at nice 19 after 205 of 1,200 jobs had finished under 36 processes; the resume logic skips finished jobs; no effect on results.
  (5) E4/E5 replicate counts, written BEFORE any leak-cell engine result was read: under VM contention (load average 150-200 on 56 cores) the planted run was cut to 40 replicates per leak cell (9 cells, 360 jobs) and the 220 null replicates that had finished before the restart (pre-registered 100 and 300). Thresholds are unchanged; the intervals are wider. The null cell's false-LEAK rate is judged on 220 replicates.

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
All inputs are programme 1's test splits (development/burned). No sealed number exists; no ledger entry exists. Result files: `research/adversarial/results/S/` (rescore_*.json/md, harness_selection_bias.json, harness_evidence_budget.json, toy_budget.json, summary_real.json).
Passes over the data: `rescore` on kth_ilic / kth_scenario / utdmhad / charades / their other protocols run twice (second pass with the final analysis files, byte-identical JSON), HMDB51 three times (plain mAP, conditional AP, none changed the first accuracy-dataset results); `run_real select`, `run_real budget`, `run_real toy` once each.

### 8.1 E1: does programme 1's verdict move? (primary protocol `within`; MDE80 printed beside every number)
Reproduction gate: recomputed headline = printed headline within 0.1 point in 58 of 58 accuracy cells (kth_ilic 15, kth_scenario 14, utdmhad 15, charades 14; the dino releases, have attackers outside the declared suite and are listed as dropped) and in 15 of 15 HMDB51 cells (conditional AP). The loaders and the metric are therefore faithful.
A2 = programme 1's own rule (best of the 2 retrained attackers on the test), H = honest cross-fit validation-selected value, D = A2 - H.

| dataset (independent units) | non-saturated reproduced cells | verdict MOVES (|D| > MDE80(D)) | cells where D's CI excludes 0 | max abs D | median MDE80(D) | adjacent orderings resolved, A2 / honest (of) | median MDE80 of adjacent gaps, honest |
|---|---:|---|---:|---:|---:|---:|---:|
| kth_ilic (150 recordings) | 12 | none | 3 | 1.07 pt | 1.03 pt | 7 / 6 (14) | 6.4 pt |
| kth_scenario (150) | 12 | pose_canon_down16_e24 (D +0.87, MDE80 0.83) | 2 | 0.99 pt | 1.32 pt | 6 / 6 (13) | 5.7 pt |
| utdmhad (213 clips, 8 persons) | 11 | pose_canon_down32 (D +2.02, MDE80 1.89) | 5 | 2.02 pt | 1.89 pt | 6 / 6 (14) | 8.0 pt |
| charades (2,720 videos, 90 homes) | 13 | none (D = 0 in every cell: r3d is always selected) | 0 | 0.00 pt | 0 | 7 / 7 (13) | 2.1 pt |
| hmdb51_vp, conditional AP (1,530 videos) | 15 | all 15 flagged, UNINTERPRETABLE (see below) | n/a | 2.05 pt | 1.2 pt | 2 / 2 (14) | 2.1 pt |

- Selection (best-of-2 against validation-selected) moves programme 1's headline by at most 2.0 points on any reproduced accuracy cell, and in privacy units by at most 0.023 (0.011 on KTH): 2 of 48 non-saturated reproduced accuracy cells move by more than their MDE80, both by 0.9 and 2.0 points. The K = 2 selection effect is not what limits programme 1; test-unit noise is.
- Orderings: programme 1's ranking of releases is mostly NOT resolved at the unit counts it has: on the two KTH splits 13 of 27 adjacent gaps are resolved (under honest values 12 of 27), the median adjacent gap needs 5.7-6.4 points to be resolved against printed gaps of 0-5 points; UTD-MHAD 6 of 14 (median MDE80 8.0 points). Charades resolves 7 of 13 orderings at 2.1 points, HMDB51 2 of 14.
- Headline pairs on kth_ilic (best-of-2 / honest, difference with MDE80): canonical stick figure x16 minus learned BDQ x64 -1.7 (7.4) / -2.6 (8.2), not resolved; x16 minus canonical x32 +2.8 (4.5) / +2.2 (5.4), not resolved; x16 minus x8 -4.5 (5.0) / -4.4 (6.0), resolved; x16 vs its seed-1 and 24-epoch variants +0.3 / -0.2 (3.4-3.9), no difference; BDQ x64 minus BDQ x32 -25.8 (9.2), resolved. On kth_scenario x16 minus x32 +4.3 (5.3) / +5.0 (6.3), resolved. These reproduce the stress test's paired MDE80s (5-8 points) with the S machinery. The programme-1 claim "learned BDQ x64 does not beat the canonical stick figure x16 on KTH" is neither confirmed nor refuted at 150 recordings: the observed gap is inside the printed MDE.
- HMDB51: D is -1.3 to -2.1 points in all 15 releases INCLUDING the black control (A2 below H). That is a property of the metric, not of selection: mean AP is not additive over units and is higher on a half-sample (chance AP rises as the sample shrinks), so the cross-fit value computed on halves is biased upward. The pre-registered "moves" flag is therefore not interpretable for HMDB51 and no verdict is drawn from it; the comparison on the same sample size is a new analysis and is not done here. The adjacent-ordering counts are unaffected (the offset cancels).
- Bonferroni upper bound (K = 2) on the full test: KTH canonical x16 13.9% printed, 18.0% bound (4.5 x chance); black control 4.0% -> 8.5%. The certificate can say "at most about 4-5 x chance", never "private".
- K2r (one attacker explains the suite): r3d alone is within its MDE80 of the honest suite value in 48 of 48 non-saturated reproduced cells (100%, per dataset 11-13 of 11-13); frame_r50 alone in 24 of 48 (50%). K2r TRIGGERS for r3d: on programme 1's data the K = 2 suite adds nothing over r3d alone (reported, not a kill of S for CXR).
- Exploratory protocols (not primary): cross_utility, hardest, cross_scenario/scene/trial all reproduce (15/15, 14/14 cells); verdicts move only for utdmhad cross_trial pose_canon_down32 and cross_utility pose_canon_down8; KTH hardest (50 recordings) resolves 2-3 of 14 adjacent orderings.

### 8.2 E2: selection bias on the harness (halves: selection on one half of the recordings, scored on the other)
- K = 2 over 52 non-saturated release cells: median optimism (best-of-2 on the selection half minus the same attacker on the other half) 0.03 point (IQR -0.06 to 0.34), 6 cells at >= 1 point, maximum 2.7 points (utdmhad black, bootstrap CI -0.3 to 5.6); median inflation (best-of-K on the scoring half minus validation-selected) 0.00. Per dataset median / max optimism: kth_ilic 0.23 / 1.08, kth_scenario 0.08 / 0.87, utdmhad 0.18 / 2.74, charades -0.01 / 0.07 (point units, halves of 75 / 75 / 106 / 1,360 units).
- Attacker families (one release with its seed, 24-epoch and DINO variants, same test items; K drawn at random from 10-11 attackers), optimism with bootstrap CI over recordings: kth_ilic K = 2 / 5 / 10: 0.59 [-0.95, 3.48], 1.75 [0.46, 4.69], 2.78 [0.71, 6.58]; kth_scenario 0.71 [-1.56, 3.75], 0.87, 2.08 [0.13, 6.20]; utdmhad (213 clips) K = 2 / 5 / 11: 0.70, 2.35, 5.12 [1.50, 10.12]; charades 0.08, 0.47, 0.81 [0.00, 2.55]. Inflation (best-of-K reported on the scoring half over the validation-selected value) is of the same size: kth_ilic 0.73 / 1.89 / 2.67 at K = 2 / 5 / 10.
- Pre-registered rule: "selection correction matters" iff the K = 2 median optimism >= 1 point (it is 0.03: no) or the largest family-pool optimism is >= 2 points (it is 2.8 on kth_ilic and 5.1 on utdmhad: yes). So: negligible at K = 2, material from K about 5-10 at these unit counts (before scaling to the full test: these are half-split values at 75-106 units; bias shrinks roughly as 1/sqrt(n), so by a 1/sqrt(n) heuristic, selecting on 150 units and scoring on 150 would show about 0.7 of them; not measured).
- The harness holds at most 11 retrained attackers per family: K up to 30 is covered only by the synthetic engine run (8.7).

### 8.3 E3: evidence budget (identity datasets; module validated on the toy first)
- Toy validation passed: dependence detected at K = 10 in 20/20 seeds for rho = 0.5 and 0.9 and in 2/20 for rho = 0; shuffled-person control at most 0.056 (chance 0.04); mean diverse / same-session accuracy at K = 10: rho 0: 0.635 / 0.629; rho 0.5: 0.579 / 0.212; rho 0.9: 0.540 / 0.159.
- Controls (black release). Original rule (upper CI >= 1/C + MDE80 at any point): FAILS in `same_session` pooling for charades (K = 6..20) and kth_ilic (K = 15, 20); `random`, `diverse`, `same_record` pass everywhere; kth_scenario and utdmhad pass. Cause: a constant predictor scores 1/(eligible persons) and the eligible set shrinks with K (e.g. 1 of 25 persons at K = 20). Tightened rule (points need >= max(10, 40%) eligible persons, per scheme): kth_ilic passes; charades `same_session` still fails at K = 6 and 8 (2.2% and 2.8% against chance 1.1% with 46 and 36 of 90 persons eligible), so charades has no valid same-session claim; its `diverse`/`random` curves are valid.
- Gain (kill i): in 87 non-saturated cells (K = 1 accuracy 6-90%), the `diverse` accuracy gain from K = 1 to K = 10 is inside its MDE80 in 21 cells (24%), not >= 90%: kill (i) does not fire. By dataset: kth_ilic median gain +28.9 points (MDE80 15.0), inside MDE in 1/24; charades +20.3 (8.5), 0/20; utdmhad +35.6 (14.7), 3/19; kth_scenario +8.5 (12.7), 17/24.
- The kth_scenario / kth_ilic contrast is the evidence-budget finding: kth_scenario's test is one session (scenario d2) per person, kth_ilic's mixes scenarios. Pooling clips of one session saturates (+8.5 points at K = 10), pooling across sessions does not (+28.9). Canonical stick figure x16 (kth_ilic, r3d): 13.9% per clip -> 19%, 26%, 31%, 34%, 40%, 42%, 47%, 51% at K = 2, 3, 4, 6, 8, 10, 15, 20 (diverse pooling); the independent-evidence toy gives 55% at K = 10 (13.5% per window): the real curve sits between the independent and the shared-nuisance regimes.
- Independence test (diverse minus same_session at the largest K both can form with the eligibility floor; kth_ilic K = 10 with 21 persons, charades K = 8 with 36): dependence detected (95% interval excludes 0) in 9 of 24 non-saturated kth_ilic cells, with the sign positive in 22 of 24; charades 2 of 20 (sign positive in 10 of 20; same-session control invalid there). Most cells are under-powered (MDE80 of the paired gap 0.1-0.3 at 21-36 persons). Reading: a shared session nuisance is present in KTH (diverse pooling beats same-session pooling in 22/24 cells) but 25 persons resolve it in only 9 cells.
- N* (target 0.9, diverse pooling) is reached within K <= 20 in only 26 non-saturated cells; N*_obs / N*_pred (ln C / calibrated per-clip information) has median 1.8 (range 0.9-9.1) and lies inside [0.5, 2] in 14 of 26 cells, so kill (ii) does not fire: N*_pred is not a substitute for the measured N*. Example: canonical x16 on KTH needs about 19 clips for 50% (95% interval from the person bootstrap, right-censored at K = 20), N*_pred 13.
- Black control for the K-axis: `diverse` and `random` curves at chance (<= 4.0% at 25 persons) for every K on KTH and UTD-MHAD.

### 8.4 K1, and what the certificate can and cannot resolve, by n
Zero-leak resolution floor (smallest Bonferroni bound the design can return, observed rate = chance, units independent; 25-way chance 4%, 95% one-sided): chance + 6.3 points at 150 units and K = 12 (K = 2: +4.5; K = 30: +7.1); +6.7 at 144 (MotionSense test trials), +8.8 at 96 (clean released-checkpoint trials); +3.4 at 360; +1.6 at 1,424 and +0.8 at 4,707 (CXR scale, K = 12). n needed for a floor of at most chance + 5 / 3 / 2 points: 205 / 457 / 908 independent units at K = 12 (111 / 259 / 510 at K = 2; 237 / 558 / 1,109 at K = 30); 90-way chance: 122 / 269 / 459 at K = 12. Paired comparison of two releases: MDE80 6.4 points at 150 KTH recordings, so 5 / 3 / 2 points need about 245 / 680 / 1,540 recordings by 1/sqrt(n) scaling (Charades measured 2.1 points at 2,720 videos).
- K1(a) HAR scale: HOLDS. Median honest adjacent-gap MDE80 is 6.4 points (kth_ilic), 5.7 (kth_scenario), 8.0 (utdmhad), above the 5-point bar. On KTH, MotionSense and UTD-MHAD S can only report "not resolved" for differences below 5-8 points; it resolves large leaks and orderings (BDQ x64 vs x32).
- K1(b) CXR scale: NOT triggered. Floor at 1,424 patients, K = 12 is chance + 1.6 points, at 4,707 it is +0.8 (arithmetic). This is a statement about the width of the interval, not about detection of real leaks (the negative-control builder owns that).

### 8.5 Locked evaluator (engine tests)
5 unit tests including a subprocess client that holds no labels, a Ladder rule, a query budget, a single sealed query bound to an artifact digest, and the shuffled-label null of the search loop. 200 loops of 40 random designs on shuffled labels (25 classes, 150 dev / 600 sealed items): best dev score shown 0.076 (chance 0.04), sealed final 0.040: the selection-bias floor of the loop is +3.6 points on the dev split and exactly zero on the sealed one.

### 8.6 E4-E6: synthetic engine and reproducibility (planted data = unit test of the statistics, NOT a power claim)
Run: 220 null replicates and 360 leak replicates (9 cells x 40: bias 0.18 / 0.31 / 0.60, gain 0.053 / 0.092 / 0.19, spectral 0.17 / 0.24 / 0.41; suite-best true accuracy about 6-7%, 9-12%, 22-30% against chance 4%), SUITE12, n_test = 150 / 600 / 1,425 / 3,000 sessions (4 windows each). Raw rows: `planted_engine.jsonl.gz`; table: `summary_planted.json`.

**E4 (pass/fail).** Original wording (clauses pooled over n <= 1,425) and a stricter per-n reading, side by side:
| n sessions | Bonferroni suite bound covers the best declared attacker's true accuracy (need >= 93%) | max-t bound | 95% CI of the selected attacker covers its true value (need >= 90%) | null false-LEAK (need <= 8%, 220 reps) | label-shuffled false-LEAK (need <= 8%, 360 reps) | selected value minus its true value |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 100% | 100% | 96.1% | 2.7% | 2.5% | +0.1 pt |
| 600 | 99.7% | 99.7% | 92.8% | 2.7% | n/a | 0.0 |
| 1,425 | 99.7% | 99.2% | 88.3% | 3.2% | 5.8% | 0.0 |
| 3,000 | 96.9% | 96.7% | 81.9% | 5.0% | n/a | 0.0 |
- Original wording: PASS. Bonferroni coverage 99.8% pooled (minimum cell 97.5% at n <= 1,425), CI coverage 92.4% pooled, both false-LEAK rates <= 5.8%, bias <= 0.1 point. K3 does not fire.
- Stricter per-n reading: the CI-coverage clause FAILS at n = 1,425 (88.3%) and n = 3,000 (81.9%). Diagnostic (exploratory, not a rescue): the spread of (selected - true) / SE is 0.97, 1.10, 1.19, 1.45 at n = 150 / 600 / 1,425 / 3,000, which implies a roughly constant extra error of 0.4 point (0.41, 0.38, 0.43 at n >= 600), the size of the sampling error of the 2,500-session truth sample itself. The shortfall is therefore consistent with the truth sample's noise, which grows relatively as the certificate's SE shrinks, but a small under-coverage of the percentile CI at large n is not excluded. What S ships as its guarantee is the suite bound (>= 96.9% everywhere) and the one-sided LEAK test (false-LEAK <= 5.8%), both of which pass; the two-sided CI of the selected attacker should be quoted with this caveat at n >= 1,425.
- The Bonferroni bound is conservative (about 100% against 95% nominal) and the max-t bound is not tighter in this setting (q = 2.6-2.8 against z = 2.64 at K = 12; attackers are weakly correlated).
- Exploratory and CIRCULAR (the planted channels map onto the declared features, F10): the LEAK verdict fires in 35-57% of replicates at n = 150 and 90-95% at n = 1,425 when the best attacker's true accuracy is 6-7% (2-3 points over chance), and in >= 95% from 9-12% at n = 150; the one-line statistics (kNN-5 and LDA on raw windows) match the suite within 1 point on the bias channel (LDA) and are 1.5-24 points below it on the gain and spectral channels. No power claim for real leaks follows; no K2 verdict is drawn.

**E5 (truth-based selection bias, 30 fitted attackers, 150 validation sessions).** Mean over replicates with 95% bootstrap intervals, points of accuracy; null = no identity signal, leak = pooled over the nine leak cells.
| scored on n sessions | K | best-of-K on the scored split minus its attacker's true accuracy (winner's curse) | best-of-K minus validation-selected (inflation) | validation-selected value minus its true accuracy | selection loss (true best minus selected) |
|---|---:|---|---|---|---|
| 150, null | 2 / 10 / 30 | 0.40 [0.33, 0.47] / 1.17 [1.10, 1.25] / 1.56 [1.47, 1.64] | 0.46 / 1.22 [1.16, 1.30] / 1.63 [1.52, 1.73] | -0.05 / -0.06 / -0.09 | 0.11 / 0.27 / 0.35 |
| 150, leak | 2 / 10 / 30 | 0.32 [0.21, 0.43] / 1.09 [0.96, 1.23] / 1.51 [1.35, 1.66] | 0.28 / 0.98 [0.92, 1.05] / 1.36 [1.24, 1.50] | +0.05 / +0.09 / +0.14 | 0.13 / 0.41 / 0.56 |
| 1,425, null | 2 / 10 / 30 | 0.13 / 0.40 [0.36, 0.43] / 0.52 [0.47, 0.56] | 0.14 / 0.40 / 0.55 [0.51, 0.59] | -0.02 / -0.03 / -0.06 | 0.10 / 0.28 / 0.35 |
| 1,425, leak | 2 / 10 / 30 | 0.04 / 0.16 [0.11, 0.22] / 0.27 [0.21, 0.34] | 0.13 / 0.46 [0.43, 0.50] / 0.60 [0.53, 0.68] | 0.00 / 0.00 / +0.02 | 0.12 / 0.42 / 0.56 |
- Pass: the validation-selected value is unbiased (|bias| <= 0.14 point at every K, n and cell type; threshold 0.5). The best-of-K reported on the scored split is optimistic by 1.1-1.2 points at K = 10 and 1.5-1.6 at K = 30 with 150 sessions, by 0.2-0.4 and 0.3-0.5 at 1,425.
- Pre-registered rule "selection correction matters iff inflation at K = 10 >= 1 point at n = 150": null 1.22 [1.16, 1.30] yes, leak 0.98 [0.92, 1.05] on the boundary. So at KTH/MotionSense scale (150 units) the best-of-K reported on the test set adds about 1 point at K = 10 and 1.4-1.6 at K = 30, the size of programme 1's 2-3 point effects; at CXR scale it adds 0.5 point. Choosing on a separate split costs 0.3-0.6 point of true accuracy at K = 10-30 (selection loss), the price of the honest number.
- Real-data comparison: the half-split optimism on the harness (E2, 75 + 75 units) is 2.1-2.8 points at K = 10 on KTH families against 1.2 here at 150 + 150, consistent with a 1/sqrt(n) scale-down.

**E6 (reproducibility).** (a) Fixture certificate digest `f860e80038f2...` identical on the laptop (numpy 2.5, py3.12) and on the VM (numpy 1.21, py3.10) after all runs. (b) `freeze --check` passed before every run (laptop and VM). (c) LITERAL FAIL: the per-release outputs of `rescore`, `run_real` and `summarize` do not embed the evaluator sha256 (only `certify` does); the sha, the analysis hashes, the tags and the exact commands are recorded in `results/S/provenance.json`. Fix going forward: any certificate produced by `certify` prints it.

### 8.7 Verdicts
- Primary endpoints: E1 done (selection moves programme 1's headlines by <= 2 points; 2 of 48 non-saturated accuracy cells move beyond MDE80; the binding limit is test-unit noise: 5.7-8.0 points MDE80 on adjacent gaps at 150-213 units; HMDB51 not interpretable); E2 negligible at K = 2, material at K >= 5-10 (2.1-5.1 points at half-split sizes); E3 evidence accumulates (kill i and ii did not fire) but a shared session nuisance halves it on KTH and the independence test resolves it in 9 of 24 cells; E4 PASS (original wording; per-n CI clause fails at n >= 1,425, caveated); E5 PASS; E6 (a), (b) pass, (c) literal fail.
- K1 holds at HAR scale (150 units), is not triggered at CXR scale (arithmetic). K2 undecided (planted data cannot carry it; carried to the negative-control manifest); K2r triggers (r3d alone equals the K = 2 suite on programme 1's data).
- Does S earn its place? As an evaluator for R-CXR: yes on statistical grounds (a bound that covers, a verdict that does not fire on null or shuffled labels, an honest selected value, MDE printed beside every number, floor of chance + 1.6 points at 1,424 patients) but its detection power for real, unseen leak types is NOT established and belongs to the negative-control builder. For R-IMU and KTH: it earns its place only as the reporter of what is unresolved (it cannot certify "private" below chance + 6 points at 150 units; programme 1's orderings with gaps of 0-5 points are unresolved). It does not rescue the KTH headline comparison: more attackers do not help, more units do (cross-fit rotation, 4 x attacker cost, buys about 1.4 x; 245 / 680 recordings needed for 5 / 3 point paired MDE).
- No new attacker families on KTH were started: E1 shows no release where selection or an extra retrained attacker (programme 1's headline attacker was always frame_r50 or r3d) moves the verdict; the headroom is unit count, not attacker count.


## 9. Record
- 2026-11-04 ~19:05Z: tag `s-suite-eval-v1` (annotated, pushed) points to commit `6cfc65e` (a main-agent commit that swept S.md, `s_suite/FREEZE.json` and the code in; their content at that commit is byte-identical to this pre-registration; `python -m s_suite.freeze --check` passes against it). That commit is the prereg_commit for every later ledger check. No real-data number existed at that time.
