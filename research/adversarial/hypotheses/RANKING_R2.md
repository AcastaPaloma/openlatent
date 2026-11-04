# Hypothesis ranking, ranker pass 2 (2026-11-04, after AUDIT_findings.md F1-F18, FOCUS.md amendments A1-A6, docs/data_licences/WHITELIST.md)

`RANKING.md` (pass 1) is kept unchanged. Where the two conflict, this file supersedes it. Sections 6 and 7 of pass 1 (control list, group definitions) still apply except where noted below.
Read in full for this pass: AUDIT_findings.md, FOCUS.md including A1-A6, WHITELIST.md, the journal tail, `worker.py` (`hf_allow`), the current job specs and results directories, `src/s_suite/run_real.py`, `src/h_imu/suite.py`. I ran nothing on the VM and no git command.

## 0. Evidence boundary

* No programme-2 model number exists. `results/` holds `h_imu/power.json` (zero-GPU power audit), `h_imu/test_windows.json`, one PD smoke job (`h_pd_smoke`: exit 0, 52.4 s for `--limit 1024` windows including start-up) and an empty `h_r0_pd_pub` (running). `results_notes/` still holds only `H.md`; **there is no `T.md` and no `S.md`**, so Builder T has no committed pre-registration and Builder S has no committed checks (C1-C8) yet.
* Every likelihood is my prior (the audit asks for priors to be assigned or reviewed by someone other than the author: this is that review, blind to any result). MEASURED = stress test or audit replication (single seed, single split, hand features, RF); ESTIMATED = planning number.
* **Cost correction to pass 1.** The smoke job bounds PD sampling at about 0.3 GPU-h per 21.7k-window release (52 s per 1,024 windows, linear scaling, ESTIMATED from one job). Pass 1 budgeted 3.5 GPU-h for six releases; 1.5-2 is enough. This is part of why the PD control is now cheap to keep.

## 1. What the amendments change (the five effects that reorder the table)

**E1. R-IMU has a new primary object, and the old one is a control (A1, F1).** The identity-targeted Anonymizing AutoEncoder (arXiv 1810.11546; published claim identity below 7% at activity accuracy above 92%, 24 users) is the defence under audit; PrivDiffuser becomes a labelled cross-threat-model control (its only protected attribute is gender), and "arm B" (PD retrained with identity guidance) a learned arm. There is no hypothesis file for the AAE yet: **H-IMU-007 (proposed below) is the new top R-IMU row.** Before it is a replication target, Builder H must read the AAE paper's own attacker and split protocol: if its identity attacker was trained on raw data (as PD's was) its <7% is also an off-threat-model number. I did not read that paper.
**E2. The IMU headline needs a fresh partition, and P-PD cannot supply it (A1, F2, F3, F7).** Trials 11-16 and every stress-test number are burned (development). Builder H's v0 compares floors chosen on folds A+B with fold O, which is the burned set; that is superseded for in-programme defenders (AAE, floors, arm B): headline = salted rank folds `sha256(subject|activity|salt)`, salt committed by the main agent **after the floor list is frozen**. Fold O survives only as the clean arm for the released PD checkpoint (96 trials, paired MDE80 about 9-13 points; Builder H MC 0.12 at p 0.1, 0.16 at p 0.3). Static/dynamic strata need sit/std, which PD's 4-activity protocol excludes, so the AAE arm needs a second protocol (call it P-AAE: the six MotionSense activities, the AAE paper's channels). That restores the generator's six-activity numerics for this arm only. Static stratum = 96 trials (4 per subject): MDE80 about 0.12-0.16 by Builder H's formula, so defenders can be compared within it only for large effects.
**E3. R-CXR cannot be cleanly three-way split with the released PriCheXy weights (A2, F4).** The released generator saw all official train_val patients (28,008, of whom 11,878 are multi-visit). Patients it never saw are the official test list (1,424 multi-visit), which is also a burned set. Splitting those 1,424 into attacker-train and sealed-test leaves about 700 each: under-trained attackers (they understate leakage) and MDE about 0.03 AUC. So the clean headline needs **PriCheXy-Net retrained on our own three-way split** (about 8 GPU-h per mu, ESTIMATED, range 4-12) and the released-weights arm becomes the sensitivity arm. My arithmetic (ESTIMATED, Builder T must confirm): excluding the 1,424 burned test patients leaves 11,878 multi-visit patients; a 35/40/25 split gives about 4,160 defender-train, 4,750 attacker-train, 2,970 sealed, paired MDE80 about 0.015 AUC (scaling the 0.021 at 1,424). Floors need no defender retrain, so floors-first is unaffected.
**E4. The ChestX-ray14 hold is lifted, but only under conditions, and the real gate is now Builder T's pre-registration (A4, WHITELIST).** Remaining conditions: (a) licence snapshot committed, (d) integrity check against an official copy (at least 500 sampled files; the official bucket is Requester Pays, so this needs the owner's billing project or a Kaggle copy), (e) every `t_` spec sets `hf_allow: ["*.json","*.log","*.md"]`, no trained attackers, defended images or per-patient embeddings leave the VM. The download can start after (a); no image number should be believed before (d), before `T.md` carries the machine-checkable template of A2, and before the salt is committed. **Consequence of `hf_allow`:** `preds.npz` is not in the allow-list, so CXR scores must be written as `scores_*.json` keyed by hashed IDs, or the allow-list extended by the main agent; the replication auditor needs raw result files.
**E5. S gets a mandatory zero-GPU deliverable and loses its self-graded calibration (A3, F8-F10).** New row H-S-006 (re-score programme 1's `preds.npz`) answers the charter's first falsifiable clause and ranks first. The suite builder's planted leaks (bias, gain, spectral line, matched one-to-one to its own features) become a harness smoke test only (S-001a); the minimum-detectable-dose claim moves to the negative-control builder with a sealed manifest and at least two unseen leak families injected into REAL defended releases (S-001b), which cannot start until defended releases exist. Every sealed number additionally needs the evaluator freeze (git tag + sha256), the sealed-query ledger, and the prereg-ancestor check; and G stays closed (A6).

## 2. Primary endpoints (F18, proposal for the main agent; all else labelled exploratory, Holm inside each focus)

* **R-IMU:** P1 replication of the AAE's published claim under its own protocol (a gate, not a tested hypothesis); P2 sealed worst-suite identity of the AAE at activity accuracy at least 0.92 against the 7% claim; P3 AAE vs the floor envelope (Builder H WIN_L/WIN_U/K2 rules on the salted partition); P4 retrained suite vs frozen raw-trained attacker (IMU-002); P5 static vs dynamic leakage gap. PD control, split diagnostics, IMU-004, arm B: exploratory.
* **R-CXR:** gate = MN-v2 validity + CXR-002 stage 0; P1 CXR-004 on matched negatives (sealed, suite worst case vs PriCheXy's own attacker); P2 CXR-003 floor envelope vs PriCheXy (hierarchical, only if P1 passes); P3 demographic share E (CXR-001). Everything else exploratory.
* **S:** P1 H-S-006 per-release verdict-moves; P2 H-S-002 clauses 1-3; P3 H-S-003 N*. S-001b is a calibration statement, not an endpoint.

## 3. Ranking table (pass 2)

Columns: Rank (old rank in brackets), Δ = status versus pass 1 (**NEW**, **CHANGED**, same), likelihood = probability the decision-relevant core clause holds (pass-1 value in brackets where moved), App and Info/GPU-h as in pass 1 (Info = H2(L) x App/5 / max(cost, 0.25); tiers decide order, gates precede what they gate), cost = staged / full GPU-h, [C] = must pass before an exciting result is believed.

| Rank | ID | Δ | Likelihood | App | Info/GPU-h | Cost | Blocked by | Verdict |
|---:|---|---|---|---:|---:|---|---|---|
| 1 (-) | **H-S-006** re-score programme 1's stored `preds.npz` (KTH ilic/scenario, UTD-MHAD, Charades; HMDB51 once added to `run_real.DATASETS`) with cluster bootstrap, best-of-K, evidence budget; state per release whether the programme-1 verdict moves by more than the printed MDE | **NEW** (A3, charter clause 1) | 0.75 that at least one headline comparison is inside the printed MDE or flips (stress test already shows the headline pair, canonical x16 vs learned bdq x64, at -1.7 +- 2.7); 0.2 that a verdict flips with resolution | 4 | 2.6 | 0.25 / 0.5 | none (predictions-only reuse is on the whitelist) | RUN NOW (`run_real cert|select|budget`) |
| 2 (1) | H-S-001a harness null smoke (null at chance, shuffled/constant) | **CHANGED** (split from S-001; calibration claims removed) | 0.85 | 2 | 1.0 | 0.25 / 0.25 | none | RUN NOW (`--procs 8`) |
| 3 (15) | **H-IMU-006'** static/dynamic strata and what static identity is made of (measured premise: static de-meaned RF 47.9%, dynamic 91.7%, single seed, 48 trials, F7). New falsifier: after per-channel spectral whitening plus band-limiting to at most 5 Hz, static identity falls to at most 15% while dynamic loses under 10 points | **CHANGED** (old prediction at most 12% already falsified; now a mandatory report stratum) [C] | 0.45 on the new claim (either outcome informative) | 4 | 3.2 | 0.1 / 0.1 | none (MotionSense MIT cleared); needs P-AAE six-activity protocol | RUN NOW (CPU, dev partition, exploratory until the salt exists) |
| 4 (2) | H-IMU-003 split memorisation | **CHANGED** (PD split is TRIAL-level, all 24 users in both parts, not window-level; also read the AAE paper's split) [C] | 0.80 | 4 | 1.15 | 0.5 / 0.5 | none | RUN NOW |
| 5 (4) | H-S-003 evidence budget | same (feeds S-006; KTH WITHIN K at most 4, ACROSS at most 6) | 0.45 mixed / 0.75 core | 4 | 2.6 | 0 | none | RUN NOW (inside S-006) |
| 6 (5) | H-S-002 selection bias of the max, loop null | same, plus sealed-query ledger semantics | 0.70 core (0.35 as written) | 4 | 2.8 | 0.25 (+0.5 real) | real K = 10-48 needs IMU cache | RUN NOW (synthetic + loop null); real half after H-IMU-007 |
| 7 (3) | H-S-005 one-line statistic | **CHANGED** (CXR metadata line on MN-v2 now doubles as validity-gate part 1) [C] | 0.60 | 3 | 2.3 | 0.25 / 0.5 | image line: CXR download | RUN NOW (CSV-only); image line after H-CXR-001 |
| 8 (8) | H-CXR-001 demographic + device attacker, **MN-v2 pair-feature matched design**, three-way split, validity gate | **CHANGED** (A2: matching on pair-feature distribution so each stratum has equal positive/negative frequency of same-view, same-dimension, same-spacing flags; patient counts per role printed; dev fold = official test list, sealed fold = hash fold nobody has seen) [C] | 0.70 | 5 | 0.22 | 4 / 4 (CPU part 0) | CPU part none; estimators: download conditions + `T.md` | RUN NOW (CSV-only: MN-v2 construction, true-metadata and device-only GBM at most 0.56 with CIs, role counts); estimators BLOCKED |
| 9 (-) | **H-IMU-007** AAE re-audit (proposed, no file yet): (i) replicate identity below 7% at activity above 92% under the paper's own protocol (K0); (ii) retrained suite on the AAE release, sealed salted partition, worst-suite identity at least 0.20 at activity at least 0.92; (iii) AAE vs the floor envelope (activity-LDA-k + dither, quantiser, PCA, noise) | **NEW** (A1) | (i) 0.70, (ii) 0.55, (iii) AAE within MDE of the envelope 0.60; core = (ii) 0.55 | 5 | 0.25 | 4 / 4 (dev replication 1 + sealed 3) | `H.md` amendment (P-AAE, salted partition, floor list frozen), main agent's salt, evaluator freeze, AAE code availability (re-implement if none) | RUN AFTER H-IMU-006' and H-IMU-003 (dev replication now; sealed numbers after salt) |
| 10 (7) | H-IMU-002 retrained suite vs frozen raw-trained attacker, now on the AAE release | **CHANGED** (target) | 0.75 | 5 | 0.8 | 1 incr. / 3 | H-IMU-007 releases | RUN AFTER H-IMU-007 |
| 11 (13) | H-CXR-005 acquisition fingerprint, now the border-only-CNN leg of the validity gate on MN-v2 (at most 0.56) | **CHANGED** (promoted from late control to gate; runs BEFORE any Siamese) [C] | 0.60 that the gate passes first try (0.50 for all original clauses) | 5 | 1.4 | 0.7 / 1.5 | download conditions (a),(d); `T.md` | RUN AFTER download conditions, before H-CXR-002 |
| 12 (9) | H-CXR-002 stage-0 gate: raw Siamese on MN-v2 beats the best metadata/device/estimated-demographic line by at least 0.03 AUC | **CHANGED** (premise "0.994 cannot be demographics" is contradicted by age+sex = 0.983; mechanism rewritten as "residual after matching is anatomy plus unmatched acquisition"; prior [0.88] to 0.70, main agent says about 0.6) [C] | 0.70 | 3 | 0.13 | 4 / 4 (raw estimators + 5 folds dev/sealed once) | H-CXR-005 validity, H-CXR-001 | RUN AFTER H-CXR-005 |
| 13 (10) | H-CXR-004 attacker-side re-audit, held-out family | **CHANGED** (stage a on floors + released-weights sensitivity arm; clean PriCheXy arm needs the retrain) | 0.55 on matched negatives; 0.95 on random negatives | 5 | 0.33 | 3 / 12 | H-CXR-002 | RUN AFTER H-CXR-002 |
| 14 (11) | H-CXR-003 floors vs PriCheXy-Net | **CHANGED** (floors-only stage a is unaffected; the clean learned-defence arm costs +8 GPU-h for a three-way-split retrain, one mu first) | 0.60 | 5 | 0.24 | 4 / 20 (+8 retrain) | H-CXR-002; retrain: main agent's call after stage a | RUN AFTER H-CXR-002 (stage a); retrain next window |
| 15 (14) | H-X-001 dither / lambda=0 envelope | **CHANGED** (first concrete cell is the AAE with and without its identity term plus dither; margin must be at least the IMU MDE, about 0.10, so it is judged across cells) [C] | 0.70 (falsification 0.30) | 5 | 0.59 | 1.5 incr. | H-IMU-007 | RUN WITH H-IMU-007 |
| 16 (6) | H-IMU-001 arm A, PD released checkpoint | **CHANGED** (demoted to CROSS-THREAT-MODEL CONTROL; 96 clean trials; MDE80 9-13 printed; exploratory; cheap now) | 0.70 | 2 | 0.23 | 1.5 / 1.5 | none (R0 job already queued) | RUN NOW at current priority, labelled control; no WIN claim |
| 17 (12) | H-CXR-007 visit gap / near-duplicates | same [C] | 0.40 | 2 | 0.8 | 0.5 | scores of 002/004 | RUN AFTER H-CXR-004 |
| 18 (16) | H-S-004 saturation, view vs architecture | same (families must be views, F16) | 0.55 core | 4 | 1.6 | 0.5 / 4 | H-IMU-002 preds | RUN AFTER H-IMU-002 |
| 19 (1b) | **H-S-001b** planted leaks injected into REAL defended releases, at least two unseen families, sealed manifest | **NEW** (A3; negative-control builder) | 0.70 that the suite misses at least one unseen family at some dose | 3 | 0.5 | 1 / 4 | negative-control builder launched; defended releases exist (IMU after H-IMU-007, CXR after floors) | RUN AFTER negative-control builder + H-IMU-007 releases |
| 20 (17) | H-CXR-006 pooled visits | same (mean-pool only) | 0.40 | 2 | 0.8 | 0.5 / 4 | H-CXR-003 fits | RUN AFTER H-CXR-003 |
| 21 (19) | H-IMU-001 arm B identity-guided PD retrain | **CHANGED** (learned arm; in-programme retrain so it can be cross-fit on the salted partition) | 0.10 resolvable learned win | 5 | 0.06 | 8 | H-IMU-007 headroom; K1 | RUN AFTER H-IMU-007 only if the AAE leaves headroom over the envelope |
| 22 (18) | H-G-002 CXR latent LOFO | **CHANGED** (A6: G stays CLOSED) | 0.12 | 5 | 0.18 | 3 | A6 | PARK (re-open trigger below) |
| 23 (20) | H-IMU-004 quantiser-alone share | same | 0.40 | 2 | 0.10 | 4 | unverified C-AAE numbers; AAE now primary | PARK (trigger: H-IMU-007 (iii) shows the AAE above the envelope; then attribute quantiser vs AE) |
| 24 (21) | H-IMU-005 learned obfuscator, held-out utility | same | 0.07 | 3 | 0.03 | 7 | MDE; six-activity design | PARK |
| 25 (22) | H-G-001 IMU LOFO gate | same | 0.05 | 3 | 0.06 | 3 | MDE 0.12 at fold O; A6 | KILL |
| - | H-X-002 stop rule | **CHANGED** (adopt with the A-gates and primary endpoints of section 2) | 0.85 no learned line | - | - | 0 | all gates | ADOPT |

**Re-open trigger for G (the one place it is powered):** CXR-003a shows a learned defence below the floor envelope by at least 0.05 AUC, or H-X-001 is falsified in at least two cells. Families must then be defined as views and frozen first (F16); today's G-002 families G1 (MLP/Siamese head) and G2 (kNN) are one view.

### Row notes (only where the reason is not in section 1)

* **H-S-006.** `run_real.py` (Builder S) already has `cert`, `select` and `budget` subcommands on the harness; HMDB51 (`hmdb51_vp`) is not in its `DATASETS`. Output per release: the programme-1 headline margin with its recording-cluster SE and MDE80, whether the better of the stored families flips, and the evidence-budget curve. Add new attacker families on KTH only if this shows headroom (A3).
* **H-IMU-006'.** Single seed, 48 trials in the audit; replicate on all 96 static trials with the three rank folds. The whitening/band-limit test separates "sensor-noise fingerprint" (falls) from "posture/tremor" (survives); either outcome decides what a defence can claim on static windows.
* **H-IMU-007.** Cheapest informative sequence: dev replication of the published claim (hours, CPU/GPU small) decides whether the AAE is a sound target at all; only then the sealed salted-partition run. The AAE's identity term gives X-001 a natural lambda contrast.
* **H-CXR-001/-002.** The validity gate (true-metadata GBM, device-only GBM, border-only CNN all at most 0.56 with CIs, on the dev fold, before any image model) is a hard precondition of H-CXR-002. My probability that MN-v2 passes all three on the first try is about 0.6; failure triggers re-matching on the dev fold only.
* **H-CXR-003/-004.** F5 of pass 1 still holds: report L = (AUC - 0.5)/(AUC_raw - 0.5) and class near-raw ties as "both ineffective". Also state, as A2 requires, that the verdict covers one learned defence at three mu.
* **H-X-001.** With IMU MDE about 0.10 per cell, one AAE cell can only support "no resolvable difference"; the falsification (adversary beats dither by at least the MDE in two independent cells) needs a CXR cell as the second.

## 4. What is blocked by each gate

| Gate | Blocks | Does not block |
|---|---|---|
| **G1 download conditions** (a) licence snapshot committed, (d) integrity check vs an official copy (billing project or Kaggle), (e) `hf_allow` in every `t_` spec | image download start ((a) only), first image number ((d)) | CSV-only work (MN-v2 build, metadata GBMs, role counts), PriCheXy repo/weight fetch (GPL-3, separate process) |
| **G2 pre-registrations**: `T.md` with the A2 template (three-way counts, pair-feature matching, validity gate, machine-checkable kill rule, `verdict_original`/`verdict_tightened`); `H.md` amendment for P-AAE, salted partition, frozen floor list | any model number in the focus that is not labelled development | development-set runs, clearly labelled exploratory |
| **G3 salt and sealed-split custody** (main agent commits the salt after the floor list is frozen; CXR hash fold nobody has looked at) | every headline (sealed) IMU and CXR number | dev partitions (trials 11-16, official test list) |
| **G4 evaluator freeze + sealed-query ledger + prereg-ancestor check** (S) | every sealed certificate | S-006 and all unsealed re-analyses |
| **G5 negative-control builder** with sealed manifest | S-001b and any minimum-detectable-dose statement | S-001a smoke |
| **G6 independent replication auditor** re-derives each headline from raw result files | acceptance of any headline | running jobs |
| **G7 IMU power** (Builder H K1: empirical MDE80_L at fold O at most 0.15; salted pooled design about 0.08-0.10) | IMU-004, IMU-005, any IMU "beats/matches" claim outside large gaps | descriptive numbers, UNRESOLVED reporting |
| **G8 A6: G closed** | H-G-002 (parked), H-G-001 (killed) | - |
| **G9 AAE code and protocol** (Builder H reads 1810.11546; re-implement if no code) | H-IMU-007 timing | PD control, strata, split diagnostics |

Pass-1 control list still applies (C1-C12), with these changes: C2 headroom gates and R0 are required of the AAE arm and (as a control) of PD; C5 MN-v1 validity becomes MN-v2 validity with the border-only CNN leg; C7 fingerprint moves ahead of H-CXR-002; C11 "defender-seen" now also covers CXR (released weights saw all official train_val patients); add C13 burned-set flag (trials 11-16, official test list, PriCheXy's 5,000 pairs are development); C14 evaluator freeze/ledger; C15 static/dynamic strata on every IMU table.

## 5. Revised queue for the next 24 GPU-hours (4 A6000)

Same rules: spend, not wall-clock; one job per GPU worker; CPU threads at most 8; `yaml.safe_load` before queueing; priorities 10-30; `<5` is the main agent's. Firm items total about 20 GPU-h; I do not fill the window (the PriCheXy-Net retrain, about 8 GPU-h, is deliberately outside it, see Phase 2e). If the CXR conditions are not met, firm spend is about 10 GPU-h (IMU and S only).

| Phase | Order | Job names | GPU-h | GPU | Starts when | Decides |
|---|---:|---|---:|---|---|---|
| 0 CPU, now | 0a | `r_s006_rescore_p1_{kth_ilic,kth_scenario,utdmhad,charades}` (`run_real cert|select|budget`) | 0 | none | now | charter clause 1; unit of analysis; evidence-budget curves (S-003) |
| | 0b | `r_s001a_null_smoke`, `r_s002_truthinfl_synth`, `r_s002_loopnull_shuf` | 0 | none | now | harness null at chance; selection bias |
| | 0c | `h_s0_dev_floors` (floors list + headroom/oracle/controls on the **dev** partition, P-PD and P-AAE, exploratory) | 1.0 | GPU0 | now | freeze the floor list (so the main agent can commit the salt); H1-H4 |
| | 0d | `h_s0_strata_split` (H-IMU-006' strata + whitening test, H-IMU-003 diagnostics, 96 static trials) | 0.3 | GPU0 | now | C3, C15; static-identity composition |
| | 0e | `t_s0_mnv2_meta_dev` (CSV only: MN-v2, true-metadata and device-only GBM with CIs, role counts) | 0 | none | now | validity gate part 1; the three-way patient counts |
| 1 IMU | 1a | `h_r0_pd_pub` (already queued) and `h_pd_pub_s0` (control; fold-O releases only needed) | 0.5 | GPU1 | now | PD control, K0 for PD; exploratory |
| | 1b | `h_aae_replicate_dev` (AAE per its paper's protocol, 3 seeds) | 1.0 | GPU1 | after G9 | H-IMU-007 (i) |
| | 1c | `h_aae_sealed_f{0,1,2}` (AAE + frozen floors + retrained suite, strata static/dynamic, 3 rank folds x 3 seeds, X-001 arms lambda=0 and dither) | 3.5 | GPU0, GPU1 | after G2, G3, G4 and 1b passes | P2, P3, P5; IMU-002; X-001 cell 1 |
| | 1d | `h_pd_control_foldO` (pd_pub s0-2 + pd_plain on the 96 clean trials, suite) | 1.0 | GPU1 | after 1a | PD control row; exploratory |
| 2 CXR | 2a | `t_dl_cxr14`, `t_prep256`, `t_integrity_sha500` (0 GPU, 3 CPU-h) | 0 | none | G1 (a) then (d) | data |
| | 2b | `t_validity_gate_dev` (true-metadata, device-only, border-only CNN on MN-v2, dev fold) | 0.7 | GPU2 | 2a | validity gate; H-CXR-005 |
| | 2c | `t_est_age_sex_raw_f{0..4}` 1.5; `t_siam_raw_mnv2_dev` then `_sealed` 2.5 | 4.0 | GPU2, GPU3 | 2b passes; G2, G3 | H-CXR-001, H-CXR-002 gate; STOP CXR if gap below 0.03 |
| | 2d | `t_rel_floors_{rw2,rw3,blur_s4,down_f4}` + `t_pcx_released_sensitivity` 0.5; `t_util_*` 1.0; `t_att_f1_*` 2.0; `t_dino_cache` 1.0; `t_est_defended` 1.0 (F5 NCC on CPU) | 5.5 | GPU2, GPU3 | 2c passes | H-CXR-003a, H-CXR-004a, H-CXR-007, mean-pool H-CXR-006 |
| 3 reserve | 3a | `r_s001b_*` first dose-response on IMU floors/AAE releases (negative-control builder) 1.0; `h_suite_gru` 1.0; buffer 1.0 | 3.0 | any | after G5 / 1c | S-001b; IMU-002 completeness |
| Outside the window | 2e | `t_pcx_retrain_mu0.01_split3` (three-way split, about 8 GPU-h, ASSUMED range 4-12) | 8 | one GPU | after 2d, main agent's call | the clean learned-defence arm of H-CXR-003/-004 |

Subtotal: 1.3 + 0.5 + 1.0 + 3.5 + 1.0 + 0.7 + 4.0 + 5.5 + 3.0 = 20.5. No arm B, X-001 beyond cell 1, H-IMU-005, G-002 in this window.

Config sketches (entrypoints marked NEW are pending the builders; the existing job specs use `$OUT` and `$HOME/venvs/e/bin/python`; every `t_` spec must carry `hf_allow`):

```yaml
name: r_s006_rescore_p1_kth_ilic            # existing CLI (s_suite/run_real.py); HMDB51 not yet in DATASETS
priority: 10
cmd: "$HOME/venvs/e/bin/python -m s_suite.run_real select --root $HARNESS_DIR --out $OUT && $HOME/venvs/e/bin/python -m s_suite.run_real budget --root $HARNESS_DIR --out $OUT"
hf_path: runs/r_s006_rescore_p1_kth_ilic
---
name: h_aae_replicate_dev                    # NEW entrypoint (Builder H, after G9); dev partition = exploratory
priority: 12
cmd: "OMP_NUM_THREADS=8 $HOME/venvs/e/bin/python research/adversarial/src/h_imu/aae.py replicate --protocol P-AAE --partition dev --seeds 0,1,2 --out $OUT"
hf_path: runs/h_aae_replicate_dev
---
name: h_aae_sealed_f0                        # NEW; refuses to run without the committed salt and evaluator tag (G3, G4)
priority: 13
cmd: "OMP_NUM_THREADS=8 $HOME/venvs/e/bin/python research/adversarial/src/h_imu/aae.py sealed --fold 0 --salt_file $SALT --eval_tag $EVAL_TAG --strata static,dynamic --out $OUT"
hf_path: runs/h_aae_sealed_f0
---
name: t_validity_gate_dev                    # NEW entrypoint (Builder T); BLOCKED until G1 (a),(d) and T.md
priority: 16
cmd: "$HOME/venvs/e/bin/python research/adversarial/src/t_cxr/validity_gate.py --design mnv2 --partition dev --legs metadata,device,border_cnn --out $OUT"
hf_path: runs/t_validity_gate_dev
hf_allow: ["*.json", "*.log", "*.md"]        # REQUIRED for t_ jobs (WHITELIST); scores as scores_*.json keyed by hashed ids, no .npz/.pt
---
name: t_siam_raw_mnv2_sealed                 # NEW; x5 folds; sealed query goes through the ledger
priority: 18
cmd: "$HOME/venvs/e/bin/python research/adversarial/src/t_cxr/run_gate.py --attacker siamese_r50 --release raw --design mnv2 --partition sealed --seeds 0,1,2 --out $OUT"
hf_path: runs/t_siam_raw_mnv2_sealed
hf_allow: ["*.json", "*.log", "*.md"]
```

## 6. Disagreements with the amendments and with pass 1

1. **H-CXR-002 prior.** I put 0.70, the main agent 0.6, the generator 0.85. The premise text is contradicted and must be rewritten, but the gate (at least 0.03 over a metadata floor held at most 0.56 by construction) fails only if over-matching removes anatomy or the unmatched-acquisition residual is the whole signal. The decision does not hinge on 0.6 versus 0.7.
2. **Reason given for closing G (A6).** "MDE 7-8 points" is the IMU/KTH figure; on CXR the MDE is about 0.015-0.02 AUC, so G is testable there. The verdict (closed now) is right because no defender class has shown headroom over the floor envelope, not because of power; I therefore PARK G-002 with an explicit re-open trigger instead of calling it closed for good.
3. **Builder H's v0 conflicts with A1.** v0 picks floors on folds A+B and compares on fold O, which is the burned set, and P-PD (four dynamic activities, two magnitude channels) cannot produce static/dynamic strata. A tightening amendment is needed (salted partition for in-programme defenders; P-AAE protocol; fold O kept only for the PD control). Until then, Builder H numbers are development numbers.
4. **A2 is not satisfiable with the released PriCheXy weights** at good power (E3). It silently adds an 8 GPU-h retrain (outside this window) or leaves a headline of about 700 sealed patients (MDE about 0.03). The main agent should choose; I recommend the retrain after floors-first stage a.
5. **`hf_allow` vs the S evaluator and the replication auditor.** The allow-list drops `.npz` and `.pt`; CXR scores must be JSON keyed by hashed IDs (or the list extended), and the auditor reads the VM copy.
6. **S-001 as the S builder's deliverable.** After F10 the suite builder's planted leaks prove only that its own features see its own leaks; keep them as a smoke test (S-001a) and do not quote any minimum detectable dose until S-001b with unseen families exists. The old S-001 MDD band (18-30% oracle accuracy) is withdrawn.
7. **Static stratum power.** Stratifying by static/dynamic halves the cells; the static stratum has 96 trials (MDE about 0.12-0.16). F7's 47.9% is single seed, 48 trials (SE about 7 points); the falsification of the old prediction is robust, the size is not. Report strata, but do not rank defenders inside the static stratum.
8. **Cost correction.** PD sampling is cheap (about 0.3 GPU-h per release, one smoke job), so pass 1's 3.5 GPU-h was too high; the PD control stays in the queue at about 1.5 GPU-h in total, labelled exploratory.
9. **The AAE claim is itself unverified by me.** Neither the audit nor I read 1810.11546's attacker protocol or split; if the "<7%" was measured by an attacker trained on raw data or under a random window split, H-IMU-003 and H-IMU-002 apply to the AAE's own claim as much as to PD's.
10. **Primary endpoints (section 2) are my proposal**, not an amendment; the main agent should confirm or replace them before the first sealed number, since F18 makes them part of the pre-registration.
