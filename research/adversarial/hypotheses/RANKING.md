# Hypothesis ranking, programme 2, ranker round 1 (2026-11-04)

Inputs read in full: CHARTER, FOCUS, journal 2026-11-04, `hypotheses/INDEX.md` and all 22 `H-*.md`, `literature/SYNTHESIS.md`, `literature/contrarian_stress_test.md`, the head of `contrarian_missed.md`,
`results_notes/H.md` (Builder H pre-registration v0, committed before any model number), the Builder H/S/T source files that exist now (`src/h_imu`, `src/s_suite`, `src/t_cxr`), programme 1's `RANKING_R2.md` (format and heuristic).
I ran nothing on the VM and no git command. The only computations I made are the binary-entropy table behind the info column and a hand evaluation of Builder H's analytic power formula.

## 0. Evidence boundary (read before the numbers)

* **No programme-2 result exists.** `results_notes/` holds only `H.md` (pre-registration, section 5 "Results: none yet"); there is no `S.md`, no `T.md`. Every likelihood below is my prior from design reading and programme 1, not a measurement.
* **`research/adversarial/AUDIT_findings.md` does not exist yet** (Auditor-2 has not written it), so the ChestX-ray14 licence clearance is still pending. Everything that needs the 45 GB images is BLOCKED.
* **MotionSense is cleared by Builder H** (`H.md` section 0: MIT, identity attacks allowed, GO; WISDM CC BY 4.0 GO as a replication set). Residual: the IoTDI'19 consent text was not checked; ask Auditor-2 to ratify in `AUDIT_findings.md`. So R-IMU is not blocked by licence, only by its own stage-0 gates.
* Labels: MEASURED = stress test on real artefacts; ANALYTIC = formula; ESTIMATED/ASSUMED = planning numbers; "my prior" = ranker's probability. "Builder H MC" = Monte-Carlo in `H.md` 2.7.
* GPU-h = worker-hours, staged cost (the cost of the version I recommend running first) unless two numbers are given as "staged / full".

## 1. Five facts that move the ranking

**F1. Builder H's pre-registration changes the IMU testbed under the generator's feet.** From `H.md` 2.1-2.9 (read from the PrivDiffuser repo): PD's MotionSense protocol is **4 activities (dws, ups, wlk, jog), 2 channels (|rotationRate|, |userAcceleration+gravity|), window 128, stride 10**, test trials {11,12,15,16} = **96 test units** (fold O); the shipped checkpoint trained on every other trial, so folds A and B are *defender-seen*. The independent unit is the **subject (24)**, not the trial. Consequences: (i) the generator's numbers (raw identity 74-78%, LDA-5 code, 144 test trials, 6 activities) belong to the stress test's 12-channel/6-activity setting; with 4 activities the label-supervised floor has at most k = 3 dims; (ii) PD conditions on a 60-d embedding of the input window and a gender label, so identity can pass through the conditioning vector (Builder H's `pdz60` arm tests exactly that); (iii) PD's shipped evaluation models are MLPs on the flat window, not the CNN of the paper text.

**F2. Subject clustering makes IMU resolution about 2x worse than the generator's MDE.** Builder H MC (`H.md` 2.7, rho 0.3, r 0.5): paired MDE80 = **0.12 (p 0.1) to 0.16 (p 0.3) at fold O (96 trials)** and 0.08-0.10 on the pooled 264 trials. I hand-evaluated the same closed form and got the same 0.108-0.144 (analytic cells), so it is the same formula, not independent confirmation. The generator used 7.5 (n = 144) and **4.8 points (cross-fit)**; rotating trials re-tests the same 24 subjects, so cross-fit buys about 13% (0.108 to 0.094), not the 25%+ that H-S-002 clause 4 needs. Hypotheses with thresholds at or below about 8 points on IMU (IMU-005, G-001, X-001's 5-point margin, S-004's "within MDE80", S-005's "within MDE80") cannot be resolved by design. Builder H's own kill K1 (empirical MDE80_L at fold O above 0.15) has, in my judgement, about a 35% chance of firing.

**F3. Most S and IMU hypotheses are analyses over one cache, not experiments.** The cache is `preds.npz` per (arm, fold, attacker, seed) of the Builder H P-PD suite. IMU-002, IMU-003, S-002, S-003 (IMU arm), S-004, S-005 and the IMU half of X-001 read it. The only expensive items are PD sampling (about 3-4 GPU-h, the long pole) and, on CXR, the release x attacker matrix. Group by cache, not by file name (section 2).

**F4. The CXR protocol gate is mostly CPU and needs only the 11 MB CSV that is already on the VM.** MN-v1 construction, its metadata-validity check (Builder T `metadata_floor`, must be at most 0.56 at every matching tier), and the true-metadata lines on PriCheXy's own pairs (MEASURED 0.983) need no images. Note `pairs.py` tiers 2-3 relax to age within 1-2 y and (W,H) within 10-20% without pixel spacing at the loosest tier: validity must be checked per tier, not once.

**F5. Floor-versus-learned comparisons on CXR can be vacuous.** A retrained attacker adapts to a small random smooth warp. I expect PriCheXy mu = 0.01 *and* the equal-strength random flow both to leak near raw under a retrained suite (LAPIS, scout-read: PriCheXy worst case 0.826). "The floor envelope is within 0.03 of PriCheXy" would then hold because both fail, not because floors defend. H-CXR-003 and H-CXR-004 must report L = (AUC - 0.5)/(AUC_raw - 0.5) and class a tie near L = 1 as "both ineffective", never as "the floor matches the learned defence".

## 2. Groups (shared experiments, caches, dependencies)

| Group | Members | Shared object | Gate |
|---|---|---|---|
| **G1 S-synthetic (no data, CPU)** | S-001 (synthetic arm), S-002 (synthetic truth + shuffled-label loop null + harness K <= 4), S-003 (KTH/Charades/UTD-MHAD) | Builder S `planted.py`, `run_planted.py`, `harness_io.py`, `evidence_budget.py`; HF `harness/*/eval/*/preds.npz` (78 files, programme-1 artefacts, permitted by the data-use gate) | none |
| **G2 IMU P-PD cache** | IMU-001 (arm A + floors), IMU-002, IMU-003, IMU-004 (floor arms only), S-004/S-005/S-002/S-001 (IMU halves), X-001 (noise/quant floors), IMU-006 (12-channel side check) | Builder H stage-0 floors and suite `preds.npz`; PD releases (`pd_pub`, `pd_plain`, `pd_w1_1.8`, `pd_w2_0.5`) | H1-H4 headroom, R0 reproduction (K0), K1 power |
| **G3 CXR stage 0** | CXR-001, CXR-002, CXR-005, S-005 (CXR half), S-001 (CXR arm) | 256-px uint8 bank, MN-v1 triples, OOF age/sex estimators, raw Siamese-R50 per fold | download gate (images); CSV-only parts free |
| **G4 CXR release ladder** | CXR-003, CXR-004, CXR-006, CXR-007, S-002/S-004 (CXR halves), X-001 (CXR), G-002 (shares DINOv2-L latent cache with F4) | release x attacker score matrix (saved pair scores) | G3 gate (CXR-002), download gate |
| **G5 learned lines (conditional)** | IMU-001 arm B, IMU-004 (AE arm), IMU-005, G-001, G-002 | encoders/adapters; need X-001 arms alongside | headroom and X-001, MDE |
| **G6 governance** | X-002 | reads gates | none |

Dependency chains: G1 -> (validity of every suite number); G2: stage 0 (floors, oracle, controls, power) -> R0 -> PD sampling -> suite -> IMU-001A/002/004; arm B only if floors leave headroom and K1 did not fire. G3 -> G4 -> G-002. IMU-006 and IMU-003 are side controls.

## 3. Axes

* **Likelihood** = my probability that the decision-relevant core clause holds (the generator's prior in brackets). Where the file is a conjunction of several clauses I give the core and say so in the row notes.
* **App (0-5)** = D (0-2: changes what we train or choose, or how we report) + U (0-1: unlocks or prunes at least 3 follow-ups) + T (0-1: transfers across R-CXR, R-IMU, S) + L (0-1: is or uses a learned method/attacker).
* **Info/GPU-h** = H2(likelihood) x (App/5) / max(cost, 0.25), H2 = binary entropy in bits. A heuristic with a 0.25 floor, so free analyses score 1-3. **Tiers decide order; the number breaks ties inside a tier; a gate precedes what it gates.**
  Tier 0 = control that must pass before any number is believed; Tier 1 = low-GPU measurement on existing artefacts; Tier 2 = headline audit; Tier 3 = conditional learned line; Tier 4 = park/kill.
* **[C]** marks a control that must pass before an exciting result is believed (section 6).

## 4. (a) Ranking table

| Rank | ID | Group / tier | Likelihood [gen] | App | Info/GPU-h | Cost GPU-h (staged / full) | Blocked by | Verdict |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | H-S-001 [C] planted-leak dose-response and null | G1 / 0 | 0.65 [0.70] (null at chance 0.85; MDD band 0.65) | 3 | 1.1 | 0.5 / 4 | CXR arm: download gate; IMU donor-swap arm: G2 stage 0 | RUN NOW (synthetic arm); IMU arm RUN AFTER H stage 0; CXR arm BLOCKED |
| 2 | H-IMU-003 [C] window-split memorisation | G2 / 0 | 0.80 [0.75] | 4 | 1.2 | 0.5 / 0.5 | none (MotionSense MIT, Builder H GO) | RUN NOW (already in `H.md` 2.10 a,b) |
| 3 | H-S-005 [C] one-line statistic vs suite | G1+G2+G3 / 0 | 0.60 [0.60] (CXR half 0.80; IMU low-leak half power-confounded) | 3 | 2.3 | 0.25 / 0.5 | CXR image-estimated line: CXR-001 | RUN NOW (IMU stat + CXR metadata on CSV); image line RUN AFTER H-CXR-001 |
| 4 | H-S-003 evidence budget | G1 / 1 | 0.45 mixed as written; 0.75 core [0.50] | 4 | 2.6 | 0 / 0 | none (HF preds are ours) | RUN NOW (KTH WITHIN limited to K <= 4); MotionSense arm RUN AFTER H-IMU-002 |
| 5 | H-S-002 [C] selection bias of the max, search-loop null | G1 / 0-1 | 0.70 core, 0.35 as written [0.75] | 4 | 2.8 | 0.25 (+0.5 real) / 1 | real K = 10-48 needs G2/G4 preds | RUN NOW (synthetic truth + shuffled-label loop null); real half RUN AFTER H-IMU-002 |
| 6 | H-IMU-001 arm A [C] floors frontier + PD release | G2 / 2 | 0.65 [0.70] (restated in Builder H's WIN/K terms) | 5 | 0.16 | 6 / 6 | stage 0 gates H1-H4, R0 (K0), K1 | RUN NOW (stage 0, CPU); PD sampling RUN AFTER H1-H4 + R0 |
| 7 | H-IMU-002 retrained suite vs frozen raw-trained attacker | G2 / 2 | 0.75 [0.70] | 5 | 0.8 | 1 incr. / 3 | H-IMU-001 stage 0 floors; PD column after sampling | RUN AFTER H-IMU-001 (stage 0) |
| 8 | H-CXR-001 [C] demographic + device attacker; MN-v1 validity | G3 / 0 | 0.70 [0.65] | 5 | 0.22 | 4 / 4 (metadata part 0) | image estimators: download gate | RUN NOW (CSV-only: MN-v1 validity, true-metadata lines); estimators BLOCKED |
| 9 | H-CXR-002 [C] stage-0 gate: identity beyond demographics | G3 / 0 | 0.88 [0.85] | 3 | 0.11 | 3 / 3 | download gate; H-CXR-001 | RUN AFTER H-CXR-001 (BLOCKED until clearance) |
| 10 | H-CXR-004 attacker-side re-audit, held-out family | G4 / 2 | 0.55 on MN-v1; 0.95 on random negatives [0.75] | 5 | 0.33 | 3 / 12 | download gate; H-CXR-002 | RUN AFTER H-CXR-002 (stage a: F1, F4, F5, F6 on pcx mu=0.01, rw, raw) |
| 11 | H-CXR-003 floors vs PriCheXy-Net | G4 / 2 | 0.60 [0.55] | 5 | 0.24 | 4 / 20 | download gate; H-CXR-002 | RUN AFTER H-CXR-002 (stage a: equal-strength random flow x3 mu + blur s4 + down x4) |
| 12 | H-CXR-007 [C] visit gap / near-duplicates | G4 / 0 | 0.40 [0.40] | 2 | 0.8 | 0.5 / 0.5 | saved scores of 002/003/004 | RUN AFTER H-CXR-004 (free) |
| 13 | H-CXR-005 [C] acquisition fingerprint | G3 / 0 | 0.50 [0.55] | 3 | 0.4 | 1.5 / 1.5 | download gate | RUN AFTER H-CXR-002 |
| 14 | H-X-001 [C] dither / lambda = 0 envelope | G2+G4+G5 / 0 | 0.70 [0.70] (falsification 0.30) | 5 | 0.44 | 2 incr. / 4 | needs a learned arm to compare | RUN WITH the first learned arm; dither floors are already inside 001 and 003 |
| 15 | H-IMU-006 [C] static-window posture signature | G2 / 0 | 0.70 [0.70] | 2 | 1.4 | 0 / 0 | none | RUN NOW (CPU filler; 12-channel side check, not in P-PD) |
| 16 | H-S-004 saturation, view vs architecture | G1+G2+G4 / 1 | 0.55 core, 0.35 as written [0.55] | 4 | 1.6 | 0.5 / 4 | H-IMU-002, H-CXR-004 preds | RUN AFTER H-IMU-002 (analysis only; no extra ladder rungs unless the greedy curve still rises) |
| 17 | H-CXR-006 pooled visits | G4 / 3 | 0.40 [0.55] | 2 | 0.8 | 0.5 / 4 | H-CXR-003 fits | RUN AFTER H-CXR-003 (mean-embedding pooling only); learned aggregator PARK |
| 18 | H-G-002 CXR latent LOFO gate | G5 / 3 | 0.12 opens, 0.65 gap exists [0.15] | 5 | 0.18 | 3 / 3 | download gate; CXR-002/-004a; X-001 arms | RUN AFTER H-CXR-004 (conditional learned line) |
| 19 | H-IMU-001 arm B identity-guided PD retrain | G5 / 3 | 0.10 learned win resolvable [0.20] | 5 | 0.06 | 8 / 8 | H-IMU-001A headroom; K1 not fired | RUN AFTER H-IMU-001 A (only if floors leave headroom and K1 did not fire) |
| 20 | H-IMU-004 quantiser-alone share | G5 / 4 | 0.40 [0.50] | 2 | 0.10 | 4 / 4 | unverified C-AAE numbers; no code | PARK (quantiser/dither arms run inside 001 for free) |
| 21 | H-IMU-005 learned obfuscator, held-out utility | G5 / 4 | 0.07 [0.12] | 3 | 0.03 | 7 / 7 | needs 6-activity design; MDE | PARK (trigger: arm B survives X-001, or measured MDE <= 8) |
| 22 | H-G-001 IMU LOFO gate | G5 / 4 | 0.05 opens, gap exists already MEASURED [0.12] | 3 | 0.06 | 3 / 3 | MDE 0.12 at fold O | KILL |
| - | H-X-002 stop rule | G6 | 0.75-0.85 no learned line [0.75] | - | - | 0 | all gates | ADOPT WITH AMENDMENTS (section 9) |

### Row notes (only where I differ or the reason is not obvious)

* **S-001.** The null part ("val-selected member exceeds the Bonferroni bound in at most 8% of 200 null replications, shuffled/constant at chance") is a harness-bug detector and gates every suite number. The MDD band (oracle accuracy 18-30%) assumes trial units; with subject as the unit (Builder H) it moves up and depends on the within-subject error correlation, which Builder H's empirical power audit measures. Builder S's `run_planted.py` defaults to `--procs 24`: the charter caps CPU threads at 8, run with `--procs 8`.
* **IMU-003.** Builder H's folds are trial-held-out by construction, so the identity half is largely a documented diagnostic (T2 vs T1, `H.md` 2.2). The gender half reproduces PD's own all-24-users protocol against subject-disjoint T3; at 24 subjects it is descriptive only (MDE 0.11-0.26).
* **S-005.** Falsifier "line within MDE80 of the suite in at least 70% of IMU low-leak cells" conflates unresolved with equal; at fold O MDE80 is 0.12-0.16, so it will fire for lack of power. Report as UNRESOLVED. The CXR half is powered (MDE80 about 0.02 AUC) and is the informative one.
* **S-003.** KTH has about 4 clips per test recording (598 clips / 150 recordings, MEASURED) and 6 recordings per person in the test split, so WITHIN pooling is limited to K <= 4 and ACROSS to K <= 6; the K = 6 WITHIN comparison in the file is infeasible. 25 persons make N* coarse. Zero GPU, one engineer-day, and it feeds the unit-of-analysis decision for every other file.
* **S-002.** As written it is a conjunction of four clauses (my as-written probability 0.35). Clause 4 (cross-fit cuts the paired MDE80 by at least 25%) most likely fails because of F2; clauses 1-3 (optimism at K = 10, sealed split removes at least 80%, shuffled-label loop max 7-10%) are the useful ones at 0.70.
* **IMU-001 A.** PD is a gender-guided defender being audited for identity, a secret it was not built to hide, so "PD sits above the floor frontier" is foreseeable (0.65-0.70). The informative parts are (i) the floor frontier on PD's own input (headroom gates H1 raw L at fold O at least 0.40, my P(pass) 0.75; H3 utility at least 0.90), (ii) `pd_pub` vs `pdz60` (does the diffusion step add anything over its conditioning vector), (iii) the K1 power verdict. Use Builder H's WIN_L/WIN_U/K2 rules for the verdict; the generator's "at least 15 points above LDA-5" numerics do not transfer (F1).
* **IMU-002.** Cheap given the cache: Builder H already scores a frozen shipped Keras MLP ("frozen raw-trained attacker", `H.md` 2.10 f) and a main suite with cnn1d/mlp_flat/trees/knn/linear plus a held-out transformer/SVM. GRU is optional (about 1 GPU-h). Prior on "median understatement at least 20 points" 0.75 after programme 1 (KTH frozen understated by a median 47 points, as cited in the H-IMU-002 file; I did not re-derive it).
* **CXR-001.** Core = real age/sex estimators give a demographic+device AUC at least 0.80 on PriCheXy random negatives and at most 0.56 on MN-v1. The 0.885 figure is a SIMULATION (age sd 5 y); it must be measured before it is quoted. This is the file that earns the protocol change.
* **CXR-002.** Pass probability is high, so information per GPU-h is low, but it is the stage-0 kill gate for roughly 40 further GPU-h. Execute as a pair with CXR-001 (the estimator defines "best demographic line").
* **CXR-004.** Split P1: on random negatives it is near certain (the demographic line alone is about 0.885 vs PriCheXy's own 0.577, so +0.08 over PriCheXy's attacker is almost free); on MN-v1 I give 0.55. Only the MN-v1 half is informative. P4 (defence-agnostic pretrained attacker) 0.45, the only new learned-attacker idea in R-CXR; run it in stage b, not stage a.
* **CXR-003.** See F5. Stage a (equal-strength random flow at three mu, blur s4, x4 downsample; 4 releases at about 0.75 GPU-h with the utility classifier, plus the cheap attackers) is a hard stage gate: if the floor envelope is within 0.03 of PriCheXy at 2 of 3 mu *and* PriCheXy's L is below 0.8, stop the 16 GPU-h remainder.
* **CXR-006.** Clause 4 of the file predicts the learned aggregator adds at most 0.02 over mean-embedding pooling; take the hint and do not build it. Lift is also capped by AUC saturation if the single-pair AUC is already above 0.9, which lowers my prior to 0.40.
* **CXR-007.** Free once scores exist; prior is a two-part pattern (flat raw, steep defended), so either part failing is informative, which is why it is a control.
* **X-001.** The dither and quantiser arms are already floor arms in Builder H (`noise_s*`, `quant_b*`) and in H-CXR-003, so X-001's incremental cost is the lambda = 0 AE arm that only exists once a learned arm does (Builder H's `dae*` is wave 2).
* **S-004.** The "within MDE80 with at most 3 members" test is satisfied trivially on IMU (MDE 0.12+); only the CXR half is informative. The analysis-only version costs 0.5 GPU-h.
* **G-001 KILL.** Its first half (a held-out-family gap exists) is already MEASURED by the stress test (a 5-dim LDA code: LR 7.6% vs RF 34%); its second half (population defender beats the controls by at least 7 points) is below the fold-O MDE80 of 0.12. A gate whose answer is fixed by power is not worth 3 GPU-h. Re-open only if Builder H's empirical paired SE gives MDE80 at most 7 points on the LOFO contrast.
* **G-002.** The only place G is testable (MDE about 0.02 AUC at 1,424 patients). Keep it conditional behind CXR-004a and the X-001 arms.

## 5. Blocked list

* **BLOCKED on Auditor-2 licence clearance of ChestX-ray14 and the main agent's release** (AUDIT_findings.md absent at ranking time): the 45 GB download and everything downstream: CXR-001 estimators, CXR-002, -003, -004, -005, -006, -007, S-001 CXR arm, S-002/S-004 CXR halves, S-005 CXR image-estimated line, G-002.
  Free before the gate (CSV-only, already on the VM, or synthetic): MN-v1 construction and per-tier metadata validity, true-metadata lines on PriCheXy's 5,000 pairs, S-005 CXR metadata line, evaluator unit tests, PriCheXy repo/weights fetch (GPL-3, separate process).
* **Gated by Builder H's own stage 0, not by a licence:** PD sampling (R0 reproduction K0, headroom H1-H4), IMU-001 arm B and IMU-005 (headroom, K1).
* **No GPU and no download needed (programme-1 harness `preds.npz` exist, 78 files):** S-003, S-002 (harness K <= 4 + synthetic), S-001 synthetic arm.
* **Excluded from identity work by the data-use gate:** CT-RATE, NLST/TCIA, Capture-24, BOXRR-23.

## 6. Controls that must pass BEFORE an exciting result is believed

| Control | File | Guards | Pass condition |
|---|---|---|---|
| C1 evaluator null | S-001 null arm, S-002 loop null, standing constant/shuffled inputs | every suite number | null violation of the Bonferroni bound at most 8% over 200 replications; shuffled-label search max within chance + 1.5 SD; constant/shuffled at chance for every member |
| C2 headroom/oracle and R0 | Builder H gates H1-H4, K0 | every IMU number | raw L at fold O at least 0.40; controls at most 0.10; utility at least 0.90; PD's own published numbers reproduced (R0) |
| C3 split memorisation | IMU-003 | any window-level or PD-protocol attribute claim | report T2 vs T1 inflation; no window-split number is carried; PD's 49.96% gender is read as a memorising attacker |
| C4 power | Builder H K1, `H.md` 2.8 | any "PD beats/matches floors" | empirical MDE80_L at fold O at most 0.15, else print UNRESOLVED |
| C5 MN-v1 validity | CXR-001 (CPU part) | every CXR image number | true-metadata GBM at most 0.56 on MN-v1 at every tier actually used |
| C6 stage-0 headroom | CXR-002 | all of R-CXR | raw Siamese on MN-v1 beats the best demographic+device line by at least 0.03 AUC |
| C7 confounds | CXR-005, CXR-007 | any raw or defended AUC | fingerprint AUC and near-duplicate share (NCC > 0.95) reported beside the headline; a defended AUC concentrated in adjacent visits is not "linkage" in general |
| C8 one-line statistic | S-005 | the claim that the suite adds value | printed beside every suite number; CXR metadata line beside every pair AUC |
| C9 dither / lambda = 0 | X-001 | any learned win (arm B, IMU-005, G-002) | learned point beats the dither frontier at matched utility by at least MDE80 in at least 2 independent cells, else the win is a bottleneck |
| C10 held-out family | Builder H 2.4, CXR-004 F5 | "worst case" claims | held-out family not more than 0.05 above the main suite, else the suite is non-saturating and the stricter verdict rules |
| C11 defender-seen flag | Builder H 2.2 | any PD number | folds A/B are defender-seen; only fold O is the unseen reading; CXR PriCheXy arm has fixed test patients (seeds replicate training noise only) |
| C12 posture signature | IMU-006 | any claim that IMU identity is gait | only relevant to the 12-channel suite; P-PD collapses attitude into magnitudes |

## 7. (b) Ordered queue for the next 24 GPU-hours on 4 A6000

Interpretation: 24 GPU-h of spend (6 wall-hours on 4 GPUs), the size of the cheap-gate stage, not 24 wall-hours (96 GPU-h), which no gate justifies; if the CXR gate stays closed, expected spend is about 10 GPU-h and the other GPUs stay idle (charter stop rule: do not fill the queue). One job per GPU worker, CPU threads at most 8, YAML validated with `yaml.safe_load`, priorities 10-30 (below 5 is reserved; a new focus's first jobs may use 4 once).

| Phase | Order | Job name(s) | GPU-h | GPU | Gate to start | Decides |
|---|---:|---|---:|---|---|---|
| 0 CPU, now | 0a | `r_s001_planted_synth_{null,bias,gain,spectral}` | 0 | none | none | C1; MDD band; blind-spot check |
| | 0b | `r_s002_truthinfl_synth`, `r_s002_loopnull_shuf` | 0 | none | none | C1; optimism at K = 10; sealed-split bias |
| | 0c | `r_s003_evbudget_{kth_ilic,kth_scenario,charades,utdmhad}` | 0 | none | HF snapshot (38 MB) | unit of analysis (recording vs clip); outcome A/B/C |
| | 0d | `h_s0_ppd_stage0` (controls, oracles, all floors x main suite, folds O/A/B, T2/T3, empirical MDE, frozen Keras reading) | 1.0 | GPU0 | none | C2, C3, C4; K0/K1; floor frontier |
| | 0e | `t_s0_mnv1_meta` (MN-v1 per-tier validity, true-metadata lines, S-005 CXR metadata line) | 0 | none | none (CSV on VM) | C5; protocol shortcut as a by-product |
| | 0f | `h_s0_static12ch` (IMU-006) | 0 | none | none | C12 |
| 1 IMU PD | 1a | `h_r0_pd_gate` (PD's loader, shipped models, w1 7.8, w2 0.05) | 1.0 | GPU0 | 0d passes H1-H4 | K0; if it fails no PD claim |
| | 1b | `h_rel_pd_pub_s0` (folds O,A,B, set A*), `h_rel_pd_plain_s0`, `h_rel_pd_w1_1.8_s0`, `h_rel_pd_w2_0.5_s0`, `h_rel_pd_pub_s{1,2}` (fold O) | 3.5 | GPU0 (and GPU1 once 1a is done) | 1a passes | releases (the long pole) |
| | 1c | `h_suite_pd_{pub,plain,w1,w2}` (cnn1d, mlp_flat on GPU; trees n_jobs 8) | 2.0 | GPU1 | 1b | WIN_L/WIN_U/K2 (IMU-001A); IMU-002; IMU-004 floor arms |
| | 1d | CPU analyses on the cache: `r_s002_real_imu`, `r_s004_imu`, `r_s005_imu`, `r_s003_motionsense` | 0 | none | 1c | S-002/-004/-005 real halves |
| 2 CXR, BLOCKED | 2a | `t_dl_cxr14`, `t_prep256` | 0 (3 CPU-h) | none | AUDIT clearance + main agent release | data |
| | 2b | `t_est_age_sex_raw_f{0..4}` 2.0; `t_siam_raw_mnv1_f{0..4}`, `t_siam_raw_rand_f{0..4}` 3.0 | 5.0 | GPU2, GPU3 | 2a | CXR-001 estimators, CXR-002 gate (C5, C6). STOP CXR if gap below 0.03 |
| | 2c | `t_rel_{pcx_mu0.01,rw2_mu0.01,rw3_mu0.01,blur_s4,down_f4}` 0.5; `t_util_*` 1.25; `t_att_f1_*` 2.5; `t_dino_cache` 1.0 (F4 heads CPU); `t_est_defended_pcx` 1.0; F5 NCC CPU | 6.3 | GPU2, GPU3 | 2b gate passes | CXR-003a, CXR-004a, CXR-007 (free), CXR-006 (mean-pool) |
| | 2d | `t_fp_{raw,pcx}` (acquisition fingerprint) | 1.5 | GPU3 | 2b | C7 |
| 3 reserve | 3a | `h_suite_pd_gru` (IMU-002 sequence model) 1.0; `r_s001_planted_imu_donor` 0.5; buffer | 2.5 | any | after 1c | IMU-002 completeness; S-001 IMU arm |

Subtotal 1.0 + 6.5 + 12.8 + 2.5 = 22.8; the remaining about 1 GPU-h is a buffer. Phases 0 and 1 run now and do not wait for CXR. G-002, IMU-001B, X-001's AE arms and everything in Tier 3/4 are deliberately not in this queue: they start only after the named gates read out.

Config sketches (entrypoints marked NEW are pending Builder H/T/S; `pd_release.py r0|rel` and `s_suite.run_planted` already exist):

```yaml
name: h_s0_ppd_stage0                      # NEW entrypoint (Builder H); folds O,A,B; arms raw, ctrl_*, oracle_*, lda{1,2,3}, act{8,60}, pcaphi, rp, noise, quant, ds, demean, pdz60
priority: 12
cmd: "python3 research/adversarial/src/h_imu/stage0.py --folds O,A,B --suite main --heldout transformer,svm_rbf --seeds 0,1,2 --n_jobs 8 --out runs/h_s0_ppd_stage0"
hf_path: runs/h_s0_ppd_stage0
---
name: h_r0_pd_gate
priority: 14
cmd: "python3 research/adversarial/src/h_imu/pd_release.py r0 --out runs/h_r0_pd_gate"
hf_path: runs/h_r0_pd_gate
---
name: h_rel_pd_pub_s0
priority: 15
cmd: "python3 research/adversarial/src/h_imu/pd_release.py rel --out runs/h_rel_pd_pub_s0 --name pd_pub --w1 7.8 --w2 0.05 --seed 0 --set astar"
hf_path: runs/h_rel_pd_pub_s0
---
name: r_s001_planted_synth                 # existing CLI; override --procs to respect the 8-thread cap
priority: 11
cmd: "python3 -m s_suite.run_planted --out runs/r_s001_planted_synth.jsonl --reps 120 --null-reps 300 --procs 8"
hf_path: runs/r_s001_planted_synth
---
name: t_siam_raw_mnv1_f0                   # NEW entrypoint (Builder T); x5 folds; BLOCKED until clearance
priority: 18
cmd: "python3 research/adversarial/src/t_cxr/run_gate.py --attacker siamese_r50 --release raw --negatives mnv1 --fold 0 --nfold 5 --seeds 0,1,2 --out runs/t_siam_raw_mnv1_f0"
hf_path: runs/t_siam_raw_mnv1_f0
```

## 8. (c) The three hypotheses whose falsification would most change the programme

1. **H-X-001 (dither/rate envelope).** Falsified if an adversarial arm beats the noise-dithered, adversary-free code at matched utility by at least MDE80 in at least 2 independent cells on cross-fit with 3 seeds (my P about 0.30). Consequence: programme 1's mechanism ("the adversary adds nothing beyond a bottleneck") does not transfer; the learned line becomes the programme's main line (arm B, G-002, IMU-005 get their stage-2 budget, the charter's second falsifiable clause is alive, and "train defenders against adversaries" replaces "train dithered codes" as the recommendation). It is the only file whose falsification makes the programme *grow*. Currently untestable (no learned arm yet), which is why it must run alongside the first one.
2. **H-CXR-002 (stage-0 gate).** Falsified if the raw Siamese beats the best demographic+device line on MN-v1 by less than 0.03 AUC (my P about 0.12), or if the random-pair-trained Siamese collapses toward the demographic line. Consequence: R-CXR stops at about 8 GPU-h spent instead of 45, the programme's deliverable becomes the protocol result (MN-v1 plus the demographic attacker; metadata alone gives AUC 0.983 on PriCheXy's own pairs, MEASURED), and with R-IMU likely unresolved (F2) X-002's stop rule is probably triggered.
3. **H-CXR-004 (the attacker suite moves the defence verdict).** Falsified if the suite's worst case is under +0.03 AUC over PriCheXy's own attacker on both protocols and the ranking is unchanged (as-written P about 0.10; P that the MN-v1 half alone misses +0.08 about 0.45). Consequence: the single-attacker evaluation was adequate in the one domain with statistical power, the programme's founding premise ("strengthening the suite moves measured privacy") is dead outside IMU, and instrument S shrinks to a reporting checklist.
   Near misses: H-CXR-003 (PriCheXy beating the floor envelope by at least 0.05 at all three mu, P about 0.2, would also reopen the learned line but overlaps X-001) and H-IMU-002 (a frozen raw-trained attacker being adequate, P about 0.25, would remove the suite's justification on IMU only).

## 9. (d) Disagreements with the generator

1. **IMU MDE.** The generator carries 7.5 points (n = 144) and 4.8 points (cross-fit n = 360) into IMU-005, G-001, X-001, S-002 clause 4 and S-004. With the subject as the independent unit (Builder H) the figure is about 0.12 at fold O and 0.08-0.10 pooled (F2). The affected thresholds are unresolvable by design; I KILL G-001, PARK IMU-005 and IMU-004, and require UNRESOLVED language rather than "within MDE80 therefore equal" in S-004 and S-005.
2. **IMU testbed numerics.** IMU-001/-004/-005/-006 are written for 12 channels and 6 activities (LDA-5, "raw 74-78%", held-out pairs of 6 activities). P-PD has 2 magnitude channels, 4 activities, LDA at most 3 dims, 96 test units, and a 4-activity design leaves nothing to hold out in IMU-005. Verdicts should use Builder H's pre-registered WIN/K rules; the numeric predictions should be discarded until stage 0 measures raw identity under P-PD.
3. **IMU-001 arm A is partly foreseeable.** PD hides gender, not identity, and conditions on a 60-d embedding of the window. The informative contrasts are `pd_pub` vs `pdz60` vs the floor frontier and the power verdict, not "PD leaks identity".
4. **CXR-004 P1.** Prior 0.75 mixes a near-certain half (random negatives, because the demographic line alone is 0.885 simulated) with an uncertain one (MN-v1, 0.55). Only the second can move the programme.
5. **CXR-003 vacuity (F5).** "Floor envelope within 0.03 of PriCheXy" can be true because both fail; report normalised L and class near-raw ties as "both ineffective". The 20 GPU-h should be staged with a hard gate after 4.
6. **S-003 K range.** WITHIN pooling at K = 6 is infeasible on KTH (about 4 clips per recording); use K <= 4 WITHIN, K <= 6 ACROSS.
7. **S-002 is a four-clause conjunction (0.35 as written);** clause 4 fails under subject clustering. Judge it on clauses 1-3.
8. **CXR-006 learned aggregator.** The file itself predicts it adds at most 0.02 over mean pooling; do not build it.
9. **IMU-004 and IMU-005 PARK.** C-AAE has no code and its "ADPCM-only 32/41/83%" is unverified; IMU-005's prior is 0.07, not 0.12, once resolvability is counted.
10. **Charter clause 1 has no hypothesis on programme 1's own HAR releases.** The stress test already shows the headline pair (canonical x16 vs learned bdq x64) is unresolved (difference -1.7, SE 2.7, MEASURED) and that the better of two families flips in 5 of 15 KTH releases. Add this as a section of the S-003 output (CPU, stored `preds.npz`), not as a new 20-30 GPU-h video re-attack, which I do not recommend before the IMU and CXR gates read out.
11. **Priors moved:** CXR-001 0.65 to 0.70, CXR-002 0.85 to 0.88, CXR-006 0.55 to 0.40, IMU-002 0.70 to 0.75, IMU-005 0.12 to 0.07, G-001 0.12 to 0.05, X-002 "no learned line" 0.75 to about 0.85.

## 10. (e) Stop condition for programme 2, in concrete terms

Evaluate at the readout of the cheap gates (stage 0 plus stage a of CXR-003/-004). Programme 2 **stops** (write up, release the VM, no further GPU) when all four of the following hold, with unresolved counted as fail and original and tightened verdicts printed side by side (no loosening):

* **S-CXR.** CXR-002 fails (gap below 0.03 AUC, or no MN-v1 tier gives a metadata AUC of at most 0.56), **or** CXR-003a shows the floor envelope within 0.03 of PriCheXy at 2 of 3 mu at L below 0.8 **and** CXR-004a shows no more than +0.05 AUC over PriCheXy's own attacker on MN-v1. If the clearance has not arrived 72 hours after this ranking, R-CXR is dropped and the metadata-only protocol finding is reported.
* **S-IMU.** Builder H K0 or K1 fires, or K2 (no PD arm attains WIN_L or WIN_U under the main suite).
* **S-G.** G-002 was not opened (because S-CXR fired) or fails (D_pop does not beat the lower floor envelope by at least 0.03 AUC on the held-out family in at least 3 of 4 rotations, 3 seeds, 5 folds).
* **S-X.** No replicated cell where the adversarial arm beats the dither frontier by at least MDE80 (C9).

Budget caps: 25 GPU-h for the gate stage; 45 GPU-h in total unless a learned line survives, then one stage-2 attempt of at most 5 GPU-h, pre-registered before it starts. Wall-clock cap: three working days after the CXR clearance (or after Phase 1 if CXR never clears).
Three outcomes: **STOP** (all four hold; my P about 0.30; spend about 20-25 GPU-h; deliverables are MN-v1 and the demographic attacker, window-split memorisation, the floors frontier on PD's input, the planted-leak calibration and the evidence-budget numbers); **CONTINUE AS ZERO-GPU WRITE-UP** (learned readings fail but an audit result or the instrument survives, my P about 0.55); **CONTINUE ONE LINE** (a learned win beats the lower envelope of all floors on the held-out family by at least MDE80 with 3 agreeing seeds and passes C9, my P about 0.10 to 0.15, which then becomes the single continuing line).
My probabilities: no learned line survives 0.85; K1 fires 0.35; CXR-002 passes 0.88; CXR audit survives 0.80.

## 11. Things to verify before any write-up

* LAPIS (2610.40722) numbers, the 0.885 simulated demographic AUC, C-AAE percentages and PD's paper-vs-code attacker description are scout-read or simulated; verify against primary sources.
* Builder S and T have not written their notes; if either changes a protocol (units, folds, tiers), re-rank the affected rows.
* The info column is a heuristic: it favours free analyses and penalises gates that are expected to pass (CXR-002) even though they guard the largest budget.
