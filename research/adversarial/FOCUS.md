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

---
# AMENDMENTS 2026-11-04 ~18:20Z VM clock (Auditor-2, research/adversarial/AUDIT_findings.md F1-F18). These SUPERSEDE the blocks above wherever they conflict.
All are made BEFORE any programme-2 model number exists, so none loosens a kill; each is a tightening or a re-specification. Original verdict text is kept in git history (commit 80afd9b / 5c5c7f0).

## A1. R-IMU is RETARGETED (F1, F2, F3, F7)
- Primary defence = the identity-targeted Anonymizing AutoEncoder on MotionSense (Malekzadeh et al., arXiv 1810.11546; published claim: identity < 7% at activity accuracy > 92%, 24 users). Check code availability, re-implement from the paper otherwise; the replication target is its own published claim.
  PrivDiffuser (arXiv 2412.14499) protects GENDER only and "identity is not a protected attribute in any experiment": it stays only as a labelled CROSS-THREAT-MODEL CONTROL or as the learned arm B retrained with identity guidance. It is not a re-audit of an identity claim.
- Burned-set ledger: MotionSense trials 11-16 (the last-trial split) and the stress test's 144-trial numbers are DEVELOPMENT data. The floor taxonomy ("activity-LDA codes bite; de-meaning/magnitude/16x downsampling do not") came from that split and is EXPLORATORY.
  The headline uses a fresh subject-clustered partition: folds ranked by sha256(subject|activity|salt) with the salt committed by the main agent AFTER the floor list is frozen, or the claim moves to WISDM/UCI-HAR.
- Released checkpoints cannot be cross-fitted (PrivDiffuser trains on trials 1-9): the released-checkpoint arm is restricted to trials 11, 12, 15, 16 (4 activities, 96 clean trials; paired MDE80 about 9-13 points); cross-fit only for defenders retrained inside the programme.
  Every IMU number prints its MDE; gaps below ~10 points on arm A are unresolvable by design.
- Measured premise (F7, single seed, 48 trials): static windows (sit, std) identify subjects at 47.9% after de-meaning (chance 4.2%), dynamic 91.7%. So MotionSense identity is not just gait (sensor noise, placement, session state). Every IMU result is reported stratified STATIC / DYNAMIC and worded
  "same-session identity". H-IMU-006's prediction (<= 12%) is already falsified; rewrite it as a measured premise with a new falsifier.
- Second IMU set is NAMED: WISDM biometrics (CC BY 4.0, 51 subjects), after the phone-model confound check.

## A2. R-CXR pre-registration must be machine-checkable (F4, F5, F6, F14)
- Three patient-disjoint sets: defender-train / attacker-train / sealed-test. Attacker-trained-on-defender-seen-patients is a SENSITIVITY arm only. State how many multi-visit patients each role gets (13,302 split three ways leaves the attacker pool well below the 1,424-test comfort zone).
- Pair design: one positive and one matched negative per patient, or a patient-cluster bootstrap with B >= 2000; strata for view mix (PA/AP), original-dimension mix and follow-up gap. Matching is on the PAIR-feature distribution (same-view, same-dimension, same-spacing flags, age gap, sex) so that each stratum has equal positive and negative frequencies
  (identical-dimension-for-all-negatives would make "different device" evidence for the positive class: only 58% of PriCheXy's positive pairs share dimensions).
- Validity gate BEFORE any image model is trained: true-metadata GBM, device-only GBM and border-only CNN all <= 0.56 AUC on the matched-negative set, with CIs. The burned CXR sets (official test list, PriCheXy's 5,000 test pairs) are development data; the headline fold is a patient-hash fold no one has looked at.
- Stage-0 gate (copied from the stress test into FOCUS): the real-image matched-negative AUC must exceed the metadata floor by >= 0.03, else R-CXR stops. H-CXR-002's premise "0.994 cannot be demographics" is contradicted (age + sex alone give 0.983); its prior is lowered to ~0.6 and its mechanism text is rewritten.
- Machine-checkable kill rule (template for every pre-registration): frozen comparator list; matching rule "floor point = linear interpolation of the lower envelope at the learned point's worst-suite leakage, both axes with CI"; margin delta fixed from the printed MDE; verdict rule as an inequality on the lower/upper 95% CI bound of (floor - learned);
  control tolerance = max(1.96 SE of the control, design-effect-aware MDE); number of distinct learned defences stated (a tie across PriCheXy-Net at three mu says nothing about learned defences in general); `verdict_original` and `verdict_tightened` fields in every result.json; "utility beyond the 14 labels" operationalised before use.

## A3. Instrument S: integrity and the missing charter question (F8, F9, F10)
- ZERO-GPU DELIVERABLE (the charter's first falsifiable clause): re-score programme 1's stored `preds.npz` (KTH, UTD-MHAD, Charades, HMDB51) with the sealed-suite machinery (cluster bootstrap, best-of-K, evidence budget) and state, per release, whether the programme-1 verdict moves by more than the printed MDE.
  Add 1-2 new attacker families on KTH only if that step shows headroom.
- Evaluator freeze: git tag + sha256 of the evaluator recorded in the pre-registration and printed in every certificate; `prereg_commit` must be an ancestor of the commit that produced the first sealed number (script-checked); a sealed-query ledger outside builder-writable directories refuses a second sealed query for the same release digest unless a declared amendment exists.
- Planted-leak calibration is owned by a NEGATIVE-CONTROL BUILDER (new role) holding a sealed manifest with at least two leak families the suite builder has not seen (cross-channel phase coupling, nonlinear amplitude-frequency interaction, a stealth leak in a band the declared features do not read), injected into REAL defended releases; report detection vs dose including the miss rate for unseen families.
- Hypothesis priors are assigned or reviewed by someone other than the author.

## A4. Data and outputs (F11, F12): see docs/data_licences/WHITELIST.md (the licence steward is the main agent). The class rule is replaced by a named whitelist. Dual-use outputs are gated: NO public release of trained chest-X-ray linkage attackers, defended images, or per-patient embeddings;
publish scores/predictions keyed by hashed IDs, code and aggregate tables; use "linkage/retrieval" wording; never run cross-dataset linkage. The worker's `hf_allow` job option enforces this for t_ jobs (["*.json","*.log","*.md"]). ChestX-ray14 is CLEARED with conditions (a)-(f) in the whitelist; MotionSense and WISDM CLEARED.

## A5. Corrections to this programme's own text (F13)
- The LAPIS inference that 12,295 patients are "92% of multi-image patients" and that 4,707 is near the largest test n is CONTRADICTED: LAPIS's test set covers all patients and only ~43% have repeated studies (about 2,030 qualify). CXR still has enough power.
- "PrivDiffuser's split is window-level" is wrong: it is TRIAL-level (train trials 1-9, test 11-16) with only 4 activities.
- "LAPIS 0.826 vs PriCheXy 0.577" compares different protocols (LAPIS's undefended reference is 0.974 against 0.818 in the original paper): not like-for-like.
- "Under 10% of action-recognition privacy papers use adaptive attackers" is at most 3 of 32 video papers.
- VERIFIED: PriCheXy-Net 81.8 -> 57.7 AUC, VoicePrivacy 25-44%, C-AAE numbers (ADPCM-only percentages unverifiable), ChestX-ray14 counts, MotionSense 24 subjects and the floor numbers (replicated).

## A6. Roles added (audit section 5)
Licence steward = main agent (WHITELIST.md); sealed-split custodian = main agent (commits the salt after the floor list is frozen; builders never see the sealed labels before the pre-registration); negative-control builder = new agent (A3); independent replication auditor = a fresh agent that re-derives each builder's headline from raw result files before the main agent accepts it (first use: Auditor-1 on programme 1; then on H, T, S).
G stays CLOSED: its gate cannot be tested at the available power (MDE 7-8 points) and F16 notes that "families" must be views, not hyperparameters.

---
# OUTCOMES 2026-11-05 ~02:40Z VM clock (builder final reports; every number is on the stated split; nothing in programme 2 was graded on a sealed split except T's rotations 1-2 and the sealed rotations of the T2 wave)

## R-IMU (Builder H, burned fold O = 96 clean trials of the SHIPPED checkpoints, NOT sealed; H.md, results/h_imu/result.json)
- Published claims reproduce under their own threat models (AAE: frozen raw-trained identity CNN 4.07% vs 6.98% published; PrivDiffuser own protocol exact). Under a RETRAINED attacker suite the shipped anonymising autoencoders are re-identified: msda 0.826 [.733, .906] at activity .930; rep 0.878 [.819, .934] at .973 (trial level). Cheap floors dominate them: activity-LDA-3 .365/.935, predlabel .049/.950, act60_r1 .094/.949. PrivDiffuser (a gender defence) ties the 2-bit predicted-activity label as an identity control (0.049).
- verdict_original = verdict_tightened = NO-GO-MATCHED (the floor matches or beats the defence); K0/K1 do not fire (MDE80_L 0.121); three distinct learned defences evaluated. Not done: sealed arm B, WISDM, wave-2 dae floors. MotionSense has 24 subjects; same-session identity includes non-gait signal (static windows 47.9%).

## R-CXR (Builder T; T.md section 7.7, results/t_cxr/t2/; 4,585 patients pooled over 3 sealed rotations for the comparison; intervals v1 percentile with v2 x1.10 alongside)
- LITERAL VERDICTS: T1 NO-GO (KT1: border-only attacker AUC 0.738 > 0.56; deviation D1). T2 literal (verdict_original = verdict_tightened) NO-GO at G2, control clause 4: the shuffled-label family-1 (Siamese) AUC 0.540 (repaired control 0.530) vs tolerance 0.021 (D3, corrected by A8/A9: the tolerance is mis-calibrated for the Siamese family, design offset ~0.03; 'F1' = attacker family 1, not an F-measure). EVERYTHING BELOW IS POST-HOC SENSITIVITY after that literal failure.
- Identity linkage beyond demographics is large: raw 0.885 vs metadata floor 0.528; interior carries it (raw minus frame-zeroed 0.011, MDE80 0.010); different-detector stratum 0.860 (same-detector 0.905); the demographic-feature attacker explains 36% of the excess.
- Published defence does not replicate under its own protocol: R0 with released weights, 5 retrained verifiers: raw 79.7 +- 2.3 vs 81.8 published (replicated); pcx_mu0.01 67.0 +- 3.9 vs 57.7 published (NOT replicated, 9.3 points weaker).
- Retrained pcxr_mu0.01 (epoch-1 checkpoint = the reconstruction-pretrained generator after two adversarial epochs; learned displacement RMS 0.66 px, max 1.27 px; the zero-padding border artefact is in EVERY warp-pipeline release) vs nominated floors, worst-case linkage L [utility U]: raw 0.887 [0.776]; pcxr 0.860 [0.783]; rw3_mu0.01 0.853 [0.781]; down_f2 0.868; rw3_mu0.005 0.871; down_f4 0.805 [0.786]. The defender lowers L by 0.027 (MDE80 0.008) but is DOMINATED by plain 4x downsampling (0.055 less leakage at equal utility). KT2 (post hoc) fires: dU -0.004, dL -0.059 (margins 0.02 / 0.03); also with F6 in L, with the labels14 floor, and under the nearest-floor rule. labels14 code floor: L 0.690 at raw utility 0.776 (strongest cheap floor, a code release, sensitivity comparator).
- T2 does NOT test: one retrained defender at one operating point (mu 0.01), mostly the pretrained generator, single seed; a 250-epoch schedule, mu 0.005/0.001, other defenders, and floors outside the four nominated (blur, quantisation, jpeg, noise, pca). The bank is from the HF mirror and cannot be checked against an official copy. Budget ~8.6 of 10 worker-hours.
- Process record: A6 order slip (a sealed job started before the nomination table, killed after ~5 s, no score computed or read, ledger entry consumed under a declared amendment); G2 and all post-G2 results are post hoc; floors were nominated from validation only (A7, committed before the sealed wave).

## Instrument S (Builders S and N)
- Evaluator v2/v3 certify seen leak families (0 false alarms in 224 and 106 nulls; targeted families 20/20 or 32/32 at the primary dose) and MISS unseen ones (v1: stealth tone 8/32; v3: window-order dependence 0/20, phase-amplitude coupling 7/20). Verdict: a LEAK verdict is trustworthy; a missing LEAK is not evidence of no leak; v3 is a leak detector, not an assurance instrument. On programme 1's HAR splits (150 recordings) it resolves only differences of >= 5-8 points; at chest X-ray scale its zero-leak floor is chance +1.6 points.

## Programme-2 reading (provisional, scoped)
In both settings where a published privacy figure could be re-measured (wearable IMU, chest X-ray), the figure depended on how the attacker was trained, and cheap floors (a label-supervised 3-d code; a 4x downsample) matched or beat the defences at equal utility. This is evidence about the tested defences and the floors, on development splits plus the T2 sealed rotations, not a statement about all defences. G stays closed.

## UPDATE 2026-11-05 ~05:50Z VM clock: R-IMU sealed arm B and H3-016 (Builder H; H.md 6.6 and 7.4; supersedes "Not done: sealed arm B" above)
- Sealed arm B (re-implemented AAE, config C1, retrained per salted fold, 3 folds, 264 trials, 24 subjects): `verdict_original` = `verdict_tightened` = NO-GO-MATCHED. aae_re L 0.948 [0.923, 0.970] at U 0.926 vs `predlabel` L 0.042 at U 0.937 (raw 0.973/0.957). One configuration, one seed per fold, S2 under-trained (monitor 0.27); 21 of the frozen sealed floors had run (the rest can only lower the envelope). Not shown: that no AAE configuration can hide identity.
- H3-016: dither composes = UNRESOLVED (original = tightened); `predlabel` flat at every evidence budget (account 0.042); undithered label codes saturate at account level (0.92-0.96); quote dithered floors at their evidence budget.
- Still not done: WISDM (stopped by its own device-confound gate, 0.667 >= 0.50), wave-2 `dae*` floors, a second sealed defender configuration.

## UPDATE 2026-11-05 ~07:40Z VM clock: R-IMU arm B2 (Builder H; H.md 8.1; results/h_imu/armB2_result.json; pre-registered H-B6 before any number)
- Second seed of C1 and the second configuration C2 = (4,1,1) of the re-implemented AAE on the SAME sealed folds: L 0.937 and 0.914 (arm B 0.948), U 0.934 and 0.943 (arm B 0.926); all three MATCH on both axes, family `verdict_original` = `verdict_tightened` = NO-GO-MATCHED, L range 0.034 [0.016, 0.053] (stable by the <= 0.10 rule). The claim 'arm B is not a single-sample artefact' survives its pre-registered kill criteria. C2's Delta_L is wide (-0.111 [-0.835, +0.057]) because its utility is above the label-only floors'. Scope: one re-implemented family, two configurations, 24 subjects; same-activity, same-session identity (H3-017). Still not done: wave-2 `dae*` floors, 14 sealed floors stopped for CPU, other training schedules.

## UPDATE 2026-11-05 ~09:58Z VM clock: R-CXR T3 (Builder T; T.md 8.11-8.12; results/t_cxr/t3/; pre-registered 4203bd5; ROLES RE-DRAWN OVER THE SAME 4,588 PATIENTS, NOT INDEPENDENT OF T2)
- `verdict_original` = `verdict_tightened` = "G2' PASSED; NO-GO at KT2": none of the five defenders (T2's pcxr_mu0.01; mu 0.01 seed 1 C1 and C2; mu 0.005 C1 and C2) beats the floors envelope (99% Bonferroni lower bounds; margins 0.02 U / 0.03 L fixed beforehand). T1 NO-GO (D1) and T2 literal NO-GO (D3) beside it. G2' passes only under the D4 control tolerance 0.06 (calibrated from T2 numbers, post-data).
- Best defender: mu 0.01 seed 1 last-epoch checkpoint, L 0.807 at U 0.784, TIES plain 4x down-sampling (0.806 / 0.785); raw 0.883 / 0.775; labels14 code 0.688 / 0.775. C1 (epoch 1) leaks more than C2 (exploratory); seeds differ by 0.027.
- Not tested: 250-epoch recipe, other defenders, quantisation / PCA / dppix / rw1 floors on sealed rotations. Budget 8.56 vs 8.34 worker-hours (2.6% over, disclosed).

## UPDATE 2026-11-05 ~11:55Z VM clock: R-CXR T4 (Builder T; T.md 10.1 and 11.2; results/t_cxr/t4/; roles re-drawn over the same 4,588 patients, NOT independent of T2 / T3)
- T4a attacker-scaling ladder: down_f4 worst-case linkage 0.806 / 0.817 / 0.852 at R1 / R2 / R4 (rise +0.045 [0.033, 0.059], still accelerating): chest X-ray LINKAGE LEVELS ARE LOWER BOUNDS. raw 0.883 -> 0.908; labels14 +0.013. H3-018 Leg-2 clause FAILS. T3 ordering survives at R4 (floor leaks 0.011 less than the defender, 99% [-0.022, -0.001]).
- T4b arm B (N3s repaired design; N3 failed its metadata control, A12 / D5): raw 0.887 on label-vector-matched negatives vs 0.886 on N2 (drop -0.001): identity survives matching on the 14-finding label vector; labels14 0.654 [0.640, 0.669] is borderline against the 0.65 threshold.
- Not tested: R3, long schedule, visit pooling, other attacker families, multi-seed rungs. Worker-hours 4.13 of 5.0.
