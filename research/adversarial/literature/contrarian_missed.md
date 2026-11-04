# Contrarian scout: directions missed by the six phase-1 scouts (programme 2, 2026-11-04)

Task: find a direction that beats the weakest of S (sealed suite + certificate), R (floors-first re-audit of HAR/IMU/CXR defences) and G (defender trained on an attacker population, tested on a held-out
family), ending in human activity recognition (HAR) and CT / medical imaging. Read first: CHARTER.md, SYNTHESIS.md, the six scout files (candidate and "not pursue" sections), research/fields/FOCUS.md.
Nothing was run on the VM; one local toy simulation (section 6, labelled). Fetch summaries come from a small model, so every number below must be re-read from the source before it enters a paper.
Read depth: OPENED = page or HTML opened with WebFetch; SNIP = search snippet only (not used for any claim, listed in section 8).

## 0. Verdict

**Nothing beats S, R or G.** The best new candidate, E (evidence budget: session-level / multi-window attackers and a leakage-composition law), scores 26 on the scouts' rubric, which ties the top of the S/R/G band
(23-27) rather than beating its weakest member, and it is naturally a module of S (first gate and headline field), not a fourth focus. Next: U (defence-agnostic pretrained attacker) 24, Y (cue-ablation
battery) 24, T (tuning-free in-context attacker) 24. A large-n identity testbed (Nymeria, 264 participants) is a data upgrade for S/R/G, not a direction, and has licence and size questions.
Confidence that no better direction exists among the fields I could reach: about 88-90% (section 7 says what I did not reach). The lock proposed in SYNTHESIS.md (S + R, G conditional) survives; I recommend
four amendments (section 5).

Effort: 36 WebSearch queries (extended mode, 2024-2026 first), 21 pages opened, about 25 directions screened, 1 toy simulation, the six scout files and FOCUS.md read.

## 1. Directions screened (all user-listed probes plus others)

| # | Probe | Queries (approx., overlapping) | What I found (OPENED unless SNIP) | Outcome |
|---|---|---|---|---|
| 1 | Canaries / planted leaks for representations | 1 | 2024-26 canary work is LLM, RAG and DP-training (ContextLeak 2512.16059, CanaryBench 2601.18834, one-run canary crafting 2605.27292: all SNIP). No learned attacker on embedding canaries; leakage scout D6 already parks it | covered by agentic D1 / leakage D6 |
| 2 | Model-stealing / representation-inversion benchmarks | 3 | Model extraction literature attacks the model, not a privacy transform (SNIP). Adaptive model inversion (2610.07677, OPENED): simple adaptive changes raise leakage 1.16-6.59x on FaceScrub, e.g. NegLS AttAcc@1 0.105 -> 0.692 | supports R's premise in another domain; no new direction |
| 3 | Robustness-vs-privacy transfer | 2 | 2610.07677 claims robustness is an attack-agnostic proxy for reconstruction vulnerability once the attack is adapted (pseudo-R2 0.317 -> 0.740); 2607.12354 and an "Inverted Leakage" chapter are SNIP and say robustness reshapes attribute leakage | classifier-inversion setting, not identity from latents; a one-line statistic (LDA accuracy) is the likely better proxy; dropped |
| 4 | Certified / randomised-smoothing analogue for identity | 1 | Nothing on certified attribute-inference defence (all results were certified-robustness-to-perturbation). FSInfo (2504.10016, OPENED): training-free Cramer-Rao-style leakage floor for reconstruction, idealised unbiased attacker, noisy-channel assumption, no attribute inference tested | leakage D5 (Gaussian ceiling) already covers the useful part |
| 5 | Sequential / streaming attackers | 3 | Real gap. VR motion (2407.18378, OPENED): attacker sums log-probabilities over all windows; 100% per-user accuracy at 1 FPS, 1 m rounding and dropped dimensions while per-sample accuracy falls; "no experiment varies the amount or duration of test data". Wrist accelerometer snippet: 58% per frame -> 68% with session vote (SNIP, secondary) | **candidate E** |
| 6 | Multi-modal fusion / linkage across modalities | 3 | 2026 survey 2608.04501 says cross-modal vectors "remain largely unexplored" (SNIP via search summary); DeSPITE (2506.13897, SNIP) aligns skeleton, point cloud, IMU, text for matching. Open synchronised IMU+video+skeleton with many identities: Nymeria only | **candidate X** (needs data) |
| 7 | Linkage across datasets | 2 | No open pair of datasets shares people. Within-dataset cross-session linkage is the feasible form (VR identifiability falls with delay, 232 users, 8 weekly sessions: 2303.01430 OPENED) | folded into S split protocol (same-session vs across-day) |
| 8 | Federated HAR privacy | 1 | 2609.02971 (OPENED, Sep 2026, VTC): server recovers client identity from weight deltas at about 1.000 accuracy on UCI HAR; clip-then-noise (sigma 0.1-0.2) gives near-random with under 5% FL accuracy cost; client count K not stated, attackers not stated as retrained | killed by the cheap floor (clip+noise is the defence); threat model weak; dropped |
| 9 | Stackelberg / game-theoretic solvers | 1 | Classical LP "privacy games" (1402.3426, SNIP); no 2025-26 paper with a learned adversary on time series | covered by suites D (PSRO) as an arm of G |
| 10 | Cryptographic / secure inference | 1 | Small proofs of concept (HE gait recognition, Sci Rep 2025; HE+MPC gestures, 2602.07936: SNIP); no learned adversary exists to evaluate | fails the "needs a learned adversary" gate; dropped |
| 11 | Counterfactual / cue-ablation tests | 1 | Only sensor-placement ablations of gait models (SNIP); no use as a privacy-suite control | **candidate Y** |
| 12 | Data valuation of attackers | 1 | Re-identification Shapley values exist for tabular record linkage (SNIP); nothing for sensor or image attackers | no usable lit; low value; dropped |
| 13 | Medical shortcut audits (site, scanner) as attacker | 2 | CT FM papers report external validation only (SNIP); programme 1 closed A and E on exactly this shape (adversary adds nothing beyond bottleneck/noise; per-source whitening a strong floor) | dropped (FOCUS.md precedent) |
| 14 | Synthetic-data release audits | 2 | CheXGenBench (2505.10496, OPENED): one Siamese ResNet-50 re-ID score on 11 generators, real-pair reference 0.59 (Sana 0.551, SD V1-5 0.572), retraining on synthetic not stated, MIMIC-CXR (credentialed DUA), no MIA/attribute attack; "Privacy is a claim, not a property" (2609.01273, OPENED) is a norms paper | R-like for generators; open-data generator training costs GPU-days; dropped to 18-19 |
| 15 | LLM-generated attackers | 2 | Agentic scout covers AutoMIA etc. New hits are LLM red-teaming of agents (SNIP) | covered |
| 16 | Benchmark-construction lessons | 2 | RobustBench admission rules plus invited external adaptive attacks; AutoAttack parameter-free ensemble (2010.09670, SNIP); "Attacker Moves Second" (2510.09023, SNIP via scout): adaptive optimisers beat 12 defences >90% | design hygiene for S (section 5, A4) |
| 17 | Unlinkability standards in biometrics | 1 | ISO/IEC 30136 lineage; Gomez-Barrero et al. (2311.04633, OPENED): gallery-free mated/non-mated likelihood-ratio measure D_sys, "maximum across linkage functions" when several are tested, K=10 key databases, only 4 biometric schemes | **component K** for S's report |
| 18 | Defence-agnostic / meta-learned attackers | 2 | DAST (2603.12840, OPENED, speech): attacker pretrained on 8 voice-conversion systems disjoint from 7 test anonymisers; MP-PAR meta-learning for action recognition (SNIP) | **candidate U** |
| 19 | Held-out attacker for defenders (G prior art) | 2 | 1906.05675 (SNIP via search summary): discrete attacker set M, "attacker overfitting" reported; no 2026 held-out-family paper found | G stays novel as a protocol; prior art is older |
| 20 | Embedding translation (vec2vec, 2505.12540) as attacker | 1 | Unsupervised text-embedding translation (SNIP); nothing on medical or sensor latents; Focus D (embedding privacy) was NO-GO | dropped (19-20) |
| 21 | Tabular in-context models as attackers | 3 | Tabular FMs studied only as victims (membership leakage: 2606.31474 SNIP). TabICLv2 (2602.11139, OPENED): open code for inference, pretraining and data generation, weight licence not stated; TabPFN-2.5 weights non-commercial, 10-class native limit, many-class wrappers (SNIP) | **component T** |
| 22 | Large-n open identity data for HAR | 4 | Nymeria (264 participants, 4-8 recordings each, Xsens 17 IMUs, Aria IMUs, observer video): OPENED, see section 4. BOXRR-23 (105,852 users, 4.7M recordings, 4.71 TB): OPENED, licence conflict | **data upgrade N** |
| 23 | Gallery-size / population scaling of re-ID | 1 | No study extrapolating accelerometer or CXR identifiability to large galleries (snippet: kinematics uniqueness depends on population, VR identifiability depends on delay) | verification-style metric (TAR@FAR, D_sys) is the fix; fold into S |
| 24 | Identity unlearning in embeddings | 1 | Face-identity unlearning (2512.13317, SNIP): approximate unlearning methods leave embedding identity structure | closed-form/erasure floors already known to fail on unseen units (FOCUS A, C); dropped |
| 25 | Wearable-IMU foundation models as attackers | 1 | Inertia-1 (2607.06617), HARBench, WHAR Arena exist (SNIP); no identity-leakage study | already FM scout D1 |
| 26 | LLM / VLM reading raw sensor data for attributes | 1 | Wearable VLM defence TGAP (2608.28691, image tokens) only (SNIP) | FM scout D5 (18) |

## 2. New candidates, scored (same rubric as the scouts: V value for robust suite, D open data + licence for HAR/CT, Det deterministic verification, C compute on 4x A6000 (5 = cheapest), L fit for learned methods, N novelty/gap)

| ID | Candidate | V | D | Det | C | L | N | Total | Kind |
|---|---|---|---|---|---|---|---|---|---|
| E | Evidence budget: session-level / multi-window attackers, leakage-composition law N* | 5 | 5 | 5 | 5 | 3 | 3 | **26** | module of S (gate 0) |
| U | Defence-agnostic pretrained attacker (DAST analogue for HAR/CT), held-out defence families | 4 | 4 | 4 | 4 | 5 | 3 | **24** | rung of L / held-out family for G |
| Y | Counterfactual cue-ablation battery ("what does the suite use?") | 4 | 5 | 5 | 5 | 2 | 3 | **24** | controls of S |
| T | Tuning-free in-context attacker (TabICLv2 / TabPFN-2.5) as a suite anchor | 3 | 5 | 4 | 5 | 3 | 4 | **24** | member of S |
| K | Gallery-free unlinkability measure D_sys (2311.04633) as the suite's headline field | 3 | 5 | 5 | 5 | 1 | 3 | **22** | report format of S |
| X | Cross-modal fusion linkage (anonymised video/skeleton to IMU) | 4 | 2 | 4 | 3 | 4 | 5 | **22** | needs Nymeria cleared |
| Z | Synthetic-release audit (CXR / IMU generators under a retrained suite, planted canaries) | 3 | 3 | 3 | 2 | 4 | 3 | **18** | R variant |
| F | Federated HAR client-identity inference | 3 | 5 | 4 | 5 | 2 | 2 | **21** | killed by clip+noise floor |
| V | Vec2vec-style unpaired translation attacker on released latents | 2 | 3 | 3 | 4 | 4 | 4 | **20** | arm of FM D4 |
| N | Large-n testbed (Nymeria skeleton/IMU, 264 people) | 4 | 3 | 4 | 3 | 3 | 3 | **20** | data upgrade, see section 4 |

For comparison, the scouts' own scores: S 23-26, R 24-27, G 24-27, L 24-25. Scores are priors; mine rest mostly on abstract-level reads, like theirs. A +-2 swing would not change the verdict
except that E could reach 28 if the gate-0 result (below) is large, which is exactly why E should be run first, inside S.

## 3. Write-ups of the surviving candidates

### E. Evidence budget (26): aggregate over windows, and measure how many windows it takes
- Gap: HAR defence papers (PrivDiffuser, CLOAK, ImmCOGNITO, PMR per the HAR scout's protocol table) report per-window or per-clip numbers. A released stream, a longitudinal CT/CXR series or a KTH person has many records;
  the realistic attacker pools them. Nair et al. (2407.18378) do pool (log-probabilities summed over windows) and find 100% per-user identification under 1 FPS, 1 m rounding and dropped dimensions even as
  per-sample accuracy falls; they also state that no experiment varies the amount of test data. Nebbia (CXR scout) shows the same on the gallery side (R@1 18.6% at 2 timepoints, 70.3% at 7+). Nobody has put the
  probe-side budget on a privacy suite's axes.
- Deliverable: for each release and each suite member, the curve identification-vs-K (K = windows/clips/images pooled per probe, bootstrap over persons) and N* = smallest K reaching a pre-declared TAR@FAR or top-1; a
  check that N*_pred = ln(C)/I_window from the calibrated per-window V-information (leakage D4) matches the measured N*; the defender ranked by N*, not by per-window accuracy. Learned parts: per-window attackers
  plus a learned set aggregator (attention over windows); G extension: a defender must also be tested against the set attacker, since a per-window adversary at chance does not imply a pooled one.
- Gate 0 (no GPU, hours): programme 1 stored only top-1, so rerun the 3-4 existing KTH releases (blur x16, canonical stick x16, BDQ pool) with per-clip posteriors saved, pool K = 1..20 clips per person (separately within-scenario and across-scenario),
  and compare with the shuffled-label and black-input controls (must stay at chance for every K).
- Kill floors (pre-register): (i) clips of one person are one effective unit (within-scenario correlation), so pooled accuracy at K=10 is inside MDE80 of K=1 in at least 90% of cells; the finding is then "effective K approx 1",
  which still changes the suite's unit of analysis to the session; (ii) N*_pred matches N* in every cell, so E is a derived certificate field rather than a discovery (cheap, still useful).
- Why it could move programme 1: the canonical stick figure at x16 had 13.7% per-clip identity on 25 people (chance 4%). My toy (section 6) shows that, under independent evidence, 13.5% per window becomes 55% at K=10 and 98% at K=50,
  and under a shared session nuisance carrying 50% / 90% of the variance it stays at 20% / 15%. The real data decide which regime applies; that is the experiment.
- Data and licence: everything already in the harness (KTH, UCI HAR 30 subjects, MotionSense 24, WISDM 51, ChestX-ray14 about 3.6 images per patient). No new access.

### U. Defence-agnostic pretrained attacker (24)
- Evidence: DAST (2603.12840, speech) pretrains an identity encoder on VoxCeleb2, then on 2.6M utterances converted by 8 voice-conversion systems that are disjoint from the 7 test anonymisers, then fine-tunes on 5-100% of the target
  system. Stage II alone (no target data) matches per-system from-scratch training on one system and is worse on six of seven; the three-stage model with 10% target data beats the VPAC top-1 row on 6 of 7 systems.
  Single run, no seeds, optimiser/dropout/data scale confounded with Stage II, and the paper's own "outperforms on all seven" claim fails at T8-5 (26.56 vs 26.39).
- Transfer: pretrain on a distribution of cheap random obfuscations (blur, quantise, downsample, noise, random warp, PCA) of large-n sets, fine-tune on a released representation with 5-10% labels, and use the pretrained model as a
  held-out attacker family for G and as the L-ladder's "defence-aware" rung. Weights go to HF (deterministic evaluation, shareable).
- Kill floors: per-release retrained LDA, kNN, small CNN and a random-init copy at the same labelled budget; if the pretrained attacker does not beat the lower envelope by more than MDE on low-label cells, drop. Overlaps
  FM scout D1 (pretrain x adaptation) and suites B (L2), which is why it is a rung, not a focus.

### Y. Counterfactual cue-ablation battery (24)
- A fixed list of cue interventions applied to the released view before attackers retrain: IMU: gravity/orientation removal, per-window de-meaning, accelerometer-magnitude only, band-limit, cadence normalisation, time-reversal, axis permutation,
  device-matched splits (HHAR/WISDM phone vs watch); CXR/CT: border and marker masking, histogram matching, downsampling, HU-window clip, FOV crop. Report which fraction of the suite's worst case survives each.
- Programme 1 evidence that this matters: brightness-only 22.4% identity, CREMA-D channel signal 76%, scene = identity on Charades, LDA the strongest attacker. Literature found: only sensor-placement ablations of gait models (SNIP).
- Kill: surviving fraction approx 1 for every cue (no single cue explains the leak), or a one-line statistic matches. Learned fit is weak (attackers only), hence 24 and a place inside S's control set.

### T. Tuning-free in-context attacker (24)
- TabICLv2 (2602.11139) releases inference and pretraining code and reports no-tuning gains over RealTabPFN-2.5; TabPFN-2.5 weights are non-commercial, native 10-class limit, many-class wrappers (SNIP). Identity has 25-264 classes, so TabICLv2 or the
  TabPFN-3 160-class decoder (SNIP) is required. As a suite member it removes per-release hyperparameters, which shrinks the selection-bias budget (leakage D3) and mimics AutoAttack's parameter-free design. Kill: LDA/kNN equal within MDE.
  Verify weight licence and class limit before use (not stated on the abstract page I opened).

### K. Adopt the biometric unlinkability measure for S's report (22)
- Gomez-Barrero et al. (2311.04633): D_sys = E over mated scores of D(s), from the likelihood ratio of mated vs non-mated linkage scores with prior ratio omega (omega = 1 is the stated worst case); with several linkage functions, the system value is the maximum. That is S's
  worst-case-over-suite in an ISO/IEC 30136 vocabulary, gallery-size free, and computable from the saved verification scores. Limits stated by the authors: depends on the linkage function, database and omega; only four schemes tested; no sensor/medical example.
  Use it as an extra headline field next to top-1 and TAR@FAR rather than a project.

## 4. Data and licence notes (new facts only)

- **Nymeria** (2406.09905 and projectaria.com, OPENED): 264 participants, 1,200 sequences (about 4-8 each), about 15 min each, Xsens MVN Link suit (17 inertial trackers, 240 Hz body pose), Aria glasses IMUs, observer video, 50 locations; CC BY-NC 4.0 (project page and GitHub); email-gated download; the licence file text I fetched has no clause on identifying individuals and does not address models or weights (read through a small-model fetch, re-check the actual file); the legacy README names sequences `<date>_<session_id>_<fake_name>_<act_id>_<uid>`, so a participant pseudonym exists in the file name (consistency across a person's sequences unverified); total about 70 TB with downloadable "data groups" whose sizes are not listed; ethics section says annotators were trained to recognise participants from mocap renderings. This is the only open source I found that would lift HAR identity from 24 to 264 people with cross-session splits. Confounds: location and scenario co-vary with session. Owner decision needed.
- **BOXRR-23** (2310.00430, OPENED): 105,852 users, 4,717,215 recordings (about 44.6 per user, my arithmetic), 4.71 TB `.xror`, head and hand tracking from Beat Saber and Tilt Brush; CC BY-NC-SA 4.0 plus a data use agreement that, per the page, bars deanonymisation and contacting users and inferring sensitive attributes, although the authors produced it for identification research. Same conflict class as Capture-24. Use only if the owner accepts the DUA reading; it is the only way to test attacker saturation and selection bias at n approx 10^5.
- **Neighbour-modality defence audit material (R-like)**: Deep Motion Masking (2311.05090, OPENED): trained against frozen similarity models, evaluated on 1,000 users x 10 train/10 test recordings of one game, baselines only MetaGuard; adaptive LSTM attack lifts MetaGuard from 81.3% (oblivious) to 96.3%; DMM 3.7% / 0.1%; no noise-only or quantisation baseline (cheap floors absent, exactly R's premise).
- **Chest data**: CheXGenBench uses MIMIC-CXR (credentialed, DUA); ChestX-ray14 remains the only open longitudinal CXR set found (CT scout). Nothing new for CT.

## 5. Recommended amendments to the shortlist (instead of a new focus)

- A1. S gets "evidence budget" as a first-class axis (gate 0 above): suite numbers are reported per probe size K and as N*, with the session as the independent unit for bootstrap. Cheapest experiment in the programme, likeliest to move a programme-1 conclusion.
- A2. S's report includes D_sys (K) and TAR@FAR beside top-1; the controls include the cue-ablation battery (Y); optional tuning-free anchor member (T).
- A3. G gets two additions: a defence-agnostic pretrained attacker (U) as a held-out family, and a set-attacker (pooled) as one more held-out family; both are read-only evaluators, so G's kill rule is unchanged.
- A4. S's governance copies RobustBench's two anti-gaming devices (admission rules that exclude defences passing only because they target the suite's input view; a standing invitation for external adaptive attacks) and adds attack-failure indicators (accuracy must not fall as capacity, budget or K rise; shuffled-label and black-input at chance).
- A5. Put the Nymeria licence/ID check to the owner before the HAR power analysis is fixed: if cleared, S's MDE for IMU/skeleton falls from approx 24 to 264 independent units.

## 6. Toy simulation (my arithmetic, local, CPU; not data)

25 identities, 16-D Gaussian class means, scale 0.20, per-window noise N(0,1), optimal Gaussian attacker that knows the nuisance fraction rho of a shared per-session offset; 3,000 trials per cell; seed 1.
Per-window top-1 about 13.5% (chance 4%). Pooled accuracy by K = (1, 5, 10, 20, 50): rho = 0: 0.135, 0.348, 0.554, 0.785, 0.976; rho = 0.5: 0.132, 0.186, 0.196, 0.208, 0.202; rho = 0.9: 0.137, 0.149, 0.145, 0.141, 0.151.
Reading: evidence accumulation is either decisive or irrelevant depending on shared nuisance, so it must be measured, and the real quantity is the effective number of independent windows. Script: /private/tmp/claude-501/-Users-kuanw-Me-code-openlatent/7467a5d1-83aa-4aea-a58e-08e14af19e26/scratchpad/accum_toy3.py (scratch, not committed).

## 7. What would change my mind, and what I did not reach

- A result showing G's population-trained defender beats the floor envelope on a held-out family in the first cheap gate would raise G; the opposite (programme 1 predicts about 15% success) leaves S + R.
- If gate 0 shows pooled K=10 accuracy within MDE of K=1 everywhere, E drops to 22 and the shortlist is unchanged.
- Not reached: ECG/EEG/PPG biosignal identity (outside the stated HAR/CT scope), mmWave/CSI beyond the HAR scout, audio-visual fusion, paywalled 2026 medical audits (Nature, Lancet), non-English literature, and anything the US-only search tool did not surface. About half of the extended-search hits were reference lists or paywalled pages, which weakens every "no paper found" statement here; treat gaps as search-limited.
- Residual risk: scouts' scores and mine are priors from abstract-level reads. Data access from the VM (Nymeria, ChestX-ray14, programme-1 artefacts) is the stress-test scout's job.

## 8. Sources

OPENED (arXiv ID or URL): 2609.02971 (federated IMU client-identity inference); 2603.12840 (DAST, abstract and HTML); 2311.05090 (Deep Motion Masking, HTML); 2310.00430 (BOXRR-23); 2407.18378 (data degradation on motion re-ID, abstract and HTML); 2303.01430 (VR identifiability over time, abstract);
2406.09905 and https://www.projectaria.com/datasets/nymeria/ and https://github.com/facebookresearch/nymeria_dataset (main, legacy branch, licence file) (Nymeria); 2610.07677 (adaptive model inversion); 2504.10016 (FSInfo); 2609.01273 (synthetic-data privacy position paper);
2505.10496 (CheXGenBench); 2311.04633 (unlinkability framework); 2602.11139 (TabICLv2, abstract). The BOXRR datasheet PDF could not be read (binary).
SNIP (search snippets only; not relied on): 2512.16059, 2601.18834, 2606.10481, 2605.27292, 2609.13499, 2608.09001; 2606.31208, 2606.26021, 2606.31474; 2505.12540; RobustBench 2010.09670, 2510.09023, 2505.14534; 2607.12354, 2608.04501, 2506.13897, 1906.05675, MP-PAR; 2512.13317, 2609.27115; 2607.06617; 2607.12545;
2606.07210 (large-scale per-speaker re-identification risk in speech anonymisation: per-unit tail analysis, relevant to S, not opened); TabPFN-2.5 (2511.08667) and TabPFN-3 (2605.13986) class and licence statements; wrist-accelerometer session-vote figures (58% -> 68%).
