# FOCUS (locked 2026-11-04 08:15Z)

Basis: six phase-1 scouts, then two contrarian scouts (missed fields: ~25 fields, nothing decisively better, ~85-90%
confidence; stress test: A and C go with changes, B no-go as stated). Evidence in `literature/`. Change this file only with a
journal entry that says why, and only when a result or a newer scout falsifies a premise below.

## Focus A: source-invariant embeddings, PathoROB first   [CLOSED as a negative result, 2026-11-04 ~14:15Z VM clock]
- Outcome (Builder A, results_notes/A.md, "ROUND 2"): the only lead (post-hoc task-agnostic adversarial AE on held-out classes) is DEAD.
  * Seed-0 gains (+0.083 midnight/Tolkach, +0.164 midnight/TCGA) were measured against the LDA family only; against the lower envelope of ALL inductive floors (per-centre whitening, PCA, random projection, dithered codes)
    they are +0.021 [+0.002,+0.046] and +0.007 [-0.081,+0.097]; the midnight/TCGA gain does not reproduce at seed 1 (+0.011 vs LDA, -0.014 vs the envelope). The two AE picks were chosen post hoc, best of 10.
  * 3 seeds, validation-selected AE vs floor envelope, held-class accuracy: midnight/TCGA -0.056 [-0.108,-0.010], midnight/Tolkach -0.002 [-0.011,+0.010]; neither reaches the +0.02 margin.
  * Cross-fit (rotating sealed test, K=4, pooled out-of-fold): midnight/TCGA -0.028 [-0.063,+0.001] (MDE80 0.048, excludes gains above ~+0.05); midnight/Tolkach +0.003 [-0.007,+0.012] (MDE80 0.014, null resolved).
  * Dither control: noise-dithered label-free codes with no adversary match or beat the adversarial AE at equal leakage; reconstruction-only AE is the weakest family.
  * Gate G1 (LEACE leaves L_worst 0.63-0.89 under retrained attackers) stays a valid by-product: closed-form erasure fitted on seen units does not hold against retrained attackers on held-out units.
  * Not run: H2-015 selection-null, phikon-v2 holdout, dinov2-L cross-fit. Parked on the VM: ~27 gate seed replications.
- Original text kept below for the record.

- A3 RESULT (Builder A, A.md "A3 RESULTS", ~04:05Z VM clock; pre-registered before any A3 number; 0.53 worker-hours; midnight/Tolkach and midnight/TCGA, 4-fold cross-fit, dense floor ladder, lam 10/30/100, 2160 encoder steps; the adversary is fit on one half of the adapter-train slides and the confusion loss is applied to the other half):
  * 'Trained' criterion (calibrated): the noise-latent floor on held-out slides is chance (0.50 Tolkach, 0.25 TCGA), unlike the same-unit 0.67; a held-out-slide adversary reads centre from raw PCA latents (0.75 Tolkach, 0.46 TCGA), so the criterion is not vacuous. Trained: all three lam on TCGA and lam 30 on Tolkach (lam 10 and 100 on Tolkach narrowly miss). Trained does NOT mean removed: evaluation attackers still reach worst-case leakage L 0.34-0.50.
  * Endpoints vs the full inductive envelope: NO WIN (A3-DEAD under the pre-registered rules): Tolkach utility is ceiling-limited (headroom 0.011), delta -0.03 to -0.05; TCGA utility is underpowered (MDE 0.056-0.083 vs the 0.04 margin), delta -0.02 to +0.01; retention -0.08 to -0.24 everywhere; selection nulls have no positive gain to calibrate. Original verdicts stand beside it (held-class label utility DEAD, retention alive-weak).
  * ATTRIBUTION (corrects my earlier wording both ways): against the NO-ADVERSARY AE at matched leakage, a TRAINED adversary adds held-class utility of +0.05 to +0.064 (MDE 0.05-0.06) in 3 of 4 trained configs, at a retention cost of 0.02-0.13. So within the AE family a converged adversary changes something, i.e. 'the adversary adds nothing' is false once the adversary actually trains; but it still does not beat the cheap floors.

## Focus A (original text): source-invariant embeddings, PathoROB first   [GO-WITH-CHANGES]
- Keep: tissue/diagnostic label.  Hide: medical centre (and stain) signature.
- Benchmark: PathoROB (arXiv 2507.17845; 3 ungated HF datasets, ~1.9 GB, fixed seeds). Drop BatchBench (sealed, server-side, 30-257 GB).
  Second domain only after A works on PathoROB: astronomy instrument latents (GalSim + HSC/Legacy cross-matches) and Cell Painting (JUMP), as generality checks.
- Headroom is NOT the Robustness Index (saturated 0.87-0.93). It is a *retrained* centre probe (linear probes reach 88-98%
  on frozen embeddings; linear erasure fools linear probes but a retrained nonlinear probe keeps 2.2x chance).
- Learned encoders: adapter E(z_FM) with (i) rate bottleneck, no adversary (mandatory control), (ii) adversarial with retrained
  attackers, (iii) VQ/discrete bottleneck, (iv) centre-aware contrastive. Floors: raw embedding, LEACE/ISP, DANN.
- Kill (REVISED 2026-11-04 11:15Z, see journal): the original comparator (LEACE) is too weak. Gate cells (7 of 36, seed 0) show LEACE leaves
  L_worst 0.63-0.89 under retrained attackers (headroom exists vs LEACE), but a label-supervised LDA projection (C-1 dims) keeps raw-level label
  accuracy at L_worst 0.00-0.20. With 2-4 classes a label-only code is 1-3 dimensions and carries no centre signal, so "keep the label,
  hide the centre" is trivially solved in-distribution. A learned adapter is therefore only worth having if it beats the LDA / label-only arm on
  something LDA cannot do: (i) utility on label sets NOT used to fit the adapter (held-out classes, finer labels, a second task),
  (ii) retention of non-label structure (kNN retrieval / probe on held-out attributes) at equal centre leakage, (iii) unseen centres at test time.
  Stop A if, over 3 seeds with bootstrap over slides/cases, no learned adapter beats the LDA arm at matched dimensionality on (i) or (ii)
  while holding L_worst within 0.05 of it. The LEACE comparison stays as a floor, not as the kill test.
- Original kill text (superseded): "does not beat LEACE on worst-case centre-probe accuracy at equal label accuracy".

## Focus C: identity-free emotion, CREMA-D first   [CLOSED as a negative result, 2026-11-04 ~12:50Z VM clock]
- Outcome (Builder C, results_notes/C.md sections 2a-2d; WavLM-Base+, 91 actors, bootstrap over actors): (1) no label-supervised learned encoder (rate, adversarial, VQ) beats the
  label-only LDA-5 projection (UAR 0.743, identity 7x chance) and none passes the pre-registered held-out-emotion criterion; (2) the pre-registered task-agnostic adversarial
  autoencoder round fails too: 0 of 3 seeds for the validation-selected config (d32, lam 3: U_HE 0.326 [0.307,0.345] vs matched LDA, dU_HE -0.208 [-0.240,-0.179]) and for every
  adversarial config; the only positive difference is the NO-adversary AE (a reconstruction effect that leaks 34.8x chance and is ineligible under the selection rule).
  By-products worth keeping: full-clip CREMA-D identity is 76% recording-channel signal (non-speech alone 56x chance), LDA is the strongest identity attacker, closed-form LEACE fitted on
  train actors leaves 53x on unseen actors, oracle per-actor prosody normalisation is a strong floor (0.535 UAR at 2x chance, a different threat model).
- Also done (Builder C final): speech-trimmed AE variant, 3 seeds: held-out-emotion 0.294 at 3.1x chance vs trimmed matched comparator 0.525 (21.9x), dU_HE -0.231 [-0.261,-0.201], FAIL.
  Post-hoc noise-dithered label-free control (one seed, not replicated): a dithered no-adversary AE code matches or beats the adversarial AE at equal leakage, so the adversary adds nothing.
  Seeds 1-2 of the rate/adv d16 cells are still queued; they cannot change the verdict.
- Original text kept below for the record.

## Focus C (original text): identity-free emotion, CREMA-D first   [GO-WITH-CHANGES]
- Keep: emotion (6 classes).  Hide: actor identity (91 actors), plus sex/age/race attributes as secondary secrets.
- Data: CREMA-D (ODbL, ~2.5 GB, VM-reachable). RAVDESS as the second set. We build the benchmark ourselves (prior WACV 2025 /
  ICME 2026 pipelines use frozen validators, shared-actor splits, gated data, no checkpoints).
- Protocol: emotion test on actor-disjoint split; identity attacker trained on the attacker's own held-in actors/sentences, scored on held-out sentences/emotions of the same actors; attackers retrained on the obfuscated latents.
  Normalise: chance (1/91), task-label-only baseline, raw reference. Worst case over linear, MLP, kNN, and a different-backbone attacker.
- Modalities: audio first (WavLM/HuBERT frozen features, cheapest), then face video crops.
- Baselines to reproduce under *retrained* attackers: blur, optical flow, landmarks, VoicePrivacy-style anonymisation. Floors: LEACE, rate bottleneck.
- Kill: if the raw reference does not beat the task-label baseline on identity (not measurable) or no learned method beats the rate bottleneck.

## Focus D: embedding privacy for retrieval   [opened 2026-11-04 11:30Z, replaces B']
- Keep: retrieval utility (BEIR nDCG@10 on the released vectors).  Hide: recoverable text / PII (inversion, token recovery, PII span recovery).
- Why: the contrarian scout scored it 26/30; existing 2024-26 defences (Eguard, SPARSE) use fixed attackers, the flaw we found in Finding 1.
  Attackers must be retrained on the defended vectors (MLP token-bag, few-shot ALGEN-style, Vec2Text-style inversion).
- Lessons imported from A and B': (1) pre-declare a headroom check against the strongest trivial floor (PCA-d, random projection, label/query-agnostic
  linear erasure, noise) before fixing any margin; (2) a pre-registered kill rule with the comparator named in advance; (3) an oracle arm to bound the room.
- Two-hour kill test per literature/contrarian_missed_fields.md before spending more.

- STATUS (2026-11-04, Builder D; updated after audit F-01/F-10, 3 seeds): verdict_original = NO-GO as pre-registered, a GATE OUTCOME. H1 passes (worst retrained attacker R-precision on raw vectors 0.183 vs 0.004 prior);
  H2 fails at the oracle's own retention (0.70): the oracle (rare words deleted before embedding) keeps only 0.738 of nDCG@10 and still has normalised leakage 0.593, headroom **0.039 [-0.003, 0.077]** (3 seeds; the earlier 2-seed 0.041 is superseded) < 0.10.
  The same H2 passes at tau 0.90 (headroom 0.155 [0.109, 0.198]) and 0.97 (0.416 [0.376, 0.451]) with the oracle as a reference only (its retention is 0.74). Mechanism: rare words are retrieval-critical, so secret and utility overlap.
  The post-hoc learned `adv` arm COLLAPSED in all 24 fits (retrieval loss at ln N from step 250; failed implementation, results_notes/D.md 7.1): it is NOT evidence that the adversary adds nothing. The valid `bott_mlp` control trains normally.
  D2 (repaired arm: confusion term normalised by its initial value + collapse check; lambda 0.001-3, d 32-512, 2 seeds; K1-K4 at tau 0.90/0.97 pre-registered in D.md section 7): the arm trains (48 of 50 fits; the 2 lambda-3 d-512 fits collapsed and are excluded) but FAILS the learned kill test: at tau 0.90 L(floor) - L(adv) = 0.015 [-0.003, 0.034] vs margin 0.0775, at tau 0.97 no adv config reaches the threshold. lambda <= 0.1 equals the no-adversary bottleneck at every d; lambda 1-3 lowers the co-trained adversary's success but costs 0.1-0.3 retention and the retrained attackers recover as much or more (d 256: R_C 0.169 vs 0.154). verdict_original (NO-GO, headroom 0.039 at tau 0.70) stands next to it. All 10 D2 jobs finished. Other adversary designs and Vec2Text-class attackers untested.
  Digit/number reframe (H2-009): absolute H1 fails (n1 raw excess 0.010; prior 0.204 because the top tokens are "1", "2", "95"); scale-aware re-analysis on the rare-number tier n1s: raw 0.041 vs prior 0.016 (ratio 2.6 [1.5, 4.1]) PASSES H1s, but deleting digits removes only ~20% of that excess and the floors match the oracle (headroom 0.06 [-0.20, 0.33], undecidable at this power). Attacker suite is weak (3,716 docs, no Vec2Text); not a resolved null.

## Focus E: single-cell batch integration (SC-0, check-only)   [NO-GO for stage 2, never locked; 2026-11-04 ~15:05Z VM clock]
- Builder E (results_notes/E.md; pre-registration d114264 before any real-data number; amendments A1-A3 logged, none relaxes a kill condition): 3 datasets (scIB Immune, scIB Pancreas: CC BY 4.0 via figshare API; HLCA core: CC BY 4.0
  under the CELLxGENE policy, platform-wide), 3 seeds, 3 study-grouped folds; HLCA 97k cells / 107 donors / 11 studies. No learned arm was run.
  SC-K0 instrument: clear on HLCA (median floor-vs-floor MDE80 0.035); Immune/Pancreas use batches as units (MDE80 0.116 / 0.098), descriptive only.
  SC-K1 measurability: fires on HLCA and Immune under E's capped reading of "3x chance" (raw attacker 0.670 / 0.752 / 0.937 vs threshold 0.843 / 0.850 / 0.850); controls pass.
  SC-K2 under-determination: FIRES on 3 of 3 (ldaC_pca16 keeps r_fine 0.77 [0.75, 0.79] on HLCA, 0.97 Immune, 0.91 Pancreas; label-free pca16 keeps the same share). K2 alone decides.
  SC-K3 headroom: clears (HLCA: scVI U 0.816 at L 0.80 beats the V-selected inductive floor by +0.116 [0.099, 0.136]; Harmony at L 0.37 has no floor near its leakage). Note the rule "best reference by U" is lenient toward firing.
- By-product: per-batch centring and whitening RAISE retrained nonlinear leakage (L 1.54 on HLCA; an independent random-forest check agrees), unlike PathoROB where per-centre whitening was a strong floor (L 0.05-0.58): per-source
  normalisation is not a universally safe floor against nonlinear attackers.
- Caveats: scIB context scores for HLCA exist for fold 1 only; scVI is a single seed; Pancreas counts were non-integer; the three seeds share the same held-out units.

## Focus E2 (reopened attempt, Auditor-1 F-06) RESULT ~22:00Z VM clock: E2-NO-GO, RESOLVED (E.md 6.11; pre-registered grid: 3 seeds x 3 folds x 13 inductive configs, HLCA; aggregates results/e_e2/e2_final.{md,json})
- Delta (V-selected inductive encoder minus the inductive floor envelope at the scVI point) = -0.037 [-0.045, -0.026] with MDE80 0.020 against the 0.05 margin: a well-powered null in the registered direction (upper bound below the margin). verdict_original (SC-0 NO-GO) stands beside it.
- Post hoc (E2-B), labelled and kept OUT of the registered pool: extra beta/lambda fill configs (arms e2x_*). The registered adversaries failed the amendment-A2 memorisation-floor check at lambda <= 16, so nothing is claimed about adversaries.
- Reading: an inductive map did not close the gap to the transductive scVI point on HLCA at these settings; the single-cell headroom is transductive for the encoder family tried.
- Builder E final details: best eligible registered encoder vae_b2 (U 0.701, L 0.68) sits below the V-selected envelope (noise_r0.5 / dith32_r0.5, U 0.738, L 0.81); study-cluster interval [-0.051, -0.016]; 0 of 3 seeds reach the margin; realised MDE80 0.020 so NOT underpowered; no eligible encoder at the Harmony point (L <= 0.42); E2-K2' (leakage-matched under-determination) does not fire (best label-only/PCA arm is 0.146 below target); trained-check A2: 0 of 8 adversarial and placebo arms pass (in-sample noise floor 0.26-0.40, unseen-donor scoring at chance), so this is NOT an adversarial null.
  POST HOC (E2-B, commit 3445d8c, outside the registered pool): vae_b0.5 / cvae_b0.5 reach U 0.755 at L 0.77, Delta +0.016 [+0.009, +0.023], about a fifth of the gap to scVI and below the 0.05 margin; larger lambda (16, 64) still does not train. Not tested: a converged adversary, larger encoders, scFM inputs, non-VAE objectives. Cost ~2.6 slot-hours.

## Focus F, H3-007 STAGE 0 (CPU, metadata only, 416 s; pre-registered in F.md 11 at 04:13Z VM clock BEFORE any number; results F.md 11.4, results/f_jump/h3_007_stage0.json) ~04:50Z: OPEN on all three tests; THE F3/F4 JUMP POSITIVE IS NOT INTERPRETABLE YET
- A0 well position: 28.0% of cross-source positive well pairs share Metadata_Well in each of the 3 splits, vs 0.31% chance (sources 3-6-8 share 94-97%, sources 2-10 68%, across the two groups 0.3%).
- Position-only code with no profile information (split s0, F protocol): 3-number code (row, column, edge flag) U_H 0.324; exact 384-d one-hot position 0.314; within-plate shuffled positions 0.050 (control passes); iid noise 0.046. This EQUALS the best F2 supervised adapter (0.317-0.321) and beats the best linear floor ldaP_16 (0.182) by +0.13.
- B0 near-duplicates: 3.7% of held-out compounds have a Tanimoto >= 0.8 neighbour among training compounds (threshold 2%).
- Reading: the retrieval protocol (same-compound wells in other sources vs same-plate negatives) rewards the shared plate layout. The F3 claim (supervised adapter beats same-label floors; 10% break-even) cannot stand until stage 1 (position-restricted positives or position-matched negatives, near-duplicate pruning, source-level negatives; ~0.6-1 GPU-h) shows how much survives. F4 verdicts are PROVISIONAL (4 of 7 jobs done, not aggregated). My earlier summaries calling JUMP 'good news' are superseded by this block.

## Focus F, H3-007 STAGE 1 and F4 RESULTS (Builder F, F.md 12.6, results/f_jump/s1.{md,json}, commit efd1862; 3 splits, plate-cluster CIs) ~05:50Z VM clock: PARTIAL SURVIVAL, NO COLLAPSE; NO UNSEEN-SOURCE GENERALISATION
- U_H by protocol (P0 original / Pa position-restricted positives / Pb position-matched negatives / Pc near-dup pruned / Pd source-level negatives / Pabc combined), mean over splits: b10 adapter 0.198 / 0.174 / 0.208 / 0.201 / 0.218 / 0.188 vs b10 same-label LDA 0.141 / 0.122 / 0.164 / 0.143 / 0.140 / 0.147; b100 adapter 0.270 / 0.242 / 0.348 / 0.274 / 0.419 / 0.321 vs same-label LDA 0.163 / 0.142 / 0.186 / 0.166 / 0.164 / 0.167. Position-only exact one-hot: 0.313 / 0.043 / 0.033 / 0.314 / 0.268 / 0.016; shuffled-position control equals the noise embedding (0.04-0.07) in every protocol, so the protocols are valid.
- Gain over the full envelope as a fraction of P0: b10 0.87 (Pa), 0.52 (Pb), 1.01 (Pc), 1.64 (Pd), 0.47 (Pabc); absolute dU +0.029 at P0, +0.015 at Pb, +0.014 at Pabc (plate CI 0.009-0.017), above the 0.010 margin in 3/3 splits. b100: 0.91 under Pa, 1.5-2.4 under Pb/Pd/Pabc. Target2-only arm: 0.62-0.79 of P0, and -0.012 / -0.014 vs same-label LDA under Pb / Pabc. The strict SURVIVES rule narrowly fails (b10 under Pb in s0 +0.0128, under Pabc +0.0113 / +0.0141 in s0 / s1; b100 under Pa +0.093 vs 0.095).
- F4 (all numbers under P0, position-confounded; F4 jobs stored only P0 scoring): with all five sources in training, break-even 2.9% [2.7, 3.2] of labels on view A and 4.3% [4.0, 4.8] on view B (the view-B 3% win comes from the leakage axis; its U-axis crossing is 4.3%). LEAVE-SOURCE-OUT: no break-even in any of the 5 folds; the 100%-label adapter is 0.013 to 0.038 BELOW the best floor.
- verdict_original NO-GO at the S4 gate; F2 literal NO-GO (G-H); F3 GO rated fragile; F4 verdict_tightened PARTIAL (view B holds, leave-source-out fails). KEPT: the supervised-adapter label-budget gain over linear floors for seen sources, attenuated by the controls. WITHDRAWN: absolute P0 retrieval U_H as a biology number (use Pa/Pb); the F3 label-free-ish GO; any unseen-source generalisation.

## Focus F: JUMP Cell Painting (check-only)   [NO-GO at the S4 gate, 2026-11-04 ~16:50Z VM clock; programme-1 stop condition met]
- Builder F (results_notes/F.md 5.1-5.8; pre-registered before any number; priority 4 used once; 3.2 worker-hours). Literal verdict: R0 passes (S2 mean AP 0.9007 / 0.4096 vs published 0.90 / 0.41); JK1 FIRES via the
  control-AP clause in 3/3 splits (|U(shuffled compounds, raw) - U(noise16)| = 0.042-0.045 > 0.01; the shuffle control sits below chance because cross-source positives face same-plate negatives); the substantive
  measurability clause passes (raw worst-of-4 source attacker 0.885 [0.865, 0.914] vs 0.60). Amendment A2 (written after the S4 numbers, relaxing that clause) was ruled a deviation (D1); the literal verdict stands.
- Post-hoc sensitivity on S5 (cannot revive the line): JK0 clears (40.4k held-out compounds, MDE80 0.0010); JK3 FIRES 3/3: Harmony (U 0.077, L 0.99) is beaten by the 16-d compound-supervised LDA (U 0.183, L 0.39),
  dU -0.106 [-0.107, -0.104]; JK2 does not fire on the registered grid (lowest L_V 0.37 at d=16, grid edge; d<=8 not run). K2 answer: LDA-301 leaks source at raw level (L 0.99). Stage 2 (label-free AE/cVAE vs the ldaP_16 envelope) not run.

## AUDIT CORRECTIONS (Auditor-1, research/fields/AUDIT_findings.md, 2026-11-04 ~18:40Z VM clock; headline numbers all re-derive from raw files, no arithmetic error)
These supersede the wording of the blocks below where they conflict. Verdicts that STAND: C (strong; the decisive negative is the held-out-emotion protocol at matched leakage), A's AE null on held-class LABEL utility (resolved on Tolkach with a ceiling: at most +0.03 available),
A gate G1 (closed-form erasure fitted on seen units fails out of sample; now 0 of 8 cells after adding the missing phikon/TCGA cell, L_worst 0.72), B' for the ONE learned latent arm tested (bounded; learned masks collapsed and were reported as failed implementation).
RETRACTED / REWORDED:
- "The adversary adds nothing beyond a bottleneck or noise code" (journal, FOCUS A/C/D, my status messages) is UNSUPPORTED. D's post-hoc adversarial arm never trained: in all 24 fits the retrieval loss sits at ln(2739) = 7.915 from step 250 (I verified this in train_log of d_ph_lrn_gtr_s0), because the confusion term (101-154 nats) dominated the retrieval loss at lambda 0.3.
  A's adversaries never reach their own objective (in-sample accuracy ends at 0.75-0.94, lam <= 10, 720 steps; a toy check removes almost nothing for lam <= 10). C's adversary is effective on unseen actors, but the dithered-code control was one seed. Restated: "with these lambda ranges and step budgets the adversary did not converge".
- "A's AE lead is DEAD" -> dead on held-class label utility; MIXED on kNN retention (the criterion FOCUS A itself named as (ii)): midnight/Tolkach cross-fit +0.095 [+0.072, +0.124] over the matched floor (MDE80 0.039), fixed-C +0.146 [+0.113, +0.180]; midnight/TCGA cross-fit +0.013 [-0.007, +0.033] (not a win); dinov2-L cross-fit not run.
- D, E, F are GATE OUTCOMES, not learned-arm nulls. D: the pre-registered H2 failed at the oracle's own retention 0.70 but passes at tau 0.90 (headroom 0.155 [0.109, 0.198]) and 0.97 (0.416 [0.376, 0.451]); the digit check failed H1 on a scale-blind absolute threshold (n1 includes ubiquitous tokens; the rare-number tier n1s is 0.0408 vs prior 0.0163, 2.5x, untested); D's learned kill test was never validly run.
  E: K2 is a utility-only gate (ldaC_pca16 keeps 0.77 of fine-label F1 at L 0.84, i.e. it hides little batch); K3 found a real +0.116 [0.099, 0.136] transductive gap; an inductive map to the scVI/Harmony point was never trained. F: the literal JK1 failure is a construction flaw of a control clause (reported), and JK3 counted "a 16-d compound-supervised LDA beats Harmony" as "no room" although that LDA gives 3.6x raw AP; no nonlinear supervised adapter was run.
- The programme-level stop was over-stated: the stop rule's own S3 clause was evaluated on 3 of 5 candidate fields (EEG and astronomy never run). Correct statement: "label-free adapters vs cheap floors are resolved NEGATIVE in A and C; B', D, E, F are gate outcomes without a valid learned arm; EEG and astronomy untested".
- Bookkeeping: job_result.json recorded the commit AFTER the run (worker.py now records the start commit and commit_after); the A phikon/TCGA protocol-2 row and a_lrn_midnight_*/a_p2x_dinov2L_camelyon jsonl have no write-up; the 2-seed D headroom 0.041 [0.001, 0.081] was stale (FOCUS D now carries the 3-seed 0.039 [-0.003, 0.077]).

## PROGRAMME 1 PARTLY REOPENED (2026-11-04 ~18:40Z VM clock; supersedes the stop below for D, E, F, A-retention, B')
New pre-registered ATTEMPTS, each reported alongside the original verdict, each tightening or repairing (never loosening a kill): F2 supervised nonlinear adapter vs the ldaP_16 envelope on JUMP S5 (<2 GPU-h); D2 adversarial arm rerun with a collapse check and sane lambda (minutes), then the pre-registered learned kill test at tau 0.90/0.97;
A-retention rescoring plus dinov2-L cross-fit, phikon-v2 holdout, the H2-015 selection null (~1 GPU-h each, extraction cached); E2 inductive map to the scVI/Harmony point (new pre-registration, not a continuation); B' adversarial arm on corr_glued/corr_noisy (the 9 parked jobs). EEG and astronomy stay parked unless the owner values breadth.

## (superseded, kept for the record) PROGRAMME 1 STOPPED (2026-11-04 ~16:50Z VM clock)
Stop rule (hypotheses/RANKING_R2.md section 7) met: AE lead dead (A), C closed, SC-0 fails K2 (E), D-digits fails H1, JUMP fails at the S4 gate (F). No further GPU work under research/fields; the shared VM stays up for programme 2.
Remaining: SUMMARY.md (scoped claims, per-focus table, limits, by-products) after the independent audit (research/fields/AUDIT_findings.md) lands.

## Focus B' REOPENED ATTEMPT (Auditor-1 F-11), result ~21:30Z VM clock: B' STAYS NO-GO under the rule as written, now on all three correlated kinds (3 seeds each; B.md appended section)
- The learned latent arm (adv) is WORSE than zero-shot SAM in every seed on corr_noisy (SAM minus adv -0.013, -0.031, -0.003) and corr_glued (-0.022, -0.051, -0.032), and about equal on corr_det (+0.023, +0.009, -0.002); the rule needs >= +0.05. Mean NMSE SAM vs adv: corr_noisy 0.830 vs 0.846, corr_glued 0.769 vs 0.804, corr_det 0.812 vs 0.802.
- Leakage: adv leaks less than SAM on corr_det (0.065 vs 0.204) and corr_noisy (0.046 vs 0.230), more on corr_glued (0.066 vs 0.016; relative sprite position, confounded by arm pose). The truly independent residual leaks ~0.00 on corr_noisy for every arm; on corr_glued raw leaks 0.05 and adv does not remove it. corr_det stays degenerate as a leakage test (nuisance is a function of the action). Rate control exists for seed 0 only on noisy/glued.
- Caveat unchanged: one learned latent arm, synthetic sprites, one fixed distractor draw, 64x64 LAM. Learned masks collapsed (failed implementation).

## Focus B': robotics latent actions   [NO-GO, decided 2026-11-04 11:30Z]
- Verdict (Builder B, results_notes/B.md): on action-correlated distractors (corr_det, 3 seeds) the best learned arm (adv) beats zero-shot SAM by 0.010 NMSE
  (rule: >= 0.05 in every seed): FAIL. Mechanism: the harm is input clutter, not leakage (raw z carries ~0 sprite information), so an adversary on z has nothing to
  remove; zero-shot SAM is no better than no mask here (0.812 vs 0.815); the exact-silhouette oracle gains only 0.048 over SAM, below the margin, so no mask-type arm
  could pass. Learned soft masks collapsed (failed implementation, not evidence). corr_noisy/corr_glued jobs already queued may finish; no more B' work.
- Original text (kept for the record): As stated (distractor-free latent actions on Distracting Meta-World) it is scooped ...
- As stated (distractor-free latent actions on Distracting Meta-World) it is scooped: a zero-shot SAM mask beats the label reference (NMSE 0.090 vs 0.142), 8+ competing methods in 12 months, no reference code, 2.4 TB data.
- Re-entry only for *action-correlated* distractors (a distractor whose motion tracks the action, where SAM-style masks cannot separate it), on a self-rendered MetaWorld/MuJoCo setup with exact nuisance ground truth.
- One-day kill test: build the env + a zero-shot-SAM baseline; if the baseline already solves it, B' is dead and its slot goes to the next field (egocentric wearer privacy, embedding privacy for retrieval, gaze, EEG, in that order).

## Standing rules for every focus
- Controls first: black/content-free input at chance; shuffled secret labels at chance; raw reference above the baseline; near-duplicate check across splits.
- Intervals: bootstrap over independent units (slides/centres, actors, episodes), 3 seeds minimum before any claim.
- Validation split for search, sealed test for the number. One result per config file, one commit per result.
- Learned methods preferred. Deterministic transforms are baselines, not contributions.
- Weights and derived embeddings to HF `KuanKuanKuan/openlatent-fields`; public datasets are never re-uploaded.

## UPDATE 2026-11-05 ~10:45Z VM clock: HLCA second dataset and the learned-mask closure (Builder F; F.md 18-21; results/f_jump/h3_003_*)
- Reduced H3-003 (HLCA, R2, d16, PCA-50): K-rule prediction refuted (dU_same +0.055..+0.204 vs predicted -0.012); controlled replication (3 seeds x 3 folds, validation-fold selection): REPLICATES, SUPCON GAIN SURVIVES against the best CE-MLP/LDA label floor (+0.060 / +0.057 at f = 10% / 100%); normalised-CE follow-up (20A): PARTIAL: +0.012 at f = 10% (explained by normalisation), +0.052 at f = 100% (rule-dependent; +0.018..+0.034 at interpolated matched leakage). Builder predictions wrong or half right twice.
- Learned source-removal on HLCA: 20B gradient reversal INERT TERM (seed 0); final attempt (section 21): ENGAGEMENT NEVER ACHIEVED in 0 of 24 cells (ramped GRL lambda 10/30, CORAL beta 3/30); line CLOSED as pre-registered. Original verdicts beside: JUMP NO-GO at the S4 gate; E's SC-0 NO-GO.
