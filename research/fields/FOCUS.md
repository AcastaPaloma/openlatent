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

- STATUS (2026-11-04, Builder D, seeds 0-1 of 3, headroom stage only): NO-GO as pre-registered. H1 passes (worst retrained attacker R-precision on raw vectors 0.183 vs 0.004 prior);
  H2 fails: the oracle (rare words deleted before embedding) keeps only 0.74 of nDCG@10 and still has normalised leakage 0.59, so headroom 0.041 [0.001, 0.081] < the 0.10 needed.
  Mechanism: rare words are retrieval-critical, so the secret and the utility overlap ("task implies the secret"). Learned kill test not run. Post-hoc learned arms (2 seeds) are queued
  and cannot flip the verdict. Reframe candidate (not started): a secret retrieval does not use (entities, numbers) built from public-licence text.

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

## Focus F: JUMP Cell Painting (check-only)   [NO-GO at the S4 gate, 2026-11-04 ~16:50Z VM clock; programme-1 stop condition met]
- Builder F (results_notes/F.md 5.1-5.8; pre-registered before any number; priority 4 used once; 3.2 worker-hours). Literal verdict: R0 passes (S2 mean AP 0.9007 / 0.4096 vs published 0.90 / 0.41); JK1 FIRES via the
  control-AP clause in 3/3 splits (|U(shuffled compounds, raw) - U(noise16)| = 0.042-0.045 > 0.01; the shuffle control sits below chance because cross-source positives face same-plate negatives); the substantive
  measurability clause passes (raw worst-of-4 source attacker 0.885 [0.865, 0.914] vs 0.60). Amendment A2 (written after the S4 numbers, relaxing that clause) was ruled a deviation (D1); the literal verdict stands.
- Post-hoc sensitivity on S5 (cannot revive the line): JK0 clears (40.4k held-out compounds, MDE80 0.0010); JK3 FIRES 3/3: Harmony (U 0.077, L 0.99) is beaten by the 16-d compound-supervised LDA (U 0.183, L 0.39),
  dU -0.106 [-0.107, -0.104]; JK2 does not fire on the registered grid (lowest L_V 0.37 at d=16, grid edge; d<=8 not run). K2 answer: LDA-301 leaks source at raw level (L 0.99). Stage 2 (label-free AE/cVAE vs the ldaP_16 envelope) not run.

## PROGRAMME 1 STOPPED (2026-11-04 ~16:50Z VM clock)
Stop rule (hypotheses/RANKING_R2.md section 7) met: AE lead dead (A), C closed, SC-0 fails K2 (E), D-digits fails H1, JUMP fails at the S4 gate (F). No further GPU work under research/fields; the shared VM stays up for programme 2.
Remaining: SUMMARY.md (scoped claims, per-focus table, limits, by-products) after the independent audit (research/fields/AUDIT_findings.md) lands.

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
