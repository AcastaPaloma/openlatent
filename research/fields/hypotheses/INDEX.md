# Hypothesis index (generator output, 2026-11-04; NOT ranked, a separate ranker scores these)

25 hypotheses: A 7, C 7, X (cross-focus A + C) 8, B' 3. Boring controls (can kill an exciting result): H-A-007, H-C-005, H-C-006, H-X-007, H-X-008 (plus the kill-test H-B-001 and the floor/premise checks H-A-001, H-C-001).
Priors are my probability that the stated prediction holds as written (all clauses), not that the idea is useful. GPU-h are throughput-based estimates for one RTX A6000, not measurements. All numeric predictions are priors to be falsified, not results.

| ID | One line | Focus | Cost (GPU-h) | Prior |
|---|---|---|---|---|
| H-A-001 | LEACE zeroes the linear centre probe but a retrained nonlinear probe keeps >= 0.30 of the raw leakage (premise of Focus A; also patch- vs slide-split inflation, unseen-centre LEACE) | A | 2 (+6 CPU) | 0.75 |
| H-A-002 | A biology-only rate bottleneck (no adversary) cuts leakage >= 40% at d=8 but never reaches L_worst <= 0.25; beats PCA-d and random projection | A | 4 | 0.50 |
| H-A-003 | A class-conditional adversary keeps >= 3 points more biology than marginal adversary / LEACE under centre-class correlation (V=0.8) | A | 3 | 0.45 |
| H-A-004 | Stain-augmentation matched-pair InfoNCE + cross-centre SupCon adapter cuts unseen-centre leakage >= 25% vs CE-only and beats LEACE | A | 6 | 0.35 |
| H-A-005 | Per-centre whitening is a hard floor on unseen centres; Tent-style entropy adaptation of the adapter adds >= 2 points where class priors are skewed | A | 2 | 0.30 |
| H-A-006 | APD and RI cannot rank raw/LEACE/ComBat/learned methods (spread < 1.5 pts); the retrained probe spans >= 0.4 and disagrees with RI rank | A | 2 (+10 CPU) | 0.60 |
| H-A-007 | BORING CONTROL: 24 colour/blur statistics carry >= 70% of the centre signal; regressing them out may match the learned adapters | A | 1 | 0.55 (0.20 that it kills) |
| H-C-001 | Premise + layer map: speaker decodable at every emotion-bearing layer (no free lunch); LEACE fitted on training actors does not erase unseen actors | C | 3 | 0.70 |
| H-C-002 | Rate does the work in audio: 6-number emotion posterior leaks <= 0.2; at matched numbers a time-resolved release leaks >= 0.1 more than a pooled one | C | 3 | 0.45 |
| H-C-003 | Matched-content contrastive (same sentence+emotion, different actor) lowers speaker leakage >= 0.15 vs CE at equal emotion | C | 5 | 0.40 |
| H-C-004 | Identity-protected is not attribute-protected: sex stays decodable on the identity frontier; removing sex costs >= 4 emotion points | C | 1 | 0.50 |
| H-C-005 | BORING CONTROL: 3-6 hand-made prosody numbers give emotion >= 40% UAR and speaker >= 4x chance (floor every learned z must beat) | C | 0 | 0.60 |
| H-C-006 | BORING CONTROL: non-speech head/tail (room, mic) identifies actors >= 5x chance; CREMA-D identity is partly a channel benchmark | C | 0 | 0.40 |
| H-C-007 | Speaker-adversarial (GRL / SISER-style) emotion encoders gain 1-4 points emotion on unseen actors but cut retrained-probe leakage by < 0.10 | C | 5 | 0.55 |
| H-X-001 | The adversary adds nothing at d <= 4 and >= 0.10 at d >= 32; threshold rate within 4x across pathology and audio (transfer of our KTH finding) | A+C | 10 | 0.45 |
| H-X-002 | Minimax against a restarted diverse attacker population lowers worst-case leakage on held-out attacker families by >= 0.10 vs a single adversary | A+C | 6 | 0.40 |
| H-X-003 | A distance-correlation penalty matches the adversary within 0.05 at <= 1/5 the cost; a CLUB MI bound is worse | A+C | 4 | 0.35 |
| H-X-004 | Occurrence-normalised VQ leaks >= 0.10 less than plain VQ; codebook dropout adds >= 0.05 and kills the usage-histogram attack | A+C | 6 | 0.30 |
| H-X-005 | Defining the secret by a dense source descriptor (stain stats / ECAPA embedding) transfers removal to unseen sources; a closed-set label adversary has a >= 0.2 seen/unseen gap | A+C | 4 | 0.40 |
| H-X-006 | Iterative nonlinear erasure (Obliviator-type) halves LEACE's worst-case leakage at a 2-6 point utility price; kernel erasure is attacker-specific | A+C | 3 | 0.35 |
| H-X-007 | BORING CONTROL: a placebo adversary (shuffled / irrelevant labels) reproduces >= 40% of the real adversary's gain | A+C | 2 | 0.30 |
| H-X-008 | BORING CONTROL: per-source mean/whitening with oracle source IDs reaches L_MLP <= 0.35 at <= 3 points utility cost, matching learned per-sample encoders | A+C | 1 | 0.50 |
| H-B-001 | Kill test: oracle-prompted SAM masks solve independent sprites but not action-correlated sprites overlapping the arm; B' survives only if they fail | B' | 15 | 0.35 (B' survives) |
| H-B-002 | LAM trained with action-correlated sprites (rho_eff ~0.9) loses >= 2x on decorrelated sprites and encodes the sprites (probe R^2 >= 0.5) | B' | 5 | 0.55 |
| H-B-003 | A 5% interventional slice or 200 action labels close >= 40-60% of the shortcut gap; Dueling readout subtraction closes < 10% | B' | 8 | 0.30 |

Total if every hypothesis ran once at its estimated cost: ~105 GPU-h (about one day of the 4-GPU VM) plus ~20 CPU-h.

## Dependency map (what must exist first)
- Builder-A (`src/a_pathorob`): embedding cache for phikonv2/phikon/midnight/dinov2L, slide/centre/label arrays, probe suite (linear, mlp, knn, tfm in `attackers.py`; the A-SUITE below adds LightGBM and RFF-SVM), adapters (`vib`, `adv_retrain`, `dann`, `vq`, `supcon_xc`, floors raw/lda/leace/isp/centre_mean), RI/APD port checked against the leaderboard. Needed by H-A-001..007, X-*.
- Builder-C (`src/c_cremad`): `splits.json` (4 actor folds, S_TR/S_TE, E_TR/E_TE, T1/T2/T3), frame-level WavLM features, pooled stats, baselines (pooling, PCA, LEACE), `evaluate.py`. Needed by H-C-001..007, X-*. Extras requested: seen-actor attacker arm (H-X-005), a TDNN sequence attacker, wavlm-large / hubert-large / emotion2vec extraction (H-C-001), ECAPA embeddings (H-X-005).
- Builder-B (`src/b_robotics`): `envgen.py`, `distractors.py` (kinds none/indep/corr_det/corr_noisy), `sam_masks.py`; extras requested: `SIGMA_CORR` argument and a `drive_from="command"` option (H-B-001..003), LAM trainer.
- Shared modules the hypotheses assume exist or get written once: GRL adversary with restarts and full warm-up (journal H18 fix), dCor / CLUB / HSIC penalties (H-X-003), product-VQ with Sinkhorn balance and codebook dropout (H-X-004), population adversary (H-X-002), kernel / Obliviator-type eraser (H-X-006).

## Suggested minimum-cost first batch (not a ranking)
Zero-GPU or near-zero: H-C-005, H-C-006, H-A-007 (the controls) and H-A-001, H-C-001 (the premises). Their outcomes decide whether the rest are worth running: if H-A-001 falsifies, A has no headroom; if H-C-006 shows a channel confound, the C benchmark needs the trimmed variant first; if H-C-005's 6-number set is on the learned frontier, report it as the hand-built winner.

## Shared definitions (all hypothesis files use these)

- **Chance / label-only baseline.** chance = 1/n_classes unless the class prior is skewed (measure it from the label file; balanced accuracy where used). Task-label-only baseline = leakage of the secret from a z that carries only the task label: chance in C (balanced crossed design) and in A at V = 0; for A under centre-class correlation use the LDA projection (`lda` floor).
- **Normalised leakage.** For an attacker k trained on the released z: L_k = (acc_k(z) - chance) / (acc_k(raw) - chance), where raw = the unmodified frozen embedding attacked by the same attacker family. L_worst = max_k L_k over the suite. 0 = chance, 1 = raw reference. Sign-flip invariant for binary secrets (take max of acc and 1 - acc only when the probe is not retrained; retrained probes make this moot).
- **Normalised utility.** U = (acc_task(z) - chance) / (acc_task(raw) - chance) with a retrained probe on the held-out units (A: biology accuracy on ID held-out slides and OOD centres; C: emotion UAR on actor-disjoint test fold).
- **A-SUITE** (Focus A attackers, always retrained on the transformed embeddings): logistic regression, MLP (2x512, dropout 0.2, early stopping on validation slides), kNN (k=20 cosine), LightGBM (300 trees), RFF-SVM (4096 random Fourier features); slide-disjoint 60/10/30 train/val/test as in the PathoROB probe protocol (2507.17845), hyperparameters chosen on validation slides only. Builder-A's `attackers.py` also has an FT-Transformer-style `tfm`.
- **C-SUITE** (Focus C attackers): LR, MLP (2x512), kNN-5 cosine, LightGBM on pooled z; x-vector-style TDNN and BiGRU-attention on sequence z; attackers enrol the sealed test-fold actors and train on S_TR x E_TR only, scored on T1 (unseen sentences, seen emotions; primary), T2 (unseen emotions), T3 (both unseen), per `src/c_cremad/data.py`.
- **Units and intervals.** Independent units are slides (within centre) in A, actors in C, episodes in B'. All CIs are bootstrap over those units (1000 resamples), never over patches or clips. Claims need 3 encoder-training seeds x 3 attacker seeds and CIs that exclude the comparison value.
- **Splits.** Search and hyper-parameters on validation; the sealed test split gives the number; one result per config file, one commit per result.
- **Equal utility.** Methods are compared at U within 0.02 of the reference or on the interpolated (U, L) frontier.
- **Fit-distribution rule.** Every eraser, adapter and normaliser is fitted on training centres/actors only; unseen sources are never used for fitting except in arms explicitly labelled "oracle".
- **Source tags used in the files.** [O] = I opened the arXiv abstract (or HTML) page in this session; [S] = opened by a literature scout and read through `literature/*.md`; [snip] = seen only in a search snippet, not relied on. Numbers from scouts are the authors' self-reported ones.

## Papers opened by me in this session (abstract pages unless noted)
2306.03819 (LEACE), 2507.17845 (PathoROB, HTML read to ~100k of 241k characters), 2603.07529 (Obliviator), 2505.15667 (Segmentation-Variant Codebooks), 2006.12013 (CLUB), 2605.20223 (Why Latent Actions Fail), 2608.06706 (Dueling World Models), 2501.05310 (speaker attributes in SSL layers), 2609.28231 (Do Center Biases Propagate?), 1603.04779 (AdaBN), 1911.08731 (Group DRO), 2312.15185 (emotion2vec), 2607.22861 (Filiot et al. pathology fine-tuning), 2006.10726 (Tent), 2004.11362 (SupCon), 2201.12191 (Kernelized Concept Erasure), 2006.07500 (Domain generalisation using causal matching), 2101.10001 (Diverse adversaries), 2607.25497 (CRoMa), 2605.01381 (concept-subspace framework), 2609.02941 (SISER), 2602.18502 (shortcut mitigation benchmark). Pages that did not contain what I wanted: 2610.30104 (BatchBench abstract shows no baseline numbers; the AdaBN +7 / DANN -0.5 figures are scout-reported), 2006.07966 (wrong ID for MatchDG; the right one is 2006.07500).

---

# H2: round-2 hypotheses (generator round 2, 2026-11-04, written after the A / B' / C / D results; NOT ranked)

19 hypotheses `H2-001` to `H2-019`: AE-lead replication and attribution 7 (001-007), C multi-task 1 (008), D reframe 1 (009), new candidate fields 4 (010-013), boring controls 4 (014-017), cross-focus procedure and stop rule 2 (018-019). Round-1 files and numbers are unchanged. Priors are my probability that the stated prediction holds as written (all clauses); GPU-h are throughput estimates, not measurements. Every numeric prediction is a prior to be falsified.

## Organising question
Where does the task NOT determine the content a representation must keep, so that label-only projections collapse and a learned task-agnostic latent is the only option? Evidence in hand (all single-seed unless stated): label-supervised LDA keeps the label and hides the source in A and C; the secret overlaps the utility in D; the harm is clutter, not leakage, in B'; the one unfalsified lead is the task-agnostic adversarial AE on held-out classes (A, midnight only, post hoc, best of 10 variants). Round 2 asks (a) is that lead real (001-007, 014-016), (b) which other fields have many downstream tasks plus a deterministic benchmark plus a trivial floor that can be tested cheaply (008-013), (c) what would we stop on (017-019).

## Definitions added by round 2 (used in the H2 files)
- **Threat models.** T = transductive: the source id (centre, actor, subject, batch) of every released item is available at release or test time, so per-source normalisation or whitening is allowed. I = inductive: unseen sources, no source id at release (the situation of a foundation-model embedding released for unknown future use). Only I-floors are kill floors for a learned encoder; T-methods (Harmony, per-source whitening, Sphering) are references and the deployable alternative.
- **Leakage-matched rule (LM-rule).** For a learned configuration with (U, L) on the sealed split, the comparator is the best I-floor point with L_ref <= L + 0.05, interpolated along each floor family's envelope (favours the baseline). I-floors: PCA-d, random projection-d, label projection + PCA padding (`lda_pca__d`), noise-dithered codes (H2-014), label-only projection.
- **SEL-V.** Configuration chosen on the validation split only: among configurations of the family with L_V <= 0.35 (A; field-specific cap elsewhere) the one with the highest V utility on the held-out tasks, ties to smaller dz then smaller lam. The sealed-split number is that configuration; best-of-N on the sealed split is a diagnostic only (H2-015).
- **Headroom fraction.** h = U_ref(T') - U_floor*(T'; L*) from floors only (on V), g = U_learned - U_floor* on the sealed split, g/h the fraction closed (H2-018). A learned arm is not run in a new cell until h is committed.
- **Held-out tasks T'.** Classes, label sets, attributes or subtasks not used to fit the encoder; probes on them are retrained on the released representation.

## Index
| ID | One line | Focus | Cost (GPU-h) | Prior |
|---|---|---|---|---|
| H2-001 | AE gain over leakage-matched inductive floors survives V-only selection, 3 seeds and phikon-v2 / phikon (>= 3 of 6 cells at +0.04) | A | 5 (+30 CPU) | 0.30 |
| H2-002 | Attribution: the adversary adds <= 0.02 AUC over a lam = 0 rate-limited AE; the nonlinear decoder, not the adversary, beats PCA-d by >= 0.03 | A + C | 3 | 0.35 |
| H2-003 | A label-free embedding-level contrastive code leaks (L >= 0.6) alone and ties the AE within 0.04 once an adversary is added | A | 3 | 0.35 |
| H2-004 | Dose-response: AE gain over label-supervised arms grows as the number of seen classes falls (Delta(2) - Delta(4) >= 0.03) | A | 2 (+10 CPU) | 0.40 |
| H2-005 | The AE keeps slide/case fingerprints: at equal held-class utility it leaks >= 0.10 more slide identity than PCA / LDA-family (second secret) | A | 1 (+6 CPU) | 0.60 |
| H2-006 | A label-free adapter trained on two datasets transfers its centre invariance to the third (leave-one-dataset-out; C arm CREMA-D -> RAVDESS) | A + C | 3 | 0.20 |
| H2-007 | A conditional-decoder VAE (source id to the decoder, no adversary) ties the adversarial AE with lower seed variance | A | 2 | 0.30 |
| H2-008 | CREMA-D multi-task probes (emotion + sentence content): LDA-emotion collapses sentence content, a speaker-adversarial label-free AE keeps it (+0.08 U_MT) | C | 1.5 | 0.20 |
| H2-009 | D reframe: with digits/entities as the secret (not used by retrieval) the oracle retains >= 0.97 nDCG and headroom >= 0.30; check first, learned stage second | D | 0.5 check / 6.5 full | 0.20 (check) |
| H2-010 | NEW FIELD N1, single-cell: inductive label-free adapter on scFM embeddings keeps fine cell types under unseen studies where coarse-label LDA collapses | new (scIB-style) | 2.5 check / 5 full | 0.15 (0.45 check) |
| H2-011 | NEW FIELD N2, Cell Painting (JUMP): adapter on well profiles retrieves replicates of held-out compounds across labs beyond inductive floors | new (JUMP) | 1 check / 4 full | 0.15 (0.70 check) |
| H2-012 | NEW FIELD N3, EEG foundation models: label-free adapter keeps 5 task decodings at low subject-identity leakage where LEACE fitted on training subjects cannot | new (EEGMMI) | 1.5 check / 4 full | 0.12 (0.35 check) |
| H2-013 | NEW FIELD N4, astronomy GalSim twin render with exact truth: adapter keeps held-out physical parameters that a parameter-supervised projection discards | new (astronomy / A domain 2) | 3 (+4 CPU) | 0.15 (0.65 check) |
| H2-014 | BORING CONTROL: noise-dithered PCA / LDA+PCA (or PCA then low-dimensional LEACE) matches the AE at equal leakage | A | 1 (+8 CPU) | 0.35 (that it kills) |
| H2-015 | BORING CONTROL: the best-of-10 pick at test inflates a null family by +0.04 to +0.09, a large share of the seed-0 lead | A | 0 (+2 CPU) | 0.35 (that it kills) |
| H2-016 | BORING CONTROL: power audit: MDE80 >= 0.10 on TCGA cells, ~0.05-0.06 on Tolkach, matched leakage resolved only to +-0.10 to +-0.20; cross-fitting halves it | A + C | 0 / 5 remedy | 0.70 |
| H2-017 | BORING CONTROL: per-actor standardisation of full WavLM features (oracle ids) is the multi-task floor (identity <= 10x chance, emotion >= 0.70) | C (A analogue exists) | 0.3 | 0.45 (that it wins) |
| H2-018 | Floors-only headroom fraction predicts where a learned latent wins: g >= 0.4 h where h >= 0.05, g <= 0.02 where h < 0.03 (4 post hoc A points fit) | cross-focus | 0 | 0.30 |
| H2-019 | STOP RULE: the central claim is false and the line stops if H2-001/014/015, the C AE round and the new-field checks all fail at pre-declared margins | programme | 0 | 0.35 |

Total if every H2 hypothesis ran in full: ~50 GPU-h (stage-1 headroom checks of 009-013 ~7 GPU-h) plus ~55 CPU-h of attacker evaluation, mostly in A.

## Dependency notes (not a ranking)
- Zero-GPU, existing artefacts: H2-016 (saved predictions), H2-015 and the dither arms of H2-014 (CPU), H2-017 (cached WavLM features), H2-018 script. H2-014 supplies the strongest comparator to H2-001, H2-006, H2-015.
- Already queued and shared: A seeds 1-2 of the midnight/dinov2 holdout cells (H2-001, H2-004), C `c_ae_*` (H2-002, H2-008 after a sentence probe is added).
- Headroom-first rule for every new field: floors and the oracle/clean reference before any learned arm (H2-009 to H2-013; H2-018 makes it a standing procedure).
- Needs the programme owner: licences and download sizes not verified by me (RAVDESS for H2-006, EEGMMI and EEG foundation-model weights for H2-012, single-cell atlas and scFM weights for H2-010, JUMP per-scenario sizes for H2-011, GalSim for H2-013); the PII-Masking-300K licence flag from the journal stays open.

## Papers opened by me while writing H2 (abstract / page level)
2606.06647 (Identity Trap in EEG Foundation Models), 2604.09787 (Learning What's Real), 2610.30104 (BatchBench abstract; no numbers on the page), PMC11297288 (Arevalo et al., JUMP batch correction, five scenarios), bioRxiv 10.64898/2025.12.19.695371 (batch effects in single-cell foundation models). Tags [S] in the H2 files mean "read by a literature scout, not opened by me"; scIB itself was not opened (the nature.com page redirected to a login).
