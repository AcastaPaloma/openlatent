# H2-012: NEW FIELD N3. EEG foundation-model embeddings: for unseen subjects and no subject id at release, a label-free adapter keeps several task decodings at a subject-identity leakage that closed-form erasure fitted on training subjects cannot reach

- Focus: candidate field (not previously killed) | Kind: headroom check first, learned arm second | Cost: ~1.5 GPU-h check, ~4 GPU-h full | Prior: 0.12 (full claim), 0.35 (headroom check passes)
- Depends on: frozen EEG foundation-model weights (LaBraM / CBraMod / REVE as in the opened audit; availability and licences not verified by me); EEGMMI from PhysioNet (109 subjects, 14 runs of several task types; licence and download size not verified here); H2-001 AE template.

## Statement
EEGMMI contains, per subject, eyes-open and eyes-closed baselines and executed and imagined movement runs of two kinds (left vs right fist; both fists vs both feet). Threat model I: subject-disjoint adapter training (70 subjects, all runs, no task labels for the AE), release for unseen subjects, no subject id at release. Downstream tasks (all retrained probes on z, scored on the 39 unseen subjects): T1 eyes open vs closed, T2 left vs right imagery, T3 left vs right execution, T4 hands vs feet execution, T5 hands vs feet imagery. Secret: subject identity (39-way among test subjects, chance 2.6%), attackers enrol on runs {R03..R08} and are scored on runs {R09..R14} (task-disjoint within subject, 4-s windows aggregated per run). Headroom check (floors only):
- He: raw frozen embeddings identify unseen subjects at >= 10x chance (measurability; FMScope reports frozen subject variance 13-89x a random null);
- Hf: the best inductive floor (LEACE of subject fitted on training subjects, subject-axis erasure, PCA-d, LDA on T2 labels) leaves identity >= 0.4 of raw (normalised) at <= 3 points mean loss on T1-T5;
- Hg: the label-supervised floor (LDA on T2) keeps <= 60% of raw's above-chance mean accuracy on T1, T3-T5.
If all pass, the claim: the label-free adversarial AE reaches identity <= 0.2 of raw at mean T1-T5 accuracy within 0.03 of the best floor's accuracy at the same identity leakage, 3 seeds, bootstrap over the 39 test subjects, and it is not beaten at its leakage level by per-subject mean/whitening in the transductive reference.

## Mechanism
EEG foundation models are released for unknown future tasks (motor imagery, sleep, emotion), so the task-agnostic setting is the natural one, and a task-label projection kills every other task. Subject identity is a large, mostly linear-plus-spectral nuisance in frozen embeddings (opened audit: subject variance 13-89x null in 12/12 model-dataset pairs; erasing the dominant subject axis raised label decoding by 6-12 points in primary cells; aperiodic 1/f removal cut subject probes by 9-19 points on two of three models). The audit also warns the other way: linear erasure may already be enough, as for LEACE on EEG in the stress-test scout, and per-subject re-centring (Riemannian alignment, per-subject standardisation) is a strong training-free transductive baseline (medical-biosignal scout, not opened by me). The learned adapter's possible niche is the inductive setting, where C showed closed-form erasure fitted on training actors leaves ~53x chance on unseen ones.

## Experiment (exact)
- Stage 0: verify EEGMMI licence/size and model weights; extract frozen embeddings at the layer used by the audit (FMScope code is public per the audit; layer-wise probing is one of its diagnostics); fix subject folds.
- Stage 1 (check): references and floors above; additionally per-subject z-scoring/whitening with subject ids (transductive reference, not a kill floor), identity attackers linear/MLP/kNN/LightGBM retrained on z, task probes linear/MLP.
- Stage 2: AE + adversary (subject ids of training subjects), cVAE, contrastive; dz in {16, 32, 64}; SEL-V on 10 validation subjects.
- Controls: constant/noise at chance for all tasks and identity; shuffled subject labels at chance; within-subject window leakage controlled by run-disjoint splits.

## Prediction and falsifier
- Prediction: He passes (>= 10x chance, 0.9); Hf passes only half the time (LEACE on training subjects keeps >= 0.4 of identity: 0.4); Hg passes (0.8); learned arm +0.02 to +0.06 mean task accuracy over the floor at equal identity leakage.
- Falsified (NO-GO at the check) if LEACE/subject-axis erasure fitted on training subjects already brings identity to <= 0.3 of raw at <= 3 points loss (the audit's linear-erasure reading holds out of sample), or if per-subject whitening dominates everything in a threat model where it is allowed.

## Cost
Embedding extraction ~1 GPU-h per model; adapters and evaluation ~0.5 GPU-h per model; ~4 GPU-h with 3 models and 3 seeds.

## Dependencies
Model weights; dataset access; owner approval of any licence terms.

## Prior and why
0.35 for headroom, 0.12 for the claim. Brain-signal identity is mostly linear and low-dimensional in these embeddings, so the programme's repeated pattern (a cheap deterministic method already works) is the likeliest outcome; the multi-task structure is real, which is why it is listed.

## Sources
[O] arXiv 2606.06647 (Identity Trap in EEG Foundation Models: FMScope, LaBraM/CBraMod/REVE, 13-89x, +6-12 points, 1/f ablation); [S] 2609.00390 NeuroPriv, 2606.09189 and literature/medical_biosignal.md (not opened by me); results_notes/C.md (LEACE fitted on training actors).
