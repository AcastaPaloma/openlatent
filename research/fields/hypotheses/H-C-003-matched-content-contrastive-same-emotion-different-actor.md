# H-C-003: Contrastive training with exactly matched (same sentence, same emotion, different actor) positives lowers speaker leakage >= 0.15 vs CE at equal emotion

- Focus: C | Kind: learned method (contrastive, object-matching) | Cost: ~5 GPU-h | Prior: 0.40 (reduction >= 0.15); 0.20 (beats the rate-matched bottleneck)
- Depends on: Builder-C frame features; sentence/emotion/actor labels; C-SUITE.

## Statement
CREMA-D is nearly fully crossed (91 actors x 12 sentences x 6 emotions, ~82 clips/actor), so for most clips there are positives that share the sentence *and* the emotion and differ only in actor. An adapter trained with InfoNCE where P(i) = {same sentence, same emotion, different actor}, hard negatives = {same sentence, same actor, different emotion}, reduces L_worst on unseen test-fold actors by >= 0.15 (absolute) versus an emotion-CE adapter of the same architecture and d at |dU| <= 0.02, and by >= 0.08 versus plain SupCon with positives = any same-emotion clip.

## Mechanism
Mahajan et al.: invariance to the domain (here actor) given the label is not enough; representations of inputs from the same underlying object across domains should coincide (2006.07500, abstract opened). Here "same object" = same text + same emotional rendering, domain = actor, observed exactly because the corpus is crossed. Inside a positive set the only things that vary are actor identity and micro-prosody, so pulling positives together discards actor. SupCon in general is a stronger classification loss (2004.11362) but its positives mix sentences and actors without that pressure. Risk: the pooled objective can discard the actor-dependent *scale* of emotion (arousal level), losing utility.

## Experiment (exact)
- Data: CREMA-D, fold 0-3 (encoder trained on 56 train actors, val 12, enrolled attackers on test fold).
- Encoder: wavlm-base-plus best layer frames -> attentive pooling -> MLP -> d = 64, L2-normalised; temperature 0.1; batches of 32 (sentence, emotion) groups x 4 actors (128 clips), 60 epochs.
- Ladder: (1) CE-only; (2) SupCon any-same-emotion; (3) matched-content InfoNCE (proposed); (4) matched-content with same-actor positives allowed (ablation: is the *different actor* part what matters?); (5) (3) + CE head; (6) rate-matched d-sweep from H-C-002; (7) LEACE; (8) (3) with an added actor adversary (X-001 style, a=1).
- Attackers: C-SUITE on pooled z, trained on S_TR x E_TR of the test-fold actors, scored on T1/T2/T3.
- Seeds: 3 x 4 folds; bootstrap over actors.
- Metric: L_worst, U, and the accuracy of the emotion head on held-out actors; also the *sentence*-probe accuracy of z (should stay low; a leak the loss may actually create).

## Prediction and falsifier
- Prediction: L_worst(3) <= L_worst(1) - 0.15 at |dU| <= 0.02 (e.g. 0.8 -> <= 0.65); L_worst(3) <= L_worst(2) - 0.08; ablation (4) is within 0.05 of (2) (the different-actor part carries the effect); (3) does not beat the rate-matched bottleneck of H-C-002 at equal d by more than 0.05; sentence-probe accuracy of z <= 2x chance.
- Falsified if L_worst(3) > L_worst(1) - 0.05 (matching buys nothing over CE), or if U(3) < U(1) - 0.05 (contrast discards emotion).

## Cost
~5 GPU-h: 8 ladder arms x 3 seeds x 4 folds at ~3-5 min per run on cached frame features.

## Dependencies
Builder-C: dataloader over cached frames with group sampler (sentence, emotion) x actors. No cross-focus dependency (the pathology analogue is H-A-004).

## Prior and why
0.40. The matching mechanism is clean and uses a corpus property no other dataset in the programme has, but contrastive losses typically reduce rather than remove nuisance; the unseen-actor attacker usually finds residual style. Beating the bottleneck is the real test (0.20).

## Sources
[O] 2006.07500; [O] 2004.11362; results/SUMMARY.md (rate does most of the work); literature/contrarian_stress_test.md 3.2 (CREMA-D structure: 12 sentences in every emotion).
