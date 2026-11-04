# FOCUS (locked 2026-11-04 08:15Z)

Basis: six phase-1 scouts, then two contrarian scouts (missed fields: ~25 fields, nothing decisively better, ~85-90%
confidence; stress test: A and C go with changes, B no-go as stated). Evidence in `literature/`. Change this file only with a
journal entry that says why, and only when a result or a newer scout falsifies a premise below.

## Focus A: source-invariant embeddings, PathoROB first   [GO-WITH-CHANGES]
- Keep: tissue/diagnostic label.  Hide: medical centre (and stain) signature.
- Benchmark: PathoROB (arXiv 2507.17845; 3 ungated HF datasets, ~1.9 GB, fixed seeds). Drop BatchBench (sealed, server-side, 30-257 GB).
  Second domain only after A works on PathoROB: astronomy instrument latents (GalSim + HSC/Legacy cross-matches) and Cell Painting (JUMP), as generality checks.
- Headroom is NOT the Robustness Index (saturated 0.87-0.93). It is a *retrained* centre probe (linear probes reach 88-98%
  on frozen embeddings; linear erasure fools linear probes but a retrained nonlinear probe keeps 2.2x chance).
- Learned encoders: adapter E(z_FM) with (i) rate bottleneck, no adversary (mandatory control), (ii) adversarial with retrained
  attackers, (iii) VQ/discrete bottleneck, (iv) centre-aware contrastive. Floors: raw embedding, LEACE/ISP, DANN.
- Kill: if the best learned adapter does not beat LEACE on worst-case centre-probe accuracy at equal label accuracy (3 seeds, bootstrap over centres/slides), stop A.

## Focus C: identity-free emotion, CREMA-D first   [GO-WITH-CHANGES]
- Keep: emotion (6 classes).  Hide: actor identity (91 actors), plus sex/age/race attributes as secondary secrets.
- Data: CREMA-D (ODbL, ~2.5 GB, VM-reachable). RAVDESS as the second set. We build the benchmark ourselves (prior WACV 2025 /
  ICME 2026 pipelines use frozen validators, shared-actor splits, gated data, no checkpoints).
- Protocol: emotion test on actor-disjoint split; identity attacker trained on the attacker's own held-in actors/sentences, scored on held-out sentences/emotions of the same actors; attackers retrained on the obfuscated latents.
  Normalise: chance (1/91), task-label-only baseline, raw reference. Worst case over linear, MLP, kNN, and a different-backbone attacker.
- Modalities: audio first (WavLM/HuBERT frozen features, cheapest), then face video crops.
- Baselines to reproduce under *retrained* attackers: blur, optical flow, landmarks, VoicePrivacy-style anonymisation. Floors: LEACE, rate bottleneck.
- Kill: if the raw reference does not beat the task-label baseline on identity (not measurable) or no learned method beats the rate bottleneck.

## Focus B': robotics latent actions, conditional   [NO-GO as stated; re-entry via kill test]
- As stated (distractor-free latent actions on Distracting Meta-World) it is scooped: a zero-shot SAM mask beats the label reference (NMSE 0.090 vs 0.142), 8+ competing methods in 12 months, no reference code, 2.4 TB data.
- Re-entry only for *action-correlated* distractors (a distractor whose motion tracks the action, where SAM-style masks cannot separate it), on a self-rendered MetaWorld/MuJoCo setup with exact nuisance ground truth.
- One-day kill test: build the env + a zero-shot-SAM baseline; if the baseline already solves it, B' is dead and its slot goes to the next field (egocentric wearer privacy, embedding privacy for retrieval, gaze, EEG, in that order).

## Standing rules for every focus
- Controls first: black/content-free input at chance; shuffled secret labels at chance; raw reference above the baseline; near-duplicate check across splits.
- Intervals: bootstrap over independent units (slides/centres, actors, episodes), 3 seeds minimum before any claim.
- Validation split for search, sealed test for the number. One result per config file, one commit per result.
- Learned methods preferred. Deterministic transforms are baselines, not contributions.
- Weights and derived embeddings to HF `KuanKuanKuan/openlatent-fields`; public datasets are never re-uploaded.
