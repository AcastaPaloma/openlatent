# B' kill test: robotics latent actions under action-correlated distractors

Builder-B notes. Branch `fields-research`. Code: `research/fields/src/b_robotics/`. Job specs: `research/fields/jobs/specs/b_*.yaml`.
Status lines are appended chronologically at the bottom; the verdict section is written last.

## Question (from FOCUS.md, Focus B')
Does a *learned* mask/latent beat a zero-shot SAM-style mask when the distractor is action-correlated (its motion tracks the
agent's action), so that appearance/segmentation alone cannot separate it from the arm? If the SAM baseline already solves
it, B' is dead.

## Setup (what was actually built)
- Env: MetaWorld v3 (MuJoCo 3.2.7, OSMesa, CPU rendering) in a venv `~/venvs/b` on the VM. 3 tasks (reach, push, pick-place),
  scripted expert + per-macro-step Gaussian exploration noise (std 0.5 on xyz) so the action is NOT a function of state.
  One macro step = 4 env steps, 16 macro steps per episode, 128x128 render from a fixed free camera (arm fills ~10-25% of the image).
  600/100/100 train/val/test episodes per task (1800/300/300 total, disjoint seeds). Label = mean executed action over the macro step.
- Exact ground truth per frame: clean frame, exact arm silhouette and object silhouette (MuJoCo segmentation), EE pixel position,
  macro action. Distractors are image-space sprites (3 opaque discs, radius 7 px, half of the palette arm-like) composited over
  the clean render; the distractor-only layer and every nuisance label are exact functions of (kind, actions, seed).
- Distractor kinds: `indep` (random-walk motion independent of the action; the case SAM solves), `corr_det` (velocity =
  fixed linear map of the action, no residual), `corr_noisy` (same plus independent noise), `corr_glued` (sprites hover on the gripper:
  EE pixel position plus AR(1) jitter; overlaps the arm, so no silhouette separates them).
- Latent action model: LAPO-style IDM(f_t, f_t+1) -> z (VAE, KL) and FDM(f_t, z) -> f_t+1 (FiLM U-Net), 64x64 input, 30 epochs, bf16.
- Utility = linear probe z -> true macro action on 2000 labelled pairs (ridge, lambda on val), NMSE on the test split, 1.0 = chance.
  Reported on xyz (the gripper dimension is sub-pixel at this resolution) and xyz+gripper; also all-label linear and MLP probes.
- Leakage = attackers retrained on frozen z (linear ridge/softmax and a 2-hidden-layer MLP), trained on the full train split, scored on test:
  residual distractor velocity (the part of sprite motion NOT explained by the action), sprite position relative to the end effector,
  sprite colour class; normalised so 0 = chance, 1 = perfect. `leak_worst` = max over targets and attackers. Total sprite velocity is
  reported as a diagnostic only (for correlated kinds it contains the action response and is not nuisance).
- Controls: shuffled action labels (utility at chance), shuffled colour labels (attacker at chance), OOD utility: the ID-trained probe applied to
  test sets with every other distractor kind (a latent that reads the action off the distractor collapses there).
- Arms: `clean` (distractor-free frames, oracle), `raw`, `rate4` / `rateb` (rate-bottleneck controls, no adversary), `sam` (zero-shot SAM ViT-B,
  oracle-prompted with 3 points from the exact silhouette, never on a decoy: the most generous zero-shot mask), `gtmask` (exact silhouette: the
  best any mask can do), `adv` / `adv30` (learned candidate: raw + attacker-confusion loss on distractor state, lambda 10 / 30).
- Seeds: LAM init/data order seed 0,1,2; the distractor realisation (sprite starts/noise) is one fixed draw per split (not replicated).
  Intervals: bootstrap over test episodes (300) for each seed; across-seed spread reported as +-1 sd (n=3).

## Kill rule (fixed before any result)
For an action-correlated kind, the learned arm survives only if, in every one of 3 seeds, NMSE(sam) - NMSE(adv) >= 0.05 AND
leak_worst(adv) <= leak_worst(sam) + 0.02 AND adv also beats the best rate-bottleneck control. If the SAM mask already reaches
the `clean` oracle utility with near-zero leakage on all correlated kinds, B' is dead without needing the learned arm.

## Log
- 08:16Z start. MetaWorld 3.0.0 + MuJoCo 3.2.7 (3.17 breaks MetaWorld eq_data), OSMesa works; EGL does not (no NVIDIA EGL libs, no /dev/dri access).
- 08:27Z data generated on CPU (1800/300/300 episodes, ~2 min with 40 procs).
- 08:36Z first GPU smoke OK (LAM trains at ~40 it/s, eval ~1 min/arm). SAM ViT-B oracle-prompted masks: IoU 0.89 vs exact arm silhouette, 6-8% of decoy pixels end up in the mask, run at ~0.8 episodes/s (GPU shared).
- 08:50Z REGIME CHECK (weak level: K=3 sprites r=7, `indep`, seed 0, 30 epochs): clean-frame oracle utility NMSE 0.687; raw LAM on distractor frames 0.750 (95% CI 0.741-0.759), leak_worst 0.08.
  The distractor costs the raw LAM only ~0.06 NMSE, which is about the size of the pre-registered margin (0.05), so no arm can show a clear win: no headroom at this strength. Weak-level SAM jobs killed; added a `strong` level (K=8, r=9, 2x speed).
- Utility ceiling for this label: a linear probe from the TRUE 3-D EE displacement over the same macro step reaches xyz NMSE 0.63/0.54/0.55 (mean 0.57); adding the previous and next displacement (controller lag) gives 0.29/0.24/0.22. So ~0.57-0.69 is the best a single-pair LAM can do on this label; the clean oracle 0.687 is close to it.
- 09:20Z strong level (K=8 sprites, r=9, 2x speed; seed 0, 30 epochs): clean 0.687, raw 0.853 (indep) / 0.810 (glued); exact silhouette (gtmask) 0.781 / 0.759. A clean-trained LAM applied to strong-distractor frames goes to 1.1-1.7 NMSE (worse than chance), so the harm is real at this level.
- 09:30Z SAM masks (ViT-B, oracle prompts, strong level, train split): IoU vs exact silhouette 0.82, arm recall 0.85, 6.5% of decoy pixels inside the mask.
- 10:05Z corr_det seed 0 (all arms, strong level): raw 0.806, rate4 0.840, rateb 0.975 (collapsed), SAM 0.814, gtmask 0.764, adv 0.791, adv30 0.795, lmask 1.000 and advmask 1.001 (collapsed to the degenerate "gate out everything that moves" solution: unsupervised soft mask has no signal to keep the arm; dropped from the remaining jobs, reported as a failed implementation, NOT as evidence about learned masks).
  Zero-shot SAM does NOT beat the raw LAM here (0.814 vs 0.806): its mask loses 15% of the arm and keeps 6.5% of the decoys; only the exact silhouette (an oracle) helps (0.764).
- Reference leakage (attackers on the TRUE action only): corr_det relative-position leak 0.05 (unavoidable), total sprite velocity 1.00 (deterministic function of the action). So in corr_det the nuisance state is fully determined by the action history and leakage there is not cleanly separable; corr_noisy / corr_glued have a truly independent residual.

## Results (strong level; 3 LAM seeds 0/1/2; +- = sd over seeds; [..] = bootstrap over the 300 test episodes, seed-mean)
Utility = test NMSE of a linear probe z -> macro action xyz from 2000 labels (1.0 = chance, lower is better). Clean-frame oracle 0.705 +- 0.017.
A clean-trained LAM fed distractor frames is far worse than chance (1.07-1.37), so the distractors do hurt.

### corr_det (primary: sprite velocity is a deterministic function of the action) - 3 seeds, complete
| arm | utility NMSE | leak_worst (best single coordinate over linear+MLP attackers; reference leak of the TRUE action alone = 0.05) |
|---|---|---|
| clean frames (oracle) | 0.705 +- 0.017 | 0 |
| raw LAM | 0.815 +- 0.009 [0.803, 0.827] | 0.118 +- 0.030 |
| rate bottleneck, z dim 4 (control, no adversary) | 0.839 +- 0.001 | 0.082 +- 0.011 |
| zero-shot SAM ViT-B mask (oracle-prompted) | 0.812 +- 0.002 [0.802, 0.821] | 0.204 +- 0.008 |
| exact arm silhouette mask (oracle) | 0.764 +- 0.006 [0.755, 0.774] | 0.231 +- 0.010 |
| adv (learned: raw + attacker-confusion loss, lambda 10) | 0.802 +- 0.010 [0.788, 0.817] | 0.065 +- 0.020 |
Paired per seed, SAM minus adv: +0.023, +0.009, -0.002 (rule needs >= +0.05 in every seed): FAIL. adv leaks less than SAM (0.065 vs 0.204) and beats the rate control by 0.03-0.05, but the utility margin is 0.010 on average.
Leakage caveat: in corr_det the sprite state is a function of the action history, so a z that knows the arm pose "leaks" sprite relative position without carrying any distractor information (reference 0.05 from the true action alone, plus pose); the cleaner leakage numbers are in corr_noisy / corr_glued (pending, see below).

### indep (distractor independent of the action; the case SAM masks were expected to solve) - 3 seeds, SAM arm pending
raw 0.859 +- 0.012, rate4 0.859 +- 0.004, adv 0.857 +- 0.008, exact silhouette 0.779 +- 0.011, clean 0.705. Residual-velocity and colour leakage of raw z are 0.00: the LAM does not encode the sprites at all.

### Why (mechanism, measured)
- The harm is input clutter, not leakage: raw z carries ~0 information about sprite velocity or colour (attackers retrained, linear and MLP) and still loses 0.10-0.15 NMSE to the clean oracle. An adversary on z has nothing to remove (adv = raw within 0.002 on indep, 0.013 on corr_det).
- Only gating the input helps, and only an exact mask gets about half of the way (0.764-0.779 vs raw 0.815-0.859 vs clean 0.705). The zero-shot SAM mask is no better than no mask on corr_det (0.812 vs 0.815): mask IoU with the exact silhouette 0.82, arm recall 0.85, 6.5-7% of decoy pixels inside the mask (sprites sit on top of the arm, so the masked frame is contaminated and the arm is clipped).
- So B' does not die because SAM "solves" the action-correlated case (it does not), but because nothing learned beats it by the rule's margin, and the margin is larger than the room a mask can give (below).

### Post-hoc exploration (NOT part of the pre-registered rule; seed 0 only unless stated; not replicated)
- lmask / advmask (learned soft mask, no labels): collapsed to chance (1.00) on corr_det seed 0. The unsupervised mask learns to gate out every pixel that moves (recon error -> 4), because nothing tells it to keep the arm. This is a failed implementation, not evidence about learned masks in general.
- sup (same 2000 action labels also train an action head on z): 2k-label probe 0.842, all-label probe 0.779: overfits the small labelled set, worse than raw at 2k labels. supadv 0.999 and smask 0.993 collapsed (confusion loss + labels, mask without a keep-the-arm signal). No claim.
- adv30 (lambda 30): 0.795 (seed 0) vs adv 0.791 on the same seed: no sensitivity to lambda.

## Verdict (against the rule as written)
**B' = NO-GO.** Rule: the learned arm must beat zero-shot SAM by >= 0.05 NMSE in every seed at equal-or-lower leakage and also beat the rate control. On corr_det (the case the coordinator named, 3 seeds, complete) the best learned arm (`adv`) beats SAM by 0.010 on average (per seed +0.023, +0.009, -0.002): fail. The mask-type learned candidates (lmask, advmask, smask) collapsed and are not a valid test of a learned mask.

**Is the rule decidable?** Only in one direction. The oracle-mask headroom over SAM is 0.048 on corr_det (per seed 0.050, 0.055, 0.039), i.e. below the 0.05 margin, so no mask-type candidate, not even a perfect silhouette, can pass; the rule could only be passed by a latent-type method that gets past the exact-mask ceiling toward the clean oracle (0.107 below SAM). The tested latent method (adv) closes 0.010 of that 0.107 (9%). A rule defined as a fraction of the SAM-to-clean gap would also fail it.

**Status of the other kinds (rule asked for corr_noisy and corr_glued too):** raw / exact-mask / adv jobs for corr_noisy and corr_glued and the SAM-mask generation for them were queued late behind the shared queue; they are NOT in these numbers. Until they land, the verdict rests on corr_det (complete) and indep (raw/mask/adv complete, SAM pending). The one case where a learned method could still plausibly differ is corr_noisy / corr_glued, where an independent residual exists; the measured mechanism (no leakage into z, harm is input clutter) says it will not.

**Caveats (all one-sided toward "learned latents cannot fix clutter"):** synthetic sprites on a MetaWorld render, one fixed draw of the distractor realisation, 64x64 LAM, 3 tasks; the label itself has a ceiling (linear probe from the true EE displacement over the same macro step: NMSE ~0.57; clean LAM 0.705); real DAVIS-style backgrounds could change the clutter/leakage balance. A different learned-mask design with a keep-the-agent signal (prompts, labels, or optical-flow-consistency) is untested and is the only B' route I would not call closed.

## What to do next
Per FOCUS.md the next field is egocentric wearer privacy (then embedding privacy for retrieval, which the contrarian scout actually scored higher, 26 vs 24, gaze, EEG). Nothing from B' needs to be carried over except the harness idea: retrained attackers with a true-action reference leak, and a pre-declared headroom check (oracle-mask gain over the baseline must exceed the margin) before any margin is fixed.

## Harness reuse (what another focus can take from `src/b_robotics/`)
- **Retrained attackers on frozen z** (`lam.py: linear_probe, mlp_probe, _softmax_probe, evaluate`): ridge/softmax and a 2-hidden-layer MLP attacker fit on the full train split of embeddings, lambda / early stopping on val, scored on test, with per-episode (numerator, denominator) parts so the bootstrap over independent units is exact (`nmse_parts`, `boot_ci`). Headline leakage = best single coordinate (`max_dim`), because pooling over many nuisance coordinates hides a leak that concerns only a few. Shuffled-label controls for both utility and leakage are built in.
- **True-action / reference leakage** (`leak_ref.py`): run the same attackers on the *task label itself* as z. Whatever they decode is leakage a perfect task-only representation carries for free (here 0.05 for sprite position, 1.00 for total sprite velocity in corr_det); report learned-z leakage as excess over this reference. Any focus whose nuisance is correlated with the task (centre vs diagnosis in A, sentence/channel vs emotion in C) should compute it.
- **Oracle-headroom check before fixing a margin** (`analyze.py`, "headroom vs SAM"): measure (best oracle gain over the baseline) vs the proposed win margin first. In B' the exact-mask gain over SAM was 0.048 < the 0.05 margin, so the rule was undecidable for mask-type methods; set margins as a fraction of oracle headroom.
- **Twin-render ground truth + OOD probe** (`envgen.py`, `distractors.py`, `GData`): clean frame, exact masks and every nuisance label are deterministic functions of (kind, actions, seed); the ID-trained utility probe is applied to every other nuisance kind (`util_by_test_kind`), which exposed that a clean-trained model collapses to 1.1-1.4 NMSE under distractors.
- **Job pattern**: one YAML per (kind, seed), arm lists as CLI args, `res_<kind>_<arm>_s<seed>.json` + `parts_*.npz`, `analyze.py DIR...` prints the table and the pre-registered rule; fold dependency steps (SAM masks -> sam arm) into one job instead of wait-loops that hold a GPU.
- Pitfalls worth knowing: MuJoCo >= 3.3 breaks MetaWorld 3.0.0 (pin 3.2.7); no NVIDIA EGL on this VM (use OSMesa, ~0.15 s/render); SAM job was CPU-thread bound until `torch.set_num_threads(4)` and GPU-side compositing; unsupervised soft masks collapse to "mask out everything that moves".

## Late results (as of 11:46Z; separate from the verdict above)
Landed: SAM arm on `corr_glued` (sprites hover on the gripper and overlap the arm), 3 seeds. SAM masks: IoU with the exact silhouette 0.83, arm recall 0.85, only 0.6% of decoy pixels in the mask (the oracle prompts avoid sprite pixels and SAM does not paint over them).
| corr_glued (strong) | utility NMSE | leak_worst |
|---|---|---|
| clean frames (oracle) | 0.705 +- 0.017 (3 seeds) | 0 |
| raw LAM | 0.810 (seed 0 only, from the regime run b_str_corr_glued) | 0.03 |
| zero-shot SAM mask | 0.769 +- 0.011 [0.759, 0.779] (seeds 0.777, 0.756, 0.774) | 0.016 +- 0.003 |
| exact silhouette | 0.759 (seed 0 only) | 0.02 |
| adv, rate4 | not run yet (b_m_corr_glued_* still queued) | |
Reading: on the overlapping case SAM is genuinely useful (0.033 better than raw) and within 0.01-0.02 of the exact mask; the exact-mask headroom over SAM is 0.018 (seed 0), far below the 0.05 margin, so for mask-type candidates the rule is again undecidable-fail. A latent method would have to reach < 0.72, i.e. nearly the clean oracle, and the adv arm never moved more than 0.015 from raw in any completed run. Leakage into z is ~0 for every arm (resid-vel 0.01), the same clutter-not-leakage picture. Verdict unchanged: NO-GO.
Still queued at hand-over (they will land in ~/work/runs on the VM and in HF `KuanKuanKuan/openlatent-fields/runs/`; fold in with `python research/fields/src/b_robotics/analyze.py ~/work/runs/b_*`): indep SAM arm (b_msam_indep_s0-2), corr_glued and corr_noisy raw / rate4 / exact-mask / adv (b_m_corr_*_s0-2), corr_noisy SAM masks + SAM arm (b_sam_corr_noisy). They cannot flip the rule on corr_det (complete, 3 seeds, FAIL); they only add coverage.

## Reopened part (Auditor-1 F-11): parked jobs restored, 18:10Z
The nine parked specs were validated (`yaml.safe_load`) and moved back to `~/queue/pending`: b_msam_indep_s0-2, b_m_corr_noisy_s0-2, b_m_corr_glued_s0-2. No arms were added (so, as before, the rate-bottleneck control exists for seed 0 of corr_noisy / corr_glued only; seeds 1-2 run raw / exact mask / adv). Landed so far:

**indep, SAM arm (3 seeds, complete)**: SAM 0.835 +- 0.004 [0.825, 0.845] vs raw 0.859 +- 0.012, adv 0.857 +- 0.008, exact silhouette 0.779 +- 0.011, clean 0.705. SAM is 0.024 better than raw and 0.022 better than adv; on this action-independent kind the zero-shot mask is the best non-oracle arm, as expected. Exact-mask headroom over SAM here is 0.056 (per seed 0.057, 0.064, 0.048), the only kind where it exceeds 0.05. The adversary has nothing to remove (residual-velocity leakage of raw z is 0.00).

**corr_noisy, SAM arm (3 seeds)**: SAM 0.830 (0.828, 0.828, 0.835; bootstrap [0.819, 0.837] for seed 0). SAM mask IoU 0.81, arm recall 0.84, 6.6% of decoy pixels inside the mask. Residual-velocity leakage through the SAM z is 0.001-0.002. raw / exact mask / adv for corr_noisy: still queued.
