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
