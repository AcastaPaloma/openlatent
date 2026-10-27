# Research journal: learned video obfuscation

Goal: a clear, credible sign that an obfuscation removes privacy-relevant information from video
(identity first; attributes next) while keeping useful video tasks, measured against **attackers
retrained on the obfuscated data** (strong protocol), not only raw-trained ones.

## Starting point (2026-10-27)

- Replication of Ilic et al. (CVPR 2024) on KTH: utility reproduces; privacy does not
  (identity 78% default / 45% targeted vs 5.46% claimed), and their protocol never retrains the
  attacker. Their comparison table copies strong-protocol numbers from BDQ (mixed protocols).
- Literature (survey in this session): strong-protocol KTH reference points are BDQ's
  supplementary (raw 92.89/94.34, downsample x16 91.22/88.86, x64 56.21/58.35, BDQ 91.11/7.15),
  with no KTH code or split published. SPAct / VITA numbers on UCF101+VISPR are reused unchanged
  across later papers. SPLAVU (ICLR'26) adapts latents of frozen video encoders, which is the
  closest prior work to a frozen-autoencoder anonymizer.

## Hypotheses

- H1: Under the strong protocol, simple fixed anonymizers (blur, downsampling, pixelation) trade
  utility and privacy along one curve: none reaches low identity accuracy with high action accuracy.
- H2: Motion-only representations (frame differences, BDQ) remove most appearance identity, but a
  video attacker recovers part of it from gait/body shape; frame attackers alone overstate privacy.
- H3: A learned anonymizer with a frozen VAE decoder (latent edit) reaches a better trade-off than
  pixel-space U-Nets, because the decoder keeps outputs on the natural-image manifold while the
  edit removes identity directions.

## Log

### 2026-10-27 06:50 UTC: first in-training signal (provisional)
- BDQ (learned blur, temporal difference, quantisation) trained against a *frame* adversary
  (Colab, p2_bdq_a1): the in-training adversary sits at ~21% identity by epoch 11, with action at 98%.
- The same BDQ trained against a *video* adversary (R3D-18, p2_bdq_a1_vadv): the adversary is
  still at 65% by epoch 8, and its entropy is only 1.76/3.22 nats. Motion-only output keeps
  identity that a spatio-temporal model reads (gait, body proportions) even while the anonymizer
  is optimised against that same model.
- Supports H2 (provisional; the retrained attackers are the real test).
- H4: anonymizers trained only against frame adversaries will show a large gap between frame
  and video attackers under the strong protocol. The video attacker number is the one to report.

### 2026-10-27 07:12 UTC: p2_unet_a1 collapsed (killed)
- Residual 3D U-Net, alpha 1, frame adversary. Training action accuracy held at 90-96% through
  epoch 10, then fell to 58 / 17 / 16.5 / 15.7 (chance 16.7) at epochs 11-14, while the adversary
  reached chance (3.6%, max entropy). The privacy term won the min-max game and f erased all content.
  The final epoch was all the code kept, so the run would have reported near-chance utility. Killed.
- Fix (train.py): keep the latest epoch whose action accuracy is within 5 points of the best so
  far; stop when it falls 30 points below the best and restore the kept epoch. Applies to every
  learned run started after this commit; p2_unet_a3 (VM4) started before it and is watched.
- H5: entropy-maximisation games need explicit utility protection (best-epoch keeping, or an
  alpha schedule that backs off when action accuracy drops); the reported trade-off of learned
  anonymizers depends heavily on the stopping rule, which papers rarely report.

### 2026-10-27 07:20 UTC: first fixed baselines (KTH, Ilic split)
| anonymizer | action (strong) | identity strong (frame / video) | action weak | identity weak |
|---|---|---|---|---|
| blur sigma 4 | 97.0 | 97.5 (97.5 / 96.3) | 16.7 | 16.6 |
| blur sigma 8 | 94.5 | 95.0 (95.0 / 94.8) | 16.7 | 8.7 |
| pixelate 4 | 97.7 | 99.0 (98.2 / 99.0) | 58.0 | 23.2 |
| pixelate 8 | 96.3 | 97.2 (96.2 / 97.2) | 16.7 | 14.0 |
- H1 confirmed in the strongest form: under the strong protocol, blur and pixelation remove
  essentially no identity (95-99%; chance 4%) and cost at most 3 action points.
- The weak protocol (Ilic et al.) reports 9-23% identity for the same outputs, overstating privacy
  by 75-90 points. Raw-trained models fail on the shifted inputs, which looks like privacy but isn't.
  This is the clearest explanation of the Ilic gap: a frozen attacker measures distribution shift.
- Concern: identity stays at 95% under sigma-8 blur. On KTH each person's videos share recording
  sessions, so the attacker may be classifying the scene/session rather than the person.
- H6: a scene-only input (per-video temporal median, people removed) still identifies the "person"
  far above chance on the Ilic split. Test: p1_bgmedian (queued) and p1_raw_scenario (running).
- H7: if H6 holds, KTH identity is partly a session-recognition task, and claims of identity
  removal on KTH must report the background-only control as the floor to beat.

### 2026-10-27 07:35 UTC: collapse guard too eager (p2_bdq_a3)
- BDQ alpha 3, lr_f 1e-2: the learned blur sigma jumped 5.2 -> 7.3 at epoch 3, coinciding with an
  adversary restart, and action fell 83.5 -> 47.7. The guard stopped training at epoch 3 and kept
  epoch 2, so the run would describe an anonymizer barely trained. Killed; requeued as
  p2_bdq_a3_lr3e-3 (gentler learnable-blur steps).
- Guard now needs 3 consecutive collapsed epochs before stopping (best-epoch keeping unchanged).
- H8: for BDQ, alpha mainly controls how fast sigma grows; a high alpha with a high lr drives
  sigma past the point where action survives within 1-2 epochs.

### 2026-10-27 07:35 UTC: p1_raw_scenario (KTH, test = scenario d2 for every person)
- Raw video: action 88.7, identity 97.7 (frame 95.2 / video 97.7); weak 92.8 / 97.7.
- Identity transfers to a recording condition never seen for that person in training, so the
  attacker isn't only memorising a person-specific session. It may still read location cues
  shared by d1/d2 (both outdoors). The scene-only control (p1_bgmedian, Ilic split; c12 on this
  split) separates person from place.
- Action is 9 points lower on this split than on Ilic's (88.7 vs ~97): scale variation (d2) is
  harder for action than for identity.

### 2026-10-27 07:50 UTC: rest of phase-1 fixed baselines, and two U-Net collapses
| anonymizer | action | identity strong (frame / video) | action weak | identity weak |
|---|---|---|---|---|
| raw | 97.5 | 99.5 (99.5 / 99.0) | 97.2 | 99.3 |
| blur sigma 2 | 97.7 | 99.5 (99.5 / 99.5) | 62.2 | 34.6 |
| downsample 4 | 97.0 | 98.2 (97.7 / 98.2) | 79.6 | 37.5 |
| pixelate 16 | 93.5 | 93.1 (90.8 / 93.1) | 16.7 | 9.9 |
| frame difference | 96.5 | 97.5 (93.0 / 97.5) | 82.9 | 13.5 |
| fixed BDQ (sigma 2, 8 levels) | 96.8 | 87.8 (68.2 / 87.8) | 18.9 | 4.0 |
- H2 confirmed for fixed motion representations: frame differences alone keep 97.5% identity,
  and fixed BDQ keeps 87.8%. Motion carries identity (gait, body shape).
- H4 confirmed in the fixed case: on fixed BDQ the video attacker is 20 points above the frame
  attacker. On every motion-only output the video attacker is the one that bounds privacy.
- Fixed BDQ under the weak protocol reads 4.0% identity (chance), while the strong protocol reads
  87.8%: the starkest weak/strong gap so far.
- p2_unet_a3 and p2_unet_a3_vadv started before the collapse guard and both collapsed around
  epoch 13 (p2_unet_a3: epoch 10 had action 94.7 / adversary 5.3, then 17.0 at epoch 13). Both
  killed and requeued on VM2/VM3 with the guard.
- H9: training against a frame adversary alone leaves gait identity; an ensemble adversary
  (frame + video) is needed. Implementing (adversary list, mean entropy).

### 2026-10-27 08:05 UTC: learned BDQ vs a frame adversary; alpha-3 U-Net vs video adversary
- p2_bdq_a1 (Colab; BDQ trained against a frame ResNet-18): action 93.5; identity strong
  frame_r50 31.3, but r3d 69.9, so privacy = 69.9. Weak protocol: 4.85 (frame).
  H4 confirmed for a learned anonymizer: optimising against a frame adversary hides appearance
  but leaves gait for a video attacker, and a weak frame-only evaluation would have reported ~5%.
- p2_unet_a3_vadv (alpha 3 vs R3D adversary, with the collapse guard) never learned action
  (17% from epoch 1): the guard compares to the best epoch, which was already chance. Killed.
  Added min_action (40 by epoch 3) so such runs stop early.
- Raw on the seq split: 99.2 / 100.0, so same-video splits make identity trivial; Ilic's split
  (99.5) is barely harder. KTH identity is saturated for raw video under every split.
- Next: ensemble-adversary runs (frame + video) for U-Net (alpha 1, 0.5), BDQ and the VAE latent edit.

### 2026-10-27 08:13 UTC: U-Net alpha 1 collapses at both seeds
- p2_unet_a1_s1 (seed 1, with the guard) collapsed at epochs 5-7 (action 93.4 -> 17.2); the guard
  stopped and kept epoch 4. Seed 0 collapsed at epoch 11. The residual 3D U-Net in the entropy
  game is unstable at alpha >= 1 on KTH whatever the seed; only the collapse epoch varies.
- H10: the instability is f learning to output a near-constant video once the adversary is
  briefly fooled; T then has no signal either. Lower alpha (0.5, queued), a lower lr_f, or a
  reconstruction/utility floor term should stabilise it.

### 2026-10-27 08:35 UTC: the scene alone identifies KTH "people" (H6 confirmed)
| run | action | identity strong (frame / video) | weak |
|---|---|---|---|
| p1_bgmedian (per-video temporal median: moving people removed, scene kept) | 59.4 | 89.3 (83.9 / 89.3) | 42.0 / 48.5 |
| p2_bdq_a1_vadv (learned BDQ vs R3D adversary) | 93.8 | 62.7 (29.8 / 62.7) | 39.1 / 5.5 |
| p1_down8 | 96.2 | 97.2 (97.2 / 94.6) | 19.1 / 10.9 |
- A frame showing only the background identifies the KTH subject 84-89% of the time: each
  subject's videos share recording sessions, so the scene acts as an ID badge. Caveat: the
  median keeps people who barely move (boxing, clapping, waving), so part of the 89% may still be
  the person. Requeued with a per-action identity breakdown (p1_bgmedian_byaction) to separate
  locomotion clips (person gone) from stationary ones.
- Consequences: (1) no obfuscation that touches only the person (Ilic-style selective masking)
  can go below ~84-89% identity on KTH under a strong attacker; (2) low KTH identity numbers in
  the literature (BDQ 7.15%) are consistent with erasing the static scene (temporal differences
  delete it), not only the person. KTH actor-ID is confounded by scene.
- Training BDQ against the video adversary helps a little (69.9 -> 62.7) but leaves a lot.
- H11: on KTH, identity = scene + gait. A credible privacy claim needs (a) a scene-only control as
  the floor and (b) a benchmark where scene is not tied to identity: VP-UCF101 attributes
  (phase 3), or KTH person crops (future).

### 2026-10-27 08:50 UTC: residual U-Net fails in every setting; VAE latent edit looks stable (in training)
- p2_unet_a1_vadv (U-Net vs R3D adversary): action 92.8, identity 92.3 (89.6 / 92.3): no privacy.
  Across five runs the residual U-Net either collapses (alpha 1 seeds 0 and 1, alpha 3, alpha 3
  vs video, alpha 1 ensemble at epoch 4) or keeps identity (alpha 1 epoch-4 checkpoint: 94.3
  frame; vs video: 92.3). A free-form pixel generator finds the degenerate "erase everything"
  solution before it finds an identity-only edit.
- In contrast, in-training numbers for the frozen-VAE latent edit: vs R3D adversary (p2_vae_a1_vadv)
  epoch 6 action 91.1 / adversary 8.8; vs frame adversary (p2_vae_a1) epoch 2 action 88.6 /
  adversary 12.1. No collapse so far. Provisional support for H3: the frozen decoder constrains
  outputs to natural images, so f can't take the degenerate path. Retrained attackers decide.
- BDQ (8 parameters) is stable but limited: best is 62.7 identity (vs video adversary).
- H12: the useful design axis is the output constraint (frozen generative decoder, few-parameter
  transform) rather than adversarial strength; unconstrained generators need a utility-preserving
  term to be usable.

### 2026-10-27 09:05 UTC: sub-quantisation utility (p2_unet_a3)
- p2_unet_a3 (alpha 3, guard restored epoch 2): co-trained T read 74.2% action from f's float
  output at epoch 2, but a fresh action model trained on the released uint8 output scored 16.7
  (chance), identity 4.0 (chance). f had encoded action in amplitudes below one 8-bit step;
  rounding to uint8 erased it.
- This is a real failure mode for adversarial anonymizers: training-time utility can be
  steganographic and vanish on release. Fix: the training loop now rounds f's output to 1/255
  with a straight-through gradient, so T sees exactly what will be released. Applies to runs
  started after this commit.
- H13: part of the U-Net's "collapse" is this effect (utility moves into sub-LSB detail while
  the adversary loses the visible content); with quantisation-aware training it may stabilise.

### 2026-10-27 09:35 UTC: BDQ moves along the curve; first UCF101 numbers
| run | action | privacy strong (frame / video) |
|---|---|---|
| p2_bdq_a3_lr3e-3 (KTH, frame adversary, alpha 3) | 89.1 | 46.2 (18.7 / 46.2) |
| p2_bdq_a1 (alpha 1) | 93.5 | 69.9 (31.3 / 69.9) |
| p2_unet_a05_ens (KTH, U-Net alpha 0.5, frame+video adversaries) | 94.1 | 78.9 (44.5 / 78.9) |
| p3_down8 (UCF101, cMAP over 5 attributes) | 72.2 | 72.1 (71.9 / 72.1) |
| p3_blur8 (UCF101) | 58.9 | 70.3 (69.4 / 70.3) |
- BDQ alpha 3 (gentler lr) trades 4 action points for 24 identity points vs alpha 1: the first
  learned run to move clearly along the utility-privacy curve. Video attacker still bounds it.
- U-Net alpha 0.5 with the ensemble didn't collapse but keeps 78.9% identity.
- UCF101: blur/downsampling cost 25-40 action points but keep attribute cMAP near 70 (raw pending).
- H14: BDQ's curve continues with alpha (5 queued, with an alpha-3 seed replicate); an alpha-3
  ensemble (Colab c15) should lower the video-attacker number further.

### 2026-10-27 09:50 UTC: VP-UCF101 calibrated against STPrivacy; leakage metric
- p3_raw: action 91.8, attribute cMAP 76.6, which matches STPrivacy's reported raw 76.62
  on VP-UCF101. Our phase-3 setup reproduces the benchmark's starting point.
- A constant predictor scores cMAP 58.7 (attribute prevalences: face 91.6, skin color 61.1,
  gender 89.8, nudity 25.1, relationship 25.7). The meaningful range is 76.6 -> 58.7, not -> 0.
  Results now report leakage = (privacy - floor) / (raw - floor).
- Fixed filters on UCF101: down4 84.5 / 75.1 (91% leakage), down8 72.2 / 72.1 (75%), blur8 58.9 /
  70.3 (65%). As on KTH, utility drops much faster than privacy.
- p2_unet_a1_ens (stopped early by the guard): 91.8 / 89.1 identity, no privacy.
- Best KTH learned result in leakage terms: BDQ alpha 3, 44% of raw identity leakage at -8 action.

### 2026-10-27 10:05 UTC: x64 downsampling calibrates against BDQ; frame difference leaks on UCF101
- p1_down64 (KTH): action 48.7, identity 38.1 (32.1 / 38.1), leakage 36%. BDQ's supplementary
  reports x64 at 56.21 / 58.35; ours is in the same regime (both lower), so our attackers aren't
  unusually weak. It's the only fixed filter below 40% identity, and it halves action.
- p1_bgmedian_scenario: 56.0 / 64.0 (64.0 / 62.0). On the scenario split, the scene alone still
  identifies the subject 64% of the time.
- p3_diff (UCF101): action 88.4, cMAP 76.0, leakage 97%. Frame differences on UCF101 keep nearly all
  attribute information (camera motion and edges carry appearance), unlike KTH's static camera.
- H15: BDQ's remaining 46% identity on KTH is gait, not scene (BDQ deletes static pixels). Test:
  p2_bdq_a3_scenario. If it holds on the scenario split, it's person-specific motion.

### 2026-10-27 10:20 UTC: KTH identity is redundant: scene alone and person alone both identify
| run | action | identity strong (frame / video) |
|---|---|---|
| p1_fgonly (person kept, scene blacked out) | 97.3 | 93.5 (92.3 / 93.5) |
| p1_bgmedian_byaction (scene only) | 62.5 | 90.6 (86.6 / 90.6) |
| p2_bdq_a1_ens (BDQ vs frame+video adversaries) | 94.1 | 65.2 (37.3 / 65.2) |
| p3_down16 (UCF101) | 53.7 | 70.3 cMAP |
- Per-action breakdown of the scene-only run: on locomotion clips (jogging / running / walking) the
  action model fails (24 / 32 / 48%), so the person really is gone, yet identity is 84-88 (frame)
  and 76-100 (video). The scene alone identifies the KTH subject. H6 confirmed without the caveat.
- Person-only output (scene erased) still gives 93.5%. KTH identity is carried redundantly by the
  scene and by the person; an anonymizer must remove both to show privacy on KTH.
- Ensemble adversaries at alpha 1 for BDQ: 65.2, between the frame-only (69.9) and video-only (62.7)
  versions. The ensemble is not stronger than the video adversary alone for BDQ.
- Direction: VP-UCF101 becomes the main benchmark. Its attributes are appearance (face, skin,
  gender) while action is largely motion, so separation is plausible. Adding UCF101 to VM1/VM3.

### 2026-10-27 10:55 UTC: VAE reconstruction alone hides nothing; KTH downsampling sweep done
- p2_vae_recon (frozen SD-VAE encode/decode, no edit): action 97.5, identity 98.7 (98.7 / 98.0),
  weak 99.0 / 99.3. Any privacy from the VAE latent-edit runs is due to the learned edit, not the
  VAE bottleneck (H3's control satisfied).
- Downsampling on KTH: x2 98.5, x4 98.2, x8 97.2, x16 92.6, x32 91.0 (action 78.6), x64 38.1
  (action 48.7). Identity only falls when action falls with it.
- Colab backlog now hardens the leads: BDQ alpha 3 seed 2; VAE ensemble seed 1 and on the
  scenario split (does the in-training chance-level adversary survive retrained attackers, and
  does it transfer to unseen recording conditions?).

### 2026-10-27 11:35 UTC: learned BDQ dominates fixed filters on KTH
| BDQ (frame adversary, lr 3e-3) | action | identity strong (frame / video) | leakage |
|---|---|---|---|
| alpha 1 | 93.5 | 69.9 (31.3 / 69.9) | 69% |
| alpha 3 | 89.1 | 46.2 (18.7 / 46.2) | 44% |
| alpha 5 | 76.9 | 35.3 (15.2 / 35.3) | 33% |
| (x64 downsampling) | 48.7 | 38.1 (32.1 / 38.1) | 36% |
- H14 confirmed: alpha moves BDQ along a utility-privacy curve that sits well above every fixed
  filter (at ~35-38% identity, BDQ keeps 28 more action points than x64 downsampling).
- The video attacker is 20 points above the frame attacker at every alpha. Training against a frame
  adversary leaves gait; this gap is what the ensemble runs are testing.

### 2026-10-27 11:45 UTC: BDQ alpha 3 is seed-sensitive
- p2_bdq_a3_lr3e-3_s1 (seed 1): action 94.5, identity 68.7 (31.6 / 68.7), vs seed 0: 89.1 / 46.2.
  A 22-point identity gap between seeds of one config. The final blur sigma doesn't explain it
  (seed 1 ended blurrier: 4.16 vs 2.69), so the learned quantisation thresholds and the adversarial
  dynamics decide the outcome.
- Consequence: no single-seed point on the curve can be claimed. Running seeds 2 (Colab) and 3,
  plus alpha 5 seed 1. Report mean and range per config.
- H16: the game has several end states for the same hyperparameters; selecting the anonymizer by a
  held-out privacy check (rather than the last epoch) would be a fairer comparison point, but must
  use data disjoint from the final test set.

### 2026-10-27 12:05 UTC: ensemble BDQ and UCF101 BDQ
- p2_bdq_a3_ens (KTH, frame+video adversaries): 96.7 / 68.6 (47.8 / 68.6), the same identity as
  frame-only alpha 3 seed 1 (68.7). The ensemble keeps a bit more action but doesn't lower video
  identity for BDQ.
- p3_bdq_a1 (UCF101): action 87.8, cMAP 75.2 (92% leakage). BDQ fails on UCF101: its learned sigma
  shrank to ~0.4 (action pulls toward detail) and camera motion keeps appearance in differences.
  BDQ is a static-camera method.
- Next: VAE latent edit replicate (p2_vae_a1_vadv seed 1) queued ahead of the seed-0 result.

### 2026-10-27 12:50 UTC: ensemble BDQ alpha 5 is the best BDQ point; person-box blackout leaks on KTH
| run | action | identity strong (frame / video) | leakage |
|---|---|---|---|
| p2_bdq_a5_ens (frame+video adversaries) | 84.4 | 37.0 (13.2 / 37.0) | 35% |
| p2_bdq_a5_lr3e-3 (frame adversary) | 76.9 | 35.3 (15.2 / 35.3) | 33% |
| p2_bdq_a10_lr3e-3 | 90.8 | 56.5 (33.8 / 56.5) | 55% |
| p1_personbox (people blacked out, KTH) | 94.1 | 96.3 (96.3 / 96.0) | 97% |
- At alpha 5 the ensemble buys 7.5 action points for the same identity. alpha 10 is worse than
  alpha 5 (non-monotonic), consistent with BDQ's seed variance; seeds of alpha-5 ensemble queued.
- Person-box blackout on KTH: action 94.1 (the black box's motion is the action), identity 96.3
  (box silhouette/gait plus the untouched scene). The standard "blackening" baseline protects
  nothing against a retrained attacker on KTH.
- H17: any obfuscation that preserves the person's motion silhouette (blackout, BDQ at low alpha,
  frame differences) preserves gait identity; only methods that distort the silhouette's dynamics
  can push identity toward chance on KTH.

### 2026-10-27 13:20 UTC: the VAE fooled its adversary, not a fresh attacker (H18)
- p2_vae_a1_vadv (frozen SD-VAE + learned latent edit, trained against an R3D adversary): in-training
  R3D accuracy ~4% (chance) over the final epochs, yet a fresh R3D trained on the released output
  reaches 75.6%; the frame ResNet-50 attacker reaches 54.0; action 96.2.
- The anonymizer learned to fool a short-trained, frequently-restarted adversary rather than to
  remove identity. In-training adversary accuracy is not a privacy measure, and results that report
  it (or a frozen attacker) overstate privacy.
- H18: keeping the adversary near convergence (k adversary updates per anonymizer update, fewer
  restarts) forces f to remove identity rather than exploit B's optimisation state. Testing b_steps=3
  for the VAE ensemble and BDQ alpha 5 ensemble.
- UCF101 blur16: 40.0 / 68.7 cMAP: utility collapses before attributes do.

### 2026-10-27 13:50 UTC: the ensemble VAE fooled both adversaries too
- p2_vae_a1_ens (VAE latent edit vs frame + video adversaries): action 97.3, identity 82.6
  (58.2 / 82.6), though both in-training adversaries were at chance (4-5%). H18 holds for the
  ensemble: f fools adversaries that are re-initialised every 3 epochs and trained one step per f step.
- Two fixes queued: b_steps=3 (more B updates per f update) and b_warmup=-1 (after every restart,
  train the fresh B for a full epoch on frozen f before f resumes, so f always faces a converged
  attacker, matching the evaluation).

### 2026-10-27 13:55 UTC: BDQ alpha 3 across three seeds
- Seeds 0 / 1 / 2: identity 46.2 / 68.7 / 66.6 (action 89.1 / 94.5 / 91.6). Seed 0 was the outlier;
  alpha 3 sits around 60% identity on average. Report means with ranges from now on.

### 2026-10-27 14:20 UTC: BDQ seeds lie on one trade-off curve; evaluation noise is small
| run | action | identity strong (frame / video) |
|---|---|---|
| BDQ a3 s3 | 68.4 | 28.9 (11.4 / 28.9) |
| BDQ a5 s0 | 76.9 | 35.3 (15.2 / 35.3) |
| BDQ a3 s0 | 89.1 | 46.2 (18.7 / 46.2) |
| BDQ a5 s1 | 90.3 | 48.0 (26.3 / 48.0) |
| BDQ a3 s2 | 91.6 | 66.6 (35.8 / 66.6) |
| BDQ a3 s1 | 94.5 | 68.7 (31.6 / 68.7) |
- The seed decides where on one action-identity curve a BDQ run ends, more than alpha does
  (seeds of alpha 3 span 28.9-68.7). The curve is steep: ~1 identity point per action point given
  up. All points dominate fixed filters (x64 downsampling: 48.7 / 38.1; x32: 78.6 / 91.0).
- Evaluation noise: p3_raw seed 0 vs 1: cMAP 76.6 vs 77.0, action 91.8 vs 90.4. Attacker training
  randomness is ~1 point, so BDQ's 20-40 point seed spread is the anonymizer, not the evaluation.
- H19: BDQ's quantiser thresholds settle into different regimes depending on early adversary
  dynamics; selecting the operating point needs a validation split, not the seed.

### 2026-10-27 14:35 UTC: all three b_steps=1 VAE variants fooled their adversaries
- p2_vae_a1 (VAE edit vs frame adversary): 97.2 / 85.1 (61.9 / 85.1). With p2_vae_a1_vadv (75.6)
  and p2_vae_a1_ens (82.6), every VAE latent-edit run with one adversary step per f step kept
  75-85% identity despite chance-level in-training adversaries.
- Adversary warm-up confirmed working (896 B-only batches on frozen f before training resumes).
- New probes of the gait channel: p1_hold4 (temporal hold: every 4th frame repeated, removing fine
  motion timing) and p2_down4_bdq_a3 (x4 downsampling chained before learned BDQ, removing body-shape
  detail before motion is quantised). H20: gait identity needs finer spatial or temporal detail than
  action recognition, so one of these moves the curve.

### 2026-10-27 16:00 UTC: DINOv2 attackers; BDQ on the scenario split
| run | action | privacy strong | leakage |
|---|---|---|---|
| p4_dino_raw (DINOv2 probe / fine-tune, raw KTH) | - | 99.3 / 99.2 | 100% |
| p4_dino_down16 (DINOv2 probe / fine-tune) | - | 87.8 / 94.0 | 94% |
| p2_bdq_a3_scenario (BDQ alpha 3, scenario split) | 55.8 | 24.7 (16.8 / 24.7) | 22% |
| p2_bdq_a5_ens_s1 (seed 1) | 86.5 | 48.0 (18.2 / 48.0) | 46% |
| p3_personbox_blur (UCF101) | 75.9 | 73.6 cMAP | 84% |
- DINOv2 fine-tuned is at least as strong as our ResNet/R3D attackers (x16 downsampling: 94.0 vs
  92.6); even a frozen linear probe reads 88%. DINO attacks on BDQ / VAE outputs are running.
- BDQ on the scenario split (test = recording condition unseen for the subject): identity 24.7,
  leakage 22%, the lowest of any run. Once the static scene is gone, identity transfers worse
  across conditions. But action drops to 55.8 (raw on this split: 88.7). Mapping the curve:
  alpha 1 and alpha-5 ensemble on this split queued.
- Ensemble BDQ alpha 5 seed 1: 48.0 (seed 0: 37.0); the seed spread persists for the ensemble.

### 2026-10-27 16:20 UTC: a stronger adversary doesn't move BDQ off its curve; VAE on UCF101
- p2_bdq_a5_ens_b3 (3 adversary steps per f step): 90.3 / 58.0 (26.8 / 58.0), on the same
  action-identity curve as the other BDQ seeds. BDQ's trade-off is a property of the BDQ family,
  not of the adversary's training state.
- p3_vae_recon (UCF101): 86.9 / 76.5 cMAP (99% leakage): the VAE round trip removes no attribute
  information. p3_vae_a1 (frame adversary, b_steps=1): 66.0 / 70.2 (65%): no better than blur.
- Next: DINOv2 attackers on VP-UCF101 attributes (raw, blur8).

### 2026-10-27 16:35 UTC: DINOv2 vs blur
- p4_dino_blur8: DINOv2 identity 97.3 on blur-8 KTH (ResNet/R3D: 95.0). DINO is the strongest attacker on every
  output so far (raw 99.3, x16 94.0, blur8 97.3): self-supervised features shrug off blur and downsampling.

### 2026-10-27 16:50 UTC: temporal hold
- p1_hold4 (every 4th frame repeated): 97.7 / 99.0 (98.8 / 99.0). Identity needs no fine motion timing; H20 (temporal
  part) rejected. The spatial part is tested by p2_down4_bdq_a3.

### 2026-10-27 17:05 UTC: DINOv2 vs the best BDQ
- p4_dino_bdq_a5_ens: DINOv2 (probe / fine-tune) 18.6 / 35.1 on the BDQ ensemble alpha-5 output, vs R3D 37.0 and
  ResNet-50 13.2 from the original run. DINO does not break BDQ: with appearance quantised away, an appearance
  model gains nothing and the video (gait) attacker stays the binding one. The BDQ number holds up.

### 2026-10-27 17:35 UTC: adversary warm-up for BDQ; DINO on low-identity BDQ
- p2_bdq_a5_ens_warm (adversary trained one epoch on frozen f after every restart): 61.4 / 23.6 (11.4 / 23.6),
  on the extrapolated BDQ curve (68.4 / 28.9 for seed 3). A converged adversary moves BDQ along its curve, not past it.
- p4_dino_bdq_a3_s3: DINOv2 13.0 on the 68.4 / 28.9 BDQ output (R3D: 28.9). On every BDQ output so far the video
  attacker, not DINO, is the binding one.
- Next: p1_diff_down8 (frame difference, then x8 downsampling): does coarse motion keep action but drop gait identity?

### 2026-10-27 17:50 UTC: leakage audit (the "obfuscation incomplete vs evaluation bug" question)
- Evaluation sanity: with train identity labels shuffled (raw video), test identity is at chance for every
  attacker: ResNet-50 2.84, R3D 1.84, DINOv2 fine-tune 2.34 (chance 4.0). Every run also asserts that train and
  test share no video. No label or split leakage in the pipeline.
- Pixel nearest neighbour (copy the identity of the most similar obfuscated train frame; no learning):
  blur8 73.1% identity, neighbour from the same recording session in 58% of test clips; down64 6.2%;
  BDQ alpha-5 ensemble 11.2%, same session 3%. Contrast-stretched panels (results/audit/) show why:
  * blur keeps the room's lighting pattern, which matches the subject's other clips from the same session:
    scene leakage that is real, not a bug (outcome 1, via a KTH property);
  * BDQ removes the scene entirely and leaves a 3-level silhouette of the moving body; the learned video
    attacker's 37% comes from silhouette shape and motion over time (gait), also real leakage;
  * down64 is 1x2 pixels per frame; a brightness-only control (p5_sanity_mean) tests whether per-frame lighting
    alone explains its 38%.
- p2_down4_bdq_a3 (x4 downsampling before BDQ): 93.1 / 65.1 (31.9 / 65.1), on the BDQ curve. Removing spatial
  detail first doesn't remove gait (H20 spatial part rejected). p4_dino_down64: 36.6 (R3D/ResNet: 38.1).

### 2026-10-27 18:40 UTC: gallery of every method; a real release bug in the VP-UCF101 VAE runs
- results/figures/gallery_{kth_1..5,ucf101_vp_1..2}.png: every finished run's released output on the same 4 test
  clips (sorted by privacy score), with contrast-stretched copies and a 4-frame temporal strip. What it shows:
  * fixed filters keep the person and scene (blur/down/pix up to x16), consistent with 91-99% identity;
  * frame difference and fg-only keep a sharp, clothed silhouette: 94-98% identity is real, not a bug;
  * scene-only (median) has no person at all and still gives 89-91% identity (KTH session/scene cue);
  * best BDQ checkpoints (24-37% identity) release a flat frame with a few gray blobs on moving limbs;
    the jogging clip keeps a whole-body silhouette, which is the gait signal the R3D attacker uses;
  * U-Net outputs are binary masks or stripe patterns that still trace the body/scene layout (79-96% identity);
  * KTH VAE edits are a gray patterned frame; the body is faint but present after contrast stretch (75-85%).
- Bug found by the gallery (VP-UCF101 VAE only): some released frames are nearly the raw frame. Cause: the VAE's
  latent edit is a 6-layer 3D conv (+-6 frames receptive field) and materialize() gave each 32-frame window only
  2 frames of left context and none on the right; edge frames saw zero padding and were left almost unedited.
  scripts/boundary_check.py on the exact release path (corr > 0.9 with raw = near-raw):
  p3_vae_a1 27.5% of windows at position 0 (1.0% of frames); p3_vae_a1_ens 57% at position 31 (1.4%).
  KTH VAE, U-Net and BDQ checkpoints: 0 near-raw frames. So the VP-UCF101 VAE numbers (66.0/70.2, 78.0/72.2)
  are contaminated; all other numbers are unaffected.
- Fix: materialize() now carries 8 frames of real context on both sides of every window (edge frames repeated at
  video start/end). Re-evaluating both checkpoints with fresh attackers: p6_vae_a1_ctx8, p6_vae_a1_ens_ctx8.
  p3_vae_a1_ens_warm (running on old code) gets the same re-evaluation when it finishes.
- p5_sanity_const (all-black frames): action r2p1d trains at chance (train acc 16-17%, chance 16.7) as it should.
