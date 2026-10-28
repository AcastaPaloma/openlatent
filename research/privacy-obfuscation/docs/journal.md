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

### 2026-10-27 18:50 UTC: brightness-only control, diff then downsample, BDQ alpha 7
- p5_sanity_mean (each frame replaced by its mean brightness, one number per frame): action 36.6, identity
  22.4 (ResNet-50 18.7, R3D 22.4; chance 4.0). Global lighting alone, i.e. which recording session, carries
  ~22% KTH identity. This is a floor: any obfuscation that keeps per-frame brightness can't go much below 22%
  on KTH, and down64's 38.1 is about half brightness. Our best BDQ (23.6) sits at this floor.
- p5_sanity_const (all-zero frames, in progress): action trains at chance, identity ResNet-50 4.01 (chance 4.0).
- p1_diff_down8 (frame difference, then x8 downsample): 95.5 / 88.3 (ResNet-50 73.6, R3D 88.3). Coarse motion keeps
  gait identity. The motion signal that carries action at 15x20 also carries identity.
- p2_bdq_a7_ens: 95.0 / 62.9 (ResNet-50 33.1). Slightly above the BDQ curve (curve at action 95 is about 68):
  a stronger privacy weight with the 2-adversary ensemble buys ~5 points without hurting action.
- H21: KTH identity has a ~22% scene/lighting floor (brightness alone). Report leakage above this floor, and
  test BDQ-style methods on the scenario split, where session cues can't transfer.
- H22: gait survives x8 downsampling of motion; p1_diff_down32 tests whether x32 (4x5 motion map) finally drops
  identity faster than action.
- H23: the alpha-7 ensemble gain is real if it holds over seeds and alpha 10: p2_bdq_a7_ens_s1, p2_bdq_a10_ens.

### 2026-10-27 18:55 UTC: converged-adversary VAE seed; DINO on the scenario-split BDQ
- p2_vae_a1_ens_b3_s1 (Colab; VAE latent edit, 2-adversary ensemble, 3 adversary steps per f step): 96.8 / 89.5
  (ResNet-50 60.9, R3D 89.5), while the in-training adversary ended at 5.3. Same failure as every VAE run: f fools
  the adversaries it trains against, but a fresh R3D trained to convergence on the output recovers ~90% identity.
  More adversary steps don't close this gap (H-VAE: the edit hides identity in a way only a short-trained
  adversary misses, e.g. a fixed pattern the fresh attacker learns to ignore).
- p4_dino_bdq_a3_scenario: DINOv2 probe / fine-tune 10.3 / 12.8 on the scenario-split BDQ (R3D 24.7). DINO stays
  below the video attacker on every BDQ output: the binding leak is temporal (gait), not appearance.
- H24: VAE latent edits are not learnable against a fixed-budget adversary; drop further VAE b-step/warm-up
  variants once the running ones finish, and spend the GPUs on BDQ alpha/seed sweeps and gait removal instead.
- Queue note: p6_vae_a1_ens_ctx8 first landed on a VM without UCF101 data; moved to 129.213.82.145.
- Colab backlog refilled: c22 p2_bdq_a7_ens_s2, c23 p2_bdq_a7_ens_warm, c24 p1_diff_down16, c25 p2_bdq_a5_ens_s3.

### 2026-10-27 19:05 UTC: temporal shuffle, more BDQ seeds, all-black control
- p5_sanity_const (all-zero frames): action 16.7 (chance 16.7), identity ResNet-50 4.0 / R3D 3.8 / DINOv2 4.0
  (chance 4.0). With no signal every model sits exactly at chance: the evaluation can't manufacture accuracy.
- p1_tshuffle (frames randomly permuted within each window): action 95.5, identity 99.0 (R3D 98.0). KTH actions
  are readable from per-frame pose, without temporal order; identity is untouched (appearance).
- p2_bdq_a5_ens_s2: 93.0 / 60.0 (ResNet-50 25.4). Seeds of the same config: 84.4/37.0, 86.5/48.0, 93.0/60.0. The
  seed moves the run along the curve, not off it; single-seed comparisons between BDQ variants are noise-level.
- p2_bdq_a5_ens_warm_s1: 78.9 / 30.3 (seed 0: 61.4 / 23.6), again on the curve.
- H25: on the best BDQ output the remaining identity is gait (silhouette dynamics), while action is readable from
  frame-level pose (tshuffle result). Shuffling frames after BDQ should then cut identity much more than action.
  Test without retraining: p6_bdq_a5_ens_tshuffle and p6_bdq_a5_ens_warm_tshuffle (frozen checkpoints, chain
  [bdq, tshuffle], fresh attackers). run.py now loads a single-stage checkpoint into a chain's learned stage and
  refuses silently-missing keys.

### 2026-10-27 19:20 UTC: frozen vs retrained attackers, all runs (results/frozen_attackers.md)
- Frozen attackers (trained on raw video only, never shown an obfuscated frame; the Ilic et al. protocol) vs
  retrained attackers, same architectures, same runs. KTH: median gap 62.4 identity points over 54 obfuscated runs;
  in 43 of 54 the frozen attacker recovers < 1/4 of what the retrained one does. Examples (frozen -> retrained):
  blur8 8.7 -> 95.0, down16 11.5 -> 92.6, frame difference 13.5 -> 97.5, VAE edit 4.0 -> 82.6, BDQ a5 ens 3.8 -> 37.0.
  Every learned method looks perfect (at chance) to a frozen attacker. VP-UCF101: median gap 9.4 cMAP points
  (attribute cMAP has a 58.7 blind floor, so the scale is compressed).
- Takeaway: the frozen protocol measures domain shift, not privacy. Only the retrained numbers are reported as results.

### 2026-10-27 19:20 UTC: VAE b3, kill of the VP-UCF101 VAE warm-up run, Colab backlog moved to Lambda
- p2_vae_a1_ens_b3 (KTH VAE edit, 3 adversary steps per f step): 97.0 / 81.8 (ResNet-50 58.7, R3D 81.8);
  frozen attackers 4.0. Fifth VAE configuration at 75-90% retrained identity: supports H24.
- Killed p3_vae_a1_ens_warm after 1 of 3 adversarial epochs (~1 h/epoch, 39.8 action at epoch 1): another VAE
  variant (H24), and it started before the materialize() fix, so its evaluation would have used the leaky
  release path. Freed the GPU for BDQ.
- The Colab watchdog is stopped; its backlog runs on Lambda instead: p2_bdq_a7_ens_warm, p2_bdq_a7_ens_s2
  (H23), p1_diff_down16 (H22), p2_bdq_a5_ens_s3 (seed spread).

### 2026-10-27 19:35 UTC: a fixed motion map matches learned BDQ; BDQ on the scenario split
- p1_diff_down32 (frame difference, then x32 downsample: a 4x5-pixel motion map, nothing learned): 93.6 / 63.9
  (ResNet-50 52.8, R3D 63.9). That is on the learned BDQ curve at high utility (BDQ a7 ens 95.0 / 62.9,
  a5 ens s2 93.0 / 60.0). Compare down32 alone 78.6 / 91.0 and diff alone 96.5 / 97.5: removing the static scene
  (diff) and the body detail (x32) together does what neither does alone. Dose so far: diff->down8 95.5/88.3,
  diff->down32 93.6/63.9; p1_diff_down16 is running.
- p2_bdq_a5_ens_scenario (test sessions are scenarios never seen in training): 78.7 / 54.0 (ResNet-50 31.7).
  Scenario-split references: raw 88.7 / 97.7, scene-only 56.0 / 64.0, blur8 77.7 / 87.7, BDQ a3 55.8 / 24.7.
- H26: what learned BDQ buys over a fixed coarse motion map is small; the information that matters (motion
  energy layout over time) is the same. Tests: p1_diff_down64 (dose), p1_diff_down32_scenario (split),
  p1_diff_down32_tshuffle (H25 on the fixed motion map: does order-free motion keep action and lose gait?).

### 2026-10-27 19:55 UTC: VAE warm-up, DINOv2 on VP-UCF101
- p2_vae_a1_ens_warm (KTH VAE edit, adversaries warmed up one epoch after each restart): 93.1 / 74.1
  (ResNet-50 51.2, R3D 74.1). Sixth VAE configuration at 74-90% retrained identity (H24 holds).
- p4_dino_ucf_raw: DINOv2 probe / fine-tune cMAP 78.9 / 78.9 (ResNet-50 76.6, R3D 75.4); p4_dino_ucf_blur8:
  70.7 / 71.4 (ResNet-50 69.4, R3D 70.3). DINOv2 is the strongest VP-UCF101 attribute attacker by 1-2 cMAP
  points; the floor is 58.7, so blur8 still leaks ~63% of the raw-minus-floor gap to DINO.
- The two VP-UCF101 VAE re-evaluations (ctx8) are materialising at ~55 videos/min (13.3k videos, ~4 h each),
  the 48-frame windows cost 40% more VAE decoding than before.
- H27: the fixed motion map (diff -> coarse downsample) that matched BDQ on KTH removes appearance attributes on
  VP-UCF101 too, where BDQ failed (87.8 / 75.2). Tests: p3_diff_down8, p3_diff_down16.
- Attacker noise floor: p1_diff_down32_s1 repeats the fixed-filter run with a different attacker seed (the
  obfuscation is identical), measuring how much the identity number moves from attacker training alone.

### 2026-10-27 20:35 UTC: temporal order is not the leak (H25 rejected); motion-map dose; BDQ a10
- Shuffling frames within each window (fresh attackers on the shuffled release):
  fixed BDQ 96.8 / 87.8 -> 95.0 / 89.1; diff->down32 93.6 / 63.9 -> 90.0 / 59.2;
  best learned BDQ (a5 ens warm) 61.4 / 23.6 -> 56.0 / 18.9. Identity barely moves, and where it does, action
  drops as much. H25 rejected: the remaining identity is not in frame order. It is in the per-frame silhouette
  shapes as a set, which is exactly the premise of set-based gait recognition (GaitSet treats a walk as an
  unordered set of silhouettes).
- Motion-map dose (frame difference then xN downsample, action / identity): x8 95.5 / 88.3, x16 94.8 / 84.6,
  x32 93.6 / 63.9 (x64 running). The drop is between x16 (8x10 px) and x32 (4x5 px): identity needs the body
  outline to span more than ~2 pixels.
- p2_bdq_a10_ens: 71.2 / 31.3 (ResNet-50 14.2), on the BDQ curve. p2_bdq_a1_scenario: 79.2 / 61.2.
- p2_vae_a1_vadv_b3_s1: 96.7 / 87.0. Seventh VAE configuration in the 74-90% band.
- H28: identity survives as body-shape statistics, so what helps is destroying per-frame shape while keeping
  motion energy. Tests: p1_diff_down20 and p1_diff_down40 (resolve the 16 -> 32 cliff), and learned BDQ followed
  by a fixed downsample, trained end to end (p2_bdq_down4_a5_ens, p2_bdq_down8_a5_ens): can BDQ learn a better
  coarse motion map than diff alone?

### 2026-10-27 20:50 UTC: two numbers per frame beat every learned BDQ at low utility
- p1_diff_down64 (frame difference, then x64 area downsample: each 120x160 frame becomes 1x2 pixels, i.e. the
  motion energy of the left and right half): action 69.9, identity 11.0 (ResNet-50 6.2, R3D 11.0; chance 16.7 /
  4.0). Per action: walking 91, running 87, jogging 71, handwaving 67, handclapping 56, boxing 47 (identity per
  action 4-17). Learned BDQ at similar utility: 68.4 / 28.9 (a3 s3), 61.4 / 23.6 (a5 ens warm). This fixed,
  two-number-per-frame signal is far off the BDQ curve, toward privacy.
  Why: down64 alone (48.7 / 38.1) keeps per-frame brightness, which carries the ~22% session floor (p5_sanity_mean);
  the frame difference removes static brightness first, so what remains is only how much motion and on which side.
- p4_dino_bdq_a5 (DINOv2 on p2_bdq_a5_lr3e-3): probe 17.9 / fine-tune 31.4 vs R3D 35.3. DINO again below the video model.
- H29: KTH action needs very little information (motion energy over time); learned obfuscators trained with a
  2-adversary game do not find this low-information solution. Tests: p1_diff_down120 (one number per frame:
  global motion energy), p1_diff_down48 (2x3 px), p1_diff_down64_s1 (attacker seed), p1_diff_down64_scenario.
- Caveat for every KTH claim: 6 coarse actions with distinct motion speeds; VP-UCF101 (p3_diff_down8/16 queued)
  tests whether this transfers to 101 actions.

### 2026-10-27 21:05 UTC: attacker-seed noise is ~1 point; BDQ ensemble + shuffle
- p1_diff_down32_s1 (same fixed filter, attackers trained with seed 1): 93.5 / 62.9 vs seed 0 93.6 / 63.9.
  Attacker training noise is ~1 point; the 37-60 spread across BDQ seeds is the obfuscator's own training noise.
- p2_bdq_a5_ens_b3_s1: 96.3 / 67.4 (seed 0: 90.3 / 58.0), on the BDQ curve.
- p6_bdq_a5_ens_tshuffle: 76.3 / 29.9 vs unshuffled 84.4 / 37.0; same pattern as the other shuffle tests (H25 rejected).
- H30: coarse motion maps hide identity only because each 16-frame clip (~1.2 s) holds about one gait cycle.
  A longer view should let a video attacker read gait frequency and phase. Tests: p1_diff_down64_stride4,
  p1_diff_down32_stride4 (16 frames at stride 4, ~2.5 s; action and identity models both get the longer view).

### 2026-10-27 21:20 UTC: motion-map dose filling in; BDQ a7 seed 2
- p1_diff_down40 (3x4 px motion map): 87.8 / 57.2 (ResNet-50 47.3). Dose so far (action / identity):
  x16 (7x10) 94.8 / 84.6, x20 (6x8) running, x32 (3x5) 93.6 / 63.9, x40 (3x4) 87.8 / 57.2, x48 (2x3) queued,
  x64 (1x2) 69.9 / 11.0. Identity falls steeply below ~3x4 px while action degrades gradually.
- p2_bdq_a7_ens_s2: 67.4 / 29.4 (seed 0: 95.0 / 62.9). The seed spread swallows the a7 gain: H23 not supported;
  alpha 7 ensemble is on the same curve as everything else.
- p2_bdq_down4_a5_ens collapsed at epoch 3-4 (action 17, adversaries 3.5); the collapse guard keeps epoch 2.
- VP-UCF101 dose for H27 extended: p3_diff_down32 (3x4 px), p3_diff_down64 (1x2 px) join p3_diff_down8/16.

### 2026-10-27 21:35 UTC: dose x20; BDQ a7 warm collapses
- p1_diff_down20 (6x8 px motion map): 95.3 / 82.4 (ResNet-50 64.0). Dose (action / identity): x8 95.5/88.3,
  x16 94.8/84.6, x20 95.3/82.4, x32 93.6/63.9, x40 87.8/57.2, x64 69.9/11.0.
- p2_bdq_a7_ens_warm: action collapsed at epoch 6 (17%); the guard kept an earlier epoch: 64.7 / 27.1, on the curve.
- Attacker-budget check for the best low-utility point: p1_diff_down64_e24 trains every model 24 epochs instead
  of 12. If identity rises well above 11, the result is attacker under-training rather than obfuscation.

### 2026-10-27 21:55 UTC: one number per frame; the x64 result replicates; scenario split
- p1_diff_down120 (one number per frame: global motion energy): action 54.2, identity 6.9 (ResNet-50 3.7;
  chance 16.7 / 4.0). Per action: running 76, handwaving 65, walking 59, jogging 51, handclapping 41, boxing 33.
- p1_diff_down64_s1 (attacker seed 1): 70.7 / 10.9 vs seed 0 69.9 / 11.0. The 1x2 px point replicates.
- p1_diff_down32_scenario: 58.0 / 42.7 (Ilic split 93.6 / 63.9; raw on the scenario split 88.7 / 97.7). On unseen
  recording conditions the coarse motion map loses much more action than identity drops.
- p2_bdq_a5_ens_s3: 92.3 / 55.2 (seeds 0-3: 84.4/37.0, 86.5/48.0, 93.0/60.0, 92.3/55.2). On the curve.
- p2_bdq_down4_a5_ens (BDQ then x4 downsample, trained end to end; collapsed after epoch 2, kept epoch 2):
  90.6 / 52.3, on the curve.
- Information ladder on KTH (fixed motion maps, numbers per frame -> action / identity):
  1 -> 54.2 / 6.9; 2 -> 69.9 / 11.0; 12 (3x4) -> 87.8 / 57.2; 15 (3x5) -> 93.6 / 63.9; 48 (6x8) -> 95.3 / 82.4.
- H31: a learned stage before coarse pooling should be able to choose *which* motion to keep. Tests:
  p2_bdq_down32_a3_ens, p2_bdq_down64_a1_ens (BDQ then fixed downsample, end to end).
  p1_diff_down64_tshuffle: with two numbers per frame, does action still need temporal order?

### 2026-10-27 22:20 UTC: the 1x2 px motion map does not generalise; first VP-UCF101 motion map
- p1_diff_down64_scenario (test scenario unseen in training): action 23.3 (chance 16.7), identity 8.8.
  On the Ilic split the same filter gave 69.9 / 11.0. Two numbers of motion energy per frame carry action only
  within familiar recording conditions (camera distance and zoom differ between KTH scenarios, so the motion
  energy of a walk in d2 looks like a different action in d1). The low-utility "win" is split-specific.
  Scenario-split reference points (action / identity): raw 88.7/97.7, blur8 77.7/87.7, scene-only 56.0/64.0,
  BDQ a1 79.2/61.2, BDQ a5 ens 78.7/54.0, BDQ a3 55.8/24.7, diff->down32 58.0/42.7, diff->down64 23.3/8.8.
  Learned BDQ holds up better across conditions than the fixed coarse maps.
- p3_diff_down16 (VP-UCF101): 63.7 / 71.0 cMAP (ResNet-50 69.9). Plain down16 53.7 / 70.3, blur8 58.9 / 70.3,
  raw 91.8 / 76.6, floor 58.7. Per-attribute AP (ResNet-50): face 96.5, skin 77.9, gender 94.5, nudity 38.5,
  relationship 47.6 (raw 97.5 / 85.2 / 96.3 / 49.2 / 54.8). Face and gender AP stay near raw because they are
  highly prevalent (AP floor = prevalence); skin colour is the attribute that actually moves.
- H32: on the scenario split, BDQ dominates fixed maps. Tests: p2_bdq_a5_ens_warm_scenario (best Ilic config),
  p1_diff_down16_scenario (does a finer motion map keep action under new conditions?).

### 2026-10-27 23:40 UTC: a fixed motion map traces the learned BDQ curve (H26 supported)
- Fixed motion maps vs learned BDQ, Ilic split (action / identity):
  diff->down32 93.6 / 63.9  vs  BDQ a7 ens 95.0 / 62.9
  diff->down48 83.6 / 37.0  vs  BDQ a5 ens 84.4 / 37.0   (identical)
  diff->down64 69.9 / 11.0  vs  BDQ a3 s3 68.4 / 28.9    (fixed is better)
  Scenario split: diff->down16 81.0 / 50.7 vs BDQ a5 ens 78.7 / 54.0 and BDQ a1 79.2 / 61.2.
  Learned BDQ, with every adversary variant tried, lands on the curve a parameter-free motion map traces. On KTH
  the learning does not buy a better trade-off; it buys a point on the same curve.
  H32 (BDQ dominates on the scenario split) not supported at the one matched point so far.
- H30 rejected: 16 frames at stride 4 (~2.5 s) don't help the attacker: diff->down32 93.6 / 57.0 (stride 2: 63.9),
  diff->down64 71.6 / 11.4 (stride 2: 11.0).
- p1_diff_down64_tshuffle: 64.2 / 11.9 (ordered 69.9 / 11.0): with only two numbers per frame, order carries
  ~6 points of action and no identity.
- p2_bdq_down8_a5_ens (BDQ then x8, end to end): 82.8 / 35.6, on the curve.
- p2_vae_a1_vadv_b3_scenario: 83.0 / 65.0. VAE again.
- p3_diff_down8 (VP-UCF101): 73.4 / 72.5 (plain down8 72.2 / 72.1). On VP-UCF101 the motion map is no better than
  plain downsampling: the attributes (face, skin, gender) are appearance and leak through any coarse frame.
- H33: the KTH action/identity trade-off is set by the data (how much of the motion layout identity shares with
  action), not by the obfuscation family. Next: fill the fixed-map curve on the scenario split (down8, down48) and
  replicate the BDQ-matching point (p1_diff_down48_s1); p1_diff_down24 fills the gap between x20 and x32.

### 2026-10-27 23:50 UTC: learned BDQ before a 1x2 px pool beats the fixed curve (H31 supported, to replicate)
- p2_bdq_down64_a1_ens (BDQ then x64 area downsample, trained end to end against frame_r18 + r3d, alpha 1):
  82.1 / 12.5 (ResNet-50 7.2, R3D 12.5). The fixed diff->down64 at the same identity gives 69.9 / 11.0, so the
  learned blur/threshold stage before the 1x2 px pool buys ~12 action points at equal privacy: the first point off
  the fixed-map/BDQ curve. With two numbers per frame the learned stage can choose *which* motion feeds each number.
  Checks queued: seed (p2_bdq_down64_a1_ens_s1), scenario split (p2_bdq_down64_a1_ens_scenario), stronger privacy
  weight (p2_bdq_down64_a3_ens), 24-epoch attackers on the frozen output (p6_bdq_down64_a1_ens_e24; run.py now
  loads a whole-chain checkpoint as is).
- p6_vae_a1_ctx8 (VP-UCF101 VAE re-evaluated on the fixed release path): 66.1 / 69.9 vs 66.0 / 70.2 with the old
  path. The ~1% near-raw edge frames moved cMAP by 0.3: the earlier number was contaminated but not materially.
- p3_diff_down32 (VP-UCF101): 42.9 / 68.9. Motion maps on VP-UCF101 follow plain downsampling (H27 not supported).

### 2026-10-28 00:05 UTC: VP-UCF101 motion maps done; learned-before-pool dose
- p3_diff_down64 (VP-UCF101, 1x2 px): see dashboard; VP-UCF101 motion-map dose (action / cMAP): x8 73.4/72.5,
  x16 63.7/71.0, x32 42.9/68.9. Same curve as plain downsampling: appearance attributes survive any coarse map.
- H31 dose: p2_bdq_down48_a1_ens (2x3 px) and p2_bdq_down120_a1_ens (one number per frame), learned BDQ before the
  pool, to compare with the fixed ladder (2x3: 83.6 / 37.0; 1x1: 54.2 / 6.9).

### 2026-10-28 00:30 UTC: scenario split: learning helps where fixed maps fail; new data and VMs
- Fixed motion maps on the scenario split (action / identity): x8 86.2/61.7, x16 81.0/50.7, x32 58.0/42.7,
  x48 44.0/26.5, x64 23.3/8.8. Learned BDQ there: a1 79.2/61.2, a5 ens 78.7/54.0, a3 55.8/24.7. At mid utility
  BDQ a3 beats the fixed curve by ~15 identity points (x32 at 58 action leaks 42.7). On the Ilic split the
  two families coincide; under a shift in recording conditions the learned blur/thresholds generalise better than
  a fixed coarse map. H32 supported at mid utility.
- p1_diff_down48_s1: 82.4 / 36.3 (seed 0: 83.6 / 37.0): replicates. p1_diff_down24: 94.3 / 77.8.
- New dataset: VP-HMDB51 (HMDB51 split 1, 51 actions; STPrivacy's five attributes on 6,766 videos; prevalence
  face 0.90, skin 0.62, gender 0.92, nudity 0.27, relationship 0.21). Videos from the MichiganNLP/hmdb HF mirror
  (official host down). First wave mirrors VP-UCF101 (configs/p7): raw, blur8, down8/16, pix8, diff, diff->down16,
  person box (black / blur), BDQ a3 ens.
- Two more A100 VMs (150.136.92.19, 129.213.90.42) bootstrapping.

### 2026-10-28 00:40 UTC: more datasets
- VP-HMDB51 prepared (VM1 first): 3,570 train / 1,530 test labelled videos, exactly STPrivacy's split 1.
- Charades added as a second identity benchmark: people filming scripted activities in their own homes, with a
  per-video subject id. The official split has disjoint subjects, so ppv/charades.py re-splits closed-set like KTH
  (90 subjects with >= 20 videos, 9,072 videos, 30% of each subject's videos to test). Utility is the 157-way
  multilabel action task (mAP; train.ACTION_KIND = "multilabel"); privacy is subject top-1 (chance 1.1%).
  Frames at 6.25 fps, 112x150 (16-frame clip = 2.6 s). Videos from AI2's public S3 (16.3 GB).
  Like KTH, train and test share homes, so the scene is part of the identity signal, which is the real threat for
  in-home footage.
- Checked and not used: Weizmann (official host unreachable), NTU RGB+D (40 subjects; license agreement required,
  only unofficial re-uploads online), CASIA-B / OU-MVLP gait (request-only), PA-HMDB51 (frame-level labels on 515
  videos; VP-HMDB51 covers the same videos at video level).
- Hypotheses for the new data: H34: on Charades, as on KTH, the scene carries most identity (homes are distinct),
  so blur/downsampling will leak heavily under retrained attackers; H35: BDQ and fixed motion maps transfer to
  VP-HMDB51 the way they did to VP-UCF101 (appearance attributes leak through coarse maps).

### 2026-10-28 00:40 UTC: attacker budget does not change the 1x2 px result
- p1_diff_down64_e24 (every model trained 24 epochs instead of 12): 71.2 / 12.7 (12 epochs: 69.9 / 11.0;
  attacker seed 1: 70.7 / 10.9). The identity number is not an under-trained attacker.
- H36: the BDQ-before-pool gain (82.1 / 12.5 vs fixed 69.9 / 11.0) may come from utility training alone (a
  learned, action-optimal pool), not the adversarial game. Tests: p2_bdq_down64_a0, p2_bdq_down48_a0 (alpha 0).

### 2026-10-28 00:50 UTC: wave queued for a 2-hour unattended window
- Every VM has 2 running + 1-4 pending jobs (all configs built and run once on a VM before queueing):
  BDQ->down64 seeds 1-3, alpha 0/3/5, 24-epoch attackers, scenario split; BDQ->down48 seed 1 and alpha 0;
  BDQ->down32/120; scenario BDQ a3 seeds 1-3; diff->down24 scenario; UCF person-blur->down8/16;
  VP-HMDB51 first wave (10 runs); Charades prepares itself on 129.213.90.42 and then queues a smoke run + 7 runs.
- VP-HMDB51 jobs on VMs that were still preparing the data are held by a watcher that releases them only after
  the prep log says HMDB READY and meta.json exists.

### 2026-10-28 00:55 UTC: scenario-split BDQ warm-up; a7 seed 1; corrected VP-UCF101 VAE ensemble
- p2_bdq_a5_ens_warm_scenario: 32.7 / 12.5. On the scenario fixed-map curve, 32.7 action sits between
  x48 (44.0 / 26.5) and x64 (23.3 / 8.8), i.e. ~17 identity by interpolation: BDQ is ~5 points better here,
  smaller than the a3 advantage at 56 action (H32 holds, more weakly at low utility).
- p2_bdq_a7_ens_s1: 95.2 / 66.7 (a7 ens seeds: 95.0/62.9, 67.4/29.4, 95.2/66.7).
- p6_vae_a1_ens_ctx8 (VP-UCF101 VAE ensemble on the fixed release path): 77.4 / 71.8 vs 78.0 / 72.2 before the fix.
  Both corrected VAE numbers moved by <0.5 cMAP; the edge-frame leak did not change any conclusion.
- Queued p3_bdq_down16_a3_ens: learned BDQ before a 7x9 px pool on VP-UCF101 (does the KTH learned-before-pool
  gain exist when the private attributes are appearance?).

### 2026-10-28 01:10 UTC: collapse-guard fix for many-class datasets; Charades ready
- Bug in the protocol, caught before any result: the collapse guard stops adversarial training when action never
  reaches 40% by epoch 3. That floor was tuned on KTH (6 classes). p7_bdq_a3_ens (VP-HMDB51, 51 classes) was
  stopped at epoch 3 while still climbing (11 -> 20 -> 32%), and p3_bdq_down16_a3_ens (VP-UCF101) was at 33.5% at
  epoch 2. Both killed and requeued with min_action 10 (also set for p8_bdq_a3_ens). Earlier VP-UCF101 BDQ runs
  (p3_bdq_*) passed 40% and are unaffected; VP-HMDB51 had no finished BDQ result.
- Charades prepared on 129.213.90.42: 90 subjects, 6,352 train / 2,720 test videos, none missing. Its first wave
  is split across three VMs; 129.213.82.145 and 150.136.92.19 prepare their own copy and release their jobs only
  when the prep log says CHARADES READY and meta.json exists.

### 2026-10-28 01:45 UTC: learned-before-pool beats the curve at high utility too (H31)
- p2_bdq_down32_a3_ens (BDQ then x32, i.e. 3x5 px): 93.0 / 40.8 (ResNet-50 18.2). Fixed diff->down32: 93.6 / 63.9;
  best plain BDQ near 93-95 action: ~95 / 63. At equal utility the learned stage before a 3x5 pool removes ~23
  identity points. With x64 (82.1 / 12.5 vs fixed 69.9 / 11.0) this is the second pool size where learning helps.
- p2_bdq_down48_a1_ens (2x3 px): 82.9 / 41.3 vs fixed 83.6 / 37.0: no gain at this size and alpha.
- p2_bdq_down64_a1_ens_scenario: 50.8 / 12.7. Scenario-split reference: BDQ a3 55.8 / 24.7, fixed x48 44.0 / 26.5,
  x64 23.3 / 8.8. Best scenario-split trade-off so far: the learned-before-pool gain holds under unseen conditions.
- Next: seed and split checks for the x32 point (p2_bdq_down32_a3_ens_s1, _scenario) and alpha 5.

### 2026-10-28 02:15 UTC: learned-before-pool replicates; one learned number per frame
- p2_bdq_down64_a1_ens_s1: 82.6 / 13.4; p2_bdq_down64_a3_ens: 82.9 / 12.5 (seed 0, alpha 1: 82.1 / 12.5).
  The BDQ->1x2 px point replicates across seed and alpha: ~82 action at ~13 identity (chance 16.7 / 4.0).
- p2_bdq_down120_a1_ens (BDQ then one number per frame): 75.9 / 8.5 (ResNet-50 5.2). Fixed one-number map: 54.2 / 6.9.
  A learned blur/threshold before the global pool buys ~22 action points at near-chance identity.
- Learned-before-pool vs fixed motion map, KTH Ilic (action / identity):
  1 number: 75.9 / 8.5 vs 54.2 / 6.9;  2 numbers: ~82.5 / 13 vs 69.9 / 11.0;  6 numbers: 82.9 / 41.3 vs 83.6 / 37.0;
  15 numbers: 93.0 / 40.8 vs 93.6 / 63.9. Gains at 1, 2 and 15 numbers; none at 6 (one alpha, one seed so far).
- p2_bdq_a3_scenario_s3: 74.2 / 44.7 (seed 0: 55.8 / 24.7). Large seed spread on the scenario split as on Ilic.
- p7_diff (VP-HMDB51 frame difference): 59.9 / 71.4 cMAP (ResNet-50 72.8). Raw reference (p7_raw) pending.
- H37: what the learned stage adds is a choice of which motion to integrate (thresholds on the blurred
  difference) that a fixed difference cannot make; the alpha-0 controls (p2_bdq_down64_a0, p2_bdq_down48_a0,
  finishing) decide whether that choice needs the adversary at all.

### 2026-10-28 02:45 UTC: the adversary is not needed for the learned-before-pool gain (H36 supported)
- p2_bdq_down64_a0 (BDQ -> 1x2 px, trained for action only, no privacy term): 81.8 / 10.7. With the 2-adversary
  game: 82.1 / 12.5, 82.6 / 13.4 (seed 1), 82.9 / 12.5 (alpha 3). The privacy comes from the capacity bottleneck
  (two numbers per frame); learning only spends that capacity on action. p2_bdq_down48_a0: 83.3 / 36.3 vs
  adversarial 82.9 / 41.3 and fixed 83.6 / 37.0.
- p6_bdq_down64_a1_ens_e24 (24-epoch attackers on the frozen BDQ->1x2 output): 82.4 / 14.5. Budget-robust.
- Scenario-split BDQ a3 seeds 0-3: 55.8/24.7, 64.5/39.3, 73.7/55.7, 74.2/44.7, spread along a curve ~8 identity
  points below the fixed maps (x24 69.0 / 50.3, x32 58.0 / 42.7).
- H38: the right object is a utility-trained encoder in front of a tiny bottleneck; adversarial training adds
  nothing at 1-2 numbers per frame. Tests: p2_bdq_down32_a0 (does the 93.0 / 40.8 point also hold without the
  adversary?), p2_unet_down64_a0 and p2_unet_down64_a1_ens (a 3D U-Net encoder before the 1x2 bottleneck: more
  action, or does a stronger encoder pack gait into the two numbers?).
