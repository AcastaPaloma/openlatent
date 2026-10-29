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

### 2026-10-28 03:30 UTC: BDQ->1x2 px holds across six variants; VP-UCF101 attributes don't move
- BDQ -> 1x2 px, KTH Ilic (action / identity): a1 seed 0 82.1/12.5, seed 1 82.6/13.4, seed 2 76.9/9.9; a3 82.9/12.5;
  a5 80.9/10.4; a0 (no adversary) 81.8/10.7; 24-epoch attackers 82.4/14.5. Mean ~81 / 12 (chance 16.7 / 4.0).
- p3_bdq_down16_a3_ens (VP-UCF101, learned before a 7x9 px pool): 62.0 / 71.3 cMAP vs fixed diff->down16 63.7 / 71.0,
  down16 53.7 / 70.3. p3_personbox_blur_down16: 48.6 / 70.4. On VP-UCF101 nothing moves the attributes far from
  ~70 cMAP (floor 58.7, raw 76.6) except destroying action: the attributes are dominated by face/gender, which are
  present in ~90% of videos and readable from coarse colour.
- p7_blur8 (VP-HMDB51): 32.0 / 68.3 cMAP (raw pending).
- Next H38 controls: p2_bdq_down64_a0_s1, p2_bdq_down64_a0_scenario, p2_bdq_down120_a0.

### 2026-10-28 04:00 UTC: first Charades baseline; more learned-before-pool seeds
- p8_raw (Charades, raw video): action mAP 15.8 (157 classes, chance 0.6); identity 93.5% top-1 over 90 people
  (ResNet-50 89.3; chance 1.1). Footage from a person's own home identifies them almost perfectly (home + person).
- p2_bdq_down48_a1_ens_s1: 87.3 / 27.9 (seed 0: 82.9 / 41.3; fixed diff->down48 83.6 / 37.0).
- p2_bdq_down64_a1_ens_s3: 80.1 / 9.7. BDQ->1x2 px over seeds 0-3 and alphas 0-5: 76.9-82.9 action, 9.7-13.4 identity.
- p7_diff_down16 (VP-HMDB51): 34.4 / 69.5 cMAP.
- Queued p8_bdq_down16_a0 (Charades: utility-only learned stage before a 7x9 px pool, H38 on in-home identity) and
  p2_bdq_down120_a0 (lost from an earlier submission; requeued).

### 2026-10-28 04:10 UTC: x32 learned-before-pool replicates across alpha
- p2_bdq_down32_a5_ens: 91.8 / 38.3 (a3: 93.0 / 40.8; fixed diff->down32 93.6 / 63.9). Seed 1, scenario split,
  alpha 0 and seed 2 (queued) pending.
- p7_pix8 (VP-HMDB51): 44.5 / 69.5 cMAP. p7_raw moved to an idle VM so the VP-HMDB51 reference lands sooner.

### 2026-10-28 04:25 UTC: x32 learned-before-pool replicates (3/3); blur in homes
- p2_bdq_down32_a3_ens_s1: 92.1 / 37.5. BDQ->3x5 px: 93.0/40.8 (a3 s0), 92.1/37.5 (a3 s1), 91.8/38.3 (a5).
  Fixed diff->down32 93.6 / 63.9: ~25 identity points removed at equal utility, replicated.
- p8_blur8 (Charades): action mAP 9.0 (raw 15.8), identity 86.9% (raw 93.5; ResNet-50 71.9). H34 supported so far:
  in-home identity survives blur (room and body shape), and the video attacker is again the strongest.
- Next: p2_bdq_down16_a3_ens and p2_bdq_down24_a3_ens (can a finer learned pool reach ~95 action at low identity?),
  p8_bdq_down32_a0 (learned bottleneck on Charades).

### 2026-10-28 04:40 UTC: one learned number per frame replicates; first VP-HMDB51 fixed filters; Charades down16
- p2_bdq_down120_a1_ens_s1: 75.8 / 9.5 (seed 0: 75.9 / 8.5; fixed one-number map 54.2 / 6.9).
- VP-HMDB51 (action / cMAP): down8 45.6 / 70.3, down16 29.7 / 66.6, person-box blur 45.4 / 69.3, pix8 44.5 / 69.5,
  blur8 32.0 / 68.3, diff 59.9 / 71.4, diff->down16 34.4 / 69.5. Raw reference running now.
- p8_down16 (Charades, 7x9 px): action mAP 8.4, identity 85.0% (raw 93.5, blur8 86.9). In-home identity survives
  coarse downsampling (room colour layout); H34 holds.
- Next: p6_bdq_down64_a1_ens_stride4 (2.5 s attack on the 1x2 px learned output), p7_diff_down8, p7_bdq_down16_a0,
  p8_down64 and p8_diff_down32 (where does in-home identity break?).

### 2026-10-28 04:55 UTC: BDQ is the first method that hides who is home (Charades)
- p8_bdq_a3_ens (Charades): action mAP 6.0 (raw 15.8), identity 24.2% (raw 93.5, blur8 86.9, down16 85.0; ResNet-50
  11.4; chance 1.1). Temporal differencing removes the static room, which carries most in-home identity; what is
  left is the moving person, and the video attacker still reads 24%. Action keeps ~40% of its mAP.
- p7_bdq_a3_ens (VP-HMDB51, full 15 epochs with the fixed guard): 63.1 / 72.7 cMAP. Like VP-UCF101 (87.8 / 74.9):
  BDQ does not hide appearance attributes on web/movie video.
- p7_personbox (VP-HMDB51): 38.4 / 68.8.
- H39: on Charades the scene is the identity; any method that removes the static background (diff, BDQ) will
  cut identity sharply, and the trade-off is then set by how much action survives. Tests: p8_bdq_a1_ens (less
  privacy weight), p8_bdq_a0 (no adversary: is it the differencing alone?), p8_diff (running).
- Replication of the best scenario-split point: p2_bdq_down64_a1_ens_scenario_s1.
- 05:05 UTC: p3_personbox_blur_down8 (VP-UCF101): 59.3 / 71.4 cMAP (plain down8 72.2 / 72.1). Blurring the person
  first costs action and buys nothing: the attributes leak through the coarse frame either way.

### 2026-10-28 05:20 UTC: the adversary matters once the bottleneck has room; VP-HMDB51 reference
- p2_bdq_down32_a0 (BDQ -> 3x5 px, utility only): 89.5 / 53.2 vs adversarial a3 93.0/40.8, 92.1/37.5, a5 91.8/38.3.
  At 15 numbers per frame the privacy term removes ~15 identity points; at 2 numbers per frame (x64) it removed
  none (a0 81.8 / 10.7 vs ~81 / 12). H38 refined: a small enough bottleneck makes the adversary unnecessary;
  with more capacity, adversarial training decides what the extra capacity carries.
- p7_raw (VP-HMDB51): action 64.9, cMAP 75.2 (ResNet-50 74.5); floor (mean prevalence) 58.8. Leakage of the first
  wave: BDQ a3 ens 85%, diff 77%, pix8 65%, person box 61%, person-box blur 64%, blur8 58%, down8 70%, down16 48%,
  diff->down16 66%. Same picture as VP-UCF101: appearance attributes survive every obfuscation that keeps action.
- Queued p2_bdq_down32_a1_ens (alpha dose at x32).

### 2026-10-28 05:40 UTC: x32 learned pool on the scenario split; fixed maps lose on Charades
- p2_bdq_down32_a3_ens_scenario: 66.2 / 31.2. Scenario-split references near this utility: fixed diff->down24
  69.0 / 50.3, plain BDQ a3 s1 64.5 / 39.3. The x32 learned-pool gain holds under unseen recording conditions.
- p8_diff_down32 (Charades): action mAP 6.8, identity 58.4% vs BDQ a3 ens 6.0 / 24.2. Unlike KTH-Ilic, on Charades
  the learned BDQ beats a fixed coarse motion map by ~34 identity points at similar action.
- Queued p2_bdq_down32_a3_ens_scenario_s1 and p8_bdq_down16_a3_ens (learned pool + adversary on Charades).

### 2026-10-28 05:55 UTC: encoder bias matters; in homes the adversary matters
- p2_unet_down64_a0 (3D U-Net encoder -> 1x2 px, utility only): 66.1 / 9.9 vs BDQ -> 1x2 px 81.8 / 10.7. With the
  adversary (p2_unet_down64_a1_ens) it collapsed to chance (16.7 / 4.0). A more expressive encoder does not help;
  BDQ's built-in temporal difference is the useful inductive bias.
- p6_bdq_down64_a1_ens_stride4 (2.5 s view on the frozen 1x2 px output): 81.8 / 12.2. Robust to longer views.
- Charades (action mAP / identity): raw 15.8 / 93.5, diff 12.9 / 89.4, diff->down16 7.9 / 75.1, diff->down32 6.8 / 58.4,
  BDQ->down32 a0 6.7 / 53.3, BDQ a3 ens 6.0 / 24.2. Frame differencing alone keeps in-home identity (handheld
  camera motion turns the static room into edges); only the adversarially trained BDQ removes it.
- Next: p8_bdq_a3_ens_s1, p8_bdq_a5_ens; KTH x32 adversary ablation p2_bdq_down32_a3_r3d / _r18 (which attacker's
  gradient carries the x32 gain: gait or appearance?).

### 2026-10-28 06:20 UTC: no adversary needed at one number per frame either; seeds for the headline points
- p2_bdq_down120_a0 (BDQ -> one number per frame, utility only): 78.1 / 7.2 (adversarial: 75.9 / 8.5, 75.8 / 9.5).
- p2_bdq_down64_a0_scenario: 52.5 / 11.2 (adversarial 50.8 / 12.7). The adversary is unnecessary at 1-2 numbers per
  frame on both splits.
- p7_bdq_down16_a3_ens (VP-HMDB51): 32.9 / 68.9 cMAP. No learned-before-pool gain on appearance attributes.
- Seeds for the headline points so they can be reported as mean +- sd: p2_bdq_down32_a3_ens_s3/_s4,
  p2_bdq_down120_a1_ens_s2, p2_bdq_down120_a0_s1, p8_bdq_a3_ens_s2.

### 2026-10-28 06:40 UTC: in Charades the home is the identity
- p8_personbox (Charades, person boxes blacked out): action mAP 10.6, identity 90.6% (raw 93.5). Removing the
  person removes almost no identity.
- p8_down64 (Charades, 1x2 px RGB per frame): 5.5 / 38.8%. Two pixels of a home's colour and light still identify
  the occupant 39% of the time (35x chance). Charades identity is the room; KTH identity was ~22% lighting floor
  plus body/gait.
- p2_bdq_down64_a0_s1: 80.9 / 12.0 (seed 0: 81.8 / 10.7). The utility-only bottleneck replicates.
- Next: p8_bdq_a10_ens (how far can adversarial BDQ push in-home identity?), p2_bdq_down48_a3_ens (resolve the
  mixed x48 results: 82.9/41.3 vs 87.3/27.9 at alpha 1).
- 06:50 UTC: p8_bdq_down16_a0 (Charades, BDQ -> 7x9 px, utility only): 8.0 / 72.5%. Unlike KTH, the bottleneck alone
  does not hide in-home identity (the room survives coarse pooling); the adversarial BDQ (24.2%) is needed.
  Queued p8_bdq_a3_r3d (video adversary only) and p8_bdq_a3_ens_e15 (50% longer adversarial training).
- 07:05 UTC: p2_bdq_down32_a3_ens_s2: 92.6 / 39.5. BDQ->3x5 px over 4 runs (seeds 0-2 a3, a5): 91.8-93.0 action,
  37.5-40.8 identity (fixed map: 93.6 / 63.9). p7_diff_down8 (VP-HMDB51): 42.6 / 70.3. Queued stronger attacks on the
  frozen x32 output: 24-epoch attackers (p6_bdq_down32_a3_ens_e24) and a 2.5 s view (p6_bdq_down32_a3_ens_stride4).

### 2026-10-28 07:40 UTC: highest-utility learned pool; a frame adversary suffices at x32
- p2_bdq_down16_a3_ens (BDQ -> 7x10 px): 95.8 / 57.0 (ResNet-50 25.6). At ~95 action: fixed diff->down16 94.8 / 84.6,
  plain BDQ a7 ens 95.0 / 62.9 and 95.2 / 66.7. The learned pool keeps raw-level action with 57% identity.
- p2_bdq_down32_a3_r18 (frame_r18 adversary only): 90.6 / 40.6, the same as the frame+video ensemble (92-93 / 38-41).
  At 3x5 px the appearance adversary alone carries the gain; the r3d-only ablation is running.
- p7_bdq_down16_a0 (VP-HMDB51): 36.0 / 69.1 (adversarial 32.9 / 68.9).
- Queued x16 seed 1, scenario split, alpha 5.

### 2026-10-28 07:55 UTC: the learned-pool curve; Charades alpha sweep
- KTH Ilic, BDQ before a pool (a3 unless noted; action / identity): x16 95.8/57.0, x24 93.0/48.8, x32 92-93/38-41
  (a1: 91.8/32.4), x48 83-87/28-41, x64 ~81/12, x120 ~77/8. A smooth curve well below the fixed motion maps
  (x16 94.8/84.6, x32 93.6/63.9, x64 69.9/11.0, x120 54.2/6.9).
- p2_bdq_down64_a1_ens_scenario_s1: 51.5 / 14.0 (seed 0: 50.8 / 12.7). Replicates on the scenario split.
- Charades BDQ (action mAP / identity): a0 12.6/88.8, a1 5.8/19.6, a3 6.0/24.2; raw 15.8/93.5. The adversary is what
  removes in-home identity, and it costs ~60% of action mAP. a5, a10, seeds and the r3d-only ablation are running.
- Next: alpha 1 at x16/x24 and a seed at x32; Charades alpha 0.5 and an a1 seed.
- 08:05 UTC: p2_bdq_down32_a3_ens_scenario_s1: 62.2 / 32.2 (seed 0: 66.2 / 31.2), replicates on the scenario split.
  p2_bdq_down32_a3_r3d (video adversary only): 92.8 / 42.1; frame-only 90.6 / 40.6; both 92-93 / 38-41. At x32 either
  adversary alone gives the gain. Queued DINOv2 and 24-epoch attackers on the frozen x16 output (most appearance left).
- 08:20 UTC: p8_bdq_a5_ens (Charades): 7.8 / 54.8% (a1 5.8 / 19.6, a3 6.0 / 24.2, a0 12.6 / 88.8). Not monotone in
  alpha: on Charades the game's outcome varies a lot from run to run (as plain BDQ did across seeds on KTH). Seeds
  for a1 and a3 are running/queued (p8_bdq_a1_ens_s1, _s2; p8_bdq_a3_ens_s1, _s2) before any Charades claim.
- 08:50 UTC: p2_bdq_down120_a1_ens_s2: 77.1 / 7.2 (BDQ -> one number per frame, 3 seeds: 75.8-77.1 / 7.2-9.5).
  p8_bdq_a3_ens_s1 (Charades): 5.8 / 21.1% (seed 0: 6.0 / 24.2). Charades BDQ a3 replicates; a5's 54.8% looks like a
  failed run of the game; p8_bdq_a5_ens_s1 queued to check. p2_bdq_down24_a3_ens_s1 queued.

### 2026-10-28 09:05 UTC: x32 robust to longer views; Charades alpha sweep complete
- p2_bdq_down32_a3_ens_s3: 92.0 / 35.8 (5 runs at x32 a3/a5: 91.8-93.0 / 35.8-40.8).
  p6_bdq_down32_a3_ens_stride4 (2.5 s view on the frozen x32 output): 92.3 / 35.5. Longer views don't help the attacker.
- Charades BDQ (mAP / identity): a0 12.6/88.8, a1 5.8/19.6, a3 6.0/24.2 and 5.8/21.1, a5 7.8/54.8, a10 6.3/32.9;
  BDQ -> 7x9 px a3 7.1/43.2. The useful region is a1-a3 (~6 / 20-24%).
- Next: DINOv2 and 24-epoch attackers on the frozen Charades a1 output (p6_dino_p8_bdq_a1_ens, p6_p8_bdq_a1_ens_e24);
  x32 a1 on the scenario split and seed 2.

### 2026-10-28 09:20 UTC: UTD-MHAD added (identity without a scene cue)
- UTD-MHAD (8 subjects x 27 actions x 4 trials, one lab, one backdrop; RGB from the authors' page): trials 1-3
  train / trial 4 test, 648 / 213 videos (3 corrupt videos in the release). Utility 27-way top-1, privacy 8-way top-1
  (chance 12.5%). Unlike KTH (lighting/scenario) and Charades (the home), identity here must come from the person.
  Prepared on 158.101.16.146, 129.151.4.50, 158.101.125.222 (129.213.89.64 cannot reach the host).
- First wave (configs/p9): raw, blur8, down16, diff->down32, BDQ a3 ens, BDQ->x32 a3 ens, BDQ->x64 a1 ens.
- H40: without a scene cue, fixed filters lose identity faster than on KTH, and the learned bottleneck keeps
  its advantage (identity here is body/gait, which BDQ+pool already removes on KTH).
- 09:25 UTC: p2_bdq_down32_a3_ens_s4: 89.8 / 55.2, an outlier. x32 a3 over seeds 0-4: action 89.8-93.0 (mean 91.9),
  identity 35.8, 37.5, 39.5, 40.8, 55.2 (mean 41.8, sd ~7). Still ~22 points below the fixed map (63.9) on average,
  but one in five runs of the game lands much worse: report mean +- sd, not the best seed.
- 09:35 UTC: p2_bdq_down120_a0_s1: 74.9 / 8.5 (seed 0: 78.1 / 7.2). p2_bdq_down16_a3_ens_scenario: 73.7 / 37.3; scenario
  learned pools x16 73.7/37.3, x32 66.2/31.2 and 62.2/32.2, x64 ~51/13 vs fixed maps x8 86.2/61.7, x16 81.0/50.7,
  x24 69.0/50.3, x32 58.0/42.7. The learned pool sits below the fixed curve on the scenario split at every size tried.
- 09:50 UTC: p2_bdq_down32_a1_ens_s1 91.3 / 41.6 (seed 0 91.8 / 32.4). p2_bdq_down48_a3_ens 87.3 / 25.3 (a1: 82.9/41.3,
  87.3/27.9): x48 sits between x32 and x64 once the bad a1 seed is set aside. p6_bdq_down32_a3_ens_e24 (24-epoch
  attackers on the frozen x32 output): 92.1 / 42.5 vs 40.8 with 12 epochs: budget-robust (+1.7).
- 10:05 UTC: p2_bdq_down16_a5_ens 96.5 / 54.0 (a3: 95.8 / 57.0). p6_dino_bdq_down16_a3_ens: DINOv2 probe 23.6, fine-tune
  4.2 on the frozen x16 output (ResNet-50 25.6, R3D 57.0): the strongest appearance attacker does not break the
  high-utility point; the binding attacker remains the video model. p8_bdq_a3_ens_s2 (Charades) 5.8 / 22.0%:
  a3 over 3 seeds 5.8-6.0 mAP / 21.1-24.2% identity. Queued p9_bdq_down16_a3_ens (UTD-MHAD), p2_bdq_down16_a5_ens_s1,
  p8_bdq_a3_r18 (Charades frame-only adversary).
- 10:20 UTC: p8_bdq_a3_r3d (Charades, video adversary only): 7.5 / 55.4% vs frame+video ensemble 5.8-6.0 / 21-24%.
  On Charades the frame (appearance) adversary is what removes the room; a video adversary alone leaves it
  (opposite of KTH, where either adversary sufficed at x32). Frame-only (p8_bdq_a3_r18) queued; p8_bdq_a1_ens_e15 queued.
- 10:35 UTC: p8_bdq_a05_ens (Charades): 7.1 / 45.6%. Charades alpha curve (mAP / identity): a0 12.6/88.8, a0.5 7.1/45.6,
  a1 5.8/19.6, a3 5.8-6.0/21-24 (3 seeds), a5 7.8/54.8 (seed 1 running), a10 6.3/32.9. Most identity goes between
  a0.5 and a1; action mAP falls from 12.6 to ~6 as soon as the adversary is on. p8_bdq_a2_ens queued.
- 10:50 UTC: x16 learned pool: a3 s1 95.0 / 61.2, a1 94.1 / 57.9 (4 runs: ~95.4 / 57.5 vs fixed x16 94.8 / 84.6).
  p8_bdq_a3_ens_e15 (Charades, 15 adversarial epochs): 5.7 / 20.6%, no change from 10 epochs.
  p9_raw (UTD-MHAD): 100.0 / 100.0. With trials held out in a single lab, every model sees near-identical repetitions:
  the dataset is a ceiling test; the obfuscated runs show how fast identity falls without a scene cue.
  gifs.py now renders VP-HMDB51, Charades and UTD-MHAD (Charades GIFs rendering).

### 2026-10-28 11:15 UTC: UTD-MHAD exposes a person-action leak; cross-action identity split
- p9_bdq_down32_a3_ens (UTD-MHAD, BDQ -> 3x5 px): 84.0 / 95.3% identity (raw 100 / 100). Without a scene cue and with
  trials held out, the attacker learns how each person performs each action; the learned pool keeps exactly that
  motion, so it cannot hide identity here. KTH's Ilic split has the same structure (same person, same action in
  train and test), so part of every KTH identity number may be this person-action signature.
- p6_dino_p8_bdq_a1_ens: DINOv2 probe / fine-tune 10.6 / 13.3% on the frozen Charades a1 output (R3D 19.6). Robust.
- p2_bdq_down24_a3_ens_s1: 93.5 / 54.8 (seed 0: 93.0 / 48.8).
- New protocol option priv_split "xaction" (run.py): identity attackers train on actions with id % 3 != 2 and are
  tested on the held-out third (all their videos). Smoke-tested. H41: on cross-action identity, raw video still
  identifies people (appearance, scene), but the learned pools fall much further, because a 1-15 number motion map
  carries person-action style and little that transfers across actions. Attack-only runs (configs/p10) on stored
  outputs: KTH raw, diff->down32, BDQ->x16/x32/x64; UTD raw, diff->down32, BDQ->x32.
- Charades GIFs added (results/gifs/charades).
- 11:20 UTC: p8_bdq_a1_ens_s1 (Charades): 4.4 / 3.6% identity (chance 1.1). Charades a1 seeds: 5.8/19.6, 4.4/3.6: the game
  sometimes removes in-home identity almost completely, at ~30% of raw action mAP. p8_bdq_a5_ens_s1: 8.4 / 66.0% (seed 0
  54.8%): at alpha 5 the Charades game reliably fails (keeps more action and identity), unlike KTH where alpha 5 was on
  the curve. p2_bdq_down24_a1_ens: 90.8 / 46.2. UTD-MHAD at 1x2 px: BDQ -> x64 a1 51.2 / 58.7, fixed diff -> x64 62.9 / 55.4
  (chance 12.5): the person-action signature survives two numbers per frame, and learning does not help here.
- 11:35 UTC: UTD-MHAD fixed filters keep everything within-action: diff->down32 97.7 / 99.5, down16 99.1 / 100 (raw 100/100).
  Cross-action versions queued (p10_utd_down16, p10_kth_down16) next to the raw and learned-pool ones.

### 2026-10-28 11:55 UTC: cross-action identity; UTD-MHAD correction
- Cross-action identity (attackers trained on 4 KTH actions, tested on the other 2; attack-only on stored outputs):
  BDQ -> x32 a3: 26.9 (within-action 40.8); fixed diff -> x32: 65.9 (within-action 63.9). The learned pool removes most
  of the identity that transfers across actions; what it leaves within-action is largely person-action style.
  The fixed motion map leaks the same identity across actions as within. H41 supported on KTH (raw, x16, x64 pending).
- Correction (UTD-MHAD): p9_down64 (1x2 RGB px per frame) 92.0 / 99.5% identity; p10_utd_diff_down32 (cross-action)
  94.8-96.2%. UTD-MHAD is NOT scene-free: each subject was recorded in a separate session, and global colour/lighting
  (plus where the subject stands) identifies them almost perfectly. The 09:20 entry's premise was wrong. Queued
  p9_mean (per-frame mean colour only) to measure this session cue directly.
- p9_bdq_down16_a3_ens 97.2 / 96.2. p2_bdq_down32_a1_ens_scenario 55.0 / 22.0 (a3: 66.2/31.2, 62.2/32.2).
  p6_bdq_down16_a3_ens_e24 (24-epoch attackers on x16): 96.0 / 57.5 (12 epochs: 57.0). Robust.
- 12:05 UTC: cross-action identity, KTH: raw 99.9, down16 99.0, fixed diff->x32 65.9, learned BDQ->x32 26.9. Appearance
  identity transfers across actions completely; only the learned pool removes most of it. UTD-MHAD cross-action: raw 100,
  down16 100. p9_mean (UTD-MHAD, per-frame mean colour only): 56.8 / 72.8% identity (chance 12.5; KTH's version: 22.4 vs
  chance 4.0): the session colour cue alone identifies most UTD subjects. p2_bdq_down16_a3_ens_scenario_s1: 76.2 / 37.0
  (seed 0: 73.7 / 37.3). Queued cross-action on plain BDQ (a5 ens warm), fixed x64, and learned x48 / x120.

### 2026-10-28 12:25 UTC: correction: three Charades runs collapsed; guard fixed for multilabel action
- p8_bdq_a1_ens_s1, p8_bdq_a1_ens_s2, p8_bdq_a3_r18 all report exactly 4.4 mAP / 3.6% identity (both attackers). These are
  floors, not results: 4.4 is the mAP of a blind predictor (each video has ~6 of 157 actions) and 3.6% is the most
  frequent subject's share. BDQ released content-free video. The 11:20 entry calling a1 s1 "near-chance identity" was
  wrong. Charades floors: action mAP 4.4, identity 3.6 (not 0.6 / 1.1 as used before).
- Cause: the collapse guard tracked per-label accuracy, which stays ~96% when nothing is predicted. Fixed (train.py):
  for multilabel actions the guard now tracks recall on positive labels, so the kept epoch is the last one that still
  predicted actions. Charades min_action set to 0 (best-epoch restore still applies). Reruns: *_g configs.
- Corrected Charades reading (mAP / identity, floors 4.4 / 3.6, raw 15.8 / 93.5): BDQ a3 seeds 5.8-6.0 / 21-24%,
  a1 seed 0 5.8 / 19.6%. Identity drops a lot, but action keeps only ~13% of its raw gap above the floor.
  Charades has no good trade-off yet.
- 12:35 UTC: cross-action identity, learned pools on KTH: x32 26.9, x48 17.0, x64 6.0, x120 4.9 (chance 4.0; within-action
  40.8 / 25.3 / 12.5 / 8.5). Fixed maps: x32 65.9 (x16, x48, x64 queued/running). p9_bdq_a3_ens (UTD-MHAD): 96.2 / 96.7.
- 12:50 UTC: cross-action identity: plain BDQ (a5 ens warm) 14.4 (within 23.6), fixed diff->x64 13.9 (within 11.0), learned
  BDQ->x64 6.0 (within 12.5). UTD-MHAD BDQ->x32 cross-action 84.0 (within 95.3; session cue). p2_bdq_down32_a1_ens_s2
  93.0 / 39.3 (a1 seeds: 32.4, 41.6, 39.3). p2_bdq_down48_a3_ens_s1 87.5 / 24.7 (seed 0 87.3 / 25.3). p9_blur8 97.7 / 100.
- Queued the hardest test: scenario split + cross-action identity (raw, fixed x16, learned x32, learned x64).
- 13:05 UTC: cross-action identity, fixed motion maps: x16 88.2 (within 84.6), x32 65.9 (63.9), x48 35.7 (37.0), x64 13.9 (11.0).
  Fixed maps leak the same identity across actions as within; learned pools drop to 26.9 / 17.0 / 6.0 / 4.9 at
  x32 / x48 / x64 / 1 px. Queued x24 for both families and a second-seed x32 cross-action test.
- 13:20 UTC: p10_kth_bdq_down16_a3_ens (learned x16, cross-action): 38.5 (within 57.5) vs fixed x16 88.2: at the
  highest-utility point the learned pool removes ~50 more points of identity that transfers across actions.
  Queued cross-action on plain BDQ a7 ens (95 / 63 point) and a guard-check rerun of Charades BDQ a3 seed 0.

### 2026-10-28 13:40 UTC: cross-action keeps separating the families; a protocol mistake corrected
- Cross-action identity (KTH Ilic): learned x24 33.1 vs fixed x24 84.3; learned x32 seed 1 20.1 (seed 0 26.9).
- p2_bdq_down24_a3_ens_scenario: 66.7 / 28.8.
- Correction: the p10_*_scenario runs (raw 99.7, fixed x16 90.6, learned x32 19.9, learned x64 9.1) rebuilt the attackers'
  split from all videos, so the attackers saw every scenario. They measure cross-action identity on obfuscators trained
  on the scenario split, not unseen conditions + unseen actions as the 12:50 entry said. Added priv_split
  "xaction_heldout" (train rows of seen actions -> test rows of held-out actions, i.e. unseen scenario AND unseen actions)
  and queued it for raw, fixed x16, learned x32 and learned x64 (p10_*_scenario_xh).
- 13:55 UTC: p10_kth_bdq_a7_ens (plain BDQ, cross-action): 43.2 (within 62.9); learned x16 at similar action: 38.5.
  p2_bdq_down16_a5_ens_s1 94.0 / 58.7. p2_bdq_down32_a1_ens_scenario_s1 71.0 / 30.7 (seed 0 55.0 / 22.0).
  p6_p8_bdq_a1_ens_e24 (24-epoch attackers on the frozen Charades a1 output): 5.9 / 23.4 (12 epochs: 19.6).
  p8_bdq_a1_ens_e15 collapsed to the floors (4.4 / 3.6); it started before the guard fix.
  Bug fixed: two jobs on one VM raced on the same frozen-attacker cache .tmp file (p10_kth_diff_down32_scenario_xh
  failed); temp files are now per-process; requeued.

### 2026-10-28 14:05 UTC: the hardest test (unseen scenario + unseen actions)
- Attackers train on the seen scenarios' videos of 4 actions and are tested on the unseen scenario's videos of the other 2
  (priv_split xaction_heldout; 1,193 train / 200 test rows). Identity top-1 (chance 4.0): raw 97.0, fixed diff->x16 48.5,
  fixed x32 40.5, learned BDQ->x16 17.5, learned x32 12.0, learned x64 7.5. Under both shifts at once the learned pools
  leak about a third of what the fixed motion maps do, and raw video still identifies 97%.
- Replications with second obfuscator seeds, a blur8 baseline and the x24 pair queued (p10_*_scenario*_xh).
- 14:20 UTC: hardest test (unseen scenario + unseen actions), identity: blur8 88.0, fixed x16/x24/x32/x64 48.5/43.5/40.5/8.0,
  learned x16 17.5 & 18.0 (seeds 0,1), x24 20.0, x32 12.0 & 12.0, x64 7.5 & 4.0, plain BDQ a3 5.5. Scenario-split action for the
  same obfuscators: blur8 77.7, fixed 81.0/69.0/58.0/23.3, learned ~75/66.7/~64/~51, plain BDQ a3 55.8. At ~64-75 action the
  learned pools leak 12-18% vs 40-48% for fixed maps; at the low end plain BDQ a3 (55.8/5.5) and learned x64 (~51/4-8) tie.
  Cross-action (Ilic): learned x16 seed 1 46.6 (seed 0 38.5); x32 a1 20.2. p8_bdq_a1_ens_s3 collapsed (started pre-fix).
- Queued: scenario-split learned pools x48 and x120 and third seeds of x16/x32/x64 (hardest-test attacks to follow).
- 14:35 UTC: hardest test: fixed diff->x48 18.0 (scenario action 44.0); plain BDQ a3 seed 1 10.5 (action 64.5).
  Charades BDQ a1.5 7.5 / 49.8%, a2 6.9 / 41.1%: across alpha 0.5-10 the Charades game gives 20-66% identity at 5.8-7.8 mAP,
  non-monotone and seed-dependent, with collapses mixed in. Guard-fixed reruns of the collapsed a1 runs queued.

### 2026-10-28 14:55 UTC: hardest test: plain BDQ ties the learned pools; Charades guard fix works
- Hardest test (unseen scenario + unseen actions), plain BDQ a3 seeds 0-3: 5.5 / 10.5 / 23.0 / 17.5 at scenario action 55.8 /
  64.5 / 73.7 / 74.2. Learned pools: ~17-18 at ~75 action, ~12 at ~64, 4-8 at ~51. In this setting plain BDQ and the learned
  pools are about even; both stay far below the fixed motion maps (x8 50.5 at 86.2, x16 48.5 at 81.0, x24 43.5 at 69.0,
  x32 40.5 at 58.0, x48 18.0 at 44.0, x64 8.0 at 23.3) and plain downsampling (x16 87.5). The learned pools' clear edge over
  plain BDQ is on the Ilic split (within- and cross-action), not under both shifts at once: correct finding 2's scope.
- p8_bdq_a1_ens_s1_g (Charades, rerun with the fixed guard): 5.7 / 19.9% (the original collapsed to 4.4 / 3.6). Guard works.
- Queued hardest tests for plain BDQ a5 ens and a5 ens warm, the VAE, 24-epoch attackers on learned x32, DINOv2 on learned x16.
- 15:10 UTC: hardest test: DINOv2 on learned x16 15.5 (R3D 17.5); plain BDQ a5 ens 24.0 at scenario action 78.7; plain BDQ
  a5 ens warm 6.0 at 32.7. Queued scenario-split plain BDQ a7 ens and learned x16 a5 (higher-utility comparison), the
  hardest test on plain BDQ a1 scenario (79.2 action), and a guard-fixed Charades a3 seed.

### 2026-10-28 15:20 UTC: new direction: keep only the task-relevant part of the frame (user's proposal)
- Motivation: on UTD-MHAD (session colour 73%), Charades (people removed 91%, 1x2 px 39%) and KTH (scene-only 89-91%) the
  background is the identity. Proposal: pick out what matters for the action (person AND task objects, e.g. a pool table),
  remove the rest, optionally obfuscate inside, then composite.
- Implementation (ppv/anonymizers.py TaskMask, type "taskmask"): frozen DINOv2 ViT-S/14 patch features per frame -> a small
  head scores every patch -> the top `keep` fraction is kept (hard top-k, sigmoid straight-through), trained by the action
  loss (alpha 0) or with the adversary (alpha 1). Optional `inner` obfuscator inside the mask (BDQ). Outside: `fill`
  gray (foreground only), blur, or raw (composite onto the real background; expected to re-leak the scene, run to measure
  exactly that). head "saliency" = unsupervised baseline (patches whose DINO features differ most from the frame mean).
  Frozen DINO/VAE weights are never checkpointed (frozen_key). Smoke-tested (forward/backward, kept fraction = keep,
  end-to-end UTD run).
- Wave p11 (keep 25%): saliency, learned a0, learned a1 on UTD-MHAD, KTH, Charades; BDQ inside the mask (UTD, KTH);
  raw-background composite (UTD); learned a1 on VP-UCF101 (attributes sit on the person, so expected not to help there).
- H42: removing the background (gray fill) cuts identity on UTD-MHAD and Charades far more than any obfuscator so far;
  the raw-background composite gives it all back; the learned head keeps task objects beyond the person.

### 2026-10-28 15:35 UTC: VP-UCF101 / VP-HMDB51 attributes are mostly implied by the action
- An "attacker" that sees no video, only the action class (per-class mean attribute vector from train), scores cMAP 77.6 on
  VP-UCF101 (raw-video attacker 76.6) and 74.3 on VP-HMDB51 (raw 75.2); blind floor 58.6 / 58.7. Per attribute (UCF):
  face 96.9, skin 83.1, gender 96.1, nudity 49.5, relationship 62.4. The benchmark's privacy labels are largely a function of
  the action (makeup -> face, horse race -> no face), so no action-preserving obfuscation can lower this cMAP; our 66-72
  results sit below the action-implied level only because they lose action. This also undercuts attribute-privacy claims
  made on these labels.
- Fair metric added (train.evaluate): action-conditional cMAP = attribute AP within each action class where the attribute
  varies, averaged; floor = mean within-class prevalence. It measures what the video reveals beyond its action.
  Unit-tested on a synthetic case. Attack-only re-evaluations queued (p12): raw, down16, diff->down16, blur8, BDQ,
  person box on VP-UCF101; raw, down16, diff->down16, blur8, BDQ on VP-HMDB51. The task-mask run (p11_ucf_mask_k25_a1)
  reports it too.
- H43: raw video reveals substantial within-action attribute information (face/skin/gender of the specific person), and
  person-removing methods (person box, task mask with adversary) reduce the conditional cMAP toward its floor while the
  standard cMAP barely moves.
- 15:45 UTC: new fixed obfuscator "pose" (ppv/anonymizers.py): torchvision Keypoint R-CNN on every 4th frame, people drawn as
  2 px stick figures on black; no face, skin, clothing, objects or room survive. Checked on 6 VP-UCF101 clips (full bodies
  -> clean skeletons; close-ups -> partial ones). Queued on VP-UCF101, VP-HMDB51, KTH and Charades (p12_*_pose).
  H44: pose removes most within-action attribute information (conditional cMAP near its floor) at a large action cost
  where objects matter (instruments, sports equipment); on Charades it removes the home entirely.

### 2026-10-28 15:40 UTC: why Charades BDQ fails: the in-game action model never learns
- p8_bdq_a3_r18_g (guard-fixed rerun) still collapsed (4.4 / 3.6). Its training log shows positive-label recall of the
  co-trained action model at 0.3% after epoch 1 and 0 afterwards: with ~6 positives among 157 labels, plain BCE lets it
  predict "no action" everywhere, so f receives no utility gradient and the adversarial term erases the video. The guard
  had no good epoch to keep. This also explains why even non-collapsed Charades BDQ runs kept only ~6 mAP (floor 4.4).
- Fix (train.py): the in-game action loss weights positives (pos_weight 20, configurable); attackers and evaluation keep
  plain BCE, so numbers stay comparable. Reruns: p8_bdq_a1_ens_pw, p8_bdq_a3_ens_pw, p8_bdq_a3_r18_pw,
  p11_ch_mask_k25_a1_pw. p10_kth_bdq_a1_scenario_xh (hardest test, plain BDQ a1, scenario action 79.2): 21.0.
- 15:50 UTC: hardest test with 24-epoch attackers on learned x32: 15.0 (12 epochs: 12.0). p2_bdq_down120_a1_ens_scenario:
  45.7 / 10.3. p8_bdq_a1_ens_s2_g (pre pos_weight): 5.8 / 22.2%.
- 16:05 UTC: hardest test on the VAE latent edit (scenario action 83.0): 61.0. Scenario-split learned pools, seed 2: x16 74.0 / 38.3
  (seeds 0,1: 73.7/37.3, 76.2/37.0), x64 53.0 / 14.8 (50.8/12.7, 51.5/14.0). p8_bdq_a3_ens_g (guard, pre pos_weight): 5.7 / 19.6%.
  Queued hardest tests for x16 s2, x64 s2, x120.
- 16:25 UTC: scenario learned pool x32 a3 seed 2: 66.3 / 29.5 (seeds 0,1: 66.2/31.2, 62.2/32.2); x32 replicates across three seeds.
  Killed p11_ch_mask_k25_a0 at adv epoch 8: the in-game action model's positive recall was 0.3% (no pos_weight), so the mask
  head got no action signal and the result would only measure an untrained mask. Requeued as p11_ch_mask_k25_a0_pw (pos_weight 20);
  the a1_pw twin already shows 61.6% recall at epoch 2. Early task-mask readings (in-game adversaries, not final): KTH a0 keeps
  identity (r3d 96.3), KTH a1 77.5, KTH BDQ-inside a1 53.0 at adv epoch 2; UTD raw-background composite 100 (expected: the session cue
  is in the background).
  Hypothesis: a task mask alone hides nothing on KTH (the person is the task region); it only helps when combined with an inner
  obfuscator, so the mask's value should show on datasets where the background carries identity (Charades homes, VP scenes).
- 16:40 UTC: hardest test on scenario learned pool x120 (scenario 45.7 / 10.3): 4.0 = chance (r3d 2.0). The hardest-test ladder for
  learned pools is now x16 17.5-18.0, x24 20.0, x32 12.0, x64 4.0-7.5, x120 4.0.
  p11_ch_saliency_k25 (Charades, DINOv2 saliency mask, keep 25% of patches, gray fill): 11.0 mAP / 91.3% identity (raw 15.8 / 93.5).
  Best Charades utility of any obfuscator so far, but identity is barely touched: unsupervised saliency picks the person, so the
  mask keeps exactly what identifies them.
  p11_utd_mask_k25_bdq_a1 (UTD-MHAD, learned mask + BDQ inside, adversary 1): 27.2 action / 92.5 identity. Worse than BDQ alone
  (96.2 / 96.7) on action; identity still near raw, so the mask doesn't remove the session cue the person carries.
  Killed p8_bdq_a3_ens_s1_g and p8_bdq_a1_ens_e15_g: both pre-pos_weight runs collapsed to 0.0 positive recall / 4-5% identity,
  the same collapse documented at 15:20; the pos_weight reruns are queued on the same VMs.
  Hypotheses: (1) a mask that picks the task region keeps the person, so on its own it can't hide identity; its use is as a
  carrier for an inner obfuscator, which keeps scene objects (tables, pool cues) at full resolution while the person is degraded.
  (2) Test this directly: invert the mask's role, i.e. obfuscate inside the person box and keep task objects raw.
  GIF check (results/gifs/charades/p11_ch_saliency_k25.gif): the saliency mask keeps the person plus the objects they handle
  (pots, dishes, the counter) and grays the rest of the room, which is what the 91.3% identity predicts.
- 16:55 UTC: hardest tests on scenario seed 2: x16 16.5 (seed 0: 17.5/18.0), x64 9.5 (4.0-7.5). Scenario x48 a3: 56.8 / 26.0.
  UTD task mask a1: 96.2 / 100; raw-background composite a0: 100 / 100 (session cue survives every mask variant).
  p12_hmdb_blur8 (attack only): attribute mAP 67.9, action-conditional cMAP 65.2 vs floor 57.5; waiting on p12_hmdb_raw for the
  raw reference. Charades p8_bdq_a1_ens_s3_g (guard, pre pos_weight): 5.9 / 23.1%, the same near-floor action as the other seeds.
  New phase p13 (inverting the mask): blur only inside detected person boxes (sigma 16) and keep the scene and handled objects raw,
  on VP-UCF101 and VP-HMDB51, with action eval. Hypothesis: action survives (objects and scene carry most UCF/HMDB classes) while
  face/skin/gender cMAP falls toward the floor, unlike every whole-frame filter so far.
- 17:10 UTC: hardest test on scenario learned pool x48 a3 (scenario 56.8 / 26.0): 16.0. Ladder x16 16.5-18, x24 20, x32 12, x48 16,
  x64 4-9.5, x120 4: the drop below ~15 needs x64.
  p12_hmdb_bdq_a3_ens (attack only): attribute mAP 73.3, action-conditional cMAP 70.4 (floor 57.5; blur8 65.2). Raw reference pending.
  Correction to the 16:55 plan: person-box blur is not new. p3/p7 already ran it (sigma 8): UCF 75.9 action / 73.6 attr mAP,
  HMDB 45.4 / 69.3; black boxes 70.4 / 73.1 and 38.4 / 68.8 (raw 91.8/76.6, 64.9/75.2). Removing the person entirely leaves
  attribute mAP within 3-6 points of raw, which is what "attributes are implied by the action/scene" predicts. Dropped the UCF p13
  duplicate (p12_ucf_personbox gives its conditional score); kept HMDB p13 (sigma 16, ~2 h) for the conditional number with action.
  Moved p12_ucf_raw / p12_hmdb_raw to VM5 so the conditional references land first.
- 17:25 UTC: KTH task mask (learned DINOv2 head, keep 25% of patches, gray fill): a0 92.5 / 96.2, a1 94.8 / 96.7 (raw 97.5 / 99.5).
  Even with an identity adversary in the loop the mask keeps identity: the patches the action needs are the person, so selecting
  patches can't trade identity for action on KTH. The mask's value is only as a carrier for an inner obfuscator (BDQ-inside run pending).
  Scenario split: plain BDQ a7 53.3 / 24.5 (a3 seeds ~55-62 / 25-31); learned pool x16 a5 74.2 / 36.0 vs a3 73.7 / 37.3: a stronger
  adversary weight doesn't move the frontier. Queued hardest tests for both.
  In-game reading worth watching: p8_bdq_a3_ens_pw (Charades, pos_weight 20) at adv epoch 5 has action positive recall 61.0 with the
  identity adversary at 4.8% (floor 3.6). If the strong eval holds, this is the first Charades obfuscator with both.
- 17:50 UTC: CORRECTION to the action-conditional floor. The reported floor (57.5) was the mean within-class prevalence. With
  ~30 test clips per (action, attribute) cell, random scores get well above prevalence in AP. Monte Carlo over the test labels
  (200 random score draws per cell, scripts in ppv/train.py and scripts/fix_cond_floor.py) gives chance = 61.3 on VP-UCF101 and
  61.9 on VP-HMDB51. ppv/train.py now computes this floor; collect.sh rewrites the floor of older results (old value kept as
  privacy_cond_floor_prevalence). The 16:55 and 17:10 entries quoting "floor 57.5" overstate the gap.
  VP-HMDB51 action-conditional cMAP (chance 61.9): raw 69.7, BDQ a3 70.4, diff->down16 66.0, blur8 65.2. So raw video reveals
  only ~8 points beyond its action on HMDB; blur8 removes about half of that but also half of action (p7: 32.0 vs 64.9), and
  BDQ removes none of it. VP-UCF101 BDQ a3: 65.2 (chance 61.3; raw pending).
  KTH: task mask + BDQ inside (a1): 80.1 / 82.9, worse than BDQ alone on both axes (96.7 / 68.6): masking first throws away
  action before BDQ sees it. UTD-MHAD learned mask a0 and saliency mask: 99.5 / 100.
  Hardest test: plain BDQ a7 scenario: 8.0 (a3 seeds 5.5-23, a1 21, a5 24); learned pool x16 a5: 19.5 (a3 16.5-18).
  Queued seed 1 of plain BDQ a7 scenario to test whether 8.0 replicates.
  New p13 on VP-UCF101 / VP-HMDB51, all with action eval: grayscale (removes skin tone and clothing colour), blur of the top third
  of each detected person box (head and shoulders, sigma 16), and both. Hypothesis: the ~8 points of attribute signal beyond the
  action sit in face and skin colour, which these remove at little action cost, unlike whole-frame blur.
- 18:05 UTC: VP-HMDB51 action-conditional cMAP (chance 61.9, raw 69.7): down16 64.7 (action p7 29.7), person boxes blurred sigma 16
  66.0 with action 41.7 (raw 64.9). VP-UCF101 blur8: 63.8 (chance 61.3; action p3 58.9; raw pending). Blurring only the people keeps
  more action than any whole-frame filter at the same conditional leakage (41.7 vs 34.4 for diff->down16 at 66.0), but still
  costs a third of HMDB action: HMDB actions are about the body (chew, smile, wave), so the person can't be blurred for free there.
  KTH saliency mask (keep 25%): 98.2 / 98.8, the same as the learned mask: selecting patches keeps the person.
  Queued: p12_hmdb_personbox (black boxes, conditional score) and seed 2 of plain BDQ a7 scenario.
- 18:30 UTC: VP-UCF101 action-conditional cMAP (chance 61.3): raw 67.1, BDQ a3 65.2, blur8 63.8, down16 63.5, diff->down16 63.5.
  Raw UCF reveals ~6 points beyond its action (HMDB ~8). Fixed filters remove ~60% of it at large action cost (p3: down16 53.7,
  blur8 58.9, diff->down16 63.7 vs raw 91.8); BDQ keeps action (87.8) and removes ~1/3.
  Charades learned task mask with pos_weight: a0 9.9 / 87.4%, a1 8.4 / 87.7% (raw 15.8 / 93.5; saliency 11.0 / 91.3). The in-game
  adversary sat at ~27% while fresh attackers reach 87%: hard top-k patch selection gives f too weak a gradient to fight them,
  and the people it keeps identify themselves.
  New p14: BDQ trained against action-conditional adversaries (ppv/train.py cond_adv). Each adversary gets a learned per-action
  logit offset; f pays alpha * relu(BCE(action-only prior) - BCE(adversary)), i.e. only for what the video adds beyond the action.
  The usual entropy objective asks f to hide attributes the action implies, i.e. to hide the action. Hypothesis: at matched
  action, p14 lowers the conditional cMAP more than p3/p7 BDQ. Smoke-tested on GPU; queued alpha 10 and 30 on UCF and HMDB.
  Also queued Charades BDQ a3 pos_weight seed 1 (seed 0's in-game reading: action recall 61, identity adversary 6.3%).
- 18:40 UTC: VP-HMDB51 p13 (with action): grayscale 65.6 action / 69.6 conditional cMAP; head-and-shoulders blur 59.2 / 69.3
  (raw 64.9 / 69.7, chance 61.9; whole-person blur 41.7 / 66.0). VP-UCF101 black person boxes: 64.8 (raw 67.1, chance 61.3).
  New code's Monte Carlo floor reproduces the offline value (61.89). Neither colour nor the face carries what the video
  reveals beyond its action. Per-attribute conditional AP minus per-attribute chance on raw video:
      HMDB  face +4.4  skin colour +8.1  gender +1.9  nudity +14.2  relationship +10.7
      UCF   face +3.2  skin colour +6.4  gender +2.0  nudity +7.2   relationship +10.1
  Gender is at chance once the action is known; the leftover signal is mostly nudity (how much skin is visible), relationship
  (more than one person in view) and skin tone (survives grayscale through luminance). Whole-person blur takes nudity to +2 and
  relationship to +7.7 on HMDB; black boxes on UCF take nudity to +0.4 but leave relationship at +8.7.
  Hypotheses: (1) relationship leaks through the count and layout of people, which survives any per-person obfuscation; only
  removing person count (one box for all people, or pose with identities merged) can hide it. (2) The p14 conditional adversary
  should concentrate on nudity and relationship, so its per-attribute numbers matter more than the mean.
- 18:55 UTC: CORRECTION to 17:25 / 18:05: Charades p8_bdq_a3_ens_pw (pos_weight 20) finished 5.7 mAP / 21.5% identity, the same
  as without pos_weight (a3 5.8 / 22.4 over 3 seeds). The in-game "action recall 61%" was not utility: with positives weighted 20x
  the in-game model predicts many positives, so recall is high while mAP for a fresh model stays at the floor. pos_weight fixed
  the collapse guard's signal, not the trade-off. Charades frontier stands: BDQ a0 12.6 / 88.8, a0.5 7.1 / 45.6, a1-a3 ~5.8 / 20;
  fixed diff->down32 6.8 / 58.4. Killed the seed-1 replicate (p8_bdq_a3_ens_pw_s1); a1_pw and a3_r18_pw continue.
  VP-UCF101 learned task mask a1: p11_ucf_mask_k25_a1 79.1 action / 74.8 attr mAP / 65.1 conditional (raw 91.8 / 77.1 / 67.1).
  VP-HMDB51 gray + head blur: 60.2 / 73.2 / 68.4 conditional (raw 64.9 / 75.6 / 69.7).
  In-game p14 readings: HMDB a10 epoch 5 action 74.6, gain over action-only 0.015; a30 epoch 4 action 65.2, gain 0.006.
  Queued p14_hmdb_bdq_cond_a3 to complete the alpha sweep (3, 10, 30).
- 19:05 UTC: VP-HMDB51 pose (stick figures on black, GIF checked: results/gifs/hmdb51_vp/p12_hmdb_pose.gif): 38.3 action /
  70.8 attr mAP / 67.6 conditional (raw 64.9 / 75.6 / 69.7, chance 61.9). Per attribute vs chance: face +5.4, skin colour +7.7,
  gender +3.1, nudity +1.9, relationship +10.3. Stick figures carry no colour, yet skin colour beyond the action is as predictable
  as from raw video (+8.1). So the beyond-action skin-colour signal is not appearance: it rides on structure the figures keep
  (framing, how much body is visible, how many people, camera motion). HMDB's own clip tags record exactly these (visible body
  part h/u/f, camera motion, number of people, viewpoint). Pose removes nudity (+1.9) but not relationship (+10.3): consistent
  with relationship = more than one person in view. Check on source overlap: 113 of ~1500 HMDB source-video names appear in both
  split-1 train and test (rough name parse), so same-source shortcuts exist but can't explain a +8 effect on their own.
  VP-UCF101 grayscale: 90.0 / 76.1 / 66.1 (raw 91.8 / 77.1 / 67.1): colour removal hides nothing.
  Hypothesis: on the VP benchmarks most of what survives beyond the action is scene structure, not identity-bearing appearance;
  the honest attribute-privacy metric needs a control that keeps structure but no appearance (pose is that control).
  Queued controls on both VP datasets: all-black frames (the conditional cMAP must land at chance: tests the metric) and
  per-frame mean colour.
- 19:20 UTC: plain BDQ a7 scenario seed 2: 60.0 / 24.8 (seed 0: 53.3 / 24.5). Queued its hardest test (seed 0 gave 8.0).
  In-game p14 readings at alpha 10 / 30 on UCF: action 93.9 / 91.2 with gain over the action-only predictor ~0.008-0.010: the
  conditional adversaries barely beat the action prior on training clips, so f is under little pressure. Strong evals pending.
- 19:35 UTC: p14 conditional adversary on VP-HMDB51 (BDQ, alpha 3 / 10 / 30): action 64.7 / 64.1 / 63.0, conditional cMAP
  69.6 / 69.8 / 68.5 (raw 69.7, plain BDQ 70.4, chance 61.9). Negative: knowing the action, the adversaries find nothing BDQ
  can remove. Fits the pose result: what survives beyond the action is structure (framing, body visibility, people count),
  which a blur/difference/quantise front end keeps by construction.
  KTH pose (stick figures, Ilic split): 97.8 action / 53.3 identity (frame_r50 34.8): body proportions and gait alone identify
  half the people, between the learned pools at x16 (57.5) and x32 (~40). Queued pose cross-action and hardest test on KTH:
  if pose identity is gait style it should fall on unseen actions like the learned pools did.
  Queued p12_hmdb_pose_1p: stick figure of only the most confident person (Pose max_people=1; checked on a 2-person clip: lit
  pixels 8748 -> 6172). Prediction: relationship drops to chance (+10.3 now) if people count is the cue.
- 19:50 UTC: metric check: VP-HMDB51 all-black frames give conditional cMAP 61.7 vs computed chance 61.9 (attribute mAP 59.5):
  the action-conditional metric and its Monte Carlo floor are right.
  Hardest test, plain BDQ a7 scenario seed 2: 8.5 (seed 0: 8.0). Replicates; plain BDQ a7 (scenario action 53-60) matches the
  learned x64 pool (51-53 action, 4-9.5 hardest).
  p10_kth_pose_scenario_xh failed: CUDA OOM, two Keypoint R-CNN jobs on one GPU with 64 frames per detector call. Pose now uses
  16 per call; requeued. Queued p12_hmdb_raw_s1 (attacker seed 1) to measure the conditional metric's seed noise.
- 20:05 UTC: conditional-metric noise: raw VP-HMDB51 attacker seed 1 gives 70.0 vs 69.7 (attr mAP 75.8 vs 75.6); differences
  of a point or more are signal. Mean colour per frame: 63.4 (chance 61.9): one number per frame carries ~1.5 of the ~8 points.
  Black person boxes on HMDB: 66.3, relationship still +9.8 over chance (raw +10.7): removing people doesn't remove how many.
  Charades p8_bdq_a1_ens_pw collapsed to the floors (4.4 / 3.6) while its in-game action recall read ~60%: pos_weight-inflated
  recall hides a content-free output from the guard. The guard should use a held-out check, not in-game recall.
  New dataset search: Weizmann actions (9 people, static scene) is the natural second KTH, but its official host times out from
  the Mac and the VMs and no verified mirror exists; not used.
  Queued: KTH pose on the scenario split (action + identity), VP-UCF101 single-person pose, raw UCF attacker seed 1.
- 20:20 UTC: KTH pose cross-action (identity on unseen actions): 27.4 (frame 23.8) vs 53.3 within actions; raw 99.9 / 99.4.
  Halves on unseen actions like the learned pools (x32: ~40 within, 23.5-26.9 across). So stick figures (97.8 action) sit at about
  the learned x32 point on identity with 6 more points of action: a strong, simple baseline the paper doesn't have.
  Plain BDQ a7 scenario seed 1: 67.5 / 45.0 (seeds 0, 2: 53.3 / 24.5, 60.0 / 24.8): a large seed spread; queued its hardest test.
  Charades BDQ a3 r18-utility pos_weight: 7.6 / 55.3, on the same frontier as a0.5 (7.1 / 45.6).
  VP-UCF101 gray + head blur: 86.4 action / 74.7 / 65.1 conditional, same as BDQ (87.8 / 74.9 / 65.2).
  VP-UCF101 black-frame control: strong frame_r50 conditional 61.8 vs chance 61.3 (metric holds on UCF too).
  Pose canonical=s (ppv/anonymizers.py): every person redrawn with template limb lengths at s x frame height from the mid-hip,
  keeping joint angles and position (visual check on KTH frames: figures same size, standard proportions). Queued KTH within and
  cross-action, and VP-HMDB51. Hypothesis: if KTH pose identity is body shape, canonical drops it toward chance; if it is
  posture/motion style it stays near 53 / 27.
- 20:35 UTC: p14 conditional adversary on VP-UCF101: alpha 10 89.7 action / 65.7 conditional, alpha 30 89.2 / 65.9 (plain BDQ
  87.8 / 65.2, raw 91.8 / 67.1, chance 61.3). Negative on both VP datasets: an action-aware adversary finds nothing more for BDQ
  to remove. Closing p14 for BDQ; the beyond-action signal is structural (see pose results).
  Hardest test (unseen scenario + unseen actions): KTH pose 21.5 (frame 17.5), next to the learned x16-x24 pools (17.5-20) and
  far below fixed maps at similar action (fixed x8 50.5). Plain BDQ a7 seed 1: 17.0 (seeds 0, 2: 8.0, 8.5): its weaker scenario
  result (45.0) carries over, so a7's 8% is seed-dependent, not a property of the setting.
  Queued KTH pose redrawn every 16 frames (within and cross-action): tests how much of pose identity is fine motion timing.
- 20:50 UTC: VP-HMDB51 single-person pose (max_people=1): 35.5 action / 68.0 attr mAP / 66.3 conditional (all people: 38.3 /
  70.8 / 67.6). Relationship vs chance: +8.3 (all people +10.3, raw +10.7). People count explains only ~2 of the ~10 points;
  hypothesis (1) of 18:40 mostly rejected. A single body's posture and placement carry "relationship" (and skin colour +6).
  VP-UCF101 head-and-shoulders blur: 85.4 / 76.0 / 66.4 (raw 91.8 / 77.1 / 67.1): face region holds nothing beyond the action.
  Queued the barest structural control, one canonical skeleton per frame on HMDB (no appearance, proportions or count), and the
  canonical-pose hardest test on KTH.
- 21:05 UTC: KTH pose ablations (Ilic; within actions / unseen actions):
      stick figures                 97.8 action / 53.3 / 27.4
      redrawn every 16 frames       93.3 / 38.1 / 29.1   (clips are 16 frames: one static posture per clip, no motion)
      canonical skeleton            pending / 21.4 across actions
  Motion timing carries within-action identity (53 -> 38) but not cross-action identity (27 -> 29); body proportions carry
  some cross-action identity (27.4 -> 21.4 canonical). A single static posture per clip still gives 93.3 action.
  KTH scenario split: stick figures 90.8 action / 38.0 identity / 21.5 hardest test, vs the learned pool x16 74.6 / 37.6 / 17.3
  (3 seeds). Same identity, +16 action; pose beats raw video on action here (88.7) because it discards the unseen scenario.
  This is the best KTH scenario trade-off found. Raw UCF attacker seed 1: 67.0 conditional (seed 0 67.1).
  Added pose rows and the new seeds / x48 / x120 / BDQ a7 rows to SUMMARY.md tables.
  Queued: seed replicates of KTH pose (Ilic, scenario), one posture per clip on the scenario split (+ hardest test),
  canonical + one posture per clip (Ilic), and one posture per clip on VP-HMDB51.
- 21:20 UTC: hardest test on canonical stick figures (KTH scenario): 16.5 (frame 12.0) vs 21.5 for plain stick figures; fixing
  limb lengths and size removes ~5 points on unseen scenario + unseen actions, landing at the learned x16 pool (17.3).
  KTH pose GIFs added (results/gifs/kth/p12_kth_pose.gif, p12_kth_pose_e16.gif): limbs only, no background; an empty tile is a
  walking clip before the person enters.
- 21:35 UTC: KTH canonical stick figures (Ilic): 97.3 action / 45.2 identity (plain stick figures 97.8 / 53.3; across actions
  21.4 vs 27.4). Fixing limb lengths and size costs nothing in action and removes ~8 identity points within actions, ~6 across.
  KTH one posture per clip, hardest test: 17.0 (all frames 21.5; canonical 16.5).
  VP-UCF101 controls: black frames 61.8 conditional (chance 61.3), mean colour per frame 62.1: no hidden channel in the pipeline.
  Queued pose and canonical pose on UTD-MHAD (within and across actions): every obfuscator so far kept UTD identity at 84-100
  through the per-session colour cue (mean colour alone 72.8); stick figures remove colour and background entirely.
  Also canonical + one posture per clip on the KTH scenario split and its hardest test.
- 22:15 UTC: pose results (a long gap between ticks left 6 VMs idle for a while; refilled now).
  KTH Ilic (action / identity):  stick figures 97.8 / 53.3, seed 1 97.8 / 53.5  |  canonical 97.3 / 45.2  |  one posture per clip
  93.3 / 38.1  |  canonical + one posture per clip 91.5 / 29.1. The last beats the learned x32 pool (~92 / ~40) and matches x48's
  identity (~25) with +4 action.
  KTH scenario (action / identity / hardest test): stick figures 90.8 / 38.0 / 21.5, seed 1 90.2 / 37.5  |  canonical 90.5 / 44.2
  / 16.5  |  one posture 84.8 / 31.7 / 17.0  |  canonical + one posture 81.7 / 27.5 / 13.5. Learned x32 pool: 62.4 / 28.7 / 12.0.
  Canonical + one posture gives ~19 more action at the same identity. Canonical alone raises scenario identity (44.2 vs 38.0):
  fixing the drawing scale removes the zoom differences between scenarios, which helps attackers generalise across them.
  UTD-MHAD: stick figures 98.1 / 100 (across actions 96.5), canonical 97.7 / 99.5 (86.4): with 8 people, body and motion
  identify everyone even without the session colour cue. UTD is not a useful privacy benchmark for any obfuscator we have.
  VP-HMDB51: canonical pose 29.1 action / 65.5 conditional (relationship +3.8, plain pose +10.3): person size (near vs far, big
  vs small figures) carried most of the relationship signal. One posture per clip: 26.1 / 66.3.
  Queued: seed replicates of canonical + one posture (Ilic, scenario), its cross-action test, every-8 variant (Ilic, scenario,
  hardest), a 4x downsample after it (Ilic, scenario, hardest), 24-epoch attackers and DINOv2 attackers on it.
- 22:30 UTC: Charades stick figures: 8.2 mAP / 32.5% identity (raw 15.8 / 93.5). Beats every non-collapsed Charades point on both
  axes except the near-floor BDQ a1-a3 (~5.8 / ~20): BDQ a0.5 7.1 / 45.6, two RGB pixels 5.5 / 38.8, person removed 10.6 / 90.6.
  Removing the room and appearance takes identity from 93 to 33; what's left is body and motion in each home.
  VP-UCF101 stick figures: 55.5 action / 74.7 attr / 66.0 conditional (raw 91.8 / 77.1 / 67.1); skin colour beyond the action
  69.2 = raw 69.2, relationship 45.4 vs raw 43.8: the HMDB finding holds on UCF, the beyond-action signal is structural.
  Queued canonical and canonical + one posture per clip on Charades.
- 22:40 UTC: VP-HMDB51 one canonical skeleton per frame (no appearance, proportions, size or people count): 26.5 action / 65.8 /
  65.1 conditional (chance 61.9). Per attribute vs chance: face +3.5, skin colour +5.5, gender +1.1, nudity +0.7, relationship
  +5.5. Skin colour is still predictable within an action from one fixed-size skeleton's posture and position. That can't be
  appearance; most likely HMDB source films (a shared actor, camera style, framing) correlate with the skin-colour label within
  an action class. The VP skin-colour label measures source, not something an appearance obfuscator can or should remove.
  Queued DINOv2 attackers on canonical one-posture stick figures, KTH scenario split.
- 22:55 UTC: DINOv2 attackers on canonical one-posture stick figures (KTH Ilic): probe 23.1, fine-tune 25.8, below R3D's 29.1.
  As with every obfuscator so far, the video attacker is the strongest; no hidden appearance cue for DINO to exploit.
  Canonical one-posture + x4 downsample, hardest test: 13.0 (without the downsample 13.5): blurring the figure adds nothing.
  Queued a seed replicate and 24-epoch attackers for the canonical one-posture hardest test.
- 23:10 UTC: canonical one-posture stick figures replicate: Ilic seed 1 92.0 / 28.3 (seed 0 91.5 / 29.1); scenario seed 1
  81.0 / 29.7 (81.7 / 27.5). Across actions (Ilic): 19.9. DINOv2 on the scenario version: probe 28.7, fine-tune 29.2 (R3D
  27.5-29.7): no stronger attacker. Adding x4 downsample: Ilic 91.8 / 25.3, scenario 79.7 / 25.8. On Ilic that beats the learned
  x48 pool (~87.4 / ~25) by ~4 action; the learned x64 pool (~81 / 11.4) still holds the low-identity end.
  Queued x8 and x16 downsamples of the canonical figures (Ilic, scenario, hardest test) to trace the pose frontier toward x64's
  identity, plus the x4 version's cross-action test.
- 23:20 UTC: robustness of canonical one-posture stick figures: 24-epoch attackers 25.1 (12-epoch 29.1: longer training only
  overfits); hardest test seed 1 14.0 (seed 0 13.5); x4 downsample across actions 15.9 (no downsample 19.9).
  One posture per 8 frames: Ilic 96.3 / 37.5, scenario 86.2 / 36.2 / 17.0: halfway between every 4 (97.3 / 45.2) and every 16
  (91.5 / 29.1); posture refresh rate is a clean action-identity dial.
  Queued: x8 / x16 cross-action and hardest tests, and seed replicates of the x4 version (Ilic, scenario).
- 01:05 UTC (Oct 29): ticks were delayed ~90 min; 7 VMs sat idle. Pose frontier results:
  KTH Ilic, canonical one-posture stick figures then downsample (action / identity / identity on unseen actions):
      none 91.5 / 29.1 / 19.9 | x4 91.8, 90.3 / 25.3, 23.6 / 15.9 | x8 89.1 / 20.1 / 13.2 | x16 89.0 / 13.2 / 12.5
  The x16 version nearly matches the learned x64 pool's identity (~11.4) with ~8 more action (~81), with no training at all.
  KTH scenario (action / identity / hardest test): x4 79.7, 79.5 / 25.8, 30.2 / 13.0 | x8 75.0 / 23.3 / 12.0 | x16 73.7 / 18.2 / 11.5.
  x16 beats the learned x48 pool on both axes (56.8 / 26.0 / 16.0); vs learned x64 (52.0 / 13.3 / 8.3): +21 action, +5 identity.
  CAVEAT: 24-epoch attackers on the scenario version gain a little (31.0 vs 27.5-29.7; hardest 16.5 vs 13.5-14.0), unlike Ilic
  (25.1 vs 29.1). Queued 24-epoch attackers on the x16 versions to check the headline.
  Charades canonical one-posture stick figures: 6.5 mAP / 16.8% identity (plain stick figures 8.2 / 32.5; BDQ a1-a3 ~5.8 / ~20;
  floors 4.4 / 3.6). Lowest Charades identity with action above BDQ's.
  Queued: seed replicates (x16 Ilic + scenario, x8 Ilic, Charades canonical), x32 (Ilic, scenario, cross-action, hardest),
  24-epoch attackers on x16 (Ilic, hardest), DINOv2 on x16, Charades canonical + x4.
- 02:10 UTC (Oct 29): canonical one-posture stick figures, robustness of the x16 version: Ilic seeds 89.0 / 13.2, 89.3 / 14.2;
  24-epoch attackers 88.3 / 13.9; DINOv2 probe 8.4, fine-tune 4.0. Scenario seeds 73.7 / 18.2, 73.0 / 19.2; DINOv2 14.0; hardest
  test 11.5, 10.5 (seed 1), 11.0 (24-epoch attackers). The x16 numbers hold.
  x32: Ilic 82.6 / 10.9 (across actions 12.3), ties the learned x64 pool (~81 / ~11.4). Scenario 66.3 / 13.3 / 8.0 hardest,
  vs the learned x64 pool 52.0 / 13.3 / 8.3: same identity on both tests, +14 action, no training.
  SUMMARY.md findings rewritten: the canonical stick figure is now finding 2 (best KTH obfuscator); learned pools are finding 3;
  Charades and VP findings updated with the pose results. VP-UCF101 single-person pose: 51.6 action / 71.4 / 65.2 conditional.
  Queued: x32 seed replicates, 24-epoch attackers and DINOv2 on x32 (scenario), x64 (Ilic, scenario, cross-action, hardest).
