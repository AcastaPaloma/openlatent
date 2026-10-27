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
