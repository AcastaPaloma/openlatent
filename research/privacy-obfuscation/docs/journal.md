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
