# H2-013: NEW FIELD N4. Astronomy instrument removal on a GalSim twin-render benchmark with exact truth: a label-free adapter keeps held-out physical parameters that a parameter-supervised projection discards

- Focus: candidate field (not previously killed; contrarian scout E2, folded into A as a second domain) | Kind: headroom check first, learned arm second | Cost: ~3 GPU-h (+~4 CPU-h rendering) | Prior: 0.15 (full claim), 0.65 (headroom check passes)
- Depends on: GalSim (pip-installable; BSD licence from memory, verify), frozen DINOv2-B (Apache-2.0, already used in A as dinov2-L); H2-001/H2-007 code; `src/b_robotics` twin-render pattern for the oracle.

## Statement
Tier T1 of the scout's plan, built here. Render 64x64 galaxy stamps from physical parameters theta = (log half-light radius r_e, axis ratio q, bulge-to-total B/T, bulge Sersic n, log total flux, clump fraction) drawn from one fixed population independent of the instrument, through three instrument classes with different pixel scales, PSF FWHM distributions (continuous within class) and sky-noise levels, 200k stamps. Each stamp has an exact CLEAN TWIN (same theta, no PSF, no noise, fixed pixel scale). Encoder: frozen DINOv2-B on the upsampled stamp. Tasks: seen = {r_e, q}; held-out future tasks T' = {B/T, n, flux, clump fraction} (ridge/MLP probe R^2, retrained on B). Secret: instrument class (3-way), PSF FWHM, sky sigma (continuous attackers R^2; L_worst = max over targets/attackers, normalised to raw). Gates and headroom check before any learned arm:
- G1: raw leaks: instrument class accuracy >= 0.95 and PSF-FWHM R^2 >= 0.6;
- G2 (input-information check, the B' lesson): U_clean(T') - U_raw(T') >= 0.10 mean R^2 (embedding of the clean twin vs the degraded stamp). If the gap is smaller, the harm is information destroyed in the input and no latent method can recover it: STOP;
- G3: the parameter-supervised projection (ridge/PLS onto seen parameters, 2 dims) keeps <= 50% of raw's R^2 on T'.
If all pass, the claim: the label-free adversarial AE (or cVAE conditioned on instrument parameters) with dz in {8, 16, 32} reaches, at L_worst <= 0.2, mean R^2 on T' at least +0.05 above the best inductive floor (PCA-d, dithered PCA, seen-parameter projection padded with PCA) and closes >= 40% of the gap U_clean - U_floor, 3 seeds, bootstrap over stamps in groups of theta draws.

## Mechanism
Physical-parameter inference across instruments is an open-ended family of tasks (redshift, morphology, mass, ellipticity are all asked of the same latent), so the content the task needs is not defined by any one label. PSF blur and noise are partly invertible (deconvolution) and partly destructive; the clean twin makes the invertible fraction measurable exactly, which pathology and audio cannot do. The opened workshop paper (Mercader-Perez et al.) separates physics and instrument latents on Legacy x HSC pairs with a dual encoder and counterfactual generation, but its page gives no leakage metric; a retrained-attacker protocol is the missing piece. B' warns that the harm may be input clutter rather than leakage, and G2 is the pre-registered guard.

## Experiment (exact)
- Rendering: GalSim, 40 CPU processes, fixed seeds; splits by theta draw (A adapter-train 45%, B probe/attacker 35%, V 20% of non-sealed, C sealed 25% of units); a "confounded" variant (theta distribution shifts per instrument, as in real surveys) is reported separately.
- Arms: raw; clean twin (oracle); seen-parameter ridge/PLS projection; PCA-d; dithered PCA; per-instrument z-scoring with instrument id (transductive reference); AE+adv, AE lam 0, cVAE (instrument parameters as decoder covariates), contrastive; dz ladder.
- Secondary utility (generative/reconstruction): linear R^2 of predicting the clean-twin embedding from z.
- Controls: constant/noise at chance; shuffled-instrument attacker at chance; theta-shuffled probes at R^2 ~ 0; train/test stamps share no theta draws.

## Prediction and falsifier
- Prediction: G1 passes (0.9); G2 passes (heavy blur at 1.2-1.8 arcsec seeing removes B/T and n; 0.65); G3 passes (0.8); learned arm +0.03 to +0.08 R^2 over floors, closing 30-50% of the clean-twin gap.
- Falsified (NO-GO at the check) if U_clean - U_raw < 0.10 (input information loss; same verdict as B'), or the PCA/dithered/instrument-z-score floors are within 0.03 R^2 of the best learned arm at L <= 0.2.

## Cost
Rendering 200k stamps ~4 CPU-h (parallel ~10 min wall); DINOv2-B embedding ~10 GPU-min; adapters ~3 GPU-h.

## Dependencies
None on other focuses; shares AE code. GalSim install on the VM.

## Prior and why
0.65 that G1-G3 pass (synthetic design chosen so physics and instrument are independent and the blur is strong), 0.15 for the claim: the same programme pattern (cheap deterministic or rate-limited floors close most of the gap) applies, and a frozen natural-image encoder may discard the faint features that carry B/T and n regardless of the adapter.

## Sources
[O] arXiv 2604.09787 (Learning What's Real: dual-encoder physics/instrument latents on Legacy x HSC, counterfactual objective); [S] literature/contrarian_missed_fields.md E2 (scout-reported numbers: physics latent ellipticity R^2 0.934, redshift 0.801; instrument latent PSF depth 0.667; not independently verified); results_notes/B.md (twin-render ground truth and oracle-headroom check).
