# CT loop journal
- 2026-11-02 20:3x: VM ok (8xA100). Train-feature extraction ~45% (923/1935), no failures, network-bound (~10 scans/GPU/85s).
  H0: reproducing dev numbers (A0 0.200 / B1 0.279 / A3 0.077) on re-extracted features validates the pipeline; a mismatch >0.03 means extraction differs (orientation/recon choice) and every later comparison is suspect.
  Next: launch valid extraction when train finishes; run A0,B1,A3 x3 seeds across GPUs.
- NOTE: running train extractors write name[:-7].npz; after DONE rename to name[:-4] convention (x_a_1.nii.npz) expected by train_arms.py; valid goes into same feats dir.
- Cycle 3: features complete (1935 train + 409 valid). Reproduction: B1 0.278 (orig 0.279), B2 0.220 (orig 0.221) -> pipeline reproduces.
  Problem: 6 arms stuck in CPU attacker phase, load avg ~250 on 124 cores (thread oversubscription). Restarted with OMP/MKL threads capped at 12.
  H1: with capped threads attacker phase finishes in minutes. If A0/A3 differ from 0.200/0.077 by >0.03 check seed variance first (orig = mean of 3 seeds).
  Next: A0..A7 seed 0 -> seeds 1,2 -> new arms.

## Cycle 4 (2026-11-02 ~21:30)  REPRODUCTION RESULT (seed 0, dev, n=1935, fit 924/eval 1011)
 | arm | now | original (3-seed mean) |
 | A0 raw | 0.207 | 0.200 | B1 PCA | 0.278 | 0.279 | B2 | 0.220 | 0.221 | A1 | 0.129 | 0.134 | A2 | 0.083 | 0.090 | A3 | 0.078 | 0.077 | A5 NoPeek | 0.139 | 0.124 | A7 LEACE | 0.102 | 0.112 |
 -> pipeline reproduced (all within 0.015). Metadata floor 0.028, chance 0.0025.
 STILL OPEN: best arm A3 is 2.8x the floor. Utility kept 0.97 (AUC 0.649 vs raw 0.648 after probe-C choice).
## Round 3 (declared BEFORE running; dev only; total configs so far 15 + these 6 = 21)
 Hypothesis H-L: patient-ID-supervised linear erasure (generalised-eigen/LDA: directions with high between-patient vs within-patient variance, found on fit patients) removes identity that the adversary misses; combined with A3 it should land nearer the 0.028 floor with utility >=0.95.
 Arms: L32A3, L64A3, L128A3 (erase k identity directions in PCA-256 then A3), L64, L128 (erasure only, no training), K0 (constant latent, degenerate check: must score chance on linkage and 0.5 AUC).
 New metrics added to every run: utility CI (patient bootstrap), nonlinear MLP-probe utility, per-label AUC.
 Kill criteria: if L*A3 top1 >= A3 (0.078) within noise, H-L rejected -> next: mutual-information/CLUB penalty and larger adversary ensembles (literature check first).

## Cycle 4 RESULTS round 3 (seed 0, dev). H-L REJECTED.
 | arm | top1 | utility AUC [95% CI] | MLP-probe AUC | sex AUC |
 | A0 raw | 0.207 | 0.648 [0.631,0.664] | 0.633 | 0.967 |
 | A3 | 0.078 | 0.649 [0.632,0.665] | 0.648 | 0.869 |
 | L32A3 | 0.101 | 0.575 | 0.574 | 0.734 | L64A3 | 0.101 | 0.557 | 0.554 | 0.631 | L128A3 | 0.086 | 0.535 | 0.537 | 0.576 |
 | L64 (erase only) | 0.193 | 0.538 | | 0.605 | L128 | 0.144 | 0.522 | | 0.524 | K0 constant | 0.000 (chance) | 0.500 | | 0.500 |
 Findings: (1) criterion 3 now checked: constant latent = chance linkage + 0.5 AUC. It exposed a scorer bug: NaN similarities were ranked as perfect hits (K0 first scored 1.0 for pca_white). Fixed with nan_to_num; no real arm produced NaN, so earlier numbers are unaffected, but old reports never ran this check.
 (2) A3 utility holds under a nonlinear probe too (MLP 0.648 vs raw 0.633) and under a patient-bootstrap CI.
 (3) Identity-discriminative directions overlap heavily with sex/disease signal: erasing them costs ~0.07-0.10 AUC and does NOT beat A3 on linkage. Linear one-shot erasure is not enough; the adapter needs to be trained.
 Hypothesis H-M: the best attacker against A3 is the siamese contrastive one (it picks the right patient most often), but the A3 training adversary is only a pair-verifier. Adding a contrastive adversary (NT-Xent via gradient reversal, SPLAVU-flavoured) should close the gap toward the 0.028 floor.
 Round 4 arms (declared before running): A12 = A3 + contrastive adversary weight 1; A13 = same weight 3. Seeds 0,1,2 each, plus A3 seeds 1,2 for a paired comparison. Total configs tried: 21 + 2 = 23.
 Kill: if A12/A13 mean top1 not below A3 mean minus ~0.01 -> reject; next try utility-constrained lambda sweep vs noise at matched utility, then second encoder / extra tasks.
 TODO after: lit search (CLUB / MI upper bound, contrastive privacy in medical imaging); run sealed exam only with frozen arm list; reconstruction attack in pixel space still not done.

## Cycle 5 RESULTS round 4 (dev, 3 seeds each)
 | arm | top1 per seed | mean | utility AUC mean | inversion top1 |
 | A3 | .078 .096 .079 | 0.084 | 0.648 | 0.060 | A12 (A3+contrastive adv w1) | .069 .060 .056 | 0.062 | 0.593 | 0.045 | A13 (w3) | .063 .068 .066 | 0.066 | 0.556 | 0.041 |
 Utility kept (raw 0.648): A3 ~1.0, A12 0.63, A13 0.38.
 Reading: the contrastive adversary lowers linkage (0.084 -> 0.062) but at a large utility cost. Old dev noise baseline B3 sigma2 was 0.064 at utility 0.581 -> A12 (0.062 @ 0.593) is NOT clearly better than plain noise at matched utility. H-M NOT supported as an improvement; it just moves along the frontier. Lesson: never compare linkage without matching utility.
 LITERATURE (searched): Privacy-Net (arXiv 1909.04087, adversarial identity-obfuscated segmentation), MICCAI-2023 chest X-ray deformation proxy (re-ID AUC 81.8->57.7 with little classification loss), Syfer (neural obfuscation), MI-bound approaches (CLUB upper bound; TURBO arXiv 2311.06527; GPP arXiv 2605.03069). None compares at matched utility on CT embeddings -> our matched-noise frontier is the right control.
 Round 5 (declared before running): (a) own noise frontier B3 sigma 0.5,1,1.5,2 x seeds 0,1 to fit linkage-vs-utility curve for noise; (b) A14 (A3 + contrastive w0.3), A15 (w0.1) x seeds 0,1,2 to fill in the A3->A12 frontier. Decision: an arm "beats noise" only if its (utility, linkage) point lies below the noise curve by more than seed spread. Total configs tried: 23 + B3 x4 + A14/A15 = 29.
 Next if nothing beats noise: (i) CLUB/MI penalty arm; (ii) bigger/other frozen encoder; (iii) pixel-space reconstruction attack; (iv) per-task heads beyond 18 labels.

## Cycle 6 RESULTS round 5 (dev, own runs; raw util 0.648; chance 0.0025; metadata floor 0.028)
 Noise curve (own, 2 seeds): sigma0.5 util .634 top1 .177 | sigma1 .611 .128 | sigma1.5 .596 .083 | sigma2 .580 .067.
 Adversarial: A3 .084 @ .648 (kept 1.00) | A15 (+contrastive w0.1) .072 @ .637 (.93) | A14 (w0.3) .067 @ .635 (.91) | A12 (w1) .062 @ .593 (.63) | A13 .066 @ .556 (.38).
 Matched utility: at ~0.635 noise gives ~0.18, A14 gives 0.067 -> adversarial adapters beat noise by ~2.7x at matched utility (well beyond seed spread 0.006). At utility >=0.95 kept only A3 (0.084 vs raw 0.207 = 60% less linkage). Still 3x the metadata floor; floor NOT reached with usable utility.
 Note: inversion-attack top1 is below the floor for A7 (0.021) and A5 (0.035) -> different attackers disagree; best-of-five latent attackers stays the headline.
 H-S (new): the adapter is trained on only 924 labelled scans; more labelled/multi-visit fit data should let the utility head find disease directions that are independent of identity and give the adversary more real pairs -> lower linkage at the same utility. Test: extract 3159 extra scans from 1300 NEW CT-RATE train patients (fit-only, never evaluated; eval gallery unchanged so numbers stay comparable). Extraction running on 8 GPUs (~1h).
 Round 6 (declared before running): A3 / A14 / A15 / A0-probe with EXTRA data, seeds 0,1,2. Control: same arms without extra data (already have). Success = lower top1 at same-or-higher utility than the matching no-extra arm by more than seed spread. Total configs so far 29 + 3x3 = 38.

## Cycle 7 (3-seed baselines now complete; dev, own runs vs ORIGINAL 3-seed means)
 A1 .134 (orig .134) | A2 .088 (.090) | A3 .084 (.077) | A5 NoPeek .129 (.124) | A7 LEACE .112 (.112) | B1 .278 (.279) | B2 .220 (.221) | A0 .207 (.200, 1 seed).
 -> every reproduced arm within 0.007 of the original mean; A3 is 0.007 worse than the original (within seed spread 0.008). Pipeline trusted.
 Extra-data extraction: ~2380/3159 scans, no failures, ~10 min left. Round 6 launches next cycle.
- Cycle 8: extra extraction DONE (3159 scans / 1300 patients, 0 failures). Round 6 launched: EXTRA=all for A3,A14,A15 x seeds 0-2 + A0,A1 seed0 (raw reference with extra data for utility-kept).

## Cycle 9 RESULTS round 6 (EXTRA data: adapter+attackers see 4083 fit scans; eval same 1011 scans; floor 0.031)
 | arm | top1 | utility AUC | MLP-probe | inversion | sex |
 | A0 raw | 0.403 | 0.688 | 0.671 | 0.106 | 0.979 | A1 | 0.132 | 0.636 | 0.632 | 0.079 | 0.850 |
 | A3 | 0.126+-0.005 | 0.684 | 0.683 | 0.061 | 0.907 | A14 | 0.107+-0.005 | 0.676 | 0.675 | 0.053 | 0.892 | A15 | 0.127+-0.003 | 0.678 | 0.680 | 0.057 | 0.914 |
 Readings: (1) H-S as stated (absolute linkage falls with more data) REJECTED: A3 0.084 -> 0.126. BUT the attacker also gets 4x data: raw linkage rose 0.207 -> 0.403. Relative linkage removed by A3: 59% -> 69%. So the OLD headline numbers used a data-starved attacker and understate raw leakage by 2x.
 (2) Utility improves with data: raw .648 -> .688, A3 .649 -> .684 (kept 0.98), A14 kept 0.94. More data does help utility, which is what the adapters utility head wanted.
 (3) OPEN RISK: is A3-small (trained on 924 scans) still safe against a data-rich attacker? Test = ATTX mode: the adapter is trained on the ORIGINAL fit set only; attackers and probes are trained on fit + extra patients. The adapter encodes the extra scans without ever having seen them.
 Round 7 (declared before running): ATTX=1 for A0, A1, B1, B3 sigma1 (seed 0) and A3, A14 (seeds 0,1,2). Success criterion: A3-small under strong attackers stays well below raw under strong attackers (A0 ATTX); otherwise the earlier 'reduced linkage' claim was an artefact of a weak attacker. Configs so far 38 + 10 = 48.

## Cycle 9b RESULTS round 7 (ATTX: adapter trained on 924 fit scans; attackers trained on 4083 scans; eval 1011; floor 0.031)
 A3 0.107+-0.003 (weak-attacker 0.084), util 0.665 | A14 0.080+-0.009 (weak 0.067), util 0.653 | A1 0.126 | B1 PCA 0.276 | raw A0 (strong attacker) ~0.403 (pending in this table; see EXTRA row).
 -> The 'reduced linkage' claim SURVIVES a 4x-data attacker: A3-small removes ~73% of raw leakage (0.403 -> 0.107), A14-small ~80% (-> 0.080). Weak-attacker numbers understated every arm by ~0.02 but raw by ~0.2. Floor 0.031 still ~2.6-3.5x away.
 PUZZLE: adapters trained WITH the extra data (EXTRA: A3 .126, A14 .107) leak MORE than small adapters under the same strong attacker (ATTX: .107, .080).
 H-T: step confound - 1500 steps x 256 = ~400 epochs at 924 scans but ~94 epochs at 4083, so the big-data adapter (and its adversary game) is undertrained. Test: EXTRA with 5000 steps for A3, A14 seeds 0,1 (file names now carry _st5000). If linkage drops to <= ATTX values at util >= .68, steps were the cause and more data + more steps is the best recipe.
 Alt H-U if not: small-data overfitting acts as extra compression; then try small adapter + stronger bottleneck (ZD=64/128) instead.
 Round 8 configs: 4 (total 52). A fresh sealed-2 exam is available: 538 unused CT-RATE train patients with 2+ scans (never extracted); old sealed set (169 valid patients) was already read once for the old arms.

## Cycle 10 RESULTS H-T (steps) + per-label utility
 EXTRA, 5000 steps: A3 0.147+-0.023 (util .661) vs 1500 steps 0.126 (util .684); A14 0.117 (util .655) vs 0.107 (.676). H-T REJECTED: more steps = MORE leakage and LESS utility (adversary game overfits / drifts).
 Raw refs: A0 EXTRA 0.403 (util .688); A0 ATTX 0.382 (util .688). B3 sigma1 ATTX: 0.134 @ util .637 -> A14-ATTX 0.080 @ .653 and A3-ATTX 0.107 @ .665 beat noise at matched utility under a strong attacker too.
 CORRECTION on utility kept: ATTX regime (adapter trained on 924 scans, probe on 4083): A3 util .665 vs raw .688 -> kept 0.88 (not 1.00; the old 1.00 came from a raw probe starved of data). EXTRA regime (adapter and probe both on 4083): A3 .684 -> kept 0.98, linkage 0.126 (69% below raw .403). Only EXTRA-A3 meets kept>=0.95.
 Per-label AUC (ATTX, mean over seeds): raw -> A3: mean change -0.023 (A14 -0.035, noise sigma1 -0.051). Biggest losses (A3): Medical material -.053, Cardiomegaly -.039, Hiatal hernia -.036, Pleural effusion -.044, Coronary calcification -.030, Interlobular septal thickening -.034. Lung nodule/opacity/pericardial effusion essentially unchanged. So obfuscation costs cardiac / pleural / device findings most, i.e. large-scale anatomy that also identifies a person.
 H-V: since 5000 steps hurt, FEWER steps (early stopping of the adversarial game) may help; test EXTRA with 500 and 1000 steps for A3 and A14, seeds 0,1 (8 runs; total configs 60). Kill: if linkage at 500/1000 >= 1500-step values (A3 .126, A14 .107) at equal/lower utility, step count is not the lever.
 Queue after H-V: (a) Pareto summary table with all arms in one regime; (b) CLUB/MI penalty arm; (c) fresh sealed-2 set (538 unused multi-visit patients) extracted only once arm list is frozen; (d) pixel-space reconstruction attack.

## Cycle 11 RESULTS H-V (fewer steps), EXTRA regime, mean over 2 seeds (1500 steps = 3 seeds)
 | arm | steps | top1 | util | kept | inversion |
 | A3 | 500 | 0.105 | .674 | .92 | .060 | A3 | 1000 | 0.112 | .685 | .99 | .055 | A3 | 1500 | 0.126 | .684 | .98 | .061 | A3 | 5000 | 0.147 | .661 | .85 | .078 |
 | A14 | 500 | 0.089 | .662 | .86 | | A14 | 1000 | 0.088 | .668 | .90 | .046 | A14 | 1500 | 0.107 | .676 | .94 | .053 | A14 | 5000 | 0.117 | .655 | .82 | .069 |
 H-V SUPPORTED: linkage rises monotonically with training steps. A3 at 1000 steps: 0.112 at kept 0.99 vs 0.126 at 0.98 (diff 0.014 > seed spread 0.005) = free gain; vs raw .403 => 72% reduction at kept>=0.95. CAVEAT: step count was chosen by looking at dev-eval linkage (4 values tried: 500/1000/1500/5000); the gain must be confirmed on a fresh set (sealed-2) before being claimed.
 Mechanism hypothesis H-W: the longer the adapter trains, the more it memorises the fit patients (utility head + adversary reset game), which makes eval-patient scans more separable. If true, input regularisation during training should give the same or bigger gain without hand-picked stopping.
 Round 9 (declared before running, EXTRA regime, 1000 steps, 3 seeds each): A16 = A3 + input dropout 0.3; A17 = A3 + Gaussian input noise 0.5; A18 = A3 + input dropout 0.6; plus A3@1000 seed 2 to complete its 3 seeds. Kill: an arm counts only if top1 < 0.112 - 0.01 at kept >= 0.95; otherwise H-W is rejected and the lever is early stopping only. Configs so far 60 + 10 = 70 (selection effect noted).

## Cycle 12 RESULTS H-W (input regularisation), EXTRA regime, 1000 steps, 3 seeds
 A3 0.110+-0.005 (kept .98) | A16 dropout .3: 0.123 (.99) | A17 gauss noise .5: 0.121 (.98) | A18 dropout .6: 0.123 (.97) | A14 0.088 (kept .90).
 H-W REJECTED: input regularisation does not help (worse by ~0.012). The 1000-step gain is early stopping of the adversarial game, not regularisation; plateau for this recipe ~0.11 at kept >= .95. Metadata floor 0.031 => still 3.5x.
 Round 10 (declared before running) = FAIR COMPARISON TABLE in the EXTRA regime (4083 fit scans, same eval 1011): each baseline gets the same tuning budget as A3 (1000 and 1500 steps, 3 seeds, best taken): A1, A2, A5 (NoPeek), A7 (LEACE); B1, B2 x3 seeds; noise B3 sigma .5/1/1.5 x2 seeds (matched-utility control in THIS regime); A8 (lambda 10) and A15 at 1000 steps; A0 seeds 1,2. 43 jobs. Total configs ~113. Decision: report mean +- std, utility kept relative to A0 EXTRA (.688), and the matched-utility noise curve.
- Cycle 13: queue13 bug (xargs -L1 joined lines with trailing blank; arm named bash; many jobs skipped). Fixed: explicit sigma placeholder x; reran only missing outputs. Earlier completed results valid.

## Cycle 14 RESULTS fair comparison, EXTRA regime (4083 fit scans; eval = same 1011 dev scans; floor .031; raw A0 top1 .380+-.016 (3 seeds), util .688)
 arm@steps: top1+-sd | kept | note   (kept = (util-.5)/(.688-.5))
 A14@1000 .088+-.005 | .90 ; A8@1000 .099+-.008 | .82 ; A14@1500 .107+-.005 | .94 ; **A3@1000 .110+-.005 | .98** ; noise s1.5 .111 | .67 ; A15@1000 .112 | .93 ; A5 NoPeek@1500 .123 | .64 ; A3@1500 .126 | .98 ; A7 LEACE@1000 .131 | .66 ; A1@1500 .139 | .71 ; noise s1 .152 | .78 ; A2@1500 .154 | .89 ; A2@1000 .162 | .94 ; noise s.5 .272 | .92 ; B2 .273 | .91 ; B1 PCA .314 | .93 ; A0 .380 | 1.00
 -> At kept >= .92 the external/cheap baselines leak 0.27-0.31 (2.5-3x A3); baselines that leak ~0.12-0.13 (NoPeek, LEACE, noise s1.5) keep only 64-69% of utility. A3@1000 is the only arm with kept >= .95 AND linkage ~.11 (71% below raw .38). Still 3.5x above floor .031: NOT anonymous.
 SEALED-2 FROZEN (FROZEN2.txt, sha256 e4e35e80..): fresh exam = 538 never-used multi-visit CT-RATE train patients (1296 scans); arms/steps fixed from dev; extraction started (8 GPUs). No sealed-2 number has been seen. Patch11 (SEAL2 mode) adds the exam as eval-only rows.
 Declared next after sealed-2: (a) MI/CLUB penalty arm; (b) second encoder or finer-resolution features for multi-task utility; (c) pixel-space reconstruction attack. Anything run after the exam is reported as exploratory.

## Cycle 15: sealed-2 queue armed (39 jobs = 13 frozen configs x 3 seeds incl. K0); starts automatically when all 8 seal2 extraction logs say DONE. Jobs file sha256 recorded in logs/seal2_started.txt on start. Results appear as runs/arm_*_seal2*.json.

## Cycle 16: SEALED-2 REPORTED (see CT experiment/ct-results-sealed2.md). Raw 0.427 -> A3@1000 0.102 (-76%), kept 0.94 linear / 1.03 MLP; A14 0.060 at kept 0.87 (MLP 0.95); cheap baselines leak 0.27-0.33; NoPeek/LEACE/noise1.5 leak 0.075-0.082 but keep 55-73%. Floor 0.034: no arm reaches it. K0 passes. Pixel reconstruction attack still not run.
 Everything below is EXPLORATORY (post-exam) and will be labelled so:
 H-X: A14 (light contrastive adversary) is the better operating point than A3 at ~equal utility on the MLP probe (kept .95 vs 1.03 but linkage .060 vs .102). Test: A14 variants at weights .1/.2/.3/.5, 1000 steps, 3 seeds on DEV eval (not the exam) to see if a point with linear-kept >= .95 and linkage < .09 exists.
 H-Y: remaining linkage is anatomy (body size, heart/diaphragm shape), not scanner. Test on dev: restrict the gallery to same-scanner/same-kernel candidates (hard negatives); if linkage stays ~ same, anatomy dominates. Also feature-view test with the global 512-d CT-FM vector g as a second encoder view (generality across inputs).
 H-Z: MI-bound (CLUB) penalty replacing the pair adversary may be more stable than adversary resets. Literature: CLUB (Cheng 2020), TURBO 2311.06527.
 Still owed: pixel-space reconstruction attack (criterion 4), utility CI used as a formal bound, second institution.

## Cycle 17 (EXPLORATORY, post-exam, DEV eval only; nothing here changes the sealed-2 report). Reproduction check was done in cycles 4-7.
 Declared before running: (H-X) A19 = A3 + contrastive adversary w0.2, A20 = w0.5, @1000 steps, 3 seeds, compared with A3@1000 (.110, kept .98) and A14@1000 (.088, kept .90) in the EXTRA regime. Question: is there a point with linear kept >= .95 and linkage < .09?
 (H-Y) New metric same_scanner_top1: linkage when the gallery is restricted to people scanned with the same scanner model + kernel as the true match (hard negatives), with its own chance level. If same-scanner linkage stays near the all-gallery number, the leak is anatomy not scanner. Run for A0, B1, A3, A14, A19, A20 x 3 seeds (files tagged _ss, so dev records are not overwritten). 18 jobs.
 Kill rules: H-X rejected if no new arm has kept >= .95 with top1 < .095; H-Y: if same-scanner top1 / chance is < half of the all-gallery top1 / chance for A3 the leak leans on scanner.

## Cycle 18 RESULTS exploratory round (DEV eval, EXTRA regime, 3 seeds, files tagged _ss). raw A0: top1 .380, util .688.
 | arm | top1 | kept lin | kept MLP | same-scanner top1 | inversion |
 | A0 | .380 | 1.00 | 1.00 | .482 | .107 | B1 | .314 | .93 | | .417 | .113 | A3@1000 | .110 | .98 | 1.03 | .231 | .056 | A14 (w.3) | .090 | .90 | .95 | .195 | .048 | A19 (w.2) | .123 | .93 | | .244 | .051 | A20 (w.5) | .078 | .85 | .91 | .165 | .040 |
 H-X REJECTED (kill rule): no new arm has kept >= .95 AND top1 < .095. Frontier of the adversarial family: (kept .98,.110) -> (.90,.090) -> (.85,.078); ~0.02 linkage per 5-8% utility. A19 is dominated (non-monotonic, seed sd .009).
 H-Y: same-scanner (hard-negative) linkage with ~38-person shortlists: A0 .482, A3 .231, A14 .195 (chance .026). Linkage does NOT collapse when scanner confounders are removed, so the leak is mostly ANATOMY not scanner. NOTE: my declared kill rule (ratio-to-chance) was ill-posed (chance changes with gallery size) - disclosed; applying the sensible reading: same-scanner top1 is 9x chance for A3.
 Nuance: against hard negatives the reduction from raw is smaller: A3 -52% (vs -71% overall), A14 -60%, A20 -66%. Part of the headline gain is removal of scanner-correlated signal; the anatomical part is harder.
 H-Q (new): the utility loss at the frontier is the utility pathway's limit, not an unavoidable price: a more expressive task head (A21: 2-layer MLP head), a wider latent (A22: 512-d), or a stronger utility loss weight (A23: BCE x3) should move the A14 point (kept .90, .090) toward higher kept at equal linkage. All with A14's adversary (w .3), 1000 steps, 3 seeds = 9 jobs. Kill: none of them gets kept >= .95 with top1 <= .095. Total configs ~140 (selection effect).
 Literature still to check: whether anatomy-level identity (organ shape) is known to be inseparable from diagnosis features in CT; will search next cycle if H-Q fails.

## Cycle 19 RESULTS H-Q (dev eval, EXTRA, 1000 steps, 3 seeds): SUPPORTED by A21.
 | arm | top1 | kept lin | kept MLP | same-scanner | inversion |
 | A3 | .110+-.005 | .98 | 1.07 | .231 | .056 | A14 | .090+-.005 | .90 | .97 | .195 | .048 | **A21 (A14 adversary + 2-layer MLP disease head)** | **.083+-.007** | **.96** | 1.05 | **.183** | .047 | A22 (512-d latent) | .095+-.002 | .95 | 1.05 | .206 | .048 | A23 (BCE x3) | .117 | .91 | 1.02 | .237 | .056 |
 Reading: the utility loss at the frontier was partly a limit of the LINEAR disease head used during training, not of the privacy pressure: a stronger head lets the adapter keep disease signal in directions the adversary does not need to touch. A21 moves the dev frontier: -25% linkage vs A3 at ~equal utility. A22 borderline (.095 at .95); A23 (loss weight only) does not help.
 CAUTION: A21 chosen on dev eval after ~140 configs; selection effect. Not claimed until confirmed on data it never touched. FROZEN3.txt written before any run: exam = the 169 original validation patients ('sealed-3'; A21 never scored there) + A21 second look on sealed-2 (disclosed as non-pristine).
 Jobs: 21 on sealed-3 (A0,B1,B3 s1.5,A3,A14,A21,K0 x3 seeds) + 3 A21 on sealed-2.

## Cycle 20 RESULTS sealed-3 (169 ORIGINAL validation patients; gallery 169, 240 queries; chance .0059, floor .042; A21 never scored here before) + A21 second look on sealed-2
 sealed-3 (3 seeds, mean+-sd; CI = bootstrap over patients): A0 .474+-.016 [.400,.544] util .702 | B1 .404 kept .90 | noise s1.5 .178 kept .67 | A3 .186+-.012 [.134,.240] kept 1.01 (MLP .703) | A14 .154+-.010 [.112,.204] kept .95 | **A21 .156+-.021 [.110,.206] kept 1.01 (MLP .702)** | K0 0.000 (chance), util .5.
 FROZEN3 test: A21 beats A3 by .030 (needed >=.015) at kept >= .95 -> met as a POINT ESTIMATE; CIs overlap heavily (169 patients) -> suggestive, not conclusive. A21 vs A14: same linkage, +6% utility kept -> consistent with the 'stronger disease head' reading. A21 still ~3.7x the metadata floor.
 Second look sealed-2 (NOT pristine): A21 .062+-.004 at util .694 (vs A14 .060 at .690, A3 .102 at .707): lands on the same frontier as A14, not a clear dominance. Honest summary: A21 = A14-level linkage with more utility on sealed-3, equal on sealed-2; benefit over A14 is small, benefit over A3 is linkage -25..40% at 0..-5% utility.
 Next (declared): (1) multi-task-model check on SAVED latents, no retraining: HistGB and kNN disease classifiers + age regression (leak) for A0/A3/A14/A21/B3 on sealed-3 -> does the utility survive other task-model families? (2) pixel/volume-space reconstruction attack still owed; plan: download ~600 low-res scans, train z->32^3 decoder, re-ID with raw attacker; decide next cycle (cost ~1.5h). (3) CLUB penalty arm. (4) literature on anatomy-level identity in CT (now needed: leak is anatomy).

## Cycle 21 RESULTS multi-task-model check on SAVED sealed-3 latents (no retraining; 3 seeds; HistGB and kNN task models, age ridge as leakage)
 First attempt was UNFAIR (raw got only its first 64 of 4096 dims for HistGB) - caught and fixed: every arm now gets a PCA-64 view of its own latent.
 | arm | HistGB AUC (kept) | kNN-25 AUC (kept) | age R2 (leak) |
 | A0 raw | .647 (1.00) | .597 (1.00) | .615 | B1 PCA | .571 (.48) | .514 (.14) | .553 | A3 | .676 (1.19) | .590 (.93) | .477 | A14 | .666 (1.13) | .577 (.79) | .426 | A21 | .682 (1.24) | .562 (.64) | .406 | noise s1.5 | .632 (.89) | .589 (.92) | .263 |
 Readings: (1) kept>1 for adapters under HistGB is NOT extra information: adapters saw the training labels (supervised), raw did not. (2) A3 is the most robust across task-model families (linear 1.01, MLP 1.0, HistGB 1.19, kNN .93). (3) A21's gain is probe-style only; nearest-neighbour retrieval utility drops to .64 -> A21 distorts local geometry (case-similarity search suffers). (4) Age stays partly decodable (R2 .62 -> .41-.48); noise hides age better (.26) but loses disease signal.
 H-AA: A21/A14 move examples around in latent space (adversary + richer head) so local neighbourhoods stop reflecting disease labels. Possible fix to test later: add a retrieval-aware utility term (supervised-contrastive on label-pattern similarity). Not run yet.
 Reconstruction-attack note (criterion 4): pixel reconstruction from z cannot add information that is not in z (data-processing inequality) and z is a function of the frozen features s; it only matters if a different attacker family reads identity better from the same information. Our attackers already include a 4x-data attacker and a feature-space decoder attack. Pixel-space version remains NOT RUN; reported as such.
 Running now (declared; precision, not tuning): sealed-3 seeds 0-7 for A3, A14, A21 (files _sealv2) with per-patient hits saved for a paired patient-bootstrap of A21-A3 and A21-A14.

## Cycle 22 RESULTS sealed-3, 8 seeds each, paired patient bootstrap (see CT experiment/ct-results-sealed3.md)
 A3 .190+-.014 (util .703) | A14 .156+-.013 (.687, kept .92) | A21 .159+-.020 (.701) ; raw .474.
 Paired: A21-A3 = -.031 [-.055,-.008] (excludes 0) ; A14-A3 = -.034 [-.062,-.005] ; A21-A14 = +.003 [-.018,+.024].
 => H-Q CONFIRMED on untouched patients: the stronger disease head lets the adapter match A14's linkage without A14's utility cost. Floor .042 still ~3.8x.
 H-AA test (declared before running): A24 = A21 + retrieval-aware term (MSE between latent cosine similarity and label-pattern cosine similarity within the batch, weight 1), A25 = weight 3. Same adversary (w .3), MLP head, 1000 steps, EXTRA dev regime, 3 seeds. Judge with multitask.py on saved latents (dev eval rows): success = kNN kept >= .85 (A3 .93, A21 .64 on sealed-3 - dev numbers computed fresh for A3/A21 too) while top1 <= .095 and linear kept >= .95. Kill otherwise. Configs ~146.

## Cycle 23 RESULTS H-AA (retrieval-aware term) on dev, EXTRA, 1000 steps, 3 seeds
 A24 (w1): top1 .083+-.004 kept .94 | A25 (w3): .106 kept .90 | A21 .083 kept .96 | A3 .110 kept .98.
 kNN-25 task model on saved dev latents: A0 .591 | A3 .576 (kept .84) | A14 .565 (.72) | A21 .564 (.70) | A24 .552 (.57) | A25 .549 (.54). HistGB: A3 1.20 | A21 1.18 | A24 1.20. Age R2 leak: .626 raw | .497 A3 | .442 A21 | .452 A24.
 H-AA REJECTED: the retrieval-aware term made kNN utility WORSE, not better (kept .57/.54 vs .70). CALIBRATION: kNN AUCs are all near chance even for raw (.591 vs .5) so 'kept' ratios on kNN magnify tiny absolute gaps (A3 vs A21 = 0.012 AUC). Do not over-read the kNN rows; report absolute AUC with them.
 Literature (searched): other open 3D CT encoders: TAP-CT (HF fomofo/tap-ct-b-3d, DINOv2-style ViT), Merlin (abdominal, ResNet-152), SPECTRE (open), VoxelFM (self-reported best, release unconfirmed), SuPreM/VISTA3D (segmentation-shaped), COLIPRI/CT-CLIP (chest, trained on CT-RATE -> contaminated). Plan: second frozen encoder to test whether the recipe (A3/A14/A21 vs raw, noise, PCA) generalises beyond CT-FM; check availability first.

## Cycle 24: second-encoder test PAUSED (needs user decision)
 Candidate: TAP-CT-B-3D (HF fomofo/tap-ct-b-3d, DINOv2-style ViT, CC-BY-NC-4.0 = non-commercial research use). Files fetched by curl (no token) into ~/tapct/model; I read the code: no subprocess / socket / urllib / eval / exec / file-write calls, only an XFORMERS env check; weights are safetensors. extract_tapct.py was DRAFTED but never saved or run (the command that would have written it was denied before it executed). Design: CLS-mean + patch-mean over 12-slice axial windows = 1536-d, resumable, same naming as feats/.
 BLOCKED: running it needs trust_remote_code=True (executes the repo's own Python on the VM, which also holds the HF token) and pip-installing transformers; the permission layer denied the first run. Not worked around. Options for the user: (a) allow it (audit above; could also run with the token file moved aside during model load), (b) skip the second encoder and instead use a different feature view of CT-FM, (c) other encoder without custom code.
 Everything else from this cycle is complete and logged. State of play: A3@1000 = robust choice (utility >= .93 under linear/MLP/tree/kNN task models); A21 = lowest linkage at full linear/MLP utility (confirmed on sealed-3 with paired bootstrap, -0.031 [-0.055,-0.008] vs A3) but weaker for kNN retrieval; none anonymous (3.7-4x the scanner floor).

## Cycle 25 (no user answer yet on TAP-CT; NOT touched). Taking option (b): different VIEW of the same frozen CT-FM: the global 512-d vector g instead of the 4096-d spatial map s.
 Question H-G: does the method ranking (A3 / A14 / A21 vs raw, PCA, noise) depend on the feature view? In the 512-d view identity cues from spatial layout are pooled away, so raw linkage should be lower and the headroom for the adapters smaller.
 Declared before running (EXTRA dev regime, dev eval, FEATKEY=g, files tagged _g): A0, B1, noise s1 and s1.5, A3@1000, A14@1000, A21@1000, K0; 3 seeds = 24 jobs. Kill: if A3/A21 do not lower linkage vs raw g by >= 50% at kept >= .90, the recipe is specific to the spatial view.
 Note: the dev regime numbers (s view) to compare against: raw .380 / A3 .110 / A14 .090 / A21 .083 / PCA .314 / noise1.5 .111 (kept .67).

## Cycle 26 RESULTS global-view (CT-FM g, 512-d) comparison, EXTRA dev regime, 3 seeds, files tagged _g. Floor .031.
 | arm | top1 | removed vs raw | kept lin | kept MLP | same-scanner | inversion | sex AUC |
 | A0 raw g | .289+-.008 | 0% | 1.00 | 1.00 | .410 | .089 | .959 | B1 PCA | .217 | 25% | .98 | .97 | .323 | | .966 |
 | noise s1 | .052 | 82% | .68 | .56 | .129 | | .861 | noise s1.5 | **.032** | 89% | .57 | .45 | .101 | | .818 |
 | A3 | .102+-.009 | 65% | .95 | .92 | .200 | .060 | .892 | A14 | .084+-.010 | 71% | .84 | .83 | .170 | | .862 | **A21** | **.077+-.004** | 73% | .90 | .90 | .156 | .046 | .842 | K0 | 0 | | 0 | 0 | | | .5 |
 H-G SUPPORTED (kill rule passed): A3 removes 65% at kept .95; A21 removes 73% at kept .90. Ordering matches the s view (A21 lowest linkage at similar utility; A3 keeps the most utility). The recipe is not specific to the 4096-d spatial map.
 NEW CAVEAT: in this view noise sigma 1.5 reaches the metadata floor (.032) at kept .57 and sigma 1 gives .052 at .68. Whether the adapters beat noise at matched utility here is NOT yet established (noise was only run down to sigma 1). Declared next: noise sigma .25, .5, .75 x 3 seeds (g view) to build the matched-utility noise curve at kept .8-.95. Claim 'adapters beat noise' on this view only if their (kept, top1) point lies below that curve by more than seed spread.
 Still no answer from the user on TAP-CT; untouched.

## Cycle 27 RESULTS noise control in the g view (3 seeds): sigma .25 -> top1 .147 (kept lin .89 / MLP .94); sigma .5 -> .088 (.80/.78); sigma .75 -> .066 (.74/.65); sigma 1 -> .052 (.68/.56); sigma 1.5 -> .032 (.57/.45).
 Matched utility: A21 (.90/.90, .077) vs noise at ~.90 ~ .13-.15 => adapters ~2x less linkage; A3 (.95/.92, .102) vs PCA (.98/.97, .217); A14 (.84/.83, .084) vs noise ~.11 (small edge); at kept <= .80 noise (.088 at .80, .066 at .74) matches/exceeds the adapters. => Adapters add value in the high-utility regime (kept >= ~.85) on both views; for aggressive privacy plain noise is as good.
 Consolidated status written to CT experiment/ct-status.md. Open decision: TAP-CT (user). Next experiment candidates: CLUB/MI penalty; scanner adversary + MLP head at kept >= .95; pixel reconstruction if requested.

## Cycle 28 (no user answer on TAP-CT; untouched). Declared before running (dev eval, EXTRA, s view, 1000 steps, 3 seeds, files _ss):
 H-S2: the part of the leak that is scanner-correlated can be removed without paying utility if the disease head is strong: A26 = A21 + scanner adversary (weight 1). Success = top1 <= .083 (A21) AND same-scanner top1 < .183 AND kept lin >= .95; otherwise rejected (earlier scanner adversaries A4/A9 cost 20-40% utility with a linear head).
 Ablation A27 = A3 (no contrastive adversary) with the 2-layer MLP disease head: how much of A21's gain over A3 (.110 -> .083) is the head alone? Reported as an ablation, no success rule.
 Reference (dev, same regime): A3 .110/.98, A14 .090/.90, A21 .083/.96 (same-scanner .183).

## Cycle 29 RESULTS (dev, EXTRA, s view, 1000 steps, 3 seeds)
 | arm | top1 | util | kept | MLP | same-scanner | sex |
 | A3 | .110+-.005 | .685 | .98 | .682 | .231 | .891 | A21 | .083+-.007 | .681 | .96 | .679 | .183 | .865 | A26 (A21+scanner adv) | .075+-.010 | .644 | **.77** | .632 | .179 | .847 | A27 (A3 + MLP head only) | .117+-.014 | .687 | .99 | .685 | .231 | .887 |
 H-S2 REJECTED: scanner adversary + MLP head still costs 19% of utility for -0.008 linkage; same-scanner linkage unchanged (.179 vs .183) -> consistent with 'leak is anatomy'.
 ABLATION: head alone does not lower linkage (A27 .117 vs A3 .110). Decomposition of A3 -> A21: +contrastive adversary: .110/.98 -> .090/.90 (A14); +MLP head: -> .083/.96 (A21). The stronger head is an ENABLER (recovers the utility that the contrastive adversary costs), not a standalone gain.
 Declared next (literature: CLUB, Cheng 2020; TURBO 2311.06527; GPP 2605.03069): H-M2: penalising an MI upper bound on I(z; patient id) with an identity classifier q(id|z) (a different mechanism from the pair verifier) lowers linkage. A28 = A27 + CLUB term (w .3), A29 = A21 + CLUB (w .3); 1000 steps, EXTRA dev, 3 seeds. Success: A29 top1 <= .075 at kept >= .95 (A21: .083/.96) or A28 top1 <= .095 at kept >= .97 (A27: .117/.99). Kill otherwise. Configs ~158.

## Cycle 30 RESULTS H-M2 (CLUB / MI-bound penalty), dev, EXTRA, s view, 1000 steps, 3 seeds
 A27 (ref) .117 kept .99 | A28 (A27 + CLUB w.3) .120+-.003 kept .92 | A21 (ref) .083 kept .96 | A29 (A21 + CLUB w.3) .142+-.009 kept .89 (same-scanner .257 vs .183).
 H-M2 REJECTED (kill rule): no improvement, and A29 is clearly worse on both axes. Untested explanation: the identity classifier q(id|z) has ~1760 classes with 2-4 scans each, so it memorises the fit patients and gives a noisy/uninformative MI signal; a pair-based or embedding-based bound might behave differently. Cost of finding out is low but I do not expect a large gain.
 Tally of rejected ideas: L* erasure, A12/A13 as free wins, 5000 steps, input regularisation (A16-18), retrieval term (A24/25), 512-d latent (A22), loss weight (A23), scanner adversary (A26), CLUB (A28/29). Positive results: A3@1000, A14, A21 (confirmed held-out), two-view generalisation, matched-noise comparison.
 ASSESSMENT: this recipe family looks saturated near linkage ~.08 at kept ~.95 (3.7x floor). Further gains probably need a different lever (different encoder -> blocked pending user, more/other-hospital data, or a different obfuscation stage such as image-level), not another adversary tweak. Recommend to the user: pause hyper-tweaking; decide TAP-CT; consider external data.
