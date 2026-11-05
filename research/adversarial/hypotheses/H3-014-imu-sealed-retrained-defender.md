# H3-014: Wearable IMU: identity-targeted defenders RETRAINED inside the programme on the salted sealed folds (a re-implemented anonymising autoencoder and an in-loop-adversary obfuscator) are dominated by the activity-label floors, and the frozen-versus-retrained attacker gap replicates

- Focus: R-IMU, "arm B" of H.md 5.9 | Kind: learned defender vs floors, sealed | GPU-h: 3.5 | Prior: 0.08 that any retrained defender beats the floor envelope by the margins; 0.85 that the attacker-protocol gap replicates
- Depends on: Builder H's `common.sealed_fold_table`, the frozen floor list (FLOOR_LIST.json sha256 fc1ba9c8...c5064) and the custodian's salt `sealed/h_imu_salt.txt` (committed after the list was frozen, ca9b9df); the AAE paper (arXiv 1810.11546) for the re-implementation (no training code is shipped; weights and evaluation notebooks are); the burned-fold result (msda 0.826 / 0.930, rep 0.878 / 0.973, floors lda3 0.365 / 0.935, act60_r1 0.094 / 0.949, predlabel 0.049 / 0.950). The salt applies only to defenders retrained inside the programme: this is that arm.

## Statement
On MotionSense (24 subjects, 6 activities, 360 trials) with the salted rank folds (per subject x activity, trials ranked by sha256 of subject|activity|trial|salt, fold = rank mod 3) and a 3-fold cross-fit (every trial in a sealed test fold exactly once; defenders, floors and attackers refit on the other two folds):
(a) a re-implemented identity-targeted anonymising autoencoder (AAE-R) does not reproduce its published identity figure against a retrained attacker suite: under the frozen raw-trained identity CNN (the paper's attacker) identity is <= 10% of windows, under the retrained suite (linear, kNN, RF, LightGBM, MLP, 1D-CNN, flat MLP, transformer) trial-level worst-case identity L >= 0.60 at activity accuracy >= 0.92;
(b) AAE-R and the in-loop-adversary obfuscator (OBF-A: window MLP / 1D-conv autoencoder with an activity head and a gradient-reversal identity term against two in-loop attackers, lambda in {0.3, 1, 3}) are DOMINATED by the label-only floor envelope on the pooled sealed folds: Delta_L = L_floor(U_defender) - L_defender <= -0.15 (the programme's m_L) and/or Delta_U <= -0.0285 (m_U), with the 95th-percentile bound below the margin; P(any learned defender instead beats the envelope with 95% lower bound > margin) = 0.08;
(c) static strata (sit, std, 96 trials per fold-pooled set) show identity above 5x chance (0.21) for every learned defender and for undithered floors but not for `predlabel`.

## Mechanism
Same as the burned-fold audit but with defenders refit on separate folds: identity lives in per-subject posture, sensor placement and gait; an autoencoder that preserves activity must keep the window fine structure that identifies the subject, a 3-d activity-LDA code throws it away. The in-loop adversary at lambda up to 3 will trade activity accuracy for lower leakage along the same frontier a dither floor traces. The re-implementation is "in spirit" (loss details of the shipped msda / rep are not public), so the verdict is on the recipe, not on the original weights.

## Experiment (exact)
- Data: MotionSense, six activities (dws, ups, wlk, jog, sit, std) so static strata exist; 50 Hz windows of 128 samples as the shipped pipeline; hand-feature view (44 features) plus raw windows.
- Defenders per fold-pair (train on two salted folds, release the third): AAE-R (conv AE: encoder -> bottleneck -> decoder, loss = reconstruction + activity CE on the code, with the paper's "replacement" mechanism for the identity-bearing part as far as the paper specifies it; seeds 2); OBF-A (3 lambdas, 1 seed, adversary population = linear and MLP identity heads on hand features and on the code, 3 restarts); both have a dithered sibling (r = 0.5, 1). Floors refit per fold: activity-LDA-k (k = 2, 3) with r in {0, 0.5, 1}, `predlabel` (2-bit activity decision), `act60`, `act60_r1`, `act8_r1` (the frozen list, unchanged). R0 gate: AAE-R must reproduce <7% identity against its own frozen raw-trained CNN on its own training-fold windows, else it is "recipe not reproduced" and (a) is not evaluated.
- Attackers: the H suite retrained on every release; L = trial-level worst case; held-out SVM and transformer as the held-out view (reported). Utility U = window-level activity accuracy of a retrained suite.
- Statistics: subject-cluster bootstrap (24 subjects) with trial resampling, B = 2000; margins m_L = 0.150, m_U = 0.0285 fixed by H-A3 and NOT changed; the realised paired MDE80_L is printed (ESTIMATED 0.06-0.07 at n = 360 by 1/sqrt(n) from 0.121 at 96).
- Seeds vs test units: test units = the 360 trials, each tested once across the 3 salted folds (cross-fit); defender seeds (2 for AAE-R, 1 per lambda for OBF-A) replicate training noise; folds replicate test units; subjects (24) are the cluster for the bootstrap. Each fold is sealed: one ledger entry per (release, fold, config); floors first, defenders second.

## Prediction and falsifier
- Predicted: AAE-R frozen-CNN identity 3-10%, retrained suite 0.55-0.85 at U 0.90-0.95; OBF-A best-lambda L 0.35-0.60 at U 0.92 (the floors reach 0.365 at 0.935 with `lda3`); Delta_L(AAE-R) <= -0.2; Delta_L(OBF-A) in [-0.30, +0.05]; static identity of `lda3` and AAE-R >= 0.21, `predlabel` <= 0.08.
- FALSIFIED (a learned defender is competitive) if any retrained defender has Delta_L >= +0.15 or Delta_U >= +0.0285 with the 95% lower bound above the margin on the pooled sealed folds with 3 folds in sign agreement, AND the dithered sibling does not reproduce it. Inconclusive if MDE80_L > 0.075 after the cross-fit (then the statement is "unresolved", not a null). The frozen-versus-retrained gap is falsified if the retrained suite reads < 0.30 on AAE-R.

## Cost / dependencies
3.5 GPU-h: AAE-R 3 folds x 2 seeds x ~12 min = 1.2 h; OBF-A 3 lambdas x 3 folds x ~10 min = 1.5 h; torch attackers 0.8 h; tabular attackers on CPU (8 threads). Needs the H3-016 and H3-017 controls first if their outcomes would change the protocol (evidence budget, adjacency); otherwise independent.

## Prior
0.08. Both burned-fold learned defenders were dominated by 0.28-0.78 in L; a retrain with an adversary is the one version not yet tested. The sealed salt exists and the floors are frozen, so a null here is the first sealed IMU claim.

## Cheapest floor that could kill it
`act60_r1` (0.094 / 0.949), `lda3` (0.365 / 0.935) and `predlabel` (0.049 / 0.950). One caveat from H3-016: the dithered floors may not compose with longer evidence; `predlabel` is the dither-free floor and is in the envelope.
