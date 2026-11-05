# H3-002: The break-even is governed by the SIZE OF THE LABEL SPACE (K classes versus code width d), not by label count: a K-sweep on JUMP predicts it, and the same curve on wearable activity labels (K = 4-6) shows privacy defenders never break even

- Focus: JUMP label budget + R-IMU | Kind: mechanism / dose-response; "label-budget curve for privacy defenders" (NEW idea) | GPU-h: 2 (JUMP arm) + ~0.5 CPU (IMU arm) | Prior: 0.60
- Depends on: F3 arms and aggregator (F.md section 8), the burned-fold O / A / B development partition of Builder H (NOT the sealed salted folds); independent of F4 and H3-001.

## Statement
Part A (JUMP). Holding the adapter and d = 8 fixed, the advantage of the supervised nonlinear adapter over the LDA fitted on the SAME labelled wells, dU_same(K), rises monotonically with the number of labelled classes K at roughly +0.04 to +0.05 mean AP per decade of K: dU_same(K) = alpha + beta * log10(K), beta in [0.025, 0.07], R^2 >= 0.9 over K in {30, 100, 300, 1000, 3000, 10000, 32000}. At fixed K = 3000, varying wells per class (1, 2, all) changes dU_same by < 0.010 (K, not sample count, is the driver).
Part B (wearable). For activity labels (K = 4-6, code width d >= 3) the label-supervised linear floor (activity-LDA-k, optionally dithered) reaches >= 95% of its 100%-label utility with <= 1% of the training windows labelled, and a nonlinear learned obfuscator with the same labels never exceeds the floor envelope by >= 0.03 window-level activity accuracy at matched trial-level identity leakage at any budget in {0.1, 0.3, 1, 3, 10, 30, 100}%: break-even budget = "none" (censored above 100%).

## Mechanism
A linear label-supervised code has at most K - 1 informative directions. When K is much larger than d the linear map must discard class structure, and a nonlinear map can keep it; when K - 1 <= d the linear code already spans all the label information and the nonlinear map has nothing extra to represent. F3's fixed-arm numbers (s0, d = 8, same-label LDA comparator) are +0.010 (K about 300), +0.031 (about 600), +0.056 (about 3,500), +0.107 (about 32,600), which is the shape this predicts. Programme 1's null results (PathoROB, CREMA-D, single-cell coarse labels) all had K small relative to d, so one rule explains both the JUMP win and the nulls, and predicts the wearable and chest X-ray label-supervised settings (K = 4-6, 14) never win.

## Experiment (exact)
Part A.
- Data: JUMP S5 (Arevalo-5), view A (encoder 0), split s0 (the F3 Fit / V / H construction, production compounds only; no Target2 classes so that K is controlled). Regimes `K{30,100,300,1000,3000,10000,32000}`: K randomly chosen T' production compounds with >= 2 Fit wells, ALL their wells labelled; nested subsets by md5. Control regimes at K = 3000: `K3000_w1`, `K3000_w2` (1 or 2 wells per class kept; classes need >= 2 wells for SupCon, so w1 uses 2 wells and w2 uses 4 where available; state the realised wells per class).
- Arms: SupCon MLP D -> 1024 -> 512 -> d, d = 8, r in {0, 0.5}, 8,000 steps, no adversary (F2/F3 code). Comparator 1: `ldaM_K_8` (PCA-256 then shrinkage-0.1 LDA on exactly the labelled wells, 8 directions). Comparator 2: E_full (floors plus ldaP at 100% labels), for context. Matched leakage and V-selection as F3 6.5.
- Metric: dU_same(K) = U_H(adapter V-selected) - U_H(ldaM_K_8 at L_V <= L_V(adapter)+0.05), compound-bootstrap and plate-cluster intervals. Fit alpha, beta by weighted least squares over the 7 K points x 3 label-draw salts (draws replicate the subset).
- Trained check: memorisation-calibrated criterion (iii) of F3 8.5, per K.
- Seeds vs test units: H compounds (about 40k) for U, plates for L; label draws (3 md5 salts) replicate the labelled subset; one adapter seed per cell (not yet replicated at seed level); the compound split is s0 only (this is a mechanism map, not a verdict).
Part B.
- Data: MotionSense, the four PD-protocol activities (so the shipped features and Builder H code apply), development partition only (fold O = trials 11, 12, 15, 16, 96 clean trials, plus development folds A / B): the sealed salted folds are NOT touched (they belong to H3-014).
- Arms: activity-LDA-3 (+ N(0, (r sd)^2) noise, r in {0, 0.5, 1}) vs a learned obfuscator (MLP AE with an activity head, d in {3, 8}, r as dither) trained on the same labelled windows at fractions f; both unlabelled windows are unused. Labelled sets are nested random subsets of training-fold windows (grouped by trial so that no trial is split).
- Metric: window-level activity accuracy U and trial-level worst-case identity L with the H suite (linear, kNN, RF, LightGBM, MLP, 1D-CNN, flat MLP; subject bootstrap); dU at matched L (tolerance 0.05) between the learned arm and the floor envelope (activity-LDA family, predlabel).
- Seeds vs test units: 24 subjects (cluster), 96 trials in fold O; MDE80_L about 0.12 (Builder H), so only large leakage differences resolve; the budget claim is on U at matched L, whose MDE80 is ~0.03 (window-level, thousands of windows). 3 seeds for the learned arm.

## Prediction and falsifier
- Predicted: beta = 0.045 (CI [0.025, 0.07]); dU_same(30) <= +0.005; wells-per-class control |effect| < 0.010. Part B: floor at f = 1% has U >= 0.95 x U(100%); learned minus floor <= +0.02 at all f; break-even "none".
- FALSIFIED (the K-rule is wrong) if beta < 0.015 or R^2 < 0.7 (flat or noisy), OR the wells-per-class control moves dU_same by >= 0.015 at fixed K (count, not K, matters), OR in Part B the learned obfuscator beats the floor envelope by >= 0.03 U at matched L in >= 2 of 3 development folds at any f (the nonlinear map does help at small K).

## Cost / dependencies
JUMP arm: 7 K values x 3 draws x 4 adapter arms is about 84 adapters, ~1.5 min each, plus ldaM and attackers: ~2 GPU-h. IMU arm: ~0.5 CPU-h tabular plus 3 small torch fits.

## Prior
0.60 (Part A 0.75, Part B 0.80). Reasons: the four F3 points already lie on a line in log K; the residual risk is that d, dither and the number of wells (which co-vary with K in F3) confound it, which is what the controls test. Part B is the standard "LDA is saturated at C - 1 dims" argument, measured at 96 trials.

## Cheapest floor that could kill it
For Part A, the wells-per-class control (a pure sample-count account) and the label-permuted memorisation arm per K. For Part B, `predlabel` (release the 2-bit decision) which is flat in f and already ties PrivDiffuser.
