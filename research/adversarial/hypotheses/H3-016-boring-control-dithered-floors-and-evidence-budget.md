# H3-016: Boring control that could kill the best wearable floors: independent per-window dither does not compose with evidence, so act60_r1 and lda3_r1 look private only at the evidence budget of one 70-window trial

- Focus: R-IMU, Instrument S (evidence-budget module) | Kind: boring control on the floors themselves | GPU-h: 0.5 (CPU-dominated, ~1.5 CPU-h) | Prior: 0.50 that account-level pooling at least doubles the normalised leakage of act60_r1
- Depends on: Builder H's stored releases or deterministic floor generators (`FLOOR_LIST.json`, seeds fixed) and the retrained tabular suite; S's evidence-budget module (H-S-003; programme-1 result: pooling clips within a session saturates, across sessions it does not, E3); N's cautions that a missing LEAK is not a clean bill. Burned/development fold O only (the sealed salted folds belong to H3-014).

## Statement
H's best floors are `act60_r1` (L 0.094 [trial-level], U 0.949), `act8_r1` (0.052 / 0.928), `lda3_r1` (0.111 / 0.726) and the dither-free `predlabel` (0.049 / 0.950). Their privacy rests on independent Gaussian noise added per window. Window-to-trial pooling in the H numbers already shows how much pooling matters for undithered codes (window-level to trial-level L: lda3 0.207 -> 0.365, act8 0.215 -> 0.510, act60 0.452 -> 0.719, msda 0.615 -> 0.826) but trial-level pooling is only ~70 windows. Claims on fold O (4 dynamic activities, 96 clean trials, 24 subjects):
(a) Account-level pooling (all of a subject's test trials, 4 in fold O, K about 280 windows) raises the dithered floors: L(act60_r1) >= 0.19 (>= 2x its trial-level 0.094) and L(lda3_r1) >= 0.20 (>= 1.8x 0.111), with 95% subject-bootstrap lower bounds above their trial-level values;
(b) the dither-free `predlabel` and `predlabel_pd` stay <= 0.08 at every K (no per-window noise, no subject information beyond the activity sequence);
(c) the long-evidence limit is bounded by the UNDITHERED code: L_account(act60_r1) <= L_account(act60) and L_account(lda3_r1) <= L_account(lda3), and the gap closes as K grows (the ratio L(r1)/L(r0) rises monotonically in K);
(d) the ordering "floors with U >= 0.92 below the released AAEs" survives at every K (msda / rep rise to >= 0.95 at account level), so this control changes WHICH floor is called best, not the NO-GO-MATCHED verdict.

## Mechanism
Per-window i.i.d. noise lowers the SNR of the code, not its information content: the attacker's log-likelihood ratio adds over windows, so the effective d' grows like sqrt(K) until the subject-specific part of the underlying code (which for act60 is large: undithered L 0.719 at trial level) is fully revealed. A deterministic quantised release carries only the activity decision and its subject information is whatever the activity sequence reveals, which does not grow with noise-averaging. Programme 1's evidence-budget result (KTH: cross-session pooling +28.9 points vs within-session +8.5) says the within-session nuisance caps the gain from windows of one recording; account-level pooling across a subject's different activities may escape that cap.

## Experiment (exact)
- Releases (regenerated from the frozen generators, fixed seeds, 2 noise seeds for dithered arms): `predlabel`, `predlabel_pd`, `act8`, `act8_r1`, `act60`, `act60_r1`, `lda3`, `lda3_r0.5`, `lda3_r1`, `lda2`, plus the shipped `msda`, `rep`, `pd_pub_s0` on fold O (identity read through the same suite).
- Attackers: the H suite as frozen (linear, kNN, RF, LightGBM, MLP, 1D-CNN where the release is window-shaped), retrained once per release on the training windows; per-window class posteriors stored; evidence aggregation by summing log-posteriors over a test bag (S's E3 rule), bag = K windows.
- Evidence axes: (i) K in {1, 2, 4, 8, 16, 32, all (~70)} windows of ONE trial, 50 draws each, both contiguous-block and random-window draws (contiguous respects session structure; the gap between the two draws is itself a measurement of within-trial nuisance); (ii) account level: all four test trials of a subject pooled (24 units, K about 280); (iii) account level with random-window subsets {70, 140, 280}.
- Metric: L_norm(K) = (acc - 1/24)/(acc_raw - 1/24) with acc_raw from the same K-pooling of the raw release (so a saturated raw does not mask growth); worst case over the suite chosen on the training-fold validation windows, scored once; subject-cluster bootstrap B = 2000. The information-accumulation fit L(K) from K <= 16 predicts K = all and K = 280 (BoN-style extrapolation check); if the fit does not predict within 0.05, the extrapolation is not claimed.
- Controls: `ctrl_const` and shuffled-label attackers at chance at every K (the tightened eligibility rule of S E3: bags need >= max(10, 40%) eligible subjects, or the black control fails); `oracle_act` at chance.
- Seeds vs test units: test units = 96 trials / 24 subjects (fold O); K draws, noise seeds (2) and attacker seeds (3 for the MLP) replicate sampling and training; the account-level claim has only 24 units: MDE80 about 0.2, so (a) is a LARGE-effect claim and the interval will be wide. If the lower bound is not above the trial-level value, the result is reported UNRESOLVED, not NO.

## Prediction and falsifier
- Predicted: L_account(act60_r1) 0.25 (range 0.12-0.45), L_account(lda3_r1) 0.22 (0.12-0.38), `predlabel` <= 0.07; log-log slope of (L - chance) vs K between 0.3 and 1.0 for dithered arms; ratio L(r1)/L(r0) rising from 0.15 (K = 1) toward 0.4-0.6 at account level for act60.
- FALSIFIED if L_account(act60_r1) <= 0.12 AND L_account(lda3_r1) <= 0.15 with the upper bounds below 0.19 / 0.20 (dither composes weakly; the floors hold at account level). FALSIFIED (b) if `predlabel` rises above 0.12 at account level (the activity sequence itself identifies subjects).

## Cost / dependencies
~1.5 CPU-h tabular plus 0.5 GPU-h for the 1D-CNN row (optional). Needs the released-window arrays on the VM. Independent of H3-014 but its result is a condition for reading H3-014's dithered floors (stated).

## Prior
0.50 for (a). The mechanism is robust; the doubt is the unit count (24 subjects, interval half-width ~0.15) and the within-session cap seen in KTH.

## Cheapest floor that could kill it
`predlabel` (dither-free; stays flat). The control is itself a statement about which floor to trust; it cannot be "killed", only found unresolved.
