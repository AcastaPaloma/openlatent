# H-S-003: Evidence budget: pooling windows within one recording is nearly worthless, pooling across independent recordings is decisive (session offset vs independent windows)

- Focus: S (instrument module; contrarian candidate E, scored 26/30) | Kind: premise / measurement | GPU-h: 0 (KTH from stored `preds.npz`; MotionSense on CPU hand features; CXR arm lives in H-CXR-006) | Prior: 0.50 (for the mixed outcome; see below)
- Depends on: HF `KuanKuanKuan/openlatent-privacy` `harness/<ds>/eval/<release>/preds.npz` + `result.json` (`test_ids`, 78 files); MotionSense windows.

## Statement
Let acc(K) be top-1 identity when K evidence units are pooled for one probe (sum of calibrated log-probabilities). Define two pooling levels: WITHIN (K clips / windows from the same recording or trial, which share a session nuisance) and ACROSS (K clips / windows each from a different recording of the same person, ideally different scenario / activity). Hypothesis: on programme-1 canonical-stick-figure x16 (KTH, best-of-2 13.9% at K = 1), WITHIN pooling at K = 6 gains <= 6 points (inside the KTH paired MDE80 of 5-8), while ACROSS pooling at K = 6 gains >= 15 points, a ratio >= 2. The measured N* (smallest K reaching 50% top-1, or TAR 50% at FAR 1%) is >= 3x the independent-window prediction N*_pred = ln(C) / I_window (C = 25 persons, ln C = 3.2 nats, I_window from calibrated per-window V-information) for WITHIN pooling and <= 1.5x for ACROSS pooling. Same pattern expected on MotionSense (24 subjects): pooling windows within a trial adds little over the trial score already used, pooling across a subject's trials of different activities adds much.

## Mechanism
Evidence accumulation is either decisive or irrelevant depending on the shared nuisance fraction (toy simulation, contrarian sec 6: per-window 13.5% becomes 55% at K = 10 and 98% at K = 50 under independence, but stays 20% / 15% when a shared session offset carries 50% / 90% of the variance). Real programme-1 numbers are per clip; real attackers pool. Nair et al. (2407.18378) pool and reach 100% per user at settings where per-sample accuracy falls, and state that nobody varied the probe size.

## Experiment (exact)
- KTH: per-clip probabilities of retrained attackers (frame_r50, r3d; dino variants where present) for releases black, raw, pose, pose_canon, pose_canon_down16/32, bdq_down64/32, motion_down32, on kth_ilic and kth_scenario. `test_ids` such as `person01_boxing_d3_1` give person, action, scenario, clip. Pooling groups: WITHIN = clips of one person-action-scenario recording; ACROSS-ACTION = recordings of one person in the same scenario, different actions; ACROSS-SCENARIO = recordings in different scenarios. K = 1..20 with replacement-free random subsets, 2000 draws per K.
- Pooling rules: sum of log-probabilities with one temperature per release fitted by cross-fit over held-out recordings; mean probability as a check; a learned scalar temperature only (no aggregator can be trained on 25 persons).
- MotionSense: RF / LightGBM hand-feature window probabilities from the rank-fold cross-fit (SUITE-IMU-8 subset); WITHIN = windows of one trial; ACROSS = windows from distinct trials of the same subject (different activities, then different trials of the same activity); caveat stated in the file: MotionSense has no confirmed cross-session data, so ACROSS here is across-trial, not across-day.
- Metric: top-1 and TAR@FAR 1%; N* with recording-cluster bootstrap (KTH 150 recordings) and trial-cluster bootstrap (MotionSense); design effect reported.
- Controls (must stay at chance for every K): black release, shuffled-label predictions, pooled-over-random-persons draws. Failure of a control means the pooling code leaks.
- Test-unit design: pooling is within the stored test split; for KTH the 150 test recordings are the units (cross-fit rotation is NOT possible from stored files, so seeds / attacker families replicate training noise only; stated). MotionSense uses the 3 rank folds.

## Prediction and falsifier
Three outcomes, pre-registered: (A) mixed (WITHIN flat, ACROSS decisive) 0.50; (B) both flat (acc(10) - acc(1) < MDE80 in >= 90% of cells, "effective K is about 1, the session is the unit") 0.25; (C) both decisive (R <= 1.5 for WITHIN too, per-clip numbers drastically understate the threat) 0.25.
- FALSIFIED (for the mixed claim) if outcome B or C obtains. B still changes the suite's unit of analysis to the recording / session; C means every programme-1 per-clip number must be re-expressed as N*.
- Secondary: |N*_pred - N*_meas| / N*_meas <= 0.5 in independent-pooling cells; if not, the information-theoretic prediction is not usable as a certificate field.

## Cost / dependencies
Zero GPU; about one engineer-day. Needs the HF snapshot (38 MB harness subset) and the MotionSense rank-fold predictions.

## Prior
0.50 for the mixed outcome; 0.75 that at least one of the two pooling levels moves acc by more than MDE80 (so the module is worth keeping). Toy simulation supports both regimes; KTH scenarios share camera and clothing, which argues for partial dependence even ACROSS-ACTION.

## Cheapest floor that could kill it
Outcome B in every cell (pooled K = 10 inside MDE80 of K = 1): then E reduces to a one-line note that the recording is the independent unit, no module needed.
