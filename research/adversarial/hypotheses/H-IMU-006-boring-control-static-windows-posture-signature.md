# H-IMU-006: Boring control: standing and sitting windows (no gait) identify subjects from phone orientation alone; de-meaned they do not

- Focus: R-IMU | Kind: boring control (the IMU analogue of CREMA-D's recording-channel signal) | GPU-h: 0 (CPU hand features) | Prior: 0.70
- Depends on: MotionSense (static trials: std and sit, 2 trials each per subject; the trial numbering 6/14 and 5/13 is from the dataset description and must be verified against the files); stress-test measurements: dynamic-activity identity does NOT depend on attitude / gravity means (all-12-channels RF 73.6% vs no-attitude de-meaned 78.5%).

## Statement
On the static activities only (std, sit; no gait cycle): (1) with the attitude (roll, pitch, yaw), gravity and mean channels retained, RF / LightGBM trial identity is >= 30% (>= 7x chance 4.17%); (2) with only userAcceleration + rotationRate, de-meaned per window, it is <= 12% (<= 3x chance); (3) the dynamic-activity result is unchanged by dropping attitude and gravity (replicating the stress test). So the identity a suite finds on static windows is a placement / session signature (phone orientation in the pocket, constant within a recording), which is not gait and which a defence might remove trivially (drop attitude + gravity) or which a suite built on all 12 channels would wrongly credit to gait.

## Mechanism
The phone sits in a front pocket in a subject-specific orientation that is constant over a trial and nearly constant across the subject's trials in one sitting; mean gravity direction and attitude carry that orientation. MotionSense has no confirmed cross-session data, so this is the same-session offset, the analogue of CREMA-D's 76% recording-channel identity.

## Experiment (exact)
- Data / units: MotionSense, 24 subjects, static activities only: 4 trials per subject (2 std, 2 sit) = 96 trials; trial unit; leave-one-trial-out cross-fit (each subject has 2 trials per activity, so train on the other trial of the same activity plus the other activity: report both "same-activity" and "cross-activity" training); seeds 3; subject-cluster and trial-cluster bootstrap B = 2000; MDE80 printed (large at n = 96: state it; the contrast (1) vs (2) is expected to be large).
- Feature views: (A) 12 channels incl. means; (B) attitude + gravity means only (6 numbers); (C) userAcceleration + rotationRate de-meaned per window (std, MAD, percentiles, 16 FFT bins); (D) dynamic activities with and without attitude / gravity (replication).
- Attackers: RF 500, LightGBM 300, LR, kNN; trial score = mean log-prob over windows.
- Controls: shuffled labels and constant input at chance; dynamic-activity raw reference above static-only.
- Test-unit design: leave-one-trial-out over the 96 static trials (all tested once); no seeds-vs-units issue because folds ARE the test units.

## Prediction and falsifier
- Predicted: (A) 35-80%, (B) 30-75%, (C) 5-12%, (D) within +-5 points of 73.6 / 78.5%.
- FALSIFIED if (A) <= 12% (no posture signature: attitude / gravity are not an identity channel in MotionSense), or (C) >= 25% (static windows carry gait-independent identity through sensor noise or small motions; then every static-window defence must be re-examined).

## Cost / dependencies
Zero GPU; one engineer-hour on top of `ms_floor2.py`. Protects R-IMU from claiming gait-leak reductions that are really orientation-channel removal.

## Prior
0.70. Phone orientation in a pocket is a strong same-session constant; the uncertainty is its magnitude with 96 trials.

## Cheapest floor that could kill it
Not applicable as a floor; it IS a control. If (A) holds, then the right IMU floor is "drop attitude, gravity and means" for static windows, and any defence's static-activity advantage must be measured against it.
