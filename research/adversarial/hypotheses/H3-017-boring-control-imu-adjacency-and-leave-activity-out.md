# H3-017: Boring control for every wearable identity number: how much of "re-identified" is same-session adjacency and same-activity posture, measured by trial-interleaved versus block-gapped versus leave-activity-out protocols

- Focus: R-IMU (protocol validity) | Kind: boring control / protocol sensitivity | GPU-h: 1 (mostly CPU) | Prior: 0.70 that leave-activity-out identity of raw is <= 0.60 while interleaved is >= 0.95
- Depends on: Builder H's tabular and torch suite and `common.py` folds; MotionSense (all 360 trials, six activities); the shipped msda / rep / PrivDiffuser releases exist only for fold O (trials 11, 12, 15, 16; the shipped checkpoints trained on trials 1-9), so the defender arm is descriptive. Uses development material only; the salted sealed folds are not touched (H3-014).

## Statement
All programme-2 identity numbers are "same-session closed-set identity": MotionSense has no across-day sessions, trial order interleaves activities, and the headline protocols train the attacker on other trials of the same subject and activity. Static-window identity (sit, std) of 7-8x chance and 48% after de-meaning is therefore partly posture/placement of that session. Four protocols on the raw release and on the floors (refit per protocol):
P-int: leave-one-trial-out within subject x activity (train on the other trials of the same activity; windows of the held-out trial scored; the interleaved case, which the salted rank folds also resemble).
P-block: train on trials 1-9, test trials 11-16 (the burned fold O analogue; a time gap of at least the 10th trial).
P-LAO: leave one ACTIVITY out: train the identity attacker on the five other activities of all subjects, test on the held-out activity (never seen as a class-conditional distribution).
P-S2D / P-D2S: train on static (sit, std), test on dynamic and vice versa.
Claims (trial-level worst-case suite accuracy, raw release): P-int >= 0.95; P-block 0.85-0.97; P-LAO <= 0.60; P-S2D <= 0.25; P-D2S <= 0.25. Floors: `lda3`, `act60_r1`, `predlabel` change by <= 0.05 across protocols (they carry little subject information to lose). The shipped `msda` and `rep` (fold O, descriptive; the attacker trained on the released windows of other activities of fold O) fall from 0.83 / 0.88 to <= 0.40 under P-LAO. Ordering floors (U >= 0.92) below msda / rep is the same in every protocol.

## Mechanism
Identity in a single session is the sum of gait (activity-specific), body and sensor placement (activity-independent: the phone sat in the front trouser pocket), orientation habits and session state. P-LAO removes everything an attacker could learn from the same activity's posture and leaves what transfers across activities; P-block removes the adjacent-trial overlap; P-int measures what both together give the attacker. The gap between P-int and P-LAO sizes how much of the headline is not a durable biometric. It cannot change the floor-versus-defender direction (the floors carry less identity in every protocol) but it changes what "re-identified 83%" means and whether H3-014's sealed rank folds (interleaved) overstate leakage for all arms equally.

## Experiment (exact)
- Data: MotionSense 24 subjects x 15 trials (dws 1, 2, 11; ups 3, 4, 12; wlk 7, 8, 15; jog 9, 16; sit 5, 13; std 6, 14); windows of 128 samples, stride 64; hand features (44) and raw windows.
- Releases: `raw`, `lda3`, `lda3_r1`, `act60_r1`, `predlabel`, `demean`, `ds8`; on fold O only (4 dynamic activities, 96 trials) the shipped `msda`, `rep`, `pd_pub_s0`.
- Attackers: linear, kNN, RF, LightGBM, MLP (3 seeds) and 1D-CNN (raw windows); worst case chosen on training-side validation windows (never on the test), scored once; trial-level pooled (about 70 windows) AND window-level reported (K = 1 column).
- Metrics: accuracy normalised by chance (1/24) and raw; subject-cluster bootstrap B = 2000 (24 clusters; MDE80 ~0.12 at 96 trials, ~0.06-0.07 at 360 trials by 1/sqrt(n)); for P-LAO each held-out activity has 24 x (2-3) trials, so per-activity intervals are wide; report pooled over activities with activity as a stratum.
- Phone-pocket confound check: the static activities have the lowest motion, so a "static posture" attacker on the mean accelerometer vector (gravity direction, 3 numbers) is run as the cheapest static floor; if it reaches >= 50% of the full suite's static accuracy, static identity is orientation, not motion.
- Seeds vs test units: units = trials (360) for P-int / P-LAO / P-S2D; 96 trials for P-block and the defender arm; subject cluster (24) for every bootstrap; attacker seeds replicate training noise; protocols differ in the training set, not in the test units.

## Prediction and falsifier
- Predicted: raw: P-int 0.97, P-block 0.92, P-LAO 0.45 (range 0.25-0.70), P-S2D 0.12, P-D2S 0.20; gravity-direction attacker explains >= 60% of static identity; `lda3` 0.37 -> 0.30 across protocols; msda under P-LAO 0.20-0.40.
- FALSIFIED (identity is a durable biometric, not a same-session artefact) if raw P-LAO >= 0.85 and P-S2D >= 0.5. Falsified the other way (identity is nearly entirely same-activity) if raw P-LAO <= 0.15 (then H3-014's and H3-015's headline numbers need an explicit durable-identity companion measure).

## Cost / dependencies
~1 GPU-h-equivalent (tabular CPU ~2 h; torch attackers 0.5 GPU-h). No download.

## Prior
0.70 for the predicted ordering. The static-identity result (F7, H 5.6) already shows an orientation channel; the magnitude of the cross-activity drop is the unknown.

## Cheapest floor that could kill it
The gravity-direction attacker (3 numbers per window) for the static strata, and `predlabel` for all protocols.
