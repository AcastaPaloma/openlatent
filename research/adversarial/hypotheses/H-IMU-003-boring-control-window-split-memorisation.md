# H-IMU-003: Boring control: window-level splits inflate both the attribute attacker and the identity attacker, so no claim from a window-split protocol is carried

- Focus: R-IMU | Kind: boring control (split effect) | GPU-h: ~0.5 (one CNN, rest CPU) | Prior: 0.75
- Depends on: MotionSense; PrivDiffuser `dataset_loader.py` behaviour (MEASURED in the stress test: per-user groups over `range(24)` in both train and test, i.e. all 24 users appear in both); C-AAE splits user-ID data "randomly ... 4:1" with all users in both parts.

## Statement
(a) Gender attribute: a gender attacker with PrivDiffuser's architecture, trained and tested on windows from all 24 users (random window split), reaches >= 90% accuracy on raw data; the same architecture under a user-DISJOINT split (train on 16-18 users, test on 6-8, 24 users total, 14 M / 10 F, chance = majority 58%) reaches <= 70% balanced accuracy, a drop >= 20 points. Hence the "49.96% after privatisation" is a statement about a memorising attacker, and gender claims on 24 users cannot discriminate defenders (MDE80 0.29 balanced accuracy).
(b) Identity: raw-data RF identity under a random 50%-overlap window split is >= 10 points (predicted 10-25) above identity under the trial-held-out split (MEASURED 73.6-78.5% for trial-held-out hand-feature RF/LR), because neighbouring windows are near-duplicates; for a defended release (activity-LDA-5, quantiser) the inflation is >= 10 points as well, so a defence "reduction" computed with a window split is smaller than the trial-held-out reduction.
(c) The same inflation exists for the C-AAE style random split.

## Mechanism
Window-level splits put neighbouring windows of one trial in both train and test; the attacker memorises a trial's drift and sensor characteristics. User-disjoint splitting removes that for attributes; trial-held-out splitting removes it for identity (still same-session: no open IMU set has several independent sessions per subject, so identity here is same-session and time-blocked; stated in every claim).

## Experiment (exact)
- Data / units: MotionSense, 24 subjects. Gender: user-disjoint 3-fold cross-fit over subjects (stratified by gender), all subjects tested once; balanced accuracy; subject-cluster bootstrap B = 2000. Identity: split variants (i) random window split, (ii) block-in-trial split (first 70% of each trial for training, last 30% for test), (iii) trial-held-out (rank folds). Attackers: PrivDiffuser CNN (4 conv + 3 FC), RF and LR on hand features. 3 seeds inside folds.
- Releases for (b): raw, activity-LDA-5 code, quantiser b = 3, PrivDiffuser release (if available from H-IMU-001).
- Metrics: accuracy / balanced accuracy and the split-inflation difference with paired bootstrap; MDE80 printed beside.
- Test-unit design: gender unit = subject (24, rotating); identity unit = trial (3 rank folds); seeds replicate training noise, folds replicate test units.
- Controls: shuffled labels at chance in every split variant (if the random-window split with shuffled labels exceeds chance, the splitter leaks); constant input at chance.

## Prediction and falsifier
- Predicted: (a) raw random-window gender 90-97%, user-disjoint 52-68%; (b) inflation 10-25 points raw, 8-20 defended; (c) matches (b).
- FALSIFIED if (a) the user-disjoint drop is < 10 points, or (b) the inflation is < 5 points on raw and on every defended release.

## Cost / dependencies
~0.5 GPU-h; none beyond MotionSense. Gate for every IMU claim: the trial-held-out / user-disjoint protocols become mandatory in all pre-registrations.

## Prior
0.75. The memorisation mechanism is certain to exist; the uncertain part is only the size (hence 10 points as the bar).

## Cheapest floor that could kill it
This is the cheapest control in the programme; nothing cheaper exists. Its failure (no inflation) would mean the split critique of PrivDiffuser / C-AAE is overstated and is reported as such.
