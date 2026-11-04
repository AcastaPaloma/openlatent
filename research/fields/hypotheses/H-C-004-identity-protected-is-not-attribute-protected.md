# H-C-004: A representation that hides actor identity still reveals sex (and partly age and race); removing sex costs >= 4 points of emotion

- Focus: C (secondary secrets from FOCUS) | Kind: attacker-side measurement + learned removal | Cost: ~1 GPU-h | Prior: 0.50
- Depends on: Builder-C frozen features, `VideoDemographics.csv` (Sex, Age, Race, Ethnicity per actor; the splits are already stratified by sex then race), the identity-protected z from H-C-002/H-C-003/X-001.

## Statement
Among released representations that sit on the identity frontier (L_worst <= 0.25 at emotion U >= 0.90 on T1), at least half have sex normalised leakage L_sex >= 0.60 under an actor-disjoint attacker (trained on encoder-train actors, tested on test-fold actors; chance = class prior), and age (3 bins) with L_age >= 0.30. A sex adversary (or LEACE on sex) cuts L_sex to <= 0.10 but costs >= 4 points of emotion UAR.

## Mechanism
Identity removal as defined by an enrolled-actor attacker targets what separates individuals; sex is a coarse attribute that is not needed to separate 22 individuals once a few other cues remain, and pitch/formant level correlates with sex and with arousal (high F0, high energy in anger/fear/happiness), so removing sex also removes part of the arousal cue. Voice-anonymisation systems show the same dissociation: gender and accent stay recoverable at near-perfect accuracy after anonymisation (VoxGuard, 2509.18413, scout-opened, self-reported), and emotion-preserving anonymisers "excel at one but not both" (2409.03655, scout-opened).

## Experiment (exact)
- Data: CREMA-D; attributes from `VideoDemographics.csv` (verify field names and counts against the file; my recollection, not verified, is 48 male / 43 female, race categories African American, Asian, Caucasian, Unknown, and Hispanic / Not Hispanic; Builder-C's `data.py` already reads `Sex` and `Race` from it).
- Representations: every z in the identity ladder with L_worst <= 0.25: posterior-6, bottlenecks d <= 24 (H-C-002), matched-content (H-C-003), adversarial (X-001/X-002), LEACE(speaker).
- Attackers: C-SUITE heads (LR, MLP, kNN, LightGBM), but *actor-disjoint*: train on encoder-train actors' z, test on test-fold actors (attribute generalisation across people, not identity re-finding). Age regression (Ridge/MLP) reported as MAE vs predict-mean baseline, plus 3-bin classification.
- Removal arms: GRL sex adversary at d=64; LEACE(sex); report emotion UAR change.
- Seeds: 3 x 4 folds; bootstrap over actors. Chance: sex 0.5 (class prior measured), race majority prior, age-bin prior; normalisation L_attr = (acc - prior)/(acc_raw - prior).

## Prediction and falsifier
- Prediction: for d = 24..64 frontier points L_sex in 0.6-0.95; posterior-6 point L_sex <= 0.4 (almost nothing but the label left); age L_age 0.3-0.6; race L_race 0.1-0.4 (few actors per race: noisy). Sex removal: L_sex <= 0.10 with emotion UAR loss 4-10 points at d=64.
- Falsified if fewer than 25% of frontier points have L_sex >= 0.6 (identity removal brings sex along), or if sex removal costs < 1 point emotion (sex is cleanly separable; secondary secret is a non-issue).

## Cost
~1 GPU-h (probes and a GRL arm on cached features).

## Dependencies
Builder-C: demographics loader; at least H-C-002 outputs.

## Prior and why
0.50. At the lowest rates sex is gone, so the claim depends on where the "frontier" points are; I set the criterion on a fraction of the frontier rather than one point. The cost-of-sex-removal clause is less certain (0.55).

## Sources
[S] 2509.18413 VoxGuard, 2409.03655 (literature/methods_transfer.md, affect_behaviour.md); [O] 2501.05310 (speaker-specific attributes in SSL layers); FOCUS.md (secondary secrets).
