# H-X-005 (A and C): Defining the secret by a dense, label-free source descriptor lets removal transfer to unseen centres and actors; a closed-set label adversary does not

- Focus: A + C | Kind: learned method (dependence minimisation against a continuous proxy) | Cost: ~4 GPU-h | Prior: 0.40
- Depends on: X-003 (best dependence penalty); a frozen speaker embedder for C; the descriptor code of H-A-007 for A; seen/unseen attacker arms in both probe suites.

## Statement
A closed-set adversary trained on K_train source labels (A: ID centres; C: 56 training actors) removes source information for those sources much better than for unseen ones: L_unseen - L_seen >= 0.20. A penalty that makes z independent of a dense source descriptor s (A: 24 stain/colour/blur/noise statistics; C: a frozen ECAPA-TDNN speaker embedding, 192-d, computed from the same clip) halves that gap (gap <= 0.10) and lowers L_worst on unseen sources by >= 0.10 relative to the label adversary at equal U.

## Mechanism
A label adversary only has to separate the K_train sources it saw, so its removal is organised around those classes; "LEACE has difficulty generalizing to unseen data" (2605.01381, abstract opened) and kernelised erasure "does not transfer to different nonlinear adversaries" (2201.12191, abstract opened); our own C protocol enrols *unseen* actors by construction. A dense descriptor places every source, seen or not, in a continuous space where nearby sources share nuisance directions, and independence from it is an unlabeled, per-sample constraint that extends to new sources. A mismatch risk: the descriptor may not capture what the attacker uses (the pretrained speaker embedder captures timbre but little prosody/timing), so the gap could shrink only along descriptor directions.

## Experiment (exact)
- A: PathoROB phikonv2_clsmean; train on ID centres; s = 24-d descriptor (H-A-007) standardised; evaluation: source-probe among all centres including OOD ones (seen vs unseen separately) with A-SUITE.
- C: CREMA-D; train on 56 train actors; s = ECAPA-TDNN (SpeechBrain, frozen) embeddings; attackers enrol (i) test-fold actors (unseen) and (ii) train actors (seen; sentence-disjoint) so L_seen and L_unseen are both measured; C-SUITE.
- Arms: CE only; label adversary (X-001 recipe); dense-proxy dCor penalty (X-003) and, if X-003's CLUB works, proxy CLUB; proxy penalty + label adversary; LEACE on s (regress-out, closed form); rate-matched bottleneck.
- Seeds: 3 x 3, 4 folds (C), 3 datasets (A). Bootstrap over sources.
- Metric: L_seen, L_unseen, the gap, U; plus the descriptor-regression R^2 of z (how much of s remains).

## Prediction and falsifier
- Prediction: label adversary gap 0.2-0.4 in both foci; proxy penalty gap <= 0.10 and L_unseen lower by >= 0.10 than the label adversary (A) and >= 0.10 (C); combined arm best on L_unseen; LEACE-on-s removes s linearly (R^2 ~ 0) but leaves L_unseen close to raw (it fixes descriptor directions only).
- Falsified if L_unseen(proxy) >= L_unseen(label adversary) - 0.03 in both foci (a dense proxy does not generalise better), or if the label adversary has a gap <= 0.05 (closed-set removal already transfers; then the whole unseen-source concern is moot, which also contradicts H-C-001(c)).

## Cost
~4 GPU-h (A 1.5, C 2.5; ECAPA embeddings of 7,442 clips < 0.1 GPU-h).

## Dependencies
Builder-C: seen/unseen attacker enrolment arms (a parameter on the existing attacker split); Builder-A: pixel descriptors; X-003 outputs.

## Prior and why
0.40. The unseen-source gap is plausible and testable; the proxy is only as good as its coverage of the attacker's cues. If it works in one focus only, the finding is still publishable as a protocol point.

## Sources
[O] 2605.01381; [O] 2201.12191; [O] 2006.12013 CLUB; [S] 2604.05296 ISP (cross-dataset transfer CelebA-20 <-> VGGFace2; literature/methods_transfer.md); [snip] ECAPA-TDNN (2005.07143, not opened).
