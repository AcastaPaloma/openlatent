# H-CXR-006: A learned set attacker that pools P >= 3 released visits restores linkage that single-pair evaluation reports as defended, but only when the pooled visits are far apart

- Focus: R-CXR (evidence budget on the medical side; companion to H-S-003) | Kind: learned-attacker re-audit | GPU-h: ~4 (reuses Siamese trunks; small attention aggregator) | Prior: 0.55
- Depends on: H-CXR-002 (stage 0), H-CXR-003 releases, download gate. Anchor (scout-read): Nebbia et al. npj Digit Med 2025 report R@1 rising from 18.6% at 2 timepoints to 70.3% at 7+ on MIDRC.

## Statement
Defence evaluations (PriCheXy, LAPIS) score single image pairs. An attacker holding P released images of the candidate patient (gallery side) and one probe image, with a learned attention aggregator over per-image Siamese embeddings, achieves on PriCheXy mu = 0.01 and on the best floor of H-CXR-003: (1) a pooled-minus-single AUC lift under MN-v1 of >= +0.10 for P = 3 when the gallery images are SPREAD over visits (largest follow-up index gaps), (2) a lift of <= +0.05 when the gallery images are ADJACENT visits (shared acquisition and near-duplicate content), (3) Rank-1 in an ~780-patient gallery at least 3x the single-pair Rank-1, (4) the learned aggregator beats plain mean-embedding cosine pooling by <= 0.02 AUC (so no learned aggregator is needed).

## Mechanism
Errors of a pairwise attacker on different visits are only partly independent: they share the patient's anatomy (independent evidence the defence blurs) and the acquisition (shared nuisance). Evidence accumulation (toy simulation: decisive under independence, flat under shared nuisance) therefore depends on visit spacing; only gallery images that differ in acquisition and time add independent evidence.

## Experiment (exact)
- Data / units: PriCheXy official-test patients with >= P + 1 images (P = 3: about 780 patients, ESTIMATED from the measured >= 3 and >= 5 counts of 69% and 43% of multi-image patients; P = 1, 2, 3, 4 reported). Unit = patient; negatives = galleries of P images from an MN-v1-matched different patient (matched on sex, view, device, age band of the gallery).
- Embeddings: Siamese-R50 trunk retrained on each release (H-CXR-003 fits reused), plus frozen DINOv2-L. Aggregators: mean embedding cosine; max-similarity; learned attention set aggregator (2-layer, trained on train-patient defended outputs with P drawn at random).
- Selection of gallery images: ADJACENT (consecutive follow-up indices), SPREAD (extremes of the follow-up index), SAME-DEVICE-ONLY vs MIXED-DEVICE.
- Releases: raw, PriCheXy mu = 0.01, random flow mu = 0.01, blur sigma 4, downsample x4.
- Metrics: AUC and Rank-1 / Rank-5 vs P; N* analogue (smallest P reaching AUC 0.9); patient-cluster bootstrap.
- Test-unit design: PriCheXy arm fixed test patients (stated: seeds replicate training noise only); floors rotate over 5 patient folds (about 2,600 patients with >= 4 images pool-wide).
- Controls: P pooled random-patient galleries -> chance; shuffled labels -> chance.

## Prediction and falsifier
- Predicted: single-pair matched AUC on PriCheXy mu = 0.01 about 0.78; SPREAD P = 3 >= 0.88; ADJACENT P = 3 <= 0.83; aggregator advantage <= 0.02.
- FALSIFIED if: pooled lift <= 0.03 for both selections (effective P about 1, shared nuisance dominates: single-pair numbers are adequate for CXR), or lift >= 0.10 for ADJACENT as well (no dependence on spacing; the independence question has a boring answer).

## Cost / dependencies
~4 GPU-h; after H-CXR-003 fits exist.

## Prior
0.55. The direction (pooling helps) is well documented; the spread-vs-adjacent contrast depends on how much of the follow-up index maps to real time gaps (ChestX-ray14 has no dates, only the visit index; stated limitation).

## Cheapest floor that could kill it
Mean-embedding cosine pooling with no learned parts reproducing the full lift: then the "learned set attacker" adds nothing and only the pooling level is reported.
