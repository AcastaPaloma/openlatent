# H-A-004: Matched-pair contrastive adapter (same patch under different synthetic stains + same-tissue-different-centre positives) cuts unseen-centre leakage >= 25% vs a CE-only adapter

- Focus: A | Kind: learned method (contrastive, object-matching) | Cost: ~6 GPU-h | Prior: 0.35
- Depends on: Builder-A cache; a second extraction pass with augmented views (pixel access to the parquet patches).

## Statement
An adapter on frozen FM embeddings trained with (i) InfoNCE between embeddings of the *same patch* under two random stain/colour augmentations (the "same object, different domain" matching condition) and (ii) SupCon over tissue labels with positives drawn from different centres, reduces the retrained-probe leakage on *unseen* (OOD) centres by >= 25% relative to a CE-only adapter of equal d at biology U within 1 point, and beats LEACE on L_worst for >= 2 of 3 datasets.

## Mechanism
Mahajan et al. show class-conditional domain invariance is not sufficient for domain generalisation and that inputs derived from the same object should share a representation across domains; with augmentation the matched object is observed exactly (2006.07500, abstract opened). Stain augmentation gives a free, exact "same tissue, different lab" pair for the stain part of the centre signature; SupCon adds label-level positives across real centres (2004.11362, abstract opened). A fine-tuning recipe at the FM level already moved PathoROB RI from 0.72 to 0.87 (2607.22861, abstract opened; technique not stated on the page), which suggests consistency to staining nuisance is learnable, but nobody has shown it as a post-hoc adapter on frozen embeddings under a retrained probe. Limitation: augmentation covers colour, not scanner blur, compression or tissue prep.

## Experiment (exact)
- Data: PathoROB 3 datasets. Re-embed K = 4 augmented views of every patch (HED jitter sigma 0.05-0.1, random gamma/brightness/contrast, Reinhard transfer to a randomly drawn *training-centre* template; never a test-centre template) with phikonv2 and phikon (midnight optional): 99,392 x 4 x 2 forward passes.
- Adapter: MLP 2048-1024-128 (d=128, also d=32), temperature 0.1, batch 1024 patches x 2 views, 100 epochs.
- Ladder: raw | CE-only adapter | stain-pair InfoNCE only | stain-pair + SupCon-across-centres | LEACE | Reinhard-normalised-pixels re-embedding (deterministic, no adapter) | conditional adversary (H-A-003 M2) | stain-pair + adversary.
- Evaluation: adapter fit on ID training slides; centre probes (A-SUITE, slide-disjoint) among *all* centres including OOD ones the adapter never saw ("source-probe"); biology accuracy ID and OOD; RI and APD for continuity with the literature.
- Seeds: 3 adapter x 3 attacker; bootstrap over slides.
- Metric: L_worst on unseen centres, normalised by raw; U.

## Prediction and falsifier
- Prediction: stain-pair+SupCon: L_worst(unseen) <= 0.75 x L_worst(CE-only) at |dU| <= 0.01; vs LEACE: lower by >= 0.05 on >= 2/3 datasets; vs Reinhard re-embedding: no difference larger than the CI (the adapter gets no more than stain normalisation does). Pure InfoNCE without SupCon loses >= 2 points of biology on Tolkach.
- Falsified if L_worst(stain-pair) >= L_worst(LEACE) - 0.02 on all datasets, or the gain over CE-only is below 10% relative (augmentation consistency does not transfer to a frozen-embedding adapter).

## Cost
~6 GPU-h: 8 view-extraction passes of ViT-B/L over 99k patches (~0.3-0.6 GPU-h each => ~3.5 GPU-h) + ~2 GPU-h adapters.

## Dependencies
Builder-A: patch pixels accessible through the same loader as `extract.py`; augmentation extraction as a queue job. A-SUITE.

## Prior and why
0.35. Strong mechanism, but the effect is concentrated in stain and the retrained probe may find scanner/compression cues; the Reinhard comparator is a hard cheap competitor (+16-19% RI in the paper).

## Sources
[O] 2006.07500 MatchDG / causal matching; [O] 2004.11362 SupCon; [O] 2607.22861 Filiot et al.; [O] 2507.17845 (Reinhard +16.2% RI, ComBat +27.4%).
