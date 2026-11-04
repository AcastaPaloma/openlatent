# H-A-007 (BORING CONTROL): 24 hand-made colour/blur statistics carry >= 70% of the centre signal; regressing them out of z matches the best learned adapter

- Focus: A | Kind: boring control that can kill an exciting result | Cost: ~1 GPU-h + CPU | Prior: 0.55 for the 70% clause; 0.20 that the control kills the learned adapters
- Depends on: Builder-A pixel access; adapters from H-A-002..004 for the comparison.

## Statement
A centre probe (MLP and LightGBM, slide-disjoint) trained on only 24 numbers per patch computed from pixels (mean/std/5th/95th percentile of R, G, B and of the H, E, D channels after colour deconvolution, Laplacian variance, high-pass MAD noise, 8x8 JPEG blockiness) reaches normalised leakage L >= 0.70 relative to the FM-embedding probe on TCGA and Tolkach. Residualising the embedding on those 24 numbers (ridge regression, fitted on training slides, no centre labels) lowers L_worst by >= 30% relative to raw but leaves L_worst >= 0.45.

## Mechanism
Much of the centre signature in pathology is stain and scanner colour (Reinhard stain normalisation raises RI by 16.2% on average and is the strongest single cheap fix besides ComBat, 2507.17845). If a handful of low-level statistics reproduces most of the centre probe, then (i) "centre" is mostly colour and a deterministic descriptor-conditional erasure is the right comparator for every learned method; (ii) a learned adapter that only matches this residualisation has learned nothing beyond a hand-built stain model, the same pattern as our canonical stick figure (Finding 2 in results/SUMMARY.md). It is a control for H-A-002..004 and X-001..X-005.

## Experiment (exact)
- Data: PathoROB patches (pixels). Descriptor extraction on CPU.
- Probes: MLP (2x256) and LightGBM on the 24-d descriptor; also logistic regression. Slide-disjoint 60/10/30; chance and raw as in H-A-001. Reference: same probes on the 2048-d embeddings of phikonv2/phikon/midnight.
- Transform arms: (1) ridge residualisation of z on the descriptors; (2) Reinhard-normalised pixels (template = pooled training centre) then re-embed phikonv2 and phikon (1 GPU-h) then A-SUITE; (3) both.
- Comparison: best learned adapter from H-A-002/003/004 on the same cells.
- Seeds: 3 attacker seeds x 3 slide partitions; bootstrap over slides.
- Metric: L_desc = (acc_desc - chance)/(acc_emb_raw - chance), and L_worst after each arm.

## Prediction and falsifier
- Prediction: L_desc(MLP or LGBM) >= 0.70 on TCGA and Tolkach, >= 0.50 on Camelyon (2 centres, weaker signal); residualisation: L_worst falls by 30-50% but stays >= 0.45; Reinhard re-embedding: L_worst >= 0.60; best learned adapter beats residualisation by >= 0.10.
- Falsified (the control kills the exciting result) if residualisation or Reinhard + residualisation is within 0.05 of the best learned adapter's L_worst at equal U on >= 2 of 3 datasets: then the adapter has only learned colour. Falsified in the other direction (L_desc < 0.4) if the centre signal is not low-level colour, which makes stain-augmentation approaches (H-A-004) a poor fit.

## Cost
~1 GPU-h (two re-embedding passes after Reinhard) + ~2 CPU-h descriptor extraction and probes.

## Dependencies
Builder-A: pixel loader; H-A-002..004 outputs for the comparison column (the 70% clause is independent).

## Prior and why
0.55 for the 70% clause (the centre probe on FMs is 88-98%; colour statistics alone usually reach 80%+ for multi-lab data in the stain-normalisation literature, which I have not opened, so treat as my prior). 0.20 that the residualisation matches the best learned adapter.

## Sources
[O] 2507.17845 (Reinhard +16.2%, ComBat +27.4%); our results/SUMMARY.md Finding 2 (hand-built beats learned).
