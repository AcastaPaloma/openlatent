# H2-003: A label-free embedding-level contrastive code keeps held-out structure as well as the AE but leaks centre unless an adversary is added; with the adversary it ties the AE

- Focus: A | Kind: learned task-agnostic control (second objective family) | Cost: ~3 GPU-h | Prior: 0.35
- Depends on: Builder A adapter loop (`adapters.py` GRID extension); H2-001/H2-002 for the AE frontier it is compared to.

## Statement
Replace the reconstruction term of the AE by an InfoNCE instance-discrimination loss on two views of the same cached embedding (feature dropout 0.2 + Gaussian noise 0.1 sigma per view, batch 1024, tau 0.1), keeping the Gaussian rate channel and the same retrained centre adversaries. Then on midnight and phikon-v2 x {TCGA, Tolkach} (held-out-class protocol, 3 seeds):
1. contrastive without adversary has L_worst >= 0.6 at dz = 32 (it keeps whatever is most instance-discriminating, which includes centre);
2. contrastive with adversary (lam in {3, 10}) is within +-0.04 of the AE-with-adversary in U_held at matched L (+-0.05), in >= 3 of 4 cells;
3. its retention-xc is >= 0.05 lower than the AE's (instance discrimination keeps fine noise rather than neighbourhood structure).

## Mechanism
Two competing explanations of the A seed-0 lead: (i) reconstruction of the whole embedding keeps non-label structure; (ii) any objective that does not see labels keeps it, and the adversary supplies the privacy. The contrastive code has no decoder, so it is a clean test: if it ties the AE then "task-agnostic + adversary" is the active ingredient and the objective is interchangeable; if it loses by a wide margin, reconstruction is doing something special; if it wins, we have a cheaper and probably more stable objective (no decoder to balance). Embedding-level views are label-free by construction (no stain augmentation, which would need pixels; that is H-A-004).

## Experiment (exact)
- Cells and grid: dz in {16, 32}, beta in {1e-3, 1e-2}, lam in {0, 3, 10}, tau 0.1; AE arms from H2-001 reused.
- Views: two independent dropout+noise draws of the same cached vector; negatives = rest of the batch drawn from adapter-train units A only (never B, V, C).
- Evaluation identical to H2-001: A-SUITE6 retrained on B, U_held, retention-xc, L_worst, unit bootstrap, LM-rule comparator, SEL-V selection.
- Sanity: contrastive code with shuffled pairing (positive = a random other vector) must collapse to chance U_held.

## Prediction and falsifier
- Prediction: L(no adversary) in [0.6, 0.9]; U_held(contrastive+adv) - U_held(AE+adv) in [-0.04, +0.04] at matched L; retention gap -0.05 to -0.15.
- Falsified if (a) contrastive without adversary has L <= 0.4 (then contrast alone already hides centre; surprising and useful), or (b) contrastive+adv exceeds the AE by >= +0.05 U_held in >= 2 cells (then reconstruction is not the key and the contrastive objective is the method to carry forward), or (c) it trails the AE by >= 0.08 in >= 3 cells (reconstruction is special).

## Cost
~3 GPU-h: 24 configs x 4 cells x 3 seeds, each ~1-3 min fit plus the CPU-bound attacker evaluation.

## Dependencies
Builder A harness; H2-001 outputs for the AE reference.

## Prior and why
0.35 for "ties within 0.04 in >= 3 of 4 cells" with the leakage clause. Embedding-level SimCSE-style objectives are known to saturate quickly on frozen features; the instance-level signal in a pathology patch embedding is dominated by slide/texture, so I expect high uncontrolled leakage and a retention shortfall.

## Sources
results_notes/A.md (patch-disjoint slide fingerprint diagnostic, item 4 of the gate section); no external paper relied on.
