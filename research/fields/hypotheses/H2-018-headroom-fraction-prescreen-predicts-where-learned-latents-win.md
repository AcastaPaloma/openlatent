# H2-018: A floors-only "headroom fraction" computed before any learned arm predicts where a learned task-agnostic latent wins: gain g >= 0.4 h where h >= 0.05, and g <= 0.02 where h < 0.03

- Focus: cross-focus (A, C, D and the four candidate fields) | Kind: unifying hypothesis and procedure (the pre-screen that every future learned arm must pass) | Cost: 0 GPU-h (analysis scripts over aggregated outputs) | Prior: 0.30
- Depends on: outputs of H2-001, H2-008-H2-013 and C's AE round; V-split floor curves for each cell; a script `headroom_fraction.py`.

## Statement
For every cell (field x encoder x dataset x downstream-task set T') define, on the validation split and with floors and a reference only:
- U_ref(T'): the raw (or, where the input degrades, the clean-twin/oracle) utility on T';
- U_floor*(T'; L): the best INDUCTIVE floor (PCA-d, `lda`/label-projection + PCA padding, random projection, noise-dithered code of H2-014) at leakage <= L;
- h = U_ref - U_floor*(L*) at the leakage level L* the learned arm will be held to (rule LM-rule, +0.05 tolerance).
Then, after the learned arm is run and V-selected, with g = U_learned(T'; L*) - U_floor*(T'; L*) on the sealed split:
1. in cells with h >= 0.05, g / h >= 0.4 in >= 60% of cells and g >= 0.04 with CI above 0 in at least one cell per field that the headroom check passes;
2. in cells with h < 0.03, g <= 0.02 in >= 80% of cells;
3. Spearman(h, g) >= 0.6 across >= 12 cells.

## Mechanism
The four failures so far share one cause (journal 12:30Z): either the task implies the secret (D rare words; C emotion structure; A in-distribution label) or a trivial method already removes the secret (LDA projection in A and C, no leakage to remove in B'), i.e. no gap between the best floor and the reference. Existing seed-0 data in A already show the structure (post hoc, AE configurations picked at test, one seed, so descriptive only): midnight/Tolkach h = 1.000 - 0.910 = 0.090, g = +0.083 (92% of h); midnight/TCGA h = 0.874 - 0.636 = 0.238, g = +0.164 (69%); dinov2-L/Tolkach h = 0.950 - 0.915 = 0.035, g = +0.002; dinov2-L/TCGA h = 0.633 - 0.669 < 0 (the floors exceed raw), g = -0.02. If the law holds, h is cheap (CPU, floors only) and decides in advance where learned arms are worth any GPU; if it fails, the headroom is not reachable by a vector map and the "oracle headroom" check is not sufficient.

## Experiment (exact)
- Procedure: before each learned arm of H2-001, H2-008-H2-013 and C's AE round is read, a committed file lists h per cell (computed on V from floors only, with the oracle/clean reference where relevant) and the leakage level L*. After the learned arm, g is added on the sealed split. No cell is dropped.
- Cells expected: A midnight/phikon-v2/phikon/dinov2-L x {TCGA, Tolkach} (8), C held-out emotion and multi-task (2-3), D digits tier (1, if H2-009 passes), single-cell, Cell Painting, EEG, astronomy (up to 4).
- Statistics: g/h per cell with bootstrap over units; Spearman across cells; a table of h, g and whether the cell passed its own pre-registered criterion.
- Controls: h must be computed without looking at C; negative-h cells reported as such (floors above raw means the probe ceiling is the limit).

## Prediction and falsifier
- Prediction: g/h in 0.4-0.9 in cells with h >= 0.05, g within [-0.02, +0.02] where h < 0.03; Spearman 0.6-0.8.
- Falsified if >= half of the h >= 0.05 cells have g/h < 0.2 (headroom exists but a learned latent cannot reach it), or any cell with h < 0.03 has g >= 0.05 with CI above 0 (the floor set missed a method, or the gain is selection noise: see H2-015).

## Cost
No GPU. One script and a standing rule: no learned arm without a prior committed h.

## Dependencies
Everything above; value is highest if run on cells in the order they arrive.

## Prior and why
0.30. The four existing points fit but they were assembled post hoc; the conjunction of three clauses over 12+ cells in unrelated fields is demanding, and the leakage-matched floors differ in resolution (H2-016).

## Sources
results_notes/A.md (PROTOCOL 2 tables: values used above), results_notes/C.md section 3, results_notes/D.md 5.1, results_notes/B.md (oracle-headroom check); journal 2026-11-04 ~12:30Z (pattern across all four attempts).
