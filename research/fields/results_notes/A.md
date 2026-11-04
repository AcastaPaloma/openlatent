# Focus A (PathoROB, source-invariant embeddings): working log

Owner: Builder-A. Code: `research/fields/src/a_pathorob/`. Job specs: `research/fields/jobs/specs/a_*.yaml`.
Everything below is a log of what was run; numbers come from `aggregate.py` on the JSONL/prediction files the jobs wrote
(HF `KuanKuanKuan/openlatent-fields`, `runs/a_*`). No in-training metric is reported as a result.

## Pre-registration (written before any adapter result was seen)

Protocol, per (encoder, dataset), seeds 0,1,2:

* Independent unit = slide (Camelyon), case (TCGA `slide_id[:12]`, Tolkach). Splits are unit-disjoint.
  `C` sealed test (25% of units per centre x class stratum, fixed across seeds), the rest split per seed into
  `A` adapter-train (45%), `B` attacker/label-probe train (35%), `V` validation (20%).
  The benchmark's unseen (OOD) centres are never used to fit an adapter; OOD1/OOD2 are unit halves for an
  unseen-centre attacker (train on OOD1, test OOD2), and the whole OOD set measures label accuracy on unseen centres.
* Secret = medical centre (closed set of the ID centres). Utility = tissue/diagnosis class. Metric = balanced accuracy.
* Attackers are retrained from scratch on the released representation of B: linear (L-BFGS softmax, L2 chosen on an
  inner unit-disjoint split), 2-layer MLP, cosine kNN, token-transformer (FT-Transformer style). Worst case = max over the four.
* Normalisation: chance (1/#centres) and the label-only baseline (predict centre from the TRUE label with B frequencies);
  `leak = (worst - floor) / (worst_raw - floor)`.
* Controls that must hold before any number is believed: constant embedding -> chance; Gaussian noise embedding -> chance;
  raw embedding with centre labels shuffled across units in B -> chance; raw embedding worst-case well above chance; the
  A-trained linear attacker is reported next to the B-trained one (adapter-overfit diagnostic).
* Selection: per method family, the configuration with the lowest worst-case centre accuracy on V among those whose V label
  accuracy is within delta (1, 2, 5 points) of the raw embedding's. Reported number = that configuration on C.
* Kill criterion A (FOCUS): the best learned adapter (vib, adv_retrain, dann, vq, supcon_xc; selected on V) must beat LEACE
  (ridge chosen on V) on worst-case centre accuracy on C at the same delta, with a paired unit-bootstrap 95% CI below 0,
  3 seeds. Otherwise stop A.
* Single-seed results are labelled "not yet replicated".

## Log

### 2026-11-04 ~08:20-09:00Z: data, embeddings, replication anchor

* Data: three ungated HF datasets in `~/work/data/pathorob` (1.8 GB). Row counts equal the paper's: camelyon 22,402 + tcga 60,690 +
  tolkach_esca 16,300 = 99,392 patches. Parquet row order matches the repo's metadata CSVs (joined on slide_id+patch_id, asserted unique).
* Encoders (frozen, feature = concat(CLS, mean patch tokens), official preprocessing Resize/CenterCrop 224): `phikonv2_clsmean`
  (owkin/phikon-v2, Owkin licence), `phikon_clsmean` (owkin/phikon), `midnight_clsmean` (kaiko-ai/midnight, MIT, bf16),
  `dinov2L_clsmean` (facebook/dinov2-large, Apache-2.0, non-pathology control). Gated models (UNI/UNI2-h/CONCH/Virchow2/H-optimus/
  Prov-GigaPath/Hibou) return `GatedRepoError` for the account behind the VM token: the click-through terms have not been accepted
  and I did not accept them on the user's behalf. Cache key = sha1(encoder spec incl. pinned HF revision + data sha256); writes are atomic.
  Embeddings are cached under `~/work/cache/pathorob`, never in `$OUT`, never uploaded.
* **Robustness Index replication (unmodified official code run on our embeddings, default k per dataset):**

| encoder | TCGA 2x2 | Camelyon | Tolkach | README leaderboard / repo reference JSON | max abs diff |
|---|---|---|---|---|---|
| phikon-v2 | 0.619 | 0.019 | 0.768 | 0.619 / 0.019 / 0.768 (also the shipped `results_summary.json`) | 0.000 |
| phikon | 0.623 | 0.011 | 0.795 | 0.623 / 0.011 / 0.795 (README) | 0.000 |
| midnight (no anchor) | 0.858 | 0.477 | 0.941 | not on the leaderboard | n/a |
| dinov2-L (no anchor) | 0.649 | 0.511 | 0.892 | not on the leaderboard | n/a |

* **RI "saturation" check (coordinator question 2).** The low RI values (0.62 TCGA, 0.01-0.02 Camelyon for the Phikon family) are
  *not* a transcription or mapping bug: they come out of the official `robustness_index.py` run on our own embeddings and match the
  README to three decimals and the repo's shipped reference JSON exactly (phikon-v2). `ri_compare.py` only prints our computed number next to
  the README value. So the FOCUS premise "RI saturated 0.87-0.93" holds for the gated top models only (Atlas 2, RudolfV 2, Virchow2);
  for the four encoders we can extract RI has plenty of headroom (Phikon Camelyon 0.01-0.02; Midnight Camelyon 0.48; TCGA 0.62-0.86).
  Consequence: a gain in RI on these encoders is not evidence for a new method (ComBat/Reinhard already move RI by +16-27% in the paper),
  and the retrained-probe protocol is needed to show something RI cannot, not because RI has no room.
* Official-code Camelyon RI runtime ~1 min per model, TCGA (paired, 112,800 rows) ~5 min, Tolkach ~0.5 min (CPU).

### Disclosure (read before the numbers)

Before the gate-cell request I had already looked at (i) the 7-config smoke on Tolkach/Phikon (`a_smoke_tolkach`) and (ii) the first ~35
configs of `a_sw1_phikon_camelyon_s0` (all methods in one grid), which includes learned-adapter rows. The sw1 grid has been withdrawn
from the queue; the gate cell (floors + controls only) is now queued separately (`a_gate_*`, priority 15-17) and its aggregate
is written below *before* I read any further learned-adapter result. The coordinator's gate rule G1 is stated in the ranking
(L_worst(LEACE) < 0.15 in >= 5/9 cells stops A). From what I had already seen, LEACE leaves L_worst ~0.8 on Camelyon/Phikon and ~0.8-0.9 on Tolkach/Phikon (one seed each), i.e. nowhere near the stop rule, but that is not the gate result.

### Measured cost of the grid (coordinator question 4)

From `a_sw1_phikon_camelyon_s0` (27 configs, the largest dataset: 22k patches x 1536-d), measured fit+eval seconds per configuration:
mean 23.6 s (fit 7.9 s, eval 15.7 s; LEACE fit 31 s because of one 1536-d eigendecomposition per rho, now cached; LDA fit 30 s; ISP/VIB/VQ/DANN 1-4 s fit).
From `a_smoke_tolkach` (16k patches): 10-20 s per config, eval 7-14 s. Eval is dominated by the four retrained attackers (linear 3 s, MLP 4 s, kNN <1 s, transformer 4-8 s)
plus the unseen-centre attacker and label probes. LightGBM + RFF-SVM add ~5-10 s per centre evaluation on GPU+CPU.
Current full grid = 12 gate + 5 control + 25 learned + 14 rate-control = 56 configs. Estimate per cell (encoder x dataset x seed):
camelyon 56 x ~32 s = 0.5 GPU-h; tcga / tolkach 56 x ~18 s = 0.28 GPU-h. Per (encoder, seed) = 1.06 GPU-h; 4 encoders x 3 seeds = **~13 GPU-h** for the
whole crossed sweep, not ~80 (the ranker's 3 min per configuration was ~6x too high). The confounded and held-out-class variants cost ~+60% of that.
Wall clock is set by queue sharing with B and C, not by this grid.


### GATE CELL, written before any learned-adapter result of the new sweep was read (2026-11-04 ~11:00Z)

**Status: PARTIAL. 7 of 36 planned cells done, all seed 0 only ("not yet replicated"): dinov2-L x {camelyon, tcga, tolkach}, midnight x {camelyon, tcga, tolkach},
phikon x camelyon. Pending: phikon-v2 (3 datasets), phikon tcga/tolkach, seeds 1-2 of everything.** Raw numbers: HF `runs/a_gate_*`, aggregation by
`aggregate.py` (unit bootstrap over the sealed test slides, B=1000; with a single seed the interval covers only test-unit sampling, not seed variation).

**G1 (stop rule: L_worst(LEACE) < 0.15 in >= 5/9 cells): NOT MET.** 0 of 7 aggregated cells; LEACE (ridge chosen on V) leaves L_worst 0.63-0.89
(CIs exclude 0.15 in every cell). With 7 of the 9 cells at 0, the rule cannot reach 5/9 any more, so A is not stopped by G1 even before seeds 1-2.
The raw reference passes too: worst-case retrained centre attacker on raw embeddings is 0.73-1.00 (chance 0.25-0.50) in every cell.

What the gate says (all one seed; each point needs seeds 1-2 before it is a claim):

1. **Linear erasure does not hold out of sample.** A linear attacker retrained on LEACE-erased held-out units still reaches 0.55-0.87 accuracy
   (L_linear 0.41-0.83), while an attacker trained on the adapter-train units A is at exact chance (column "A-trained linear" = chance in the gate
   table): the LEACE guarantee is in-sample, and with n_A of 1.9k-7.2k patches in d = 1.5-3k it does not carry over to new slides. H-A-001's "L_LR <= 0.10" clause
   therefore holds on the fit units and fails on held-out units. Non-linear attackers (MLP/kNN/transformer/LightGBM/RFF-SVM) leave L_worst at 0.63-0.89, i.e. the headroom premise holds.
2. **The strongest floor is per-centre whitening (`centre_whiten`, PCA-64 then per-centre whitening, transductive).** L_worst 0.05-0.58 (dinov2-L: 0.05-0.14; midnight 0.21-0.58; phikon camelyon 0.39) at a
   label cost of 2-9 points. It needs the centre id of every released patch and the test batch's statistics, so it is a reference, not an encoder. `centre_mean` / `centre_zscore` zero the linear attacker but nonlinear ones keep L 0.3-0.9.
3. **The task-label-only floor is already strong in the crossed design.** `lda` (supervised linear projection to K-1 dimensions, centre never seen) gives L_worst 0.00-0.20 with label accuracy within
   0-3 points of raw. The closed-set ID design (every centre has every class in equal amounts) cannot reward a method for removing a centre signal that the label never uses; any label-supervised low-dimensional code starts here.
   Consequence for the sweep: a learned adapter has to beat `lda` and the rate-limited label bottleneck, not just LEACE, and the confounded / held-out-class / unseen-centre variants are where the methods can differ.
4. **Slide fingerprints survive every transform.** The patch-disjoint diagnostic (attackers trained and tested on different patches of the *same* test slides) recovers the centre at 0.78-1.00 after LEACE, ISP, centre_mean and centre_zscore
   (and 0.28-0.95 after centre_whiten, 0.36-0.66 after lda). Part of what "centre" attackers find on new slides is slide identity, which patch-level splits would inflate.
5. **Controls.** Constant embedding: exactly chance in all 7 cells. Gaussian noise embedding: chance (0.25-0.26 / 0.50-0.51) in all 7 cells. Raw embedding above chance: yes in all 7.
   **Shuffled-secret control (unit-level permutation of the attacker's training labels) is NOT a reliable chance control at these unit counts**: individual repetitions of the worst-of-4 accuracy range 0.30-0.88
   in the two-centre cells (e.g. midnight/tolkach 0.65, 0.88, 0.54; phikon/camelyon 0.58, 0.30, 0.74; dinov2-L/camelyon 0.48, 0.39, 0.62) because a random relabelling of 12-20 attacker-train units can agree with the true centre by luck
   (this is the scientific content of the control, but its variance is too large to certify "at chance"). I therefore treat normalised leakage differences below ~0.25 as inside the null spread at these sample sizes
   and have replaced the control by an orthogonal shuffle (`ctrl_shuffle_orth`: within each true centre the permuted label is balanced across labels, so its expected agreement with the truth is zero by construction); jobs started after
   commit "A: orthogonal shuffle control" write it.
6. Cell sizes: sealed test C has 20 slides (Camelyon), 48 cases (TCGA), 12 cases (Tolkach); Tolkach is the least reliable cell.

#### Compact gate summary (balanced accuracy on sealed C; mean over the listed seeds; chance and label-only floor in the heading)


**dinov2L_clsmean / camelyon / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.500; label-only floor 0.500. Controls, worst of the 4 attackers: const 0.50; noise 0.51; shuffle__rep0 0.48; shuffle__rep1 0.39; shuffle__rep2 0.62

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.935 | 0.994 | 0.991 | 0.855 | 0.937 | 0.962 | 0.979 | 1.00 [1.00, 1.00] | 1.000 | 0.653 |
| lda | 0.925 | 0.487 | 0.481 | 0.500 | 0.493 | 0.502 | 0.455 | 0.00 [0.00, 0.38] | 0.587 | 0.333 |
| centre_mean | 0.939 | 0.500 | 0.764 | 0.655 | 0.647 | 0.433 | 0.571 | 0.53 [0.39, 0.66] | 0.997 | 0.333 |
| centre_zscore | 0.951 | 0.500 | 0.732 | 0.655 | 0.693 | 0.454 | 0.654 | 0.47 [0.36, 0.58] | 0.998 | 0.333 |
| centre_whiten | 0.907 | 0.500 | 0.521 | 0.569 | 0.513 | 0.430 | 0.533 | 0.14 [0.09, 0.20] | 0.789 | 0.333 |
| leace__rho1.0 | 0.952 | 0.702 | 0.905 | 0.765 | 0.811 | 0.694 | 0.789 | 0.82 [0.70, 0.92] | 0.984 | 0.857 |
| isp__n_iter20 | 0.946 | 0.670 | 0.795 | 0.720 | 0.766 | 0.686 | 0.733 | 0.60 [0.38, 0.78] | 0.982 | 0.852 |

**dinov2L_clsmean / tcga / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.250; label-only floor 0.250. Controls, worst of the 4 attackers: const 0.25; noise 0.26; shuffle__rep0 0.24; shuffle__rep1 0.34; shuffle__rep2 0.34

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.702 | 0.731 | 0.699 | 0.463 | 0.542 | 0.553 | 0.682 | 1.00 [1.00, 1.00] | 0.895 | 0.820 |
| lda | 0.696 | 0.345 | 0.327 | 0.304 | 0.263 | 0.311 | 0.334 | 0.20 [0.07, 0.36] | 0.358 | 0.384 |
| centre_mean | 0.716 | 0.250 | 0.426 | 0.345 | 0.342 | 0.319 | 0.305 | 0.37 [0.25, 0.49] | 0.687 | 0.603 |
| centre_zscore | 0.707 | 0.250 | 0.412 | 0.351 | 0.334 | 0.298 | 0.317 | 0.34 [0.23, 0.46] | 0.700 | 0.653 |
| centre_whiten | 0.590 | 0.250 | 0.251 | 0.296 | 0.246 | 0.258 | 0.258 | 0.10 [0.05, 0.15] | 0.283 | 0.288 |
| leace__rho0.001 | 0.692 | 0.547 | 0.553 | 0.426 | 0.417 | 0.404 | 0.517 | 0.63 [0.54, 0.73] | 0.793 | 0.816 |
| isp__n_iter20 | 0.700 | 0.536 | 0.555 | 0.372 | 0.403 | 0.374 | 0.495 | 0.63 [0.52, 0.75] | 0.770 | 0.813 |

**dinov2L_clsmean / tolkach_esca / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.500; label-only floor 0.500. Controls, worst of the 4 attackers: const 0.50; noise 0.51; shuffle__rep0 0.62; shuffle__rep1 0.81; shuffle__rep2 0.50

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.940 | 0.930 | 0.888 | 0.616 | 0.715 | 0.792 | 0.841 | 1.00 [1.00, 1.00] | 0.978 | 0.962 |
| lda | 0.970 | 0.520 | 0.538 | 0.536 | 0.510 | 0.523 | 0.546 | 0.09 [-0.05, 0.42] | 0.655 | 0.591 |
| centre_mean | 0.922 | 0.500 | 0.662 | 0.593 | 0.644 | 0.582 | 0.666 | 0.38 [0.27, 0.53] | 0.976 | 0.628 |
| centre_zscore | 0.922 | 0.500 | 0.642 | 0.584 | 0.632 | 0.550 | 0.629 | 0.33 [0.25, 0.46] | 0.971 | 0.625 |
| centre_whiten | 0.862 | 0.500 | 0.499 | 0.523 | 0.504 | 0.519 | 0.512 | 0.05 [0.01, 0.18] | 0.682 | 0.559 |
| leace__rho1.0 | 0.939 | 0.795 | 0.845 | 0.626 | 0.761 | 0.652 | 0.789 | 0.80 [0.69, 0.90] | 0.951 | 0.965 |
| isp__n_iter5 | 0.942 | 0.769 | 0.728 | 0.592 | 0.627 | 0.592 | 0.692 | 0.63 [0.48, 0.76] | 0.943 | 0.967 |

**midnight_clsmean / camelyon / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.500; label-only floor 0.500. Controls, worst of the 4 attackers: const 0.50; noise 0.50; shuffle__rep0 0.51; shuffle__rep1 0.35; shuffle__rep2 0.64

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.986 | 1.000 | 0.961 | 0.927 | 0.956 | 0.971 | 0.968 | 1.00 [1.00, 1.00] | 1.000 | 0.926 |
| lda | 0.983 | 0.506 | 0.468 | 0.500 | 0.422 | 0.508 | 0.462 | 0.01 [0.00, 0.45] | 0.610 | 0.333 |
| centre_mean | 0.985 | 0.500 | 0.953 | 0.917 | 0.885 | 0.808 | 0.902 | 0.91 [0.86, 0.95] | 0.997 | 0.333 |
| centre_zscore | 0.986 | 0.500 | 0.877 | 0.917 | 0.902 | 0.615 | 0.890 | 0.83 [0.74, 0.91] | 0.993 | 0.333 |
| centre_whiten | 0.956 | 0.500 | 0.785 | 0.790 | 0.511 | 0.609 | 0.561 | 0.58 [0.50, 0.68] | 0.954 | 0.333 |
| leace__rho1.0 | 0.985 | 0.813 | 0.931 | 0.913 | 0.904 | 0.903 | 0.923 | 0.86 [0.79, 0.95] | 0.998 | 0.932 |
| isp__n_iter5 | 0.985 | 0.888 | 0.938 | 0.889 | 0.868 | 0.905 | 0.927 | 0.88 [0.77, 0.95] | 0.996 | 0.924 |

**midnight_clsmean / tcga / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.250; label-only floor 0.250. Controls, worst of the 4 attackers: const 0.25; noise 0.26; shuffle__rep0 0.22; shuffle__rep1 0.35; shuffle__rep2 0.30

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.874 | 0.776 | 0.717 | 0.617 | 0.583 | 0.577 | 0.726 | 1.00 [1.00, 1.00] | 0.990 | 0.848 |
| lda | 0.853 | 0.269 | 0.289 | 0.304 | 0.338 | 0.321 | 0.302 | 0.17 [0.06, 0.32] | 0.483 | 0.514 |
| centre_mean | 0.879 | 0.250 | 0.633 | 0.545 | 0.485 | 0.432 | 0.542 | 0.73 [0.54, 0.94] | 0.989 | 0.899 |
| centre_zscore | 0.880 | 0.250 | 0.694 | 0.544 | 0.459 | 0.426 | 0.514 | 0.84 [0.68, 1.04] | 0.975 | 0.898 |
| centre_whiten | 0.792 | 0.250 | 0.269 | 0.361 | 0.251 | 0.339 | 0.256 | 0.21 [0.11, 0.32] | 0.597 | 0.459 |
| leace__rho0.001 | 0.869 | 0.685 | 0.658 | 0.569 | 0.446 | 0.485 | 0.635 | 0.83 [0.76, 0.90] | 0.982 | 0.843 |
| isp__n_iter5 | 0.872 | 0.697 | 0.646 | 0.571 | 0.530 | 0.546 | 0.660 | 0.85 [0.78, 0.90] | 0.987 | 0.848 |

**midnight_clsmean / tolkach_esca / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.500; label-only floor 0.500. Controls, worst of the 4 attackers: const 0.50; noise 0.51; shuffle__rep0 0.65; shuffle__rep1 0.88; shuffle__rep2 0.54

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.983 | 0.966 | 0.926 | 0.742 | 0.875 | 0.848 | 0.931 | 1.00 [1.00, 1.00] | 0.999 | 0.998 |
| lda | 0.993 | 0.510 | 0.548 | 0.560 | 0.501 | 0.571 | 0.506 | 0.13 [-0.04, 0.46] | 0.661 | 0.658 |
| centre_mean | 0.979 | 0.500 | 0.674 | 0.744 | 0.637 | 0.730 | 0.673 | 0.52 [0.33, 0.71] | 0.999 | 0.791 |
| centre_zscore | 0.978 | 0.500 | 0.658 | 0.744 | 0.606 | 0.734 | 0.672 | 0.52 [0.31, 0.70] | 0.999 | 0.806 |
| centre_whiten | 0.930 | 0.500 | 0.614 | 0.619 | 0.479 | 0.535 | 0.482 | 0.25 [0.10, 0.44] | 0.812 | 0.652 |
| leace__rho0.01 | 0.984 | 0.871 | 0.913 | 0.782 | 0.825 | 0.842 | 0.846 | 0.89 [0.78, 0.98] | 0.997 | 0.998 |
| isp__n_iter5 | 0.983 | 0.873 | 0.879 | 0.693 | 0.712 | 0.703 | 0.774 | 0.81 [0.69, 0.97] | 0.993 | 0.998 |

**phikon_clsmean / camelyon / crossed**, seeds = 1 (PARTIAL: single seed, not yet replicated); chance 0.500; label-only floor 0.500. Controls, worst of the 4 attackers: const 0.50; noise 0.51; shuffle__rep0 0.58; shuffle__rep1 0.30; shuffle__rep2 0.74

| representation | label acc | linear | mlp | knn | tfm | lgbm | rff | L_worst [95% CI] | patch-disjoint worst | unseen-centre worst |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.979 | 1.000 | 1.000 | 0.997 | 0.962 | 0.998 | 1.000 | 1.00 [1.00, 1.00] | 1.000 | 0.998 |
| lda | 0.949 | 0.479 | 0.458 | 0.500 | 0.428 | 0.428 | 0.424 | 0.00 [0.00, 0.39] | 0.644 | 0.349 |
| centre_mean | 0.978 | 0.500 | 0.845 | 0.918 | 0.798 | 0.546 | 0.899 | 0.84 [0.73, 0.92] | 1.000 | 0.421 |
| centre_zscore | 0.980 | 0.500 | 0.937 | 0.914 | 0.767 | 0.559 | 0.877 | 0.87 [0.74, 0.96] | 0.999 | 0.416 |
| centre_whiten | 0.914 | 0.500 | 0.536 | 0.697 | 0.501 | 0.574 | 0.499 | 0.39 [0.32, 0.47] | 0.951 | 0.333 |
| leace__rho1.0 | 0.979 | 0.764 | 0.815 | 0.898 | 0.763 | 0.699 | 0.856 | 0.80 [0.65, 0.92] | 1.000 | 0.998 |
| isp__n_iter20 | 0.980 | 0.762 | 0.821 | 0.912 | 0.764 | 0.828 | 0.886 | 0.82 [0.69, 0.93] | 0.999 | 0.998 |

#### Gate G1: LEACE (ridge chosen on V) normalised worst-case centre leakage L_worst on sealed C (stop rule: L_worst < 0.15 in >= 5 of 9 cells)

| encoder | dataset | variant | seeds | LEACE config | L_worst (4 selection families) [95% CI] | L_worst (+LightGBM, RFF-SVM) | L_linear | < 0.15? |
|---|---|---|---|---|---|---|---|---|
| dinov2L_clsmean | camelyon | crossed | 1 | leace__rho1.0 | 0.82 [0.70, 0.92] | 0.82 | 0.41 | no |
| dinov2L_clsmean | tcga | crossed | 1 | leace__rho0.001 | 0.63 [0.54, 0.73] | 0.63 | 0.62 | no |
| dinov2L_clsmean | tolkach_esca | crossed | 1 | leace__rho1.0 | 0.80 [0.69, 0.90] | 0.80 | 0.69 | no |
| midnight_clsmean | camelyon | crossed | 1 | leace__rho1.0 | 0.86 [0.79, 0.95] | 0.86 | 0.63 | no |
| midnight_clsmean | tcga | crossed | 1 | leace__rho0.001 | 0.83 [0.76, 0.90] | 0.83 | 0.83 | no |
| midnight_clsmean | tolkach_esca | crossed | 1 | leace__rho0.01 | 0.89 [0.78, 0.98] | 0.89 | 0.80 | no |
| phikon_clsmean | camelyon | crossed | 1 | leace__rho1.0 | 0.80 [0.65, 0.92] | 0.80 | 0.53 | no |

G1 count: 0 of 7 aggregated cells have L_worst(LEACE) < 0.15.
