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


## PROTOCOL 2: held-out classes, retention, unseen centres, matched-dimensionality LDA arm (2026-11-04 ~11:50Z)

Requested by the revised FOCUS A kill rule (LEACE is only a floor; the comparator is the label-supervised LDA arm). Implemented in
`run_sweep.py --grid p2 [--holdout]`, `aggregate_p2.py`. Definitions:

* **Held-out-class protocol** (`--holdout`): the adapter-train units A keep only the first half of the classes (TCGA 2 of 4: breast, colon seen; lung adeno, lung squamous held out;
  Tolkach 3 of 6 seen). Every adapter and the LDA arm are fit on A (seen labels only). Utility on the held-out classes = balanced accuracy of a linear probe retrained on the
  released representation of the attacker-train units B restricted to the held-out classes, scored on sealed C and on the benchmark's unseen centres. Camelyon has 2 classes and
  no finer label, so it has no held-out-class cell (its p2 job is the crossed variant with the new retention / unseen-centre columns; pending).
* **Retention** of non-label structure (`ret/knn_xc`): fraction of the 10 raw-space nearest B-neighbours (other centres only) of a C patch that are among its 30 nearest in the released space.
* **LDA arm at matched dimensionality** (`lda_pca__d`): LDA (C_seen-1 dims) plus the top PCs of the residual after removing the LDA subspace, d in {4,...,128}; plain `lda` below that.
  Label-free references: `pca__d`, `randproj__d`, per-centre whitening (`centre_whiten`, transductive). **Task-agnostic learned adapters** (`ae__`): autoencoder through a rate-limited Gaussian channel with no class labels,
  optionally with adversaries retrained from scratch each round (lam > 0). The label-supervised adapters (vib, vq, dann, adv_retrain, supcon_xc) see only the seen classes.
* **Kill test as revised:** a learned family (configuration selected on V) beats the matched LDA arm if the paired-bootstrap CI of the difference in held-class utility (i) or retention (ii) is above 0 while its worst-case leakage L is not more than 0.05 above the LDA arm's.
  I additionally report a leakage-matched comparison (each learned config vs the best LDA-family / label-free reference with L <= L_config + 0.05, chosen on C so it favours the baseline), because the dimension-matched LDA+PCA arm leaks a lot (L 0.5-0.9 for d >= 16: the extra PCs carry the centre).
* Power: held-out-class cells have 24 held-class cases in sealed C for TCGA and 12 cases for Tolkach, so held-class CIs are +-0.05 to +-0.15.

**Status: PARTIAL, seed 0 only ("not yet replicated").** Done: dinov2-L x {tcga, tolkach}, midnight x {tcga, tolkach}, holdout. Running / pending: phikon tcga holdout, the three Camelyon crossed cells,
and seeds 1 and 2 of the four finished cells (queued, jobs `a_p2h_*_s1|s2`). Controls under this protocol pass: constant and noise embeddings at chance and the orthogonal shuffled-secret control
at chance (TCGA 0.24-0.31, chance 0.25; Tolkach 0.50-0.58, chance 0.50; orthogonal shuffling has about 1/3 of the spread of the old permutation control).

**Result so far (seed 0).**

1. **Official kill-test tally: 0 of 24 (4 cells x 6 learned families, selected on V) beat the matched-dimensionality LDA arm** on held-class utility or retention (table at the end of this section). No learned family is above 0 on both CIs anywhere;
   the closest are the task-agnostic `ae` configs (held-class utility difference -0.09 to +0.009, every CI includes 0 or is negative; retention difference -0.04 to -0.13, i.e. below the LDA+PCA arm).
2. **LDA does collapse onto its training labels, but only mildly in these cells:** held-class accuracy of plain LDA vs raw is 0.544 vs 0.633 (dinov2-L, TCGA), 0.910 vs 1.000 (midnight, Tolkach), i.e. a loss of 0.09 with CIs that overlap for TCGA (24 cases).
   Label-supervised bottlenecks collapse harder (midnight/Tolkach: vib 0.814, vq 0.760 held-class accuracy, CI half-widths ~0.07; raw 1.000).
3. **The dimension-matched comparison is dominated by leakage, not by what the adapters keep.** At d >= 16 the LDA+PCA arm has L 0.5-0.9 while the learned adapters sit at L 0.1-0.5, so every learned config is "better" on L (-0.2 to -0.8) and worse on retention.
   The informative comparison is at equal leakage.
4. **Exploratory, leakage-matched, one seed, configurations picked post hoc from 10 AE variants (so not a confirmed result): the task-agnostic adversarial autoencoder is the only learned family that beats the label-only / label-free references at equal leakage, and only for midnight.**
   midnight / Tolkach, `ae__beta0.001_dl32_lam10.0` (L 0.27 [0.05, 0.47]): held-class accuracy 0.993 [0.989, 0.997] = +0.083 [+0.048, +0.126] over the best LDA-family point at L <= 0.32 (plain LDA, 2 dims, held 0.910)
   and +0.021 [+0.002, +0.046] over per-centre whitening; retention +0.243 [+0.188, +0.299] over the best LDA-family point (`lda_pca__d4`) and +0.216 [+0.140, +0.275] over PCA-4.
   midnight / TCGA, `ae__beta0.001_dl32_lam10.0` (L 0.35): held-class accuracy +0.164 [+0.040, +0.284] over `lda_pca__d8`, retention +0.119 [+0.077, +0.159] over PCA-4; `ae__beta0.001_dl8_lam0.0` (no adversary, L 0.29): +0.146 [+0.010, +0.271] over `lda_pca__d8`.
   dinov2-L (both datasets): no gain (CIs include 0 or negative). Per-centre whitening and label-free PCA-4 are strong references everywhere (PCA-4: L 0.11, retention 0.40 on dinov2-L/TCGA).
   Seeds 1 and 2 of these four cells are queued; this exploratory contrast is only worth a claim if it reappears there for the same two AE configurations.
5. Full tables (all configs, with the leakage-matched comparison for every learned config): produced by `aggregate_p2.py --runs <dirs>` (without `--compact`); HF `runs/a_p2h_*`.

Compact output of `aggregate_p2.py --compact` (references and the V-selected configuration of each learned family; intervals are 95% unit bootstraps over the sealed test units, B=1000, one seed):

#### dinov2L_clsmean / tcga / holdout  (seeds [0]; PARTIAL, single seed: not yet replicated; centre chance 0.250, floor 0.250; seen classes [0, 1], held-out classes [2, 3])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 2048 | nan | 0.969 | 0.633 [0.507, 0.735] | 0.588 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.900 | 0.618 [0.517, 0.702] | 0.567 | 0.06 [0.01, 0.11] | 0.08 | 0.401 [0.383, 0.419] | 0.389 |
| leace__rho1.0 | 2048 | nan | 0.967 | 0.636 [0.514, 0.739] | 0.620 | 0.82 [0.76, 0.88] | 0.82 | 0.952 [0.943, 0.961] | 0.963 |
| lda | 1 | nan | 0.976 | 0.544 [0.408, 0.682] | 0.588 | 0.14 [0.06, 0.31] | 0.14 | 0.052 [0.045, 0.059] | 0.040 |
| lda_pca__d4 | 4 | nan | 0.981 | 0.585 [0.483, 0.690] | 0.573 | 0.10 [0.02, 0.22] | 0.12 | 0.291 [0.272, 0.312] | 0.256 |
| lda_pca__d8 | 8 | nan | 0.974 | 0.647 [0.546, 0.743] | 0.637 | 0.27 [0.17, 0.40] | 0.27 | 0.573 [0.553, 0.595] | 0.539 |
| lda_pca__d16 | 16 | nan | 0.972 | 0.653 [0.553, 0.742] | 0.595 | 0.50 [0.40, 0.63] | 0.50 | 0.713 [0.699, 0.729] | 0.701 |
| lda_pca__d32 | 32 | nan | 0.978 | 0.660 [0.555, 0.748] | 0.565 | 0.81 [0.71, 0.91] | 0.81 | 0.706 [0.690, 0.721] | 0.708 |
| lda_pca__d64 | 64 | nan | 0.971 | 0.621 [0.522, 0.709] | 0.554 | 0.83 [0.76, 0.91] | 0.83 | 0.625 [0.607, 0.644] | 0.629 |
| lda_pca__d128 | 128 | nan | 0.961 | 0.621 [0.509, 0.712] | 0.616 | 0.89 [0.83, 0.95] | 0.89 | 0.543 [0.517, 0.570] | 0.545 |
| pca__d4 | 4 | nan | 0.879 | 0.583 [0.519, 0.655] | 0.566 | 0.11 [0.02, 0.25] | 0.11 | 0.402 [0.377, 0.427] | 0.368 |
| pca__d16 | 16 | nan | 0.938 | 0.669 [0.575, 0.756] | 0.615 | 0.44 [0.33, 0.56] | 0.45 | 0.723 [0.709, 0.737] | 0.712 |
| pca__d64 | 64 | nan | 0.919 | 0.600 [0.503, 0.682] | 0.548 | 0.84 [0.75, 0.92] | 0.89 | 0.614 [0.592, 0.635] | 0.617 |
| vib__beta0.001_dl32 | 32 | 23.2 | 0.944 | 0.561 [0.456, 0.666] | 0.548 | 0.08 [0.04, 0.19] | 0.08 | 0.053 [0.047, 0.059] | 0.041 |
| vq__K16_m4 | 64 | 16.0 | 0.935 | 0.599 [0.530, 0.679] | 0.527 | 0.12 [0.03, 0.24] | 0.15 | 0.084 [0.076, 0.093] | 0.073 |
| dann__dl64_lam3.0 | 64 | nan | 0.938 | 0.621 [0.521, 0.714] | 0.567 | 0.48 [0.37, 0.60] | 0.48 | 0.338 [0.320, 0.356] | 0.303 |
| adv_retrain__dl64_lam1.0 | 64 | nan | 0.965 | 0.640 [0.549, 0.729] | 0.544 | 0.49 [0.41, 0.58] | 0.49 | 0.322 [0.310, 0.335] | 0.312 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.968 | 0.619 [0.551, 0.680] | 0.543 | 0.39 [0.28, 0.49] | 0.39 | 0.262 [0.236, 0.292] | 0.266 |
| ae__beta0.001_dl32_lam1.0 | 32 | 133.0 | 0.933 | 0.649 [0.547, 0.743] | 0.557 | 0.51 [0.42, 0.61] | 0.51 | 0.653 [0.641, 0.669] | 0.634 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc [95% CI] | Δ retention xc [95% CI] | Δ L_worst | wins? |
|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.099 [-0.208, 0.021] | -0.653 [-0.671, -0.635] | -0.72 | no |
| vq | vq__K16_m4 | lda_pca__d64 | -0.022 [-0.117, 0.080] | -0.541 [-0.564, -0.519] | -0.71 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | 0.000 [-0.087, 0.082] | -0.287 [-0.315, -0.260] | -0.35 | no |
| adv_retrain | adv_retrain__dl64_lam1.0 | lda_pca__d64 | 0.019 [-0.030, 0.078] | -0.303 [-0.325, -0.280] | -0.34 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.001 [-0.061, 0.063] | -0.363 [-0.399, -0.324] | -0.45 | no |
| ae | ae__beta0.001_dl32_lam1.0 | lda_pca__d32 | -0.011 [-0.065, 0.039] | -0.053 [-0.064, -0.040] | -0.30 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.25; ctrl_noise 0.27; ctrl_shuffle_orth__rep0 0.25; ctrl_shuffle_orth__rep1 0.24; ctrl_shuffle_orth__rep2 0.25

#### dinov2L_clsmean / tolkach_esca / holdout  (seeds [0]; PARTIAL, single seed: not yet replicated; centre chance 0.500, floor 0.500; seen classes [0, 1, 2], held-out classes [3, 4, 5])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 2048 | nan | 0.949 | 0.950 [0.926, 0.980] | 0.976 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.921 | 0.873 [0.824, 0.913] | 0.889 | 0.07 [0.00, 0.16] | 0.07 | 0.470 [0.387, 0.532] | 0.487 |
| leace__rho1.0 | 2048 | nan | 0.952 | 0.944 [0.917, 0.977] | 0.975 | 0.91 [0.85, 0.99] | 0.91 | 0.976 [0.967, 0.985] | 0.975 |
| lda | 2 | nan | 0.972 | 0.588 [0.492, 0.654] | 0.486 | 0.06 [-0.07, 0.37] | 0.06 | 0.134 [0.100, 0.187] | 0.058 |
| lda_pca__d4 | 4 | nan | 0.961 | 0.630 [0.508, 0.698] | 0.614 | 0.07 [-0.06, 0.42] | 0.07 | 0.333 [0.277, 0.426] | 0.236 |
| lda_pca__d8 | 8 | nan | 0.948 | 0.827 [0.776, 0.887] | 0.820 | 0.15 [-0.00, 0.54] | 0.15 | 0.572 [0.522, 0.632] | 0.495 |
| lda_pca__d16 | 16 | nan | 0.972 | 0.892 [0.861, 0.927] | 0.873 | 0.14 [0.02, 0.51] | 0.20 | 0.766 [0.745, 0.790] | 0.726 |
| lda_pca__d32 | 32 | nan | 0.967 | 0.915 [0.898, 0.943] | 0.908 | 0.23 [0.08, 0.47] | 0.30 | 0.833 [0.793, 0.855] | 0.820 |
| lda_pca__d64 | 64 | nan | 0.961 | 0.922 [0.889, 0.955] | 0.938 | 0.69 [0.57, 0.81] | 0.71 | 0.804 [0.739, 0.836] | 0.816 |
| lda_pca__d128 | 128 | nan | 0.958 | 0.913 [0.882, 0.953] | 0.956 | 0.89 [0.86, 0.94] | 0.89 | 0.709 [0.618, 0.772] | 0.736 |
| pca__d4 | 4 | nan | 0.877 | 0.722 [0.598, 0.810] | 0.739 | 0.23 [0.04, 0.68] | 0.23 | 0.432 [0.373, 0.511] | 0.341 |
| pca__d16 | 16 | nan | 0.947 | 0.900 [0.878, 0.934] | 0.877 | 0.17 [-0.00, 0.50] | 0.17 | 0.759 [0.728, 0.783] | 0.730 |
| pca__d64 | 64 | nan | 0.930 | 0.919 [0.899, 0.939] | 0.933 | 0.70 [0.58, 0.83] | 0.70 | 0.767 [0.696, 0.808] | 0.791 |
| vib__beta0.001_dl32 | 32 | 27.5 | 0.966 | 0.580 [0.470, 0.626] | 0.568 | 0.05 [-0.12, 0.34] | 0.05 | 0.095 [0.075, 0.122] | 0.045 |
| vq__K64_m8 | 128 | 48.0 | 0.967 | 0.644 [0.573, 0.712] | 0.674 | 0.02 [-0.12, 0.36] | 0.02 | 0.203 [0.161, 0.278] | 0.126 |
| dann__dl64_lam3.0 | 64 | nan | 0.917 | 0.851 [0.801, 0.910] | 0.838 | 0.09 [-0.11, 0.38] | 0.09 | 0.383 [0.332, 0.450] | 0.298 |
| adv_retrain__dl64_lam10.0 | 64 | nan | 0.956 | 0.885 [0.856, 0.928] | 0.906 | 0.27 [0.13, 0.52] | 0.27 | 0.580 [0.543, 0.626] | 0.530 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.972 | 0.912 [0.893, 0.928] | 0.907 | 0.31 [0.18, 0.55] | 0.31 | 0.383 [0.340, 0.451] | 0.358 |
| ae__beta0.001_dl32_lam0.0 | 32 | 79.3 | 0.940 | 0.917 [0.898, 0.933] | 0.903 | 0.32 [0.17, 0.62] | 0.34 | 0.797 [0.769, 0.815] | 0.768 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc [95% CI] | Δ retention xc [95% CI] | Δ L_worst | wins? |
|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.335 [-0.467, -0.286] | -0.738 [-0.772, -0.677] | -0.18 | no |
| vq | vq__K64_m8 | lda_pca__d128 | -0.269 [-0.353, -0.187] | -0.506 [-0.611, -0.341] | -0.87 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.071 [-0.139, -0.021] | -0.420 [-0.498, -0.297] | -0.60 | no |
| adv_retrain | adv_retrain__dl64_lam10.0 | lda_pca__d64 | -0.038 [-0.082, -0.003] | -0.224 [-0.290, -0.119] | -0.42 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.010 [-0.042, 0.017] | -0.421 [-0.489, -0.300] | -0.39 | no |
| ae | ae__beta0.001_dl32_lam0.0 | lda_pca__d32 | 0.002 [-0.032, 0.026] | -0.037 [-0.051, -0.018] | +0.09 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.50; ctrl_noise 0.50; ctrl_shuffle_orth__rep0 0.51; ctrl_shuffle_orth__rep1 0.52; ctrl_shuffle_orth__rep2 0.55

#### midnight_clsmean / tcga / holdout  (seeds [0]; PARTIAL, single seed: not yet replicated; centre chance 0.250, floor 0.250; seen classes [0, 1], held-out classes [2, 3])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.981 | 0.874 [0.741, 0.970] | 0.780 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.949 | 0.793 [0.699, 0.867] | 0.673 | 0.23 [0.16, 0.32] | 0.23 | 0.287 [0.258, 0.315] | 0.312 |
| leace__rho1.0 | 3072 | nan | 0.982 | 0.867 [0.726, 0.967] | 0.779 | 0.93 [0.87, 0.98] | 0.93 | 0.988 [0.977, 0.996] | 0.997 |
| lda | 1 | nan | 0.976 | 0.625 [0.483, 0.760] | 0.547 | 0.02 [-0.04, 0.19] | 0.02 | 0.095 [0.082, 0.107] | 0.077 |
| lda_pca__d4 | 4 | nan | 0.981 | 0.617 [0.484, 0.737] | 0.514 | 0.31 [0.19, 0.49] | 0.31 | 0.330 [0.277, 0.386] | 0.283 |
| lda_pca__d8 | 8 | nan | 0.981 | 0.636 [0.508, 0.763] | 0.613 | 0.32 [0.22, 0.48] | 0.34 | 0.418 [0.369, 0.472] | 0.365 |
| lda_pca__d16 | 16 | nan | 0.974 | 0.731 [0.604, 0.852] | 0.676 | 0.42 [0.31, 0.58] | 0.47 | 0.507 [0.457, 0.561] | 0.484 |
| lda_pca__d32 | 32 | nan | 0.978 | 0.819 [0.703, 0.931] | 0.692 | 0.54 [0.43, 0.68] | 0.62 | 0.619 [0.567, 0.677] | 0.629 |
| lda_pca__d64 | 64 | nan | 0.960 | 0.869 [0.771, 0.948] | 0.714 | 0.61 [0.49, 0.75] | 0.61 | 0.713 [0.659, 0.765] | 0.741 |
| lda_pca__d128 | 128 | nan | 0.978 | 0.854 [0.737, 0.965] | 0.758 | 0.93 [0.86, 1.02] | 0.93 | 0.712 [0.645, 0.774] | 0.777 |
| pca__d4 | 4 | nan | 0.981 | 0.613 [0.492, 0.740] | 0.533 | 0.37 [0.24, 0.53] | 0.37 | 0.316 [0.265, 0.373] | 0.270 |
| pca__d16 | 16 | nan | 0.957 | 0.700 [0.564, 0.827] | 0.655 | 0.47 [0.36, 0.61] | 0.49 | 0.500 [0.450, 0.554] | 0.486 |
| pca__d64 | 64 | nan | 0.954 | 0.863 [0.765, 0.946] | 0.719 | 0.63 [0.49, 0.76] | 0.63 | 0.705 [0.649, 0.758] | 0.734 |
| vib__beta0.01_dl32 | 32 | 7.8 | 0.951 | 0.599 [0.529, 0.669] | 0.573 | 0.04 [0.02, 0.13] | 0.06 | 0.066 [0.056, 0.076] | 0.047 |
| vq__K16_m1 | 16 | 4.0 | 0.983 | 0.525 [0.403, 0.648] | 0.367 | 0.02 [-0.02, 0.19] | 0.05 | 0.060 [0.045, 0.076] | 0.058 |
| dann__dl64_lam3.0 | 64 | nan | 0.974 | 0.729 [0.597, 0.848] | 0.565 | 0.45 [0.34, 0.58] | 0.45 | 0.313 [0.261, 0.365] | 0.312 |
| adv_retrain__dl64_lam3.0 | 64 | nan | 0.981 | 0.753 [0.614, 0.881] | 0.789 | 0.45 [0.34, 0.58] | 0.49 | 0.500 [0.445, 0.559] | 0.490 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.982 | 0.783 [0.669, 0.882] | 0.707 | 0.40 [0.29, 0.55] | 0.40 | 0.360 [0.293, 0.428] | 0.412 |
| ae__beta0.001_dl32_lam3.0 | 32 | 129.8 | 0.964 | 0.731 [0.601, 0.840] | 0.694 | 0.47 [0.37, 0.58] | 0.47 | 0.492 [0.436, 0.547] | 0.502 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc [95% CI] | Δ retention xc [95% CI] | Δ L_worst | wins? |
|---|---|---|---|---|---|---|
| vib | vib__beta0.01_dl32 | lda_pca__d32 | -0.221 [-0.323, -0.118] | -0.553 [-0.603, -0.505] | -0.49 | no |
| vq | vq__K16_m1 | lda_pca__d16 | -0.206 [-0.333, -0.079] | -0.447 [-0.499, -0.398] | -0.40 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.140 [-0.276, -0.018] | -0.400 [-0.452, -0.344] | -0.16 | no |
| adv_retrain | adv_retrain__dl64_lam3.0 | lda_pca__d64 | -0.117 [-0.235, -0.009] | -0.213 [-0.259, -0.162] | -0.16 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.086 [-0.174, -0.002] | -0.354 [-0.419, -0.277] | -0.21 | no |
| ae | ae__beta0.001_dl32_lam3.0 | lda_pca__d32 | -0.089 [-0.197, 0.016] | -0.127 [-0.167, -0.087] | -0.07 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.25; ctrl_noise 0.27; ctrl_shuffle_orth__rep0 0.24; ctrl_shuffle_orth__rep1 0.26; ctrl_shuffle_orth__rep2 0.31

#### midnight_clsmean / tolkach_esca / holdout  (seeds [0]; PARTIAL, single seed: not yet replicated; centre chance 0.500, floor 0.500; seen classes [0, 1, 2], held-out classes [3, 4, 5])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.981 | 1.000 [1.000, 1.000] | 0.998 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.923 | 0.973 [0.947, 0.993] | 0.972 | 0.31 [0.14, 0.46] | 0.31 | 0.331 [0.238, 0.423] | 0.403 |
| leace__rho1.0 | 3072 | nan | 0.986 | 1.000 [1.000, 1.000] | 0.998 | 0.98 [0.90, 1.11] | 0.98 | 0.994 [0.991, 0.997] | 0.998 |
| lda | 2 | nan | 0.987 | 0.910 [0.868, 0.946] | 0.789 | 0.13 [-0.11, 0.50] | 0.13 | 0.204 [0.172, 0.227] | 0.111 |
| lda_pca__d4 | 4 | nan | 0.989 | 0.898 [0.831, 0.950] | 0.808 | 0.12 [-0.05, 0.50] | 0.12 | 0.375 [0.346, 0.400] | 0.284 |
| lda_pca__d8 | 8 | nan | 0.978 | 0.973 [0.947, 0.998] | 0.928 | 0.37 [0.26, 0.59] | 0.37 | 0.506 [0.472, 0.539] | 0.437 |
| lda_pca__d16 | 16 | nan | 0.990 | 0.958 [0.930, 1.000] | 0.858 | 0.71 [0.34, 0.85] | 0.73 | 0.570 [0.531, 0.621] | 0.573 |
| lda_pca__d32 | 32 | nan | 0.987 | 0.984 [0.972, 0.998] | 0.943 | 0.88 [0.80, 0.95] | 0.88 | 0.658 [0.612, 0.703] | 0.671 |
| lda_pca__d64 | 64 | nan | 0.990 | 0.988 [0.973, 1.000] | 0.994 | 0.92 [0.82, 1.03] | 0.92 | 0.715 [0.681, 0.765] | 0.757 |
| lda_pca__d128 | 128 | nan | 0.983 | 0.991 [0.976, 1.000] | 0.995 | 0.96 [0.89, 1.04] | 0.96 | 0.705 [0.651, 0.768] | 0.792 |
| pca__d4 | 4 | nan | 0.927 | 0.903 [0.807, 0.968] | 0.791 | 0.20 [0.10, 0.56] | 0.24 | 0.402 [0.369, 0.450] | 0.337 |
| pca__d16 | 16 | nan | 0.939 | 0.952 [0.907, 1.000] | 0.884 | 0.80 [0.69, 0.89] | 0.81 | 0.564 [0.518, 0.621] | 0.577 |
| pca__d64 | 64 | nan | 0.967 | 0.982 [0.958, 0.998] | 0.962 | 0.89 [0.78, 1.01] | 0.91 | 0.676 [0.629, 0.733] | 0.750 |
| vib__beta0.001_dl32 | 32 | 19.8 | 0.990 | 0.814 [0.746, 0.885] | 0.738 | 0.13 [-0.06, 0.51] | 0.13 | 0.127 [0.102, 0.148] | 0.055 |
| vq__K16_m4 | 64 | 16.0 | 0.989 | 0.760 [0.696, 0.832] | 0.774 | 0.14 [-0.01, 0.49] | 0.16 | 0.191 [0.157, 0.225] | 0.124 |
| dann__dl64_lam3.0 | 64 | nan | 0.942 | 0.928 [0.896, 0.965] | 0.867 | 0.68 [0.49, 0.82] | 0.68 | 0.418 [0.388, 0.449] | 0.347 |
| adv_retrain__dl64_lam3.0 | 64 | nan | 0.991 | 0.980 [0.962, 0.998] | 0.939 | 0.49 [0.30, 0.71] | 0.49 | 0.513 [0.468, 0.556] | 0.468 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.988 | 0.965 [0.922, 1.000] | 0.965 | 0.58 [0.36, 0.83] | 0.58 | 0.465 [0.405, 0.512] | 0.429 |
| ae__beta0.001_dl32_lam10.0 | 32 | 149.8 | 0.983 | 0.993 [0.989, 0.997] | 0.952 | 0.27 [0.05, 0.47] | 0.27 | 0.618 [0.553, 0.677] | 0.557 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc [95% CI] | Δ retention xc [95% CI] | Δ L_worst | wins? |
|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.170 [-0.228, -0.113] | -0.531 [-0.584, -0.482] | -0.75 | no |
| vq | vq__K16_m4 | lda_pca__d64 | -0.228 [-0.295, -0.162] | -0.524 [-0.583, -0.487] | -0.78 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.060 [-0.089, -0.024] | -0.298 [-0.330, -0.274] | -0.25 | no |
| adv_retrain | adv_retrain__dl64_lam3.0 | lda_pca__d64 | -0.008 [-0.020, 0.006] | -0.203 [-0.243, -0.177] | -0.44 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.023 [-0.061, 0.009] | -0.250 [-0.308, -0.217] | -0.35 | no |
| ae | ae__beta0.001_dl32_lam10.0 | lda_pca__d32 | 0.009 [-0.003, 0.023] | -0.040 [-0.092, 0.000] | -0.61 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.50; ctrl_noise 0.51; ctrl_shuffle_orth__rep0 0.51; ctrl_shuffle_orth__rep1 0.51; ctrl_shuffle_orth__rep2 0.58

##### Kill-test tally (FOCUS A, revised): learned families (selected on V) beating the matched LDA arm on (i) held-class utility or (ii) retention with L within tolerance

| encoder | dataset | variant | family | config | wins |
|---|---|---|---|---|---|
| dinov2L_clsmean | tcga | holdout | vib | vib__beta0.001_dl32 | no |
| dinov2L_clsmean | tcga | holdout | vq | vq__K16_m4 | no |
| dinov2L_clsmean | tcga | holdout | dann | dann__dl64_lam3.0 | no |
| dinov2L_clsmean | tcga | holdout | adv_retrain | adv_retrain__dl64_lam1.0 | no |
| dinov2L_clsmean | tcga | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| dinov2L_clsmean | tcga | holdout | ae | ae__beta0.001_dl32_lam1.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | vib | vib__beta0.001_dl32 | no |
| dinov2L_clsmean | tolkach_esca | holdout | vq | vq__K64_m8 | no |
| dinov2L_clsmean | tolkach_esca | holdout | dann | dann__dl64_lam3.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | adv_retrain | adv_retrain__dl64_lam10.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| dinov2L_clsmean | tolkach_esca | holdout | ae | ae__beta0.001_dl32_lam0.0 | no |
| midnight_clsmean | tcga | holdout | vib | vib__beta0.01_dl32 | no |
| midnight_clsmean | tcga | holdout | vq | vq__K16_m1 | no |
| midnight_clsmean | tcga | holdout | dann | dann__dl64_lam3.0 | no |
| midnight_clsmean | tcga | holdout | adv_retrain | adv_retrain__dl64_lam3.0 | no |
| midnight_clsmean | tcga | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| midnight_clsmean | tcga | holdout | ae | ae__beta0.001_dl32_lam3.0 | no |
| midnight_clsmean | tolkach_esca | holdout | vib | vib__beta0.001_dl32 | no |
| midnight_clsmean | tolkach_esca | holdout | vq | vq__K16_m4 | no |
| midnight_clsmean | tolkach_esca | holdout | dann | dann__dl64_lam3.0 | no |
| midnight_clsmean | tolkach_esca | holdout | adv_retrain | adv_retrain__dl64_lam3.0 | no |
| midnight_clsmean | tolkach_esca | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| midnight_clsmean | tolkach_esca | holdout | ae | ae__beta0.001_dl32_lam10.0 | no |


## ROUND 2 (2026-11-04 ~13:30Z): power audit, 3 seeds, cross-fit and dither controls

Written after reading RANKING_R2.md. Everything in this section is on the SAME sealed test units in every seed (`RandomState(1234)`), so "3 seeds" replicate training noise
(adapter init, A/B/V draw), not test-unit noise; bootstrap intervals and MDEs below are conditional on that fixed test set. Cross-fit (below) is the replication that fixes this.

### Corrections to my earlier seed-0 write-up (made because they change the lead)

* The seed-0 "leakage-matched" gains (+0.083 midnight/Tolkach, +0.164 midnight/TCGA) were measured against the **LDA family only** (plain LDA, `lda_pca__d8`), a sparse and weak comparator at that leakage.
  Against the lower envelope of **all** inductive floors (adding per-centre whitening, PCA, random projection) the same picks are **+0.021 [+0.002, +0.046]** (midnight/Tolkach, seed 0; the best floor there is `centre_whiten`)
  and **+0.007 [-0.081, +0.097]** (midnight/TCGA, seed 0; `centre_whiten` again). The per-centre-whitening contrast was already in my seed-0 exploratory table (columns "best label-free ref") and I summarised the LDA-family column only. That was a selective read.
* **Confirmed:** midnight/TCGA seed 1, `ae__beta0.001_dl32_lam10.0` (L 0.32, held 0.778): +0.011 against plain LDA (0.767), -0.014 [-0.116, +0.084] against the full floor envelope. The +0.164 does not reproduce.
* The two AE configurations were picked after looking at the seed-0 test numbers (post hoc, best of 10); the V-selected AE differs from both of them in 7 of 12 cell-seeds (the validation-selected configuration coincides with one of them in 5).

### Power audit (H2-016 step 1; paired bootstrap over the sealed test units, MDE80 = 2.8 x SE of the paired difference)

Held-class utility difference of each ae configuration vs the best inductive floor with L <= L_AE + 0.05, all ae configs x 3 seeds per cell:

| cell | sealed test units / held-class cases | SE median [min, max] | **MDE80 median [min, max]** | median half-width of the 95% CI of L_worst (matching tolerance is 0.05) |
|---|---|---|---|---|
| midnight / TCGA | 48 units, 24 held-class cases | 0.049 [0.024, 0.076] | **0.138 [0.068, 0.213]** | 0.122 |
| dinov2-L / TCGA | 48, 24 | 0.033 [0.014, 0.076] | **0.093 [0.039, 0.211]** | 0.108 |
| midnight / Tolkach | 12 cases | 0.019 [0.003, 0.037] | **0.052 [0.008, 0.103]** | 0.224 |
| dinov2-L / Tolkach | 12 cases | 0.022 [0.010, 0.045] | **0.061 [0.029, 0.125]** | 0.264 |

Reading: the TCGA cells cannot detect a held-class difference smaller than about 0.09-0.14 (a +0.04 criterion is unresolvable there), the Tolkach cells about 0.05-0.06 (borderline for +0.04), and L itself is resolved only to +-0.11 (TCGA) / +-0.22-0.26 (Tolkach), five times the 0.05 matching tolerance.
The Tolkach intervals are narrow partly because held-class accuracy is near the ceiling (midnight 0.97-1.00). These SEs are a lower bound on the total error (test-unit noise only; no training noise, no selection of the comparator by point estimate on C).
The earlier statements "0 of 24" and "+0.083" are therefore both inside or near the undetectable zone for TCGA.

### 3-seed table, V-selected families vs the matched LDA arm AND vs the floor envelope (seed-pooled V selection, MDE next to every difference)


#### dinov2L_clsmean / tcga / holdout  (seeds [0, 1, 2]; centre chance 0.250, floor 0.250; seen classes [0, 1], held-out classes [2, 3])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 2048 | nan | 0.963 | 0.617 [0.525, 0.698] | 0.627 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.881 | 0.594 [0.509, 0.674] | 0.566 | 0.05 [0.02, 0.09] | 0.05 | 0.397 [0.381, 0.413] | 0.383 |
| leace__rho1.0 | 2048 | nan | 0.961 | 0.626 [0.529, 0.714] | 0.643 | 0.75 [0.68, 0.81] | 0.75 | 0.969 [0.963, 0.974] | 0.975 |
| lda | 1 | nan | 0.976 | 0.554 [0.423, 0.679] | 0.589 | 0.07 [0.06, 0.17] | 0.08 | 0.049 [0.044, 0.055] | 0.039 |
| lda_pca__d4 | 4 | nan | 0.977 | 0.566 [0.480, 0.646] | 0.594 | 0.08 [0.04, 0.18] | 0.10 | 0.302 [0.285, 0.323] | 0.264 |
| lda_pca__d8 | 8 | nan | 0.976 | 0.602 [0.512, 0.686] | 0.607 | 0.26 [0.19, 0.36] | 0.27 | 0.556 [0.535, 0.578] | 0.518 |
| lda_pca__d16 | 16 | nan | 0.970 | 0.650 [0.551, 0.738] | 0.647 | 0.51 [0.44, 0.61] | 0.52 | 0.704 [0.691, 0.718] | 0.689 |
| lda_pca__d32 | 32 | nan | 0.973 | 0.656 [0.566, 0.741] | 0.651 | 0.80 [0.74, 0.87] | 0.81 | 0.709 [0.697, 0.719] | 0.703 |
| lda_pca__d64 | 64 | nan | 0.962 | 0.615 [0.521, 0.703] | 0.655 | 0.85 [0.80, 0.91] | 0.85 | 0.629 [0.614, 0.644] | 0.629 |
| lda_pca__d128 | 128 | nan | 0.953 | 0.618 [0.523, 0.704] | 0.659 | 0.88 [0.85, 0.92] | 0.88 | 0.542 [0.517, 0.567] | 0.546 |
| pca__d4 | 4 | nan | 0.899 | 0.571 [0.506, 0.636] | 0.589 | 0.09 [0.03, 0.20] | 0.09 | 0.417 [0.393, 0.439] | 0.370 |
| pca__d16 | 16 | nan | 0.931 | 0.662 [0.562, 0.749] | 0.654 | 0.51 [0.44, 0.60] | 0.52 | 0.706 [0.692, 0.719] | 0.694 |
| pca__d64 | 64 | nan | 0.914 | 0.611 [0.516, 0.698] | 0.656 | 0.86 [0.81, 0.92] | 0.87 | 0.617 [0.601, 0.634] | 0.617 |
| vib__beta0.01_dl4 | 4 | 9.8 | 0.968 | 0.531 [0.428, 0.626] | 0.546 | 0.05 [0.03, 0.12] | 0.07 | 0.054 [0.050, 0.059] | 0.042 |
| vq__K16_m4 | 64 | 16.0 | 0.957 | 0.588 [0.502, 0.674] | 0.518 | 0.08 [0.05, 0.14] | 0.09 | 0.089 [0.083, 0.097] | 0.076 |
| dann__dl64_lam3.0 | 64 | nan | 0.945 | 0.622 [0.550, 0.690] | 0.576 | 0.59 [0.50, 0.68] | 0.59 | 0.387 [0.370, 0.403] | 0.349 |
| adv_retrain__dl64_lam1.0 | 64 | nan | 0.969 | 0.622 [0.541, 0.696] | 0.593 | 0.46 [0.40, 0.52] | 0.46 | 0.320 [0.308, 0.331] | 0.310 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.968 | 0.619 [0.542, 0.686] | 0.612 | 0.35 [0.29, 0.43] | 0.36 | 0.248 [0.226, 0.273] | 0.244 |
| ae__beta0.01_dl32_lam0.0 | 32 | 23.4 | 0.898 | 0.653 [0.567, 0.729] | 0.639 | 0.48 [0.40, 0.55] | 0.48 | 0.255 [0.243, 0.270] | 0.229 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | vib__beta0.01_dl4 | lda_pca__d4 | -0.036 [-0.094, +0.028] (MDE 0.084) | -0.248 [-0.266, -0.231] (MDE 0.025) | -0.03 | no | -0.064 [-0.149, +0.028] (MDE 0.129) vs centre_whiten, n_near=4 | no |
| vq | vq__K16_m4 | lda_pca__d64 | -0.027 [-0.113, +0.061] (MDE 0.127) | -0.539 [-0.557, -0.520] (MDE 0.028) | -0.77 | no | -0.006 [-0.089, +0.085] (MDE 0.120) vs centre_whiten, n_near=5 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | +0.007 [-0.055, +0.074] (MDE 0.093) | -0.242 [-0.264, -0.218] (MDE 0.033) | -0.26 | no | -0.040 [-0.094, +0.016] (MDE 0.081) vs pca__d16, n_near=0 | no |
| adv_retrain | adv_retrain__dl64_lam1.0 | lda_pca__d64 | +0.007 [-0.042, +0.057] (MDE 0.071) | -0.308 [-0.327, -0.289] (MDE 0.027) | -0.39 | no | -0.001 [-0.075, +0.063] (MDE 0.099) vs randproj__d16, n_near=0 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | +0.004 [-0.028, +0.036] (MDE 0.046) | -0.381 [-0.408, -0.348] (MDE 0.043) | -0.49 | no | -0.005 [-0.072, +0.057] (MDE 0.089) vs randproj__d16, n_near=1 | no |
| ae | ae__beta0.01_dl32_lam0.0 | lda_pca__d32 | -0.003 [-0.028, +0.024] (MDE 0.037) | -0.454 [-0.468, -0.439] (MDE 0.021) | -0.33 | no | -0.010 [-0.028, +0.010] (MDE 0.027) vs pca__d16, n_near=2 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.25; ctrl_noise 0.26; ctrl_shuffle_orth__rep0 0.27; ctrl_shuffle_orth__rep1 0.24; ctrl_shuffle_orth__rep2 0.27

#### dinov2L_clsmean / tolkach_esca / holdout  (seeds [0, 1, 2]; centre chance 0.500, floor 0.500; seen classes [0, 1, 2], held-out classes [3, 4, 5])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 2048 | nan | 0.945 | 0.954 [0.939, 0.975] | 0.966 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.880 | 0.860 [0.818, 0.895] | 0.880 | 0.03 [0.02, 0.11] | 0.03 | 0.460 [0.386, 0.523] | 0.462 |
| leace__rho1.0 | 2048 | nan | 0.938 | 0.952 [0.938, 0.973] | 0.966 | 0.91 [0.86, 0.96] | 0.91 | 0.984 [0.980, 0.988] | 0.983 |
| lda | 2 | nan | 0.976 | 0.562 [0.440, 0.607] | 0.521 | 0.06 [-0.06, 0.34] | 0.06 | 0.114 [0.098, 0.139] | 0.062 |
| lda_pca__d4 | 4 | nan | 0.965 | 0.635 [0.546, 0.699] | 0.638 | 0.06 [-0.02, 0.40] | 0.06 | 0.312 [0.261, 0.395] | 0.239 |
| lda_pca__d8 | 8 | nan | 0.956 | 0.816 [0.735, 0.884] | 0.806 | 0.08 [-0.01, 0.41] | 0.08 | 0.559 [0.518, 0.612] | 0.490 |
| lda_pca__d16 | 16 | nan | 0.961 | 0.892 [0.850, 0.930] | 0.880 | 0.11 [-0.03, 0.30] | 0.13 | 0.748 [0.729, 0.773] | 0.711 |
| lda_pca__d32 | 32 | nan | 0.961 | 0.910 [0.879, 0.931] | 0.915 | 0.25 [0.12, 0.44] | 0.27 | 0.809 [0.775, 0.838] | 0.804 |
| lda_pca__d64 | 64 | nan | 0.952 | 0.919 [0.905, 0.929] | 0.932 | 0.68 [0.55, 0.78] | 0.69 | 0.791 [0.724, 0.833] | 0.797 |
| lda_pca__d128 | 128 | nan | 0.942 | 0.919 [0.899, 0.939] | 0.951 | 0.91 [0.87, 0.95] | 0.91 | 0.707 [0.616, 0.774] | 0.722 |
| pca__d4 | 4 | nan | 0.854 | 0.742 [0.625, 0.825] | 0.759 | 0.09 [-0.01, 0.44] | 0.09 | 0.422 [0.365, 0.493] | 0.339 |
| pca__d16 | 16 | nan | 0.941 | 0.888 [0.846, 0.924] | 0.883 | 0.10 [0.01, 0.31] | 0.10 | 0.745 [0.723, 0.770] | 0.718 |
| pca__d64 | 64 | nan | 0.915 | 0.910 [0.899, 0.923] | 0.928 | 0.70 [0.57, 0.79] | 0.70 | 0.762 [0.692, 0.812] | 0.772 |
| vib__beta0.001_dl32 | 32 | 27.0 | 0.972 | 0.579 [0.488, 0.633] | 0.591 | 0.04 [-0.07, 0.33] | 0.04 | 0.088 [0.078, 0.100] | 0.048 |
| vq__K64_m8 | 128 | 48.0 | 0.969 | 0.653 [0.547, 0.728] | 0.685 | 0.02 [-0.06, 0.28] | 0.02 | 0.181 [0.156, 0.222] | 0.125 |
| dann__dl64_lam3.0 | 64 | nan | 0.939 | 0.846 [0.771, 0.900] | 0.844 | 0.15 [0.03, 0.31] | 0.15 | 0.425 [0.383, 0.490] | 0.360 |
| adv_retrain__dl64_lam3.0 | 64 | nan | 0.959 | 0.891 [0.872, 0.910] | 0.908 | 0.15 [0.05, 0.38] | 0.15 | 0.511 [0.483, 0.552] | 0.482 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.974 | 0.872 [0.840, 0.893] | 0.888 | 0.26 [0.13, 0.47] | 0.27 | 0.349 [0.314, 0.404] | 0.340 |
| ae__beta0.001_dl32_lam0.0 | 32 | 78.0 | 0.934 | 0.906 [0.875, 0.928] | 0.906 | 0.30 [0.15, 0.44] | 0.31 | 0.772 [0.748, 0.791] | 0.747 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.331 [-0.417, -0.297] (MDE 0.074) | -0.722 [-0.759, -0.677] (MDE 0.056) | -0.21 | no | -0.281 [-0.360, -0.219] (MDE 0.097) vs centre_whiten, n_near=6 | no |
| vq | vq__K64_m8 | lda_pca__d128 | -0.267 [-0.369, -0.194] (MDE 0.121) | -0.526 [-0.615, -0.394] (MDE 0.159) | -0.89 | no | -0.207 [-0.299, -0.144] (MDE 0.104) vs centre_whiten, n_near=4 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.073 [-0.149, -0.018] (MDE 0.093) | -0.366 [-0.439, -0.251] (MDE 0.142) | -0.53 | no | -0.046 [-0.085, -0.016] (MDE 0.051) vs lda_pca__d16, n_near=1 | no |
| adv_retrain | adv_retrain__dl64_lam3.0 | lda_pca__d64 | -0.028 [-0.049, -0.007] (MDE 0.029) | -0.281 [-0.335, -0.185] (MDE 0.112) | -0.53 | no | -0.001 [-0.024, +0.025] (MDE 0.035) vs lda_pca__d16, n_near=2 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.047 [-0.079, -0.030] (MDE 0.034) | -0.442 [-0.503, -0.338] (MDE 0.121) | -0.42 | no | -0.038 [-0.054, -0.027] (MDE 0.023) vs lda_pca__d32, n_near=2 | no |
| ae | ae__beta0.001_dl32_lam0.0 | lda_pca__d32 | -0.004 [-0.016, +0.006] (MDE 0.017) | -0.038 [-0.051, -0.025] (MDE 0.018) | +0.05 | no | -0.004 [-0.016, +0.006] (MDE 0.017) vs lda_pca__d32, n_near=0 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.50; ctrl_noise 0.51; ctrl_shuffle_orth__rep0 0.54; ctrl_shuffle_orth__rep1 0.55; ctrl_shuffle_orth__rep2 0.49

#### midnight_clsmean / tcga / holdout  (seeds [0, 1, 2]; centre chance 0.250, floor 0.250; seen classes [0, 1], held-out classes [2, 3])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.970 | 0.872 [0.744, 0.965] | 0.778 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.931 | 0.791 [0.696, 0.859] | 0.652 | 0.18 [0.13, 0.23] | 0.19 | 0.291 [0.267, 0.315] | 0.312 |
| leace__rho1.0 | 3072 | nan | 0.972 | 0.878 [0.748, 0.972] | 0.774 | 0.83 [0.77, 0.89] | 0.83 | 0.992 [0.985, 0.997] | 0.997 |
| lda | 1 | nan | 0.970 | 0.688 [0.592, 0.777] | 0.545 | 0.02 [0.01, 0.16] | 0.03 | 0.099 [0.086, 0.111] | 0.074 |
| lda_pca__d4 | 4 | nan | 0.979 | 0.634 [0.535, 0.719] | 0.546 | 0.30 [0.20, 0.46] | 0.30 | 0.327 [0.278, 0.379] | 0.285 |
| lda_pca__d8 | 8 | nan | 0.972 | 0.650 [0.564, 0.738] | 0.626 | 0.33 [0.24, 0.45] | 0.35 | 0.414 [0.370, 0.462] | 0.376 |
| lda_pca__d16 | 16 | nan | 0.973 | 0.749 [0.631, 0.854] | 0.677 | 0.48 [0.38, 0.59] | 0.50 | 0.484 [0.439, 0.530] | 0.487 |
| lda_pca__d32 | 32 | nan | 0.959 | 0.857 [0.774, 0.930] | 0.698 | 0.59 [0.50, 0.71] | 0.62 | 0.600 [0.555, 0.651] | 0.626 |
| lda_pca__d64 | 64 | nan | 0.948 | 0.844 [0.726, 0.933] | 0.731 | 0.67 [0.58, 0.78] | 0.67 | 0.704 [0.661, 0.746] | 0.733 |
| lda_pca__d128 | 128 | nan | 0.946 | 0.847 [0.740, 0.936] | 0.758 | 0.86 [0.80, 0.93] | 0.86 | 0.722 [0.674, 0.767] | 0.777 |
| pca__d4 | 4 | nan | 0.958 | 0.624 [0.537, 0.702] | 0.557 | 0.38 [0.27, 0.51] | 0.38 | 0.325 [0.276, 0.375] | 0.293 |
| pca__d16 | 16 | nan | 0.960 | 0.719 [0.596, 0.826] | 0.671 | 0.46 [0.37, 0.59] | 0.48 | 0.480 [0.438, 0.524] | 0.487 |
| pca__d64 | 64 | nan | 0.927 | 0.836 [0.720, 0.926] | 0.730 | 0.69 [0.60, 0.79] | 0.69 | 0.694 [0.650, 0.736] | 0.727 |
| vib__beta0.001_dl32 | 32 | 18.6 | 0.972 | 0.654 [0.580, 0.719] | 0.591 | 0.05 [0.01, 0.15] | 0.07 | 0.080 [0.071, 0.088] | 0.063 |
| vq__K16_m4 | 64 | 16.0 | 0.976 | 0.675 [0.580, 0.767] | 0.541 | 0.09 [0.04, 0.21] | 0.09 | 0.119 [0.097, 0.142] | 0.098 |
| dann__dl64_lam3.0 | 64 | nan | 0.950 | 0.675 [0.573, 0.766] | 0.603 | 0.52 [0.42, 0.65] | 0.52 | 0.331 [0.292, 0.373] | 0.338 |
| adv_retrain__dl64_lam3.0 | 64 | nan | 0.962 | 0.803 [0.711, 0.881] | 0.723 | 0.39 [0.30, 0.50] | 0.41 | 0.492 [0.444, 0.546] | 0.481 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.983 | 0.778 [0.687, 0.859] | 0.690 | 0.28 [0.21, 0.39] | 0.32 | 0.372 [0.315, 0.431] | 0.402 |
| ae__beta0.001_dl32_lam0.0 | 32 | 70.7 | 0.944 | 0.802 [0.700, 0.893] | 0.683 | 0.58 [0.51, 0.67] | 0.59 | 0.588 [0.546, 0.633] | 0.622 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.204 [-0.275, -0.122] (MDE 0.109) | -0.520 [-0.567, -0.477] (MDE 0.064) | -0.54 | no | -0.034 [-0.103, +0.045] (MDE 0.108) vs lda, n_near=1 | no |
| vq | vq__K16_m4 | lda_pca__d64 | -0.169 [-0.269, -0.060] (MDE 0.154) | -0.585 [-0.625, -0.543] (MDE 0.060) | -0.59 | no | -0.012 [-0.084, +0.078] (MDE 0.119) vs lda, n_near=0 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.169 [-0.252, -0.084] (MDE 0.125) | -0.373 [-0.407, -0.333] (MDE 0.054) | -0.15 | no | -0.116 [-0.188, -0.049] (MDE 0.103) vs centre_whiten, n_near=1 | no |
| adv_retrain | adv_retrain__dl64_lam3.0 | lda_pca__d64 | -0.041 [-0.092, +0.015] (MDE 0.077) | -0.212 [-0.251, -0.169] (MDE 0.061) | -0.28 | no | +0.012 [-0.044, +0.057] (MDE 0.074) vs centre_whiten, n_near=2 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.066 [-0.136, +0.015] (MDE 0.107) | -0.332 [-0.382, -0.280] (MDE 0.076) | -0.39 | no | -0.013 [-0.074, +0.044] (MDE 0.084) vs centre_whiten, n_near=1 | no |
| ae | ae__beta0.001_dl32_lam0.0 | lda_pca__d32 | -0.056 [-0.108, -0.010] (MDE 0.071) | -0.012 [-0.044, +0.018] (MDE 0.045) | -0.01 | no | -0.056 [-0.108, -0.010] (MDE 0.071) vs lda_pca__d32, n_near=2 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.25; ctrl_noise 0.26; ctrl_shuffle_orth__rep0 0.28; ctrl_shuffle_orth__rep1 0.28; ctrl_shuffle_orth__rep2 0.28

#### midnight_clsmean / tolkach_esca / holdout  (seeds [0, 1, 2]; centre chance 0.500, floor 0.500; seen classes [0, 1, 2], held-out classes [3, 4, 5])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.974 | 0.998 [0.993, 1.000] | 0.998 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.900 | 0.980 [0.961, 0.993] | 0.976 | 0.23 [0.06, 0.42] | 0.23 | 0.340 [0.260, 0.430] | 0.414 |
| leace__rho1.0 | 3072 | nan | 0.973 | 0.998 [0.993, 1.000] | 0.998 | 0.94 [0.88, 1.07] | 0.94 | 0.995 [0.992, 0.997] | 0.999 |
| lda | 2 | nan | 0.986 | 0.861 [0.809, 0.895] | 0.823 | 0.16 [0.04, 0.41] | 0.16 | 0.195 [0.167, 0.216] | 0.117 |
| lda_pca__d4 | 4 | nan | 0.987 | 0.871 [0.796, 0.925] | 0.808 | 0.14 [0.05, 0.43] | 0.15 | 0.355 [0.328, 0.387] | 0.283 |
| lda_pca__d8 | 8 | nan | 0.982 | 0.952 [0.914, 0.988] | 0.937 | 0.33 [0.15, 0.55] | 0.34 | 0.495 [0.461, 0.539] | 0.451 |
| lda_pca__d16 | 16 | nan | 0.984 | 0.963 [0.933, 0.990] | 0.913 | 0.77 [0.54, 0.91] | 0.78 | 0.546 [0.506, 0.603] | 0.572 |
| lda_pca__d32 | 32 | nan | 0.980 | 0.989 [0.982, 0.996] | 0.975 | 0.93 [0.86, 1.05] | 0.93 | 0.617 [0.575, 0.673] | 0.668 |
| lda_pca__d64 | 64 | nan | 0.965 | 0.994 [0.984, 1.000] | 0.996 | 0.99 [0.96, 1.05] | 0.99 | 0.692 [0.651, 0.758] | 0.758 |
| lda_pca__d128 | 128 | nan | 0.957 | 0.994 [0.984, 1.000] | 0.997 | 1.02 [0.98, 1.10] | 1.02 | 0.709 [0.663, 0.781] | 0.798 |
| pca__d4 | 4 | nan | 0.927 | 0.889 [0.808, 0.948] | 0.796 | 0.20 [0.07, 0.45] | 0.22 | 0.410 [0.376, 0.457] | 0.350 |
| pca__d16 | 16 | nan | 0.942 | 0.953 [0.916, 0.987] | 0.898 | 0.78 [0.64, 0.91] | 0.79 | 0.537 [0.494, 0.596] | 0.577 |
| pca__d64 | 64 | nan | 0.921 | 0.992 [0.980, 0.999] | 0.986 | 0.98 [0.95, 1.05] | 0.99 | 0.658 [0.612, 0.729] | 0.745 |
| vib__beta0.001_dl32 | 32 | 19.3 | 0.985 | 0.823 [0.780, 0.879] | 0.760 | 0.09 [-0.06, 0.36] | 0.09 | 0.111 [0.095, 0.123] | 0.061 |
| vq__K64_m8 | 128 | 48.0 | 0.987 | 0.858 [0.815, 0.944] | 0.773 | 0.12 [-0.02, 0.40] | 0.14 | 0.217 [0.183, 0.245] | 0.141 |
| dann__dl64_lam3.0 | 64 | nan | 0.949 | 0.906 [0.852, 0.943] | 0.823 | 0.84 [0.74, 0.92] | 0.84 | 0.407 [0.365, 0.445] | 0.379 |
| adv_retrain__dl64_lam3.0 | 64 | nan | 0.990 | 0.988 [0.977, 0.996] | 0.958 | 0.53 [0.34, 0.70] | 0.53 | 0.541 [0.504, 0.584] | 0.533 |
| supcon_xc__dl64_tau0.1 | 64 | nan | 0.974 | 0.985 [0.968, 0.999] | 0.980 | 0.63 [0.46, 0.81] | 0.63 | 0.478 [0.427, 0.527] | 0.456 |
| ae__beta0.001_dl32_lam10.0 | 32 | 140.3 | 0.966 | 0.978 [0.969, 0.988] | 0.940 | 0.25 [0.06, 0.48] | 0.27 | 0.556 [0.522, 0.597] | 0.536 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | vib__beta0.001_dl32 | lda_pca__d32 | -0.166 [-0.205, -0.112] (MDE 0.062) | -0.506 [-0.570, -0.466] (MDE 0.075) | -0.84 | no | (no floor at this leakage) | no |
| vq | vq__K64_m8 | lda_pca__d128 | -0.136 [-0.177, -0.046] (MDE 0.075) | -0.491 [-0.591, -0.427] (MDE 0.115) | -0.90 | no | -0.013 [-0.055, +0.066] (MDE 0.081) vs lda_pca__d4, n_near=2 | no |
| dann | dann__dl64_lam3.0 | lda_pca__d64 | -0.088 [-0.133, -0.057] (MDE 0.058) | -0.285 [-0.350, -0.244] (MDE 0.075) | -0.15 | no | -0.086 [-0.128, -0.056] (MDE 0.054) vs randproj__d64, n_near=1 | no |
| adv_retrain | adv_retrain__dl64_lam3.0 | lda_pca__d64 | -0.006 [-0.013, +0.000] (MDE 0.009) | -0.152 [-0.197, -0.125] (MDE 0.052) | -0.46 | no | +0.001 [-0.008, +0.008] (MDE 0.011) vs randproj__d16, n_near=0 | no |
| supcon_xc | supcon_xc__dl64_tau0.1 | lda_pca__d64 | -0.009 [-0.021, +0.001] (MDE 0.016) | -0.214 [-0.269, -0.175] (MDE 0.068) | -0.36 | no | -0.002 [-0.016, +0.009] (MDE 0.018) vs randproj__d16, n_near=0 | no |
| ae | ae__beta0.001_dl32_lam10.0 | lda_pca__d32 | -0.010 [-0.018, -0.004] (MDE 0.011) | -0.061 [-0.101, -0.032] (MDE 0.050) | -0.67 | no | -0.002 [-0.011, +0.010] (MDE 0.015) vs centre_whiten, n_near=2 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.50; ctrl_noise 0.52; ctrl_shuffle_orth__rep0 0.55; ctrl_shuffle_orth__rep1 0.56; ctrl_shuffle_orth__rep2 0.50

##### Kill-test tally (FOCUS A, revised): learned families (selected on V) beating the matched LDA arm on (i) held-class utility or (ii) retention with L within tolerance

| encoder | dataset | variant | family | config | wins |
|---|---|---|---|---|---|
| dinov2L_clsmean | tcga | holdout | vib | vib__beta0.01_dl4 | no |
| dinov2L_clsmean | tcga | holdout | vq | vq__K16_m4 | no |
| dinov2L_clsmean | tcga | holdout | dann | dann__dl64_lam3.0 | no |
| dinov2L_clsmean | tcga | holdout | adv_retrain | adv_retrain__dl64_lam1.0 | no |
| dinov2L_clsmean | tcga | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| dinov2L_clsmean | tcga | holdout | ae | ae__beta0.01_dl32_lam0.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | vib | vib__beta0.001_dl32 | no |
| dinov2L_clsmean | tolkach_esca | holdout | vq | vq__K64_m8 | no |
| dinov2L_clsmean | tolkach_esca | holdout | dann | dann__dl64_lam3.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | adv_retrain | adv_retrain__dl64_lam3.0 | no |
| dinov2L_clsmean | tolkach_esca | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| dinov2L_clsmean | tolkach_esca | holdout | ae | ae__beta0.001_dl32_lam0.0 | no |
| midnight_clsmean | tcga | holdout | vib | vib__beta0.001_dl32 | no |
| midnight_clsmean | tcga | holdout | vq | vq__K16_m4 | no |
| midnight_clsmean | tcga | holdout | dann | dann__dl64_lam3.0 | no |
| midnight_clsmean | tcga | holdout | adv_retrain | adv_retrain__dl64_lam3.0 | no |
| midnight_clsmean | tcga | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| midnight_clsmean | tcga | holdout | ae | ae__beta0.001_dl32_lam0.0 | no |
| midnight_clsmean | tolkach_esca | holdout | vib | vib__beta0.001_dl32 | no |
| midnight_clsmean | tolkach_esca | holdout | vq | vq__K64_m8 | no |
| midnight_clsmean | tolkach_esca | holdout | dann | dann__dl64_lam3.0 | no |
| midnight_clsmean | tolkach_esca | holdout | adv_retrain | adv_retrain__dl64_lam3.0 | no |
| midnight_clsmean | tolkach_esca | holdout | supcon_xc | supcon_xc__dl64_tau0.1 | no |
| midnight_clsmean | tolkach_esca | holdout | ae | ae__beta0.001_dl32_lam10.0 | no |

### Per-seed replication of the two post hoc AE picks (and which configuration validation selects), same test units


#### dinov2L_clsmean / tcga / holdout  (seeds [0, 1, 2]; sealed C identical across seeds)

Pre-registered picks (the two seed-0 post hoc AE configurations) vs the best inductive floor at L <= L_AE + 0.05:

| seed | config | L | held acc [95% CI] | best floor (held acc, L) | n floor pts within +-0.05 (P3) | Δ held [95% CI] | SE | MDE80 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ae__beta0.001_dl32_lam10.0 | 0.46 | 0.624 [0.527, 0.703] | pca__d16 (0.669, 0.44) | 3 | -0.046 [-0.104, +0.011] | 0.030 | 0.084 | no |
| 0 | ae__beta0.001_dl8_lam0.0 | 0.32 | 0.615 [0.516, 0.710] | lda_pca__d8 (0.647, 0.27) | 1 | -0.032 [-0.086, +0.013] | 0.025 | 0.070 | no |
| 1 | ae__beta0.001_dl32_lam10.0 | 0.39 | 0.674 [0.588, 0.751] | randproj__d16 (0.682, 0.31) | 0 | -0.008 [-0.064, +0.044] | 0.028 | 0.077 | no |
| 1 | ae__beta0.001_dl8_lam0.0 | 0.21 | 0.647 [0.559, 0.729] | lda_pca__d4 (0.615, 0.06) | 1 | +0.032 [-0.059, +0.125] | 0.048 | 0.133 | no |
| 2 | ae__beta0.001_dl32_lam10.0 | 0.40 | 0.601 [0.492, 0.699] | randproj__d16 (0.585, 0.37) | 1 | +0.017 [-0.074, +0.109] | 0.046 | 0.130 | no |
| 2 | ae__beta0.001_dl8_lam0.0 | 0.48 | 0.565 [0.456, 0.670] | lda_pca__d16 (0.614, 0.53) | 1 | -0.049 [-0.139, +0.037] | 0.046 | 0.129 | no |

Seeds on which each pre-registered pick wins (Δ >= +0.04, lower CI > 0, >= 2 floor points near its L): ae__beta0.001_dl32_lam10.0: 0 of 3; ae__beta0.001_dl8_lam0.0: 0 of 3

AE configuration selected on validation (V held-class accuracy among ae configs whose V leakage is within +0.05 of the best inductive floor's V leakage... simplified: max V held-class accuracy among ae configs with V_L <= median V_L of the ae grid), per seed and pooled over seeds:

| selection | config | L | held acc | best floor | Δ held [95% CI] | SE | MDE80 |
|---|---|---|---|---|---|---|---|
| seed 0 V-selected | ae__beta0.01_dl8_lam0.0 | 0.32 | 0.637 | lda_pca__d8 (0.647) | -0.010 [-0.058, +0.040] | 0.027 | 0.074 |
| seed 1 V-selected | ae__beta0.001_dl8_lam0.0 | 0.21 | 0.647 | lda_pca__d4 (0.615) | +0.032 [-0.059, +0.125] | 0.048 | 0.133 |
| seed 2 V-selected | ae__beta0.01_dl8_lam0.0 | 0.40 | 0.562 | randproj__d16 (0.585) | -0.022 [-0.119, +0.070] | 0.048 | 0.134 |

Per-seed V-selected ae configuration: seed 0: ae__beta0.01_dl8_lam0.0, seed 1: ae__beta0.001_dl8_lam0.0, seed 2: ae__beta0.01_dl8_lam0.0. Most frequent config chosen on 2 of 3 seeds.

Power audit, all ae configs x seeds with a floor comparator (n=30): SE of the paired Δ held: min 0.014, median 0.033, max 0.076; **MDE80 = 2.8 x SE: min 0.039, median 0.093, max 0.211**; median half-width of the 95% CI of L_worst: 0.108 (the leakage-matching tolerance is 0.05).


#### dinov2L_clsmean / tolkach_esca / holdout  (seeds [0, 1, 2]; sealed C identical across seeds)

Pre-registered picks (the two seed-0 post hoc AE configurations) vs the best inductive floor at L <= L_AE + 0.05:

| seed | config | L | held acc [95% CI] | best floor (held acc, L) | n floor pts within +-0.05 (P3) | Δ held [95% CI] | SE | MDE80 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ae__beta0.001_dl32_lam10.0 | 0.16 | 0.890 [0.861, 0.926] | pca__d16 (0.900, 0.17) | 3 | -0.010 [-0.039, +0.019] | 0.015 | 0.042 | no |
| 0 | ae__beta0.001_dl8_lam0.0 | 0.16 | 0.862 [0.812, 0.904] | pca__d16 (0.900, 0.17) | 3 | -0.038 [-0.074, -0.007] | 0.016 | 0.045 | no |
| 1 | ae__beta0.001_dl32_lam10.0 | 0.09 | 0.880 [0.815, 0.927] | randproj__d64 (0.881, 0.14) | 3 | -0.001 [-0.043, +0.031] | 0.019 | 0.053 | no |
| 1 | ae__beta0.001_dl8_lam0.0 | 0.05 | 0.797 [0.682, 0.877] | pca__d16 (0.881, 0.03) | 8 | -0.084 [-0.148, -0.039] | 0.028 | 0.078 | no |
| 2 | ae__beta0.001_dl32_lam10.0 | 0.19 | 0.851 [0.790, 0.924] | lda_pca__d16 (0.907, 0.17) | 1 | -0.056 [-0.099, -0.011] | 0.019 | 0.052 | no |
| 2 | ae__beta0.001_dl8_lam0.0 | 0.13 | 0.827 [0.740, 0.918] | lda_pca__d16 (0.907, 0.17) | 5 | -0.079 [-0.145, -0.020] | 0.032 | 0.091 | no |

Seeds on which each pre-registered pick wins (Δ >= +0.04, lower CI > 0, >= 2 floor points near its L): ae__beta0.001_dl32_lam10.0: 0 of 3; ae__beta0.001_dl8_lam0.0: 0 of 3

AE configuration selected on validation (V held-class accuracy among ae configs whose V leakage is within +0.05 of the best inductive floor's V leakage... simplified: max V held-class accuracy among ae configs with V_L <= median V_L of the ae grid), per seed and pooled over seeds:

| selection | config | L | held acc | best floor | Δ held [95% CI] | SE | MDE80 |
|---|---|---|---|---|---|---|---|
| seed 0 V-selected | ae__beta0.001_dl32_lam3.0 | 0.17 | 0.899 | pca__d16 (0.900) | -0.002 [-0.023, +0.018] | 0.010 | 0.029 |
| seed 1 V-selected | ae__beta0.001_dl32_lam10.0 | 0.09 | 0.880 | randproj__d64 (0.881) | -0.001 [-0.043, +0.031] | 0.019 | 0.053 |
| seed 2 V-selected | ae__beta0.001_dl32_lam3.0 | 0.08 | 0.817 | pca__d16 (0.883) | -0.067 [-0.106, -0.031] | 0.019 | 0.053 |

Per-seed V-selected ae configuration: seed 0: ae__beta0.001_dl32_lam3.0, seed 1: ae__beta0.001_dl32_lam10.0, seed 2: ae__beta0.001_dl32_lam3.0. Most frequent config chosen on 2 of 3 seeds.

Power audit, all ae configs x seeds with a floor comparator (n=28): SE of the paired Δ held: min 0.010, median 0.022, max 0.045; **MDE80 = 2.8 x SE: min 0.029, median 0.061, max 0.125**; median half-width of the 95% CI of L_worst: 0.264 (the leakage-matching tolerance is 0.05).


#### midnight_clsmean / tcga / holdout  (seeds [0, 1, 2]; sealed C identical across seeds)

Pre-registered picks (the two seed-0 post hoc AE configurations) vs the best inductive floor at L <= L_AE + 0.05:

| seed | config | L | held acc [95% CI] | best floor (held acc, L) | n floor pts within +-0.05 (P3) | Δ held [95% CI] | SE | MDE80 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ae__beta0.001_dl32_lam10.0 | 0.35 | 0.800 [0.688, 0.903] | centre_whiten (0.793, 0.23) | 3 | +0.007 [-0.081, +0.097] | 0.046 | 0.129 | no |
| 0 | ae__beta0.001_dl8_lam0.0 | 0.29 | 0.782 [0.668, 0.881] | centre_whiten (0.793, 0.23) | 2 | -0.011 [-0.125, +0.101] | 0.057 | 0.160 | no |
| 1 | ae__beta0.001_dl32_lam10.0 | 0.32 | 0.778 [0.688, 0.866] | centre_whiten (0.792, 0.17) | 4 | -0.014 [-0.116, +0.084] | 0.051 | 0.142 | no |
| 1 | ae__beta0.001_dl8_lam0.0 | 0.42 | 0.704 [0.606, 0.800] | centre_whiten (0.792, 0.17) | 0 | -0.088 [-0.194, +0.039] | 0.059 | 0.165 | no |
| 2 | ae__beta0.001_dl32_lam10.0 | 0.39 | 0.840 [0.784, 0.892] | centre_whiten (0.789, 0.14) | 4 | +0.051 [-0.029, +0.151] | 0.045 | 0.127 | no |
| 2 | ae__beta0.001_dl8_lam0.0 | 0.43 | 0.560 [0.427, 0.682] | centre_whiten (0.789, 0.14) | 4 | -0.229 [-0.325, -0.142] | 0.047 | 0.131 | no |

Seeds on which each pre-registered pick wins (Δ >= +0.04, lower CI > 0, >= 2 floor points near its L): ae__beta0.001_dl32_lam10.0: 0 of 3; ae__beta0.001_dl8_lam0.0: 0 of 3

AE configuration selected on validation (V held-class accuracy among ae configs whose V leakage is within +0.05 of the best inductive floor's V leakage... simplified: max V held-class accuracy among ae configs with V_L <= median V_L of the ae grid), per seed and pooled over seeds:

| selection | config | L | held acc | best floor | Δ held [95% CI] | SE | MDE80 |
|---|---|---|---|---|---|---|---|
| seed 0 V-selected | ae__beta0.001_dl32_lam1.0 | 0.46 | 0.756 | centre_whiten (0.793) | -0.037 [-0.095, +0.021] | 0.030 | 0.085 |
| seed 1 V-selected | ae__beta0.001_dl32_lam10.0 | 0.32 | 0.778 | centre_whiten (0.792) | -0.014 [-0.116, +0.084] | 0.051 | 0.142 |
| seed 2 V-selected | ae__beta0.01_dl32_lam0.0 | 0.29 | 0.710 | centre_whiten (0.789) | -0.079 [-0.162, +0.003] | 0.043 | 0.121 |

Per-seed V-selected ae configuration: seed 0: ae__beta0.001_dl32_lam1.0, seed 1: ae__beta0.001_dl32_lam10.0, seed 2: ae__beta0.01_dl32_lam0.0. Most frequent config chosen on 1 of 3 seeds.

Power audit, all ae configs x seeds with a floor comparator (n=30): SE of the paired Δ held: min 0.024, median 0.049, max 0.076; **MDE80 = 2.8 x SE: min 0.068, median 0.138, max 0.213**; median half-width of the 95% CI of L_worst: 0.122 (the leakage-matching tolerance is 0.05).


#### midnight_clsmean / tolkach_esca / holdout  (seeds [0, 1, 2]; sealed C identical across seeds)

Pre-registered picks (the two seed-0 post hoc AE configurations) vs the best inductive floor at L <= L_AE + 0.05:

| seed | config | L | held acc [95% CI] | best floor (held acc, L) | n floor pts within +-0.05 (P3) | Δ held [95% CI] | SE | MDE80 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ae__beta0.001_dl32_lam10.0 | 0.27 | 0.993 [0.989, 0.997] | centre_whiten (0.973, 0.31) | 1 | +0.021 [+0.002, +0.046] | 0.012 | 0.032 | no |
| 0 | ae__beta0.001_dl8_lam0.0 | 0.47 | 0.919 [0.857, 0.969] | lda_pca__d8 (0.973, 0.37) | 1 | -0.054 [-0.109, -0.001] | 0.031 | 0.087 | no |
| 1 | ae__beta0.001_dl32_lam10.0 | 0.22 | 0.966 [0.944, 0.987] | centre_whiten (0.978, 0.15) | 1 | -0.012 [-0.029, -0.001] | 0.008 | 0.021 | no |
| 1 | ae__beta0.001_dl8_lam0.0 | 0.32 | 0.881 [0.802, 0.938] | centre_whiten (0.978, 0.15) | 1 | -0.097 [-0.173, -0.041] | 0.037 | 0.103 | no |
| 2 | ae__beta0.001_dl32_lam10.0 | 0.29 | 0.975 [0.967, 0.983] | centre_whiten (0.991, 0.25) | 4 | -0.016 [-0.029, -0.002] | 0.006 | 0.018 | no |
| 2 | ae__beta0.001_dl8_lam0.0 | 0.55 | 0.914 [0.840, 0.962] | centre_whiten (0.991, 0.25) | 0 | -0.077 [-0.151, -0.030] | 0.035 | 0.098 | no |

Seeds on which each pre-registered pick wins (Δ >= +0.04, lower CI > 0, >= 2 floor points near its L): ae__beta0.001_dl32_lam10.0: 0 of 3; ae__beta0.001_dl8_lam0.0: 0 of 3

AE configuration selected on validation (V held-class accuracy among ae configs whose V leakage is within +0.05 of the best inductive floor's V leakage... simplified: max V held-class accuracy among ae configs with V_L <= median V_L of the ae grid), per seed and pooled over seeds:

| selection | config | L | held acc | best floor | Δ held [95% CI] | SE | MDE80 |
|---|---|---|---|---|---|---|---|
| seed 0 V-selected | ae__beta0.001_dl32_lam10.0 | 0.27 | 0.993 | centre_whiten (0.973) | +0.021 [+0.002, +0.046] | 0.012 | 0.032 |
| seed 1 V-selected | ae__beta0.001_dl32_lam3.0 | 0.49 | 0.965 | randproj__d16 (0.990) | -0.026 [-0.043, -0.010] | 0.009 | 0.025 |
| seed 2 V-selected | ae__beta0.001_dl32_lam10.0 | 0.29 | 0.975 | centre_whiten (0.991) | -0.016 [-0.029, -0.002] | 0.006 | 0.018 |

Per-seed V-selected ae configuration: seed 0: ae__beta0.001_dl32_lam10.0, seed 1: ae__beta0.001_dl32_lam3.0, seed 2: ae__beta0.001_dl32_lam10.0. Most frequent config chosen on 2 of 3 seeds.

Power audit, all ae configs x seeds with a floor comparator (n=30): SE of the paired Δ held: min 0.003, median 0.019, max 0.037; **MDE80 = 2.8 x SE: min 0.008, median 0.052, max 0.103**; median half-width of the 95% CI of L_worst: 0.224 (the leakage-matching tolerance is 0.05).



### Cross-fit evaluation (H2-016 step 3) and dither / reconstruction-only controls (H2-014)

`run_sweep.py --crossfit 4`: the sealed test units ROTATE. Per (centre, class) stratum the units are shuffled once (`RandomState(1234)`) and cut into 4 chunks; run seed k tests chunk k, so the four folds are unit-disjoint and
together test every one of the 192 TCGA cases / 45 Tolkach cases exactly once (checked); adapters, probes, attackers are refit per fold and the A/B/V draw differs per fold. Reported numbers are POOLED OUT-OF-FOLD (bootstrap over all test units
jointly); the validation selection uses the fold-mean V metrics (seed-pooled V, ranker P4). Grid `p3`: references (raw, centre_whiten, leace, lda, lda_pca, pca, randproj), **noise-dithered label-free codes with NO adversary** (PCA-d, LDA+PCA-d, random projection, and the AE with
lam = 0 as base; release z = z0 + r * std_A(z0_j) * eps, eps fixed per patch, same noise for attacker training, probe training and test), **reconstruction-only AE** (lam 0, beta 1e-4), and the 10-config AE grid.
Cells: midnight x {TCGA, Tolkach}, 4 folds each (dinov2-L was not cross-fitted; its fixed-C 3-seed result above is already null).

Controls: constant embedding exactly 0.50 / 0.25 in every fold (pooled 0.60 on Tolkach is an artefact of pooling four different constant predictions over folds with different class mixes; per fold it is 0.500);
noise embedding 0.52 / 0.25; orthogonal shuffled secret 0.47-0.52 (Tolkach, chance 0.5) and 0.25-0.27 (TCGA, chance 0.25).

**Result (pooled out-of-fold, V-selected AE, difference in held-class utility to the best inductive floor with L <= L_AE + 0.05; floors incl. all dithered codes; MDE80 = 2.8 x SE):**

| cell | V-selected ae | L | held acc | best floor at <= equal leakage | Δ held [95% CI] (MDE80) | n floor pts near L | beats floor envelope? |
|---|---|---|---|---|---|---|---|
| midnight / TCGA (192 cases, 96 held-class) | ae dl32 lam1 | 0.46 | 0.810 | dither PCA-32, r 0.5 (0.839) | **-0.028 [-0.063, +0.001] (MDE 0.048)** | 5 | no |
| midnight / Tolkach (45 cases) | ae dl32 lam10 | 0.37 | 0.971 | dither random-proj-32, r 1 | **+0.003 [-0.007, +0.012] (MDE 0.014)** | 3 | no |

Cross-fitting cut the MDE80 from 0.07-0.14 (TCGA) and 0.02-0.10 (Tolkach) in the fixed-C analysis to 0.048 and 0.014. The Tolkach contrast is now a resolved null (an effect of +0.04 would have been seen with >99% power); TCGA excludes anything above about +0.05.
Over all 12 AE configurations per cell no configuration has a positive held-class Δ against the floor envelope with CI above 0 (TCGA: best -0.015, Tolkach: best +0.003 at CI [-0.007, +0.012]).

**H2-014 (dither, no adversary) vs the AE at equal leakage (pooled out-of-fold; columns are L, held-class acc, retention xc):** the dithered label-free codes match the adversarial AE on utility and the adversary adds nothing on held-class accuracy; the AE keeps more raw-neighbourhood structure than
*dithered* codes (random noise destroys nearest-neighbour structure), but not more than the undithered low-dimensional floors (see the retention column of the table above: best undithered floor 0.487 vs AE 0.40-0.47 at L 0.3-0.4 on TCGA).
The reconstruction-only AE (lam 0, beta 1e-4) is the weakest AE family at its leakage (TCGA dl8: held 0.715 at L 0.43 vs dithered PCA-32 0.839 at L 0.45).


#### AE configurations vs the nearest dithered label-free codes (pooled out-of-fold)

| cell | AE config | L | held acc | retention xc | nearest dithered label-free codes with |ΔL| <= 0.05: config (L, held, retention) |
|---|---|---|---|---|---|
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.0001_dl32_lam0.0 | 0.62 | 0.849 | 0.699 | (none) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.0001_dl8_lam0.0 | 0.43 | 0.715 | 0.514 | basepca_d32_r0.5 (0.45, 0.839, 0.47), baseae_beta0.001_dl32_r0.5 (0.48, 0.821, 0.40) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl32_lam0.0 | 0.62 | 0.853 | 0.637 | (none) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl32_lam1.0 | 0.46 | 0.810 | 0.565 | basepca_d32_r0.5 (0.45, 0.839, 0.47), baseae_beta0.001_dl32_r0.5 (0.48, 0.821, 0.40) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl32_lam10.0 | 0.34 | 0.803 | 0.465 | basepca_d32_r1.0 (0.33, 0.802, 0.21), baselda_pca_d32_r1.0 (0.33, 0.797, 0.20), baseae_beta0.0001_dl32_r1.0 (0.35, 0.776, 0.17) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl32_lam3.0 | 0.45 | 0.779 | 0.529 | basepca_d32_r0.5 (0.45, 0.839, 0.47), baseae_beta0.001_dl32_r0.5 (0.48, 0.821, 0.40) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl8_lam0.0 | 0.41 | 0.732 | 0.489 | basepca_d32_r0.5 (0.45, 0.839, 0.47) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl8_lam1.0 | 0.31 | 0.717 | 0.409 | basepca_d32_r1.0 (0.33, 0.802, 0.21), baselda_pca_d32_r1.0 (0.33, 0.797, 0.20), baserandproj_d32_r1.0 (0.27, 0.793, 0.25) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl8_lam10.0 | 0.19 | 0.674 | 0.240 | baserandproj_d32_r2.0 (0.17, 0.733, 0.09), basepca_d32_r2.0 (0.20, 0.712, 0.07), baselda_pca_d32_r2.0 (0.18, 0.703, 0.07) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.001_dl8_lam3.0 | 0.30 | 0.673 | 0.334 | basepca_d32_r1.0 (0.33, 0.802, 0.21), baselda_pca_d32_r1.0 (0.33, 0.797, 0.20), baserandproj_d32_r1.0 (0.27, 0.793, 0.25) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.01_dl32_lam0.0 | 0.37 | 0.768 | 0.234 | basepca_d32_r1.0 (0.33, 0.802, 0.21), baselda_pca_d32_r1.0 (0.33, 0.797, 0.20), baseae_beta0.0001_dl32_r1.0 (0.35, 0.776, 0.17) |
| midnight_clsmean / tcga / holdout_cv4 | ae__beta0.01_dl8_lam0.0 | 0.33 | 0.686 | 0.339 | basepca_d32_r1.0 (0.33, 0.802, 0.21), baselda_pca_d32_r1.0 (0.33, 0.797, 0.20), baseae_beta0.0001_dl32_r1.0 (0.35, 0.776, 0.17) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.0001_dl32_lam0.0 | 0.83 | 0.959 | 0.728 | (none) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.0001_dl8_lam0.0 | 0.46 | 0.911 | 0.582 | basepca_d32_r1.0 (0.42, 0.882, 0.21), baselda_pca_d32_r1.0 (0.43, 0.860, 0.20), basepca_d16_r0.5 (0.49, 0.857, 0.32) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl32_lam0.0 | 0.76 | 0.939 | 0.623 | (none) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl32_lam1.0 | 0.51 | 0.949 | 0.601 | basepca_d16_r0.5 (0.49, 0.857, 0.32) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl32_lam10.0 | 0.37 | 0.971 | 0.609 | basepca_d32_r1.0 (0.42, 0.882, 0.21), baseae_beta0.0001_dl32_r1.0 (0.41, 0.799, 0.18), baseae_beta0.001_dl32_r1.0 (0.41, 0.799, 0.17) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl32_lam3.0 | 0.45 | 0.931 | 0.578 | basepca_d32_r1.0 (0.42, 0.882, 0.21), baselda_pca_d32_r1.0 (0.43, 0.860, 0.20), basepca_d16_r0.5 (0.49, 0.857, 0.32) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl8_lam0.0 | 0.40 | 0.881 | 0.538 | basepca_d32_r1.0 (0.42, 0.882, 0.21), baselda_pca_d32_r1.0 (0.43, 0.860, 0.20), baseae_beta0.0001_dl32_r1.0 (0.41, 0.799, 0.18) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl8_lam1.0 | 0.22 | 0.846 | 0.484 | baserandproj_d32_r2.0 (0.17, 0.889, 0.10), basepca_d32_r2.0 (0.22, 0.722, 0.07), baselda_pca_d32_r2.0 (0.26, 0.698, 0.07) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl8_lam10.0 | 0.17 | 0.908 | 0.440 | baserandproj_d32_r2.0 (0.17, 0.889, 0.10), basepca_d32_r2.0 (0.22, 0.722, 0.07), basepca_d16_r2.0 (0.18, 0.582, 0.05) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.001_dl8_lam3.0 | 0.17 | 0.865 | 0.456 | baserandproj_d32_r2.0 (0.17, 0.889, 0.10), basepca_d32_r2.0 (0.22, 0.722, 0.07), basepca_d16_r2.0 (0.18, 0.582, 0.05) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.01_dl32_lam0.0 | 0.39 | 0.828 | 0.191 | basepca_d32_r1.0 (0.42, 0.882, 0.21), baselda_pca_d32_r1.0 (0.43, 0.860, 0.20), baseae_beta0.0001_dl32_r1.0 (0.41, 0.799, 0.18) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | ae__beta0.01_dl8_lam0.0 | 0.27 | 0.725 | 0.338 | baserandproj_d32_r1.0 (0.30, 0.969, 0.27), baselda_pca_d32_r2.0 (0.26, 0.698, 0.07), baseae_beta0.001_dl32_r2.0 (0.24, 0.646, 0.06) |

#### Cross-fit compact output (references and V-selected configuration)

#### midnight_clsmean / tcga / holdout_cv4  (seeds [0, 1, 2, 3]; centre chance 0.250, floor 0.250; seen classes [0, 1], held-out classes [2, 3])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.982 | 0.891 [0.833, 0.936] | 0.785 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.947 | 0.823 [0.776, 0.866] | 0.675 | 0.14 [0.10, 0.19] | 0.14 | 0.306 [0.289, 0.321] | 0.320 |
| leace__rho1.0 | 3072 | nan | 0.984 | 0.889 [0.829, 0.935] | 0.781 | 0.82 [0.76, 0.88] | 0.82 | 0.984 [0.973, 0.992] | 0.996 |
| lda | 1 | nan | 0.986 | 0.659 [0.592, 0.722] | 0.595 | 0.08 [0.01, 0.19] | 0.08 | 0.095 [0.088, 0.102] | 0.074 |
| lda_pca__d4 | 4 | nan | 0.989 | 0.667 [0.599, 0.731] | 0.579 | 0.32 [0.24, 0.42] | 0.32 | 0.359 [0.332, 0.386] | 0.302 |
| lda_pca__d8 | 8 | nan | 0.990 | 0.705 [0.644, 0.764] | 0.640 | 0.42 [0.34, 0.51] | 0.42 | 0.448 [0.419, 0.474] | 0.395 |
| lda_pca__d16 | 16 | nan | 0.976 | 0.811 [0.752, 0.868] | 0.710 | 0.49 [0.40, 0.57] | 0.49 | 0.552 [0.524, 0.580] | 0.523 |
| lda_pca__d32 | 32 | nan | 0.975 | 0.856 [0.798, 0.911] | 0.736 | 0.61 [0.54, 0.69] | 0.61 | 0.668 [0.641, 0.696] | 0.657 |
| lda_pca__d64 | 64 | nan | 0.973 | 0.884 [0.831, 0.931] | 0.731 | 0.72 [0.64, 0.80] | 0.72 | 0.749 [0.722, 0.775] | 0.749 |
| pca__d4 | 4 | nan | 0.985 | 0.620 [0.563, 0.684] | 0.542 | 0.36 [0.28, 0.46] | 0.36 | 0.367 [0.338, 0.396] | 0.309 |
| pca__d16 | 16 | nan | 0.952 | 0.801 [0.736, 0.862] | 0.703 | 0.49 [0.40, 0.58] | 0.49 | 0.548 [0.519, 0.576] | 0.526 |
| pca__d64 | 64 | nan | 0.954 | 0.883 [0.828, 0.932] | 0.745 | 0.70 [0.64, 0.79] | 0.70 | 0.740 [0.713, 0.765] | 0.744 |
| ae__beta0.001_dl32_lam1.0 | 32 | 106.5 | 0.959 | 0.810 [0.759, 0.856] | 0.713 | 0.46 [0.40, 0.54] | 0.47 | 0.565 [0.538, 0.591] | 0.561 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | (no config within the leakage tolerance on V) | | | | | no | | no |
| vq | (no config within the leakage tolerance on V) | | | | | no | | no |
| dann | (no config within the leakage tolerance on V) | | | | | no | | no |
| adv_retrain | (no config within the leakage tolerance on V) | | | | | no | | no |
| supcon_xc | (no config within the leakage tolerance on V) | | | | | no | | no |
| ae | ae__beta0.001_dl32_lam1.0 | lda_pca__d32 | -0.046 [-0.092, -0.006] (MDE 0.063) | -0.103 [-0.124, -0.082] (MDE 0.030) | -0.16 | no | -0.028 [-0.063, +0.001] (MDE 0.048) vs dither__basepca_d32_r0.5, n_near=5 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.25; ctrl_noise 0.25; ctrl_shuffle_orth__rep0 0.25; ctrl_shuffle_orth__rep1 0.26; ctrl_shuffle_orth__rep2 0.27

#### midnight_clsmean / tolkach_esca / holdout_cv4  (seeds [0, 1, 2, 3]; centre chance 0.500, floor 0.500; seen classes [0, 1, 2], held-out classes [3, 4, 5])


References and V-selected learned configurations (absolute numbers on sealed C):

| config | dim | rate (bits) | seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |
|---|---|---|---|---|---|---|---|---|---|
| raw | 3072 | nan | 0.985 | 0.998 [0.996, 0.999] | 0.997 | 1.00 [1.00, 1.00] | 1.00 | 1.000 [1.000, 1.000] | 1.000 |
| centre_whiten | 64 | nan | 0.919 | 0.950 [0.910, 0.976] | 0.957 | 0.22 [0.17, 0.32] | 0.22 | 0.415 [0.368, 0.465] | 0.440 |
| leace__rho1.0 | 3072 | nan | 0.986 | 0.998 [0.996, 0.999] | 0.997 | 0.91 [0.87, 0.96] | 0.91 | 0.998 [0.996, 0.999] | 0.999 |
| lda | 2 | nan | 0.981 | 0.785 [0.731, 0.835] | 0.721 | 0.11 [0.02, 0.29] | 0.11 | 0.177 [0.157, 0.195] | 0.101 |
| lda_pca__d4 | 4 | nan | 0.984 | 0.843 [0.803, 0.880] | 0.736 | 0.12 [0.05, 0.30] | 0.13 | 0.349 [0.324, 0.374] | 0.251 |
| lda_pca__d8 | 8 | nan | 0.984 | 0.954 [0.931, 0.972] | 0.916 | 0.30 [0.22, 0.39] | 0.31 | 0.514 [0.490, 0.539] | 0.424 |
| lda_pca__d16 | 16 | nan | 0.978 | 0.921 [0.871, 0.960] | 0.879 | 0.67 [0.59, 0.76] | 0.72 | 0.573 [0.542, 0.604] | 0.550 |
| lda_pca__d32 | 32 | nan | 0.984 | 0.954 [0.909, 0.984] | 0.956 | 0.84 [0.79, 0.91] | 0.86 | 0.673 [0.642, 0.706] | 0.660 |
| lda_pca__d64 | 64 | nan | 0.974 | 0.982 [0.954, 0.995] | 0.994 | 0.91 [0.86, 0.95] | 0.91 | 0.746 [0.718, 0.780] | 0.765 |
| pca__d4 | 4 | nan | 0.956 | 0.926 [0.897, 0.949] | 0.820 | 0.21 [0.14, 0.31] | 0.24 | 0.433 [0.412, 0.457] | 0.344 |
| pca__d16 | 16 | nan | 0.958 | 0.930 [0.883, 0.966] | 0.897 | 0.72 [0.64, 0.80] | 0.76 | 0.572 [0.540, 0.606] | 0.553 |
| pca__d64 | 64 | nan | 0.968 | 0.986 [0.971, 0.994] | 0.980 | 0.88 [0.82, 0.92] | 0.88 | 0.714 [0.681, 0.754] | 0.758 |
| ae__beta0.001_dl32_lam10.0 | 32 | 162.0 | 0.979 | 0.971 [0.958, 0.983] | 0.931 | 0.37 [0.29, 0.45] | 0.40 | 0.609 [0.580, 0.642] | 0.544 |

Per family, configuration selected on V (L_V within +0.05 of the matched LDA arm; then max V held-class acc):

| family | config | matched LDA | Δ held-class acc vs matched LDA [95% CI] (MDE80) | Δ retention xc [95% CI] (MDE80) | Δ L_worst | wins vs matched LDA? | Δ utility vs best inductive floor at <= equal leakage [95% CI] (MDE80), n_near | beats floor envelope (Δ >= +0.04, CI > 0, n_near >= 2)? |
|---|---|---|---|---|---|---|---|---|
| vib | (no config within the leakage tolerance on V) | | | | | no | | no |
| vq | (no config within the leakage tolerance on V) | | | | | no | | no |
| dann | (no config within the leakage tolerance on V) | | | | | no | | no |
| adv_retrain | (no config within the leakage tolerance on V) | | | | | no | | no |
| supcon_xc | (no config within the leakage tolerance on V) | | | | | no | | no |
| ae | ae__beta0.001_dl32_lam10.0 | lda_pca__d32 | +0.017 [-0.008, +0.058] (MDE 0.048) | -0.063 [-0.090, -0.039] (MDE 0.035) | -0.47 | no | +0.003 [-0.007, +0.012] (MDE 0.014) vs dither__baserandproj_d32_r1.0, n_near=3 | no |

Controls (worst of the 4 attackers, centre bacc on C): ctrl_const 0.60; ctrl_noise 0.52; ctrl_shuffle_orth__rep0 0.49; ctrl_shuffle_orth__rep1 0.52; ctrl_shuffle_orth__rep2 0.47

##### Kill-test tally (FOCUS A, revised): learned families (selected on V) beating the matched LDA arm on (i) held-class utility or (ii) retention with L within tolerance

| encoder | dataset | variant | family | config | wins |
|---|---|---|---|---|---|
| midnight_clsmean | tcga | holdout_cv4 | ae | ae__beta0.001_dl32_lam1.0 | no |
| midnight_clsmean | tolkach_esca | holdout_cv4 | ae | ae__beta0.001_dl32_lam10.0 | no |

### VERDICT on the post hoc AE lead (2026-11-04 ~14:15Z): **DEAD**

* Pre-registered margin (RANKING_R2 section 7, AE-DEAD (a): 3-seed V-selected Δ_held < +0.02 in either midnight cell): fixed-C 3 seeds, midnight/TCGA -0.056 [-0.108, -0.010] (MDE 0.071), midnight/Tolkach -0.002 [-0.011, +0.010] (MDE 0.015); met in both.
* Replication of the two seed-0 picks: 0 of 3 seeds on every cell win against the floor envelope (Δ >= +0.04, CI above 0, >= 2 floor points near L); seed 0 midnight/TCGA +0.007 [-0.081, +0.097], seed 1 -0.014 (the +0.164 / +0.011 pair in the ranking is the LDA-family-only comparator).
* Cross-fit (test-unit noise replicated, MDE 0.048 / 0.014): -0.028 [-0.063, +0.001] and +0.003 [-0.007, +0.012]; (b): dithered / undithered floors within 0.03 of the AE in both midnight cells at equal leakage.
* Not run: H2-015 selection-null (criterion (c)); phikon-v2 cells; dinov2-L cross-fit. None can revive a lead that is null in both midnight cells against a resolved MDE.
* The label-supervised families (vib, vq, dann, adv_retrain, supcon_xc) stay below the matched LDA arm and below the floor envelope in all four fixed-C cells (table above).


## AUDIT FOLLOW-UP: PRE-REGISTRATION (written 2026-11-04 ~18:15Z VM clock, BEFORE any of the numbers below it; tightening only, original verdicts stay alongside)

Trigger: Auditor-1 findings F-02, F-03, F-07, F-08, F-09, F-14 (research/fields/AUDIT_findings.md) and the AUDIT CORRECTIONS block of FOCUS.md. Original verdicts, unchanged by this block: gate G1 not met; held-class LABEL utility of the label-free AE: no gain
(cross-fit midnight/TCGA -0.028 [-0.063, +0.001], midnight/Tolkach +0.003 [-0.007, +0.012]). What is reopened: criterion (ii) retention, the F-02 adversary attribution, and the untested encoders.

**Definitions changed (hygiene, F-08).** "Inductive floors" = raw, leace, lda, lda_pca, pca, randproj and the noise-dithered codes (all fitted on adapter-train units only). `centre_whiten` is a **transductive reference** (it needs the centre id of each released patch and
the test split's own statistics) and is reported in its own column, not in the inductive envelope. Every difference gets a **headroom to raw** (raw minus the best inductive floor at the same leakage, i.e. the most any adapter could gain) and an MDE80; a cell with headroom < 0.04 on the metric
is labelled "ceiling-limited, not scored" for that metric. Pooled cross-fit L is reported as is (the Tolkach pooled constant-embedding control reads 0.60 because four different constant predictions are pooled; per fold it is 0.50), and the V size is stated.

**(2) H2-015 selection null.** For each cell (fixed-C 3-seed pools and the two cross-fit cells), statistic T_obs = max over the 12 `ae` configurations of the difference (held-class accuracy; separately kNN-retention-xc) to the best inductive floor with L <= L_c + 0.05.
Null A (frontier permutation): replace the 12 `ae` configurations by 12 inductive-floor points drawn at random with |L - L_ae| <= 0.1 (one per `ae` config, 5000 draws), compare each to the remaining floors exactly as for the AE, record the max. Null B (max-T bootstrap): over 1000 unit bootstraps,
T^b = max_c (D^b_c - D_c) where D^b_c is the difference recomputed on the resample and D_c the point estimate. A best-of-12 gain counts as "above selection noise" only if T_obs exceeds the 95th percentile of BOTH nulls.
Observed best-of-12 gains that do not exceed them are described as selection noise, whatever their point estimate.

**(3) Replications.** dinov2-L cross-fit (K=4 rotating test units, TCGA and Tolkach) and phikon-v2 held-out-class protocol (fixed C, seed 0, TCGA and Tolkach; single seed, labelled "not yet replicated"), grid `p4` (= `p3` minus 14 dithered / duplicate floor
points; keeps all 12 `ae` configurations, the reconstruction-only AE, lda_pca, pca, randproj, a reduced dither ladder, controls). Decision rules, unchanged from FOCUS A: (i) held-class utility win = Delta >= +0.04 with CI > 0 against the inductive envelope and >= 2 floor points within +-0.05 of L;
(ii) retention win = Delta >= +0.05 with CI > 0, >= 2 floor points near L, **and** held-class utility not worse than the matched floor by more than 0.03 (added: retention must not be bought with utility). Cells with headroom < 0.04 on a metric are not scored on that metric.
Combined verdict language: "dead on label utility, mixed on retention" is kept unless (3) gives retention wins in both new encoders (-> retention ALIVE) or in none of the four encoder-cells including midnight/TCGA (-> retention DEAD).

**(4) F-02 adversary convergence (one cell: midnight/Tolkach, 4-fold cross-fit).** `ae` dl32 beta 1e-3 with lam in {30, 100, 300} and 3x more encoder steps (36 rounds x 60 = 2160 steps instead of 720; the retrained adversaries and their per-round in-sample accuracy are logged as `adv_train_acc_per_round`).
"Adapter trained" criterion: the mean over folds of the LAST round's in-sample adversary accuracy is within 0.05 of chance (<= 0.55 for 2 centres). No statement about what the adversary adds or does not add is allowed for a configuration that fails the criterion (it is reported as "adversary did not converge");
for configurations that pass, they are compared with the dithered no-adversary codes and the floor envelope at equal leakage under the same rules as (i)/(ii). Reference rows lam in {0, 1, 3, 10} (12 rounds) come from the existing cross-fit run of that cell.

Budget <= 4 GPU-h (estimate: dinov2-L cross-fit 2.0, phikon-v2 holdout 0.5, F-02 0.4), at most 3 `a_` jobs at once, priority 4 once.

### AUDIT FOLLOW-UP (1): corrected verdict row, retention axis, missing rows (zero compute; aggregated 2026-11-04 ~18:40Z from the saved predictions)

**Verdict on the label-free AE in A: DEAD on held-class LABEL utility; MIXED on kNN retention (criterion (ii) of the revised FOCUS A).** Numbers use the INDUCTIVE floor envelope (centre_whiten removed from it, reported separately), the leakage tolerance 0.05, MDE80 = 2.8 x bootstrap SE, and "headroom" = raw minus the best floor at that leakage = the most any adapter could gain.
V-selected `ae` configuration per cell; "n_near" = inductive floor points within +-0.05 of the AE's L (< 2 = no comparator, P3).

| cell / design | AE config | L | utility: headroom | utility Δ vs inductive envelope [95% CI] (MDE80) | retention: headroom | retention Δ vs inductive envelope [95% CI] (MDE80) | n_near | utility win | retention win |
|---|---|---|---|---|---|---|---|---|---|
| midnight / Tolkach, cross-fit K=4 | dl32 lam10 | 0.37 | 0.029 (ceiling-limited) | +0.003 [-0.007, +0.012] (0.014) | 0.486 | **+0.095 [+0.068, +0.123] (0.041)** vs `lda_pca__d8` | 3 | not scored | **YES** |
| midnight / Tolkach, fixed C, 3 seeds | dl32 lam10 | 0.25 | 0.109 | +0.089 [+0.033, +0.174] (0.104) vs `pca__d4` | 0.590 | +0.146 [+0.112, +0.177] (0.046) vs `pca__d4` | 1 | no comparator | no comparator (P3) |
| midnight / TCGA, cross-fit K=4 | dl32 lam1 | 0.46 | 0.052 | -0.028 [-0.063, +0.001] (0.048) | 0.448 | +0.013 [-0.009, +0.034] (0.030) vs `lda_pca__d16` | 5 | no | no |
| midnight / TCGA, fixed C, 3 seeds | dl32 lam0 | 0.58 | 0.015 (ceiling-limited) | -0.056 [-0.108, -0.010] (0.071) | 0.322 | -0.090 [-0.130, -0.051] (0.058) | 2 | not scored | no |
| dinov2-L / Tolkach, fixed C, 3 seeds | dl32 lam0 | 0.30 | 0.043 | -0.004 [-0.016, +0.006] (0.017) | 0.191 | -0.038 [-0.051, -0.025] (0.018) | 0 | no comparator | no |
| dinov2-L / TCGA, fixed C, 3 seeds | dl32 lam0 (beta 0.01) | 0.48 | < 0 (a floor beats raw) | -0.010 [-0.028, +0.010] (0.027) | 0.294 | -0.451 [-0.464, -0.439] (0.018) | 2 | not scored | no |
| phikon / TCGA, fixed C, seed 0 (not yet replicated) | dl32 lam1 | 0.59 | 0.032 | -0.051 [-0.150, +0.046] (0.143) | 0.814 | +0.292 [+0.253, +0.331] (0.054) vs `randproj__d16` but utility -0.113 [-0.215, -0.015] vs the matched LDA arm | 0 | no comparator | no (n_near 0, utility cost) |
| dinov2-L / Camelyon, crossed, seed 0 (no held-out classes) | dl32 lam1 | 0.81 | -0.017 | -0.046 [-0.108, -0.011] vs `leace` | 0.034 (ceiling-limited) | -0.296 [-0.327, -0.268] | 1 | not scored | not scored |

Reading: retention is positive in exactly one cell with a valid comparator (midnight/Tolkach cross-fit: +0.095 at equal held-class utility, utility headroom only 0.029), null in the other midnight cell, and negative in the weak-headroom or weak-encoder cells; the fixed-C Tolkach +0.146 has one
floor point near its L and is not a win under P3. Dropping `centre_whiten` from the envelope moves the fixed-C midnight/Tolkach utility contrast from -0.002 (vs centre_whiten) to +0.089 on one comparator point, which is why cross-fit (3 floor points near L) is the number to quote.
Caveat that applies to every retention row: the metric is the overlap with raw-space kNN structure, which an MSE-reconstructing AE optimises directly; whether that counts as utility is an owner decision.

**Dither control on retention.** Noise-dithered label-free codes without an adversary reach the AE's leakage only by destroying neighbourhood structure: at L 0.30-0.43 on midnight/Tolkach their retention is 0.17-0.27 (random-projection-32 r1: 0.27; PCA-32 r1: 0.21; LDA+PCA-32 r1: 0.20; the AE-based code r1: 0.17-0.18) versus 0.609 for the AE (L 0.37);
on midnight/TCGA at L 0.27-0.35 the dithered codes have 0.17-0.25 versus 0.465 for AE lam10 (L 0.34). So the dither control kills the AE on held-class utility (equal) but is **not** a competitive floor on retention; the competitive retention floors are the UNdithered low-dimensional projections (`lda_pca__d8`: 0.514 at L 0.30 on Tolkach), which is what the +0.095 is measured against.

**Rows missing from the earlier write-ups.**

* `a_p2h_phikon_tcga_s0` (phikon: the worst Robustness Index, most centre headroom; seed 0, fixed C): raw held-class acc 0.715 [0.569, 0.832]; V-selected AE -0.113 [-0.215, -0.015] against the matched-dimension LDA arm (MDE80 0.146); no gain; retention row above. Not replicated (one seed).
* `a_lrn_midnight_camelyon_s0` and `a_lrn_midnight_tcga_s0` (crossed design, label-supervised families vib / vq / dann / adv_retrain / supcon_xc, one seed): all five beat LEACE on worst-case centre accuracy at equal label accuracy (V-selected best learned minus LEACE -0.39 [-0.46, -0.31] Camelyon, -0.38 [-0.45, -0.28] TCGA), but none is better than the plain LDA arm (Camelyon: LDA L 0.01 at label 0.983; vib L 0.08 at 0.972; TCGA: LDA L 0.17 at 0.853; vib L 0.10 at 0.862; adv_retrain L 0.10 at 0.861), and the adversarial families do not beat the best no-adversary bottleneck
  (adversarial minus bottleneck worst-case accuracy +0.130 [+0.061, +0.189] Camelyon, -0.003 [-0.087, +0.075] TCGA). Crossed design only: it cannot reward a method over LDA (finding of the gate).
* `a_p2x_dinov2L_camelyon_s0`: row above (control encoder; Camelyon has no held-out classes).
* `a_gate_phikon_tcga_s0` (finished before the 11:15Z tally): LEACE L_worst **0.72 [0.60, 0.81]**; **G1 is 0 of 8 cells** (was 0 of 7); conclusion unchanged. LEACE linear-attacker L 0.72; raw worst-case centre accuracy is far above the label-only floor in all eight cells.


### AUDIT FOLLOW-UP (2): H2-015 selection null for the best-of-N AE (as pre-registered above; `selection_null.py`; 5000 draws for null A (2000 used in the run reported), 1000 unit bootstraps for null B)

T_obs = best-of-N difference between an `ae` configuration and the best INDUCTIVE floor at L <= L_ae + 0.05. Null A pretends floor points are the AE (frontier roughness + best-of-N); null B is the max-T bootstrap (sampling noise + best-of-N).

| cell | metric | N ae configs | T_obs (best-of-N Δ) | best config | null A: 95th pct / p | null B (max-T bootstrap): 95th pct | above selection noise (exceeds both)? |
|---|---|---|---|---|---|---|---|
| dinov2L_clsmean / tcga / holdout | held-class acc | 10 | +0.017 | ae__beta0.001_dl32_lam3.0 | +0.022 / p=1.000 | +0.083 | no (selection noise) |
| dinov2L_clsmean / tcga / holdout | retention xc | 10 | +0.031 | ae__beta0.001_dl8_lam0.0 | +0.260 / p=1.000 | +0.018 | no (selection noise) |
| dinov2L_clsmean / tolkach_esca / holdout | held-class acc | 10 | +0.003 | ae__beta0.001_dl32_lam1.0 | +0.018 / p=0.819 | +0.045 | no (selection noise) |
| dinov2L_clsmean / tolkach_esca / holdout | retention xc | 10 | +0.011 | ae__beta0.001_dl32_lam1.0 | +0.062 / p=0.327 | +0.028 | no (selection noise) |
| midnight_clsmean / tcga / holdout | held-class acc | 10 | +0.059 | ae__beta0.001_dl32_lam10.0 | +0.059 / p=0.754 | +0.122 | no (selection noise) |
| midnight_clsmean / tcga / holdout | retention xc | 10 | +0.205 | ae__beta0.001_dl8_lam1.0 | +0.078 / p=0.000 | +0.046 | YES |
| midnight_clsmean / tcga / holdout_cv4 | held-class acc | 12 | -0.015 | ae__beta0.001_dl32_lam0.0 | +0.060 / p=1.000 | +0.074 | no (selection noise) |
| midnight_clsmean / tcga / holdout_cv4 | retention xc | 12 | +0.054 | ae__beta0.001_dl8_lam10.0 | +0.119 / p=0.825 | +0.032 | no (selection noise) |
| midnight_clsmean / tolkach_esca / holdout | held-class acc | 10 | +0.089 | ae__beta0.001_dl32_lam10.0 | +0.063 / p=0.000 | +0.075 | YES |
| midnight_clsmean / tolkach_esca / holdout | retention xc | 10 | +0.146 | ae__beta0.001_dl32_lam10.0 | +0.235 / p=0.726 | +0.070 | no (selection noise) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | held-class acc | 12 | +0.003 | ae__beta0.001_dl32_lam10.0 | +0.058 / p=0.953 | +0.037 | no (selection noise) |
| midnight_clsmean / tolkach_esca / holdout_cv4 | retention xc | 12 | +0.095 | ae__beta0.001_dl32_lam10.0 | +0.225 / p=0.602 | +0.034 | no (selection noise) |
| phikon_clsmean / tcga / holdout | held-class acc | 10 | +0.071 | ae__beta0.001_dl8_lam3.0 | +0.172 / p=1.000 | +0.174 | no (selection noise) |
| phikon_clsmean / tcga / holdout | retention xc | 10 | +0.422 | ae__beta0.001_dl32_lam0.0 | +0.322 / p=0.000 | +0.041 | YES |

Reading (all cells, V-unselected best-of-N; N = 10 configurations for the fixed-C grid, 12 for cross-fit):

* **Held-class utility, cross-fit (the cells that carry the verdict): T_obs -0.015 (TCGA) and +0.003 (Tolkach), far below both null thresholds (null A p = 1.00 and 0.95).** The best-of-12 utility gain is selection noise in both midnight cells; the utility verdict is calibrated.
* **The one "YES" on utility (fixed-C midnight/Tolkach, +0.089 vs null A 0.063, null B 0.075) is a sparse-comparator artefact:** its comparator is a single floor point (`pca__d4`, n_near = 1), the fixed-C grid has no dithered floors, and `centre_whiten` is no longer in the envelope. The cross-fit grid, which does have dithered floors (3 points near L), gives +0.003. P3 already forbids calling it a win; the null does not rescue it.
* **Retention, cross-fit midnight/Tolkach: +0.095 clears the sampling null (null B 95th percentile +0.034) but NOT the frontier null (null A 95th percentile +0.225, p = 0.60).** Retention rises steeply with L on the floor curve (lda_pca d8: 0.51 at L 0.30; d16: 0.57 at L 0.67) and the floor grid has few points between L 0.3 and 0.6, so a one-sided
  "floors with L <= L_ae + 0.05" comparison favours any point that sits slightly above the nearest floor in L. A random floor point treated as the AE shows best-of-12 "gains" of up to +0.225 by the same procedure. Consequence: the retention positive is NOT established as a property of the AE; it is equally consistent with comparing against a sparse floor frontier.
  Midnight/TCGA cross-fit retention (+0.054 best-of-12) is within both nulls.
* Fixed-C retention "YES"s (midnight/TCGA +0.205, phikon/TCGA +0.422) have n_near < 2 or a utility cost (phikon: -0.113 vs the matched LDA arm) and fail the registered retention rule; dinov2-L is within both nulls.

**Added after seeing null A (post hoc, tightening only):** a dense floor ladder for the two midnight cross-fit cells (`pca` d in {6,10,12,20,24}, `lda_pca` d in {6,10,12,20,24}, `randproj` d in {8,12,24}; same folds, same splits as the cross-fit run). The retention claim (ii) now has to survive the dense ladder
(Delta >= +0.05 with CI > 0 against the densified inductive envelope) before it is called a win; otherwise "retention: mixed" becomes "retention: not established".

### AUDIT FOLLOW-UP (4): F-02 adversary convergence, midnight/Tolkach, 4-fold cross-fit (as pre-registered: `ae` dl32 beta 1e-3, lam in {30, 100, 300}, 36 rounds x 60 = 2160 encoder steps; reference lam in {0, 1, 3, 10} with 12 rounds from the existing cross-fit run)

`f02_report.py` reads the in-sample accuracy (mean of the linear and the MLP adversary) that the freshly re-initialised adversary reaches against the adapter after each round, on the adapter-train units A (chance 0.5).

| config | rounds | first-round adv acc | last-round adv acc: mean [min, max over folds] | adapter trained (pre-registered: last round <= 0.55)? | rate (bits) | L_worst [95% CI] | held-class acc | retention xc |
|---|---|---|---|---|---|---|---|---|
| lam 0 (no adversary) | 12 | n/a | n/a | n/a | 67 | 0.76 [0.70, 0.81] | 0.939 | 0.623 |
| lam 1 | 12 | 0.697 | 0.917 [0.862, 0.939] | **no** | 86 | 0.51 [0.44, 0.60] | 0.949 | 0.601 |
| lam 3 | 12 | 0.697 | 0.879 [0.849, 0.908] | **no** | 117 | 0.45 [0.35, 0.55] | 0.931 | 0.578 |
| lam 10 | 12 | 0.697 | 0.858 [0.834, 0.892] | **no** | 162 | 0.37 [0.29, 0.45] | 0.971 | 0.609 |
| lam 30 | 36 | 0.697 | 0.832 [0.754, 0.871] | **no** | 391 | 0.33 [0.24, 0.42] | 0.954 | 0.437 |
| lam 100 | 36 | 0.697 | 0.777 [0.664, 0.866] | **no** | 327 | 0.33 [0.25, 0.42] | 0.940 | 0.403 |
| lam 300 | 36 | 0.697 | 0.811 [0.733, 0.849] | **no** | 543 | 0.34 [0.24, 0.51] | 0.951 | 0.422 |

**Result: the "adapter trained" criterion is NOT met by any configuration, including lam 300 with three times the encoder steps.** By the pre-registered rule no statement about what the adversary adds (or fails to add) is allowed for this AE; the correct wording is "with lam <= 300 and <= 2160 steps the adversary did not converge" (the earlier "adversary adds nothing" sentence stays struck).
What the sweep does show: the held-out worst-attacker leakage L stops improving at about 0.33 beyond lam 30 (0.51 at lam 1, 0.37 at lam 10, 0.33-0.34 at lam 30-300) while retention falls from 0.61 (lam 10) to 0.40-0.44, and the rate term (KL of the released channel) grows to 330-540 bits.
Against the dithered no-adversary floors (leakage-matched, held-class accuracy): lam 100 -0.029 [-0.048, -0.014] (MDE 0.024), lam 30 -0.014 [-0.034, +0.001], lam 300 -0.017 [-0.041, +0.001]; retention -0.078 to -0.111 (all below the floors). Harder adversarial training did not create a utility or retention advantage at lam >= 30.

**Calibration that qualifies the pre-registered criterion (CPU, same adversary code, the fold-0 adapter-train set A of this cell: 1600 patches, 10 slides):** a freshly trained adversary reaches in-sample accuracy (linear, MLP) of **0.574 / 0.771 (mean 0.67) on pure Gaussian-noise latents** that carry no centre information at all (memorisation of 1600 samples in 32 dimensions),
0.965 / 1.000 on latents that carry only a per-slide fingerprint plus noise of the same size as the signal, 0.982 / 1.000 on PCA-32 of the raw embedding, and 0.81 / 0.98 (r=1), 0.69 / 0.87 (r=2), 0.61 / 0.78 (r=4), 0.60 / 0.77 (r=8) on dithered PCA-32. So an in-sample accuracy within 0.05 of chance (0.55) is **unattainable by construction** at this sample size, and the logged value is largely slide memorisation:
the adversary reads the slide identity of the training slides, which implies the centre in-sample, and an adapter cannot remove slide identity from patches it still has to describe. Against the calibrated floor (0.67; criterion <= 0.72) lam 100 reaches 0.78 on average (0.66 on one fold) and lam 300 0.81: still not trained, but closer than the pre-registered threshold suggests.
Consequence: the F-02 question ("does a converged adversary add anything over rate / noise?") cannot be answered with an adversary trained and scored on the same units; it needs an adversary scored on slides it did not fit (adversary trained on half of A's slides, confusion loss on the other half). That is a new method and is listed as a hypothesis, not run.

### AUDIT FOLLOW-UP (3): dinov2-L cross-fit, phikon-v2 held-out classes, dense floor ladder, and the updated verdict (aggregated 2026-11-04 ~20:10Z)

Grid `p4` (37 configurations; all 12 `ae`, reconstruction-only AE, lda_pca / pca / randproj ladders, reduced dither ladder, 2 controls). dinov2-L: K=4 cross-fit, TCGA and Tolkach (fold 1 of the TCGA cell was killed by an over-broad `pkill` of mine and re-run as `a_aud_dinov2L_tcga_cv_fold1`; its 7 partial configurations are superseded). phikon-v2: held-out-class protocol, fixed C, seed 0 (single seed, not yet replicated).
V-selected = validation-selected `ae` configuration per cell; MDE80 = 2.8 x bootstrap SE; headroom = raw minus the best inductive floor at the same leakage; "retention win" = registered rule (Delta >= +0.05, CI > 0, >= 2 floor points near L, utility not worse than -0.03).

| cell / design | V-selected ae | L | utility: headroom, Δ vs inductive envelope [95% CI] (MDE80) | retention: headroom, Δ vs inductive envelope [95% CI] (MDE80) | n_near | utility win | retention win | best-of-N vs selection null (utility / retention) |
|---|---|---|---|---|---|---|---|---|
| dinov2-L / Tolkach, cross-fit | ae beta 1e-4 dl32 lam 0 (reconstruction-only, no adversary) | 0.20 | 0.055, +0.018 [+0.007, +0.030] (0.016) | 0.175, **+0.067 [+0.059, +0.076] (0.013)** vs `lda_pca__d32` | 2 | no (< +0.04) | **YES** | **above both nulls**: +0.034 (`ae` beta 1e-3 dl32 lam 0, L 0.16; +0.025..+0.041 vs the dithered AE) vs null A 0.020 / B 0.029; retention +0.067 vs 0.047 / 0.021 |
| dinov2-L / TCGA, cross-fit | dl32 lam1 | 0.53 | -0.013 (a floor beats raw), -0.010 [-0.029, +0.010] (0.027) | 0.279, -0.077 [-0.083, -0.071] (0.009) | 5 | ceiling-limited | no | within both nulls (+0.060 vs 0.061 / 0.051; +0.082 vs 0.251 / 0.012) |
| phikon-v2 / TCGA, fixed C, seed 0 | dl32 lam10 | 0.48 | 0.069, +0.054 [-0.035, +0.133] (0.117) | 0.819, +0.096 [+0.058, +0.133] (0.054) vs `lda_pca__d4` | 3 | no | YES (single seed) | within: +0.054 vs 0.054 / 0.155; retention +0.196 vs null A 0.326 (p 0.20) |
| phikon-v2 / Tolkach, fixed C, seed 0 | recon-only dl32 lam 0 | 0.96 | 0.000 (comparator is raw) | not scored | 10 | ceiling-limited | no | within (retention +0.261 vs null A 0.360, p 0.21) |
| midnight / Tolkach, cross-fit, **dense floor ladder** (`pca`, `lda_pca` d 6-24, `randproj` d 8-24 added) | dl32 lam10 | 0.37 | 0.029, +0.003 [-0.007, +0.012] (0.014) | 0.475, **+0.085 [+0.057, +0.115] (0.041)** vs `lda_pca__d10` (0.525 at L 0.39) | 4 | ceiling-limited | **YES** | utility within; retention +0.085 vs null A 0.199 (p 0.39; 0.54 with the undithered pool), exceeds null B 0.035 |
| midnight / TCGA, cross-fit | dl32 lam1 | 0.46 | 0.052, -0.028 [-0.063, +0.001] (0.048) | 0.448, +0.013 [-0.009, +0.034] (0.030) | 5 | no | no | within both |

Readings.

* **Dense ladder (the post hoc follow-up registered after the first null):** the midnight/Tolkach retention gap survives a floor grid with 13 extra points: +0.095 (sparse) -> +0.085 (dense), and the AE (L 0.37, retention 0.609) Pareto-dominates every inductive floor up to L 0.43 (best floor retention 0.571 at L 0.43). So the gap is not an artefact of a sparse floor curve. It still does not exceed the frontier-permutation null (p 0.39), because that null is wide for any metric that rises steeply with L (retention 0.45 at L 0.29 to 0.64 at L 0.77 for the floors);
  null B (sampling + best-of-N) is passed. Reading: a real but small retention advantage, not distinguishable from frontier roughness at the L resolution we have (+-0.08).
* **Utility:** no encoder shows a registered utility win. The only cell where best-of-N exceeds both nulls is dinov2-L/Tolkach (+0.034, below the +0.04 threshold, from an AE with lam 0, i.e. no adversary). Held-class label utility of the label-free AE: **dead, calibrated.**
* **Retention:** positive (registered rule fires) in dinov2-L/Tolkach, phikon-v2/TCGA (seed 0) and midnight/Tolkach; null in midnight/TCGA, negative in dinov2-L/TCGA; not scored in phikon-v2/Tolkach. In dinov2-L the winning configurations are the **no-adversary** ones (reconstruction-only, lam 0): the retention advantage comes from the nonlinear reconstruction bottleneck, not from the adversary.
* Budget: 4.55 worker-hours (dense ladder 1.04, dinov2-L TCGA 1.60 + 0.23 re-run, dinov2-L Tolkach 0.97, phikon-v2 0.42, F-02 0.29) against the 4.0 allowed; the overrun (0.55) is mostly the box load (load average 100-200: jobs ran about 3x slower than planned) and about 0.1 for my accidental fold-1 kill. dense-ladder TCGA and dinov2-L/TCGA dense floors were not run.

### UPDATED VERDICT on the label-free AE in A (supersedes the "DEAD" header above where they differ)

* **Held-class LABEL utility: DEAD (stands, now calibrated by the selection null; ceiling-limited in Tolkach).**
* **kNN retention (criterion (ii)): ALIVE, weak.** The pre-registered rule "retention wins in both new encoders -> ALIVE" is met (dinov2-L/Tolkach cross-fit +0.067 [+0.059, +0.076]; phikon-v2/TCGA seed 0 +0.096 [+0.058, +0.133]; midnight/Tolkach cross-fit +0.085 [+0.057, +0.115], MDE80 0.013-0.054), but the effect is small and carried by no-adversary AEs,
  it is null or negative in the TCGA cells of two encoders, the phikon-v2 cell is one seed, and only dinov2-L/Tolkach exceeds the frontier-permutation selection null. It is a statement about a nonlinear reconstruction bottleneck keeping raw-space neighbourhoods at equal leakage and equal label utility, and the metric is what an MSE-trained AE optimises (owner decision whether that is utility).
* **Adversary: untested.** F-02: not trained at lam <= 300 with 2160 steps (in-sample adversary accuracy 0.78-0.83, memorisation floor 0.67); the sentence "the adversary adds nothing" stays retracted; no claim either way.
* Gate G1: 0 of 8 cells; unchanged.
