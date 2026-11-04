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
| dinov2-L (no anchor) | see below | 0.511 | see below | not on the leaderboard | n/a |

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
