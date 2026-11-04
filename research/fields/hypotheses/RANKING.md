# Hypothesis ranking (ranker pass 1, 2026-11-04; priors only, no result numbers exist yet)

Inputs read: CHARTER, FOCUS, INDEX, all 25 `H-*.md`, `results_notes/A.md` (pre-registration, empty log), `results/` (only
`smoke_gpu_check`), `jobs/specs` (A extraction x4, A smoke, C feature extraction x3), and the code in `src/a_pathorob`,
`src/c_cremad`, `src/b_robotics`. **There are no first numbers yet**, so every "likelihood" below is my reasoning-based prior
(the generator's prior adjusted by what the code and the cited README numbers imply), not an updated posterior. Re-rank when
the triggers in section 7 fire.

## 0. How the columns are defined

* **Likelihood** = my probability that the *decision-relevant core clause* holds (named in the row notes). The generator's priors
  are for the whole conjunction of clauses as written; several are 4-6 clause conjunctions, so I also give the as-written value where
  I think it is much lower. Generator prior in brackets.
* **Applicability (App, 1-5)** = D + U + T + L. D (0-2): does the outcome change which encoder family we train next. U (0-1):
  unlocks or prunes >= 3 follow-ups. T (0-1): transfers across foci (A, C, B'). L (0-1): is itself a learned method (user preference).
  Gates and floors score low here by construction; they are ranked by gating, not by App.
* **Info/GPU-h** = H2(likelihood) x (App/5) / max(first-decision marginal GPU-h, 0.25). H2 is binary entropy in bits. It is a
  ranking heuristic, not a measurement. Cost floor of 0.25 stands in for analyst/CPU time.
* **Cost** = generator estimate -> marginal estimate after sharing the builders' already-specced grids (first-decision slice in
  parentheses where I propose staging). A-grid costs are **unmeasured**: `a_smoke_tolkach` prints `fit`/`eval` seconds; rescale every A
  number by measured/assumed (I assumed ~1.5 min attacker-suite eval + 1-3 min fit per configuration).
* **Rank rule**: tier 1 = gates (cost ~0, can invalidate everything); tier 2 = pivots that decide the encoder family (App 5) and
  the B' kill test; tier 3 = by info/GPU-h, ties broken by App; tier 4 = needs heavy new code or an upstream outcome (shifted down
  even if info/GPU-h looks good).

## 1. What is actually implemented (changes costs and blockers)

| Piece | Status in repo |
|---|---|
| A floors raw / lda / leace (4 ridge) / isp (1,5,20) / centre_mean; controls const / noise / shuffle | in `adapters.py` GRID, covers H-A-001, X-006 (INLP part), the A halves of X-008 and A-005 floor |
| A learned vib (dl 4,32 x beta 4), vq (5), dann (4), adv_retrain (6), supcon_xc (2); suite linear/mlp/knn/tfm; oodcentre + centre_onA diagnostics | in GRID, covers H-A-002, A half of X-001 (partly), supcon part of H-A-004 |
| A: correlated-V splits, conditional adversary, placebo adversary, dCor/HSIC/CLUB, population adversary, Sinkhorn-VQ+dropout, PCA-d / random-projection controls, per-centre whitening, stain-pair re-extraction, LightGBM / RFF-SVM attackers | **missing** (H-A-001..007 and the X-* files assume several of these) |
| C reference: raw, mean, smooth, pca, leace_tr, leace_oracle, ctrl_constant / length / shuffle, per-layer sweep (`make_jobs.py ref`) | specced, covers H-C-001 (a,b,c) |
| C learned: vib/vq/none, pool utt or P frames, AdvEnsemble (3 MLPs re-initialised every 10 epochs, warm-up); sweep0 = 28 configs (rate d 2..256, adv d16/d64 x 4 lam, VQ x6, seg25 x3) | specced, covers H-C-002, C halves of X-001 / C-007 / X-004 (plain VQ) |
| C evaluate: 91-way identity attackers on T1/T2/T3 with `seen` and `unseen` actor groups; sex / race / age attackers on every representation by default | present, so H-C-004 measurement and the seen-vs-unseen gap of H-X-005 are **free** on every learned run |
| C: posterior-6 reference, per-actor mean subtraction, matched-content contrastive sampler, ECAPA, LightGBM, TDNN attacker, SISER-style entropy adversary | **missing** |
| B': envgen, distractors (none / indep / corr_det / corr_noisy), sam_masks | present; **no LAM trainer, no probes, no `drive_from="command"`** |

Two protocol mismatches to fix before any number is believed:
1. H-C-001 / C-005 / C-006 state absolute thresholds for ~23 enrolled actors (e.g. ">= 0.21"). Builder-C enrols **all 91 actors**
   (chance 1/91) and reports the test-fold subset as `unseen`. State every threshold as a multiple of chance.
2. The two adversary implementations differ (A `adv_retrain`: KL-to-uniform against retrained linear+MLP; C `AdvEnsemble`:
   CE-to-uniform against 3 re-initialised MLPs), and C's "rate" configs use `beta=0` (width only, no rate penalty) while A's `vib`
   has a KL term. H-X-001's claim "threshold d* within 4x across fields" is about *rate*; compare in bits or nats (A reports
   `rate_bits_per_patch`, C reports `kl_nats` / `bits_vq`), not in d, and unify the adversary before the cross-field comparison.

## 2. Groups, mechanisms, and what can share one job or ladder

| Group | Mechanism | Members | Shared job / ladder |
|---|---|---|---|
| G0 Gates | controls and premise checks, deterministic | H-C-006, H-C-005, H-A-001, H-C-001, H-X-008, H-A-007 (+ builder ctrl_* jobs) | **LA-0**: one `run_sweep` cell per (encoder, dataset, seed) with `--grid raw,lda,centre_mean,leace,isp,ctrl_const,ctrl_noise,ctrl_shuffle` serves H-A-001, the A part of X-008, the H-A-005 floor half, and the INLP part of X-006. **LC-0**: `make_jobs.py ref` serves H-C-001 and C controls. **LX-CPU**: one CPU script for H-C-005 + H-C-006; one for H-A-007 descriptors. |
| G1 Rate | capacity does the work, no adversary | H-A-002, H-C-002, lambda=0 rows of H-X-001, plain-VQ arm of H-X-004 | **LA-1 / LC-1**: the same `vib`/`rate` grid. C `c_rate_d*` and `c_seg25_*` already exist; add posterior-6, matched-number windows, PCA-d, random projection. |
| G2 Adversary family | minimax against retrained attackers | H-X-001, H-C-007 (identical arms, merged), H-X-007 (placebo), H-X-002 (population), H-A-003 (conditional variant) | One adversary module with flags `--adv {none,grl,restart,pop}`, `--adv_labels {true,perm,attr}`, `--cond_on_y`; one grid file per focus. X-007 and X-002 are *columns* of the X-001 grid, not new ladders. |
| G3 Penalty family | non-adversarial dependence minimisation | H-X-003 then H-X-005 | One module `--pen {dcor,hsic,club}`, `--secret {label,dense_proxy}`; X-005 reuses X-003 outputs. |
| G4 Matched-pair contrastive | same object, different source | H-C-003, H-A-004 (+ `supcon_xc` already in A GRID) | Shared loss code only. Data differ, so two jobs. Same idea transfers to B' twin renders (exact with/without-distractor pairs). |
| G5 Discrete and nonlinear erasure | VQ occupancy; Obliviator / kernel | H-X-004, H-X-006 | VQ: builder `vq` arms first. Erasure: ISP is free in LA-0; Obliviator needs ~2 builder-days. |
| G6 Source-aware / test-time | per-source statistics at release or test | H-X-008, H-A-005 | H-A-005 floor half = A part of H-X-008 (merge); Tent half killed. |
| G7 Analysis of existing runs | attributes, metric validity | H-C-004, H-A-006, seen-vs-unseen gap of H-X-005 | Pure aggregation over G1/G2 outputs; no new training except a LEACE-on-sex arm. |
| G8 B' chain | latent-action shortcut | H-B-001 + H-B-002 (merge), then H-B-003 | **LB-1**: one LAM grid {kind, rho_eff, mask arm, z-dim}; B-001 is a subset of it. |

Dependency spine: LA-0 / LC-0 gates -> G1 rate rows -> G2 grid -> {placebo, population, dCor -> proxy, conditional}. G4 needs the G1
rate-matched bottleneck as comparator. G8 is independent of A and C and can use its own GPU from the start.

## 3. (a) Ranking table

Likelihood: my core-clause estimate, generator prior in brackets. Cost: generator -> marginal (first-decision). Info/GPU-h as defined in section 0.

| Rank | ID | Group | Likelihood | App | Info/GPU-h | Cost GPU-h | Blocked by | Verdict |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | H-C-006 | G0 channel control | 0.50 [0.40] | 2 (gate) | 1.60 | 0 -> 0 (0.5 CPU-h) | WAVs + splits | RUN NOW |
| 2 | H-C-005 | G0 prosody floor | 0.50 [0.60] | 2 (gate) | 1.60 | 0 -> 0 (1 CPU-h) | WAVs + splits | RUN NOW |
| 3 | H-A-001 | G0 premise / floor | 0.78 [0.75] | 3 | 0.91 | 2 -> 0.5 (+3.6 for phikon, dinov2L) | `a_extract_*`; LightGBM/RFF-SVM missing from suite | RUN NOW |
| 4 | H-C-001 | G0 premise / layer map | 0.70 [0.70] | 3 | 1.06 | 3 -> 0.5 | `c_feats_wavlm_bp`; `make_jobs.py ref` | RUN NOW |
| 5 | H-X-008 | G0 / G6 oracle-normalisation control (A+C) | 0.45 [0.50] | 3 | 1.19 | 1 -> 0.5 | caches; new `peractor` method (C), `centre_zscore/whiten` (A) | RUN NOW |
| 6 | H-A-007 | G0 colour-statistics control | 0.55 [0.55] | 2 (gate) | 0.40 | 1 -> 1 (+2 CPU-h) | pixel loader; embeddings for residualisation | RUN NOW (descriptors, probes); re-embedding after rank 3 |
| 7 | H-C-002 | G1 rate (C) | 0.60 [0.45] | 5 | 0.97 | 3 -> 1 (+ builder sweep0) | `c_feats_wavlm_bp`; posterior-6 + order-shuffle code | RUN NOW |
| 8 | H-A-002 | G1 rate (A) | 0.60 [0.50] | 5 | 0.39 | 4 -> 2.5 (A cells unmeasured) | H-A-001 refs; PCA-d, random-projection methods | RUN NOW |
| 9 | H-X-001 | G2 adversary x rate (A+C) | 0.45 [0.45] (as written 0.2) | 5 | 0.33 | 10 -> 8 (slice 3) | lambda=0 rows (ranks 7-8); unify adversary impl | RUN AFTER H-C-002/H-A-002 rows (slice may start in parallel) |
| 10 | H-C-007 | G2, merged into H-X-001 | 0.45 core, 0.2 as written [0.55] | 4 | 0.79 | 5 -> 1 (SISER + TDNN arm only) | H-X-001 | MERGE into H-X-001, RUN AFTER |
| 11 | H-B-001 | G8 kill test | 0.35 B' survives [0.35] | 3 (direction) | 0.22 | 15 -> 10 (stage 1: 2.5) | LAM trainer + probes (not in repo); SAM masks | RUN NOW stage 1 on one GPU |
| 12 | H-C-004 | G7 attributes | 0.55 core, 0.30 as written [0.50] | 2 | 1.32 | 1 -> 0.3 | any C learned runs (secrets already computed) | RUN AFTER H-C-002 (free analysis) |
| 13 | H-C-003 | G4 matched-content | 0.40 [0.40]; beats rate-matched 0.20 | 4 | 0.60 | 5 -> 4 (seed 0: 1.3) | H-C-002 comparator; (sentence, emotion) x actor sampler | RUN AFTER H-C-002 (write code now) |
| 14 | H-X-007 | G2 placebo control | 0.25 [0.30] | 4 | 0.43 | 2 -> 1.5 | H-X-001 best cell per focus | RUN AFTER H-X-001 |
| 15 | H-A-003 | G2 conditional adversary | 0.35 [0.45] | 4 | 0.25 | 3 -> 3 | correlated-V split generator; cond-adversary flag; H-A-002 | RUN AFTER H-X-001 (A half) |
| 16 | H-X-003 | G3 dCor vs adversary | 0.30 [0.35] | 4 | 0.18 | 4 -> 4 (drop CLUB arm) | H-X-001 (only if adversary gain >= 0.05 somewhere) | RUN AFTER H-X-001 |
| 17 | H-X-005 | G3 dense proxy | 0.30 [0.40]; gap-half 0.55 | 3 | 0.18 (gap half free) | 4 -> 3 | H-X-003; ECAPA extraction | gap analysis RUN NOW (free); proxy RUN AFTER H-X-003 |
| 18 | H-A-006 | G7 metric validity | 0.35 as written [0.60] | 2 | 0.19 | 2 (+10 CPU) -> 2 | methods from ranks 3, 8, 9; RI anchor | RUN AFTER H-A-002, re-scoped (section 7, item 2) |
| 19 | H-X-002 | G2 population | 0.30 [0.40] | 4 | 0.12 | 6 -> 6 | H-X-001 gain >= 0.10; LightGBM/ExtraTrees/RBF attackers; soft-kNN head | RUN AFTER H-X-001 (tier 4: new code) |
| 20 | H-B-002 | G8 merged into H-B-001 | 0.50 core [0.55] | 3 | 0.12 | 5 -> in B-001 (stage 2: 5) | H-B-001 clause (b) holds | RUN AFTER H-B-001 |
| 21 | H-A-004 | G4 stain-pair | 0.30 [0.35] | 3 | 0.09 | 6 -> 6 | H-A-007 L_desc >= 0.5; H-A-002; pixel pipeline | RUN AFTER H-A-007 |
| 22 | H-X-006 | G5 nonlinear erasure | 0.35 core, 0.1 as written [0.35] | 2 | 0.37 (GPU only) | 3 -> 1 (+2 builder-days) | Obliviator code or reimplementation | PARK (INLP part free in LA-0) |
| 23 | H-X-004 | G5 balanced VQ + dropout | 0.20 [0.30] | 3 | 0.07 | 6 -> 6 | builder VQ perplexity < 0.5 K and VQ >= vib at equal bits | PARK |
| 24 | H-B-003 | G8 remedies | 0.10 as written, R2 alone 0.40 [0.30] | 3 | 0.07 | 8 -> 8 | H-B-002 measured gap; `drive_from="command"` | PARK |
| 25 | H-A-005 | G6 TTA | floor 0.55, Tent 0.20 [0.30] | 2 | 0.14 | 2 -> 0 (floor merged) | n/a | KILL Tent half; MERGE floor half into H-X-008 |

Merged total if everything not parked or killed runs: ~61 GPU-h new + ~17 GPU-h builder-C grids (ref 3, sweep0 4.7, seeds12 9.4)
+ the unmeasured A grid. Generator's figure was ~105 GPU-h with overlapping rows counted twice.

### Row notes (why my number differs, what decides it)

* **H-C-006 (0.50)**: 91-way, 150 ms of non-speech. 5x chance is only 5.5%, easy to clear if any session/mic difference exists.
  Escalation rule I propose: >= 10x chance (>= 11%) or >= 0.5 of the speech-segment accuracy means the C primary numbers must be reported on trimmed audio too.
* **H-C-005 (0.50)**: speaker >= 4x chance from 3-6 numbers is near certain (0.85); emotion UAR >= 0.40 / 0.45 from 3 / 6 numbers is a coin flip (acted data, pYIN noise). The informative direction is the other one (hand-built prosody on the learned frontier), which I put at ~0.1.
* **H-A-001 (0.78 core; 0.3 for all five clauses)**: linear probe to chance after LEACE is near a theorem (0.85); the retrained MLP/kNN keeping >= 0.30 in >= 7/9 cells is the real bet (0.75); "max over four attackers" biases toward >= 0.30, which helps the prediction but inflates the headroom estimate, so also report per-attacker. Camelyon (2 ID centres, rank-1 erasure) is the surest cell.
* **H-C-001 (0.70)**: (a) 0.9, (b) 0.7, (c) 0.55. `leace_oracle` and `c_layers` already exist; wavlm-large / hubert-large / emotion2vec are not extracted (hubert-base and wav2vec2-base are queued and serve as the cross-backbone check).
* **H-X-008 (0.45)**: C is the likelier win (balanced emotions per actor make the actor mean a clean offset) but it subtracts only the mean while the pooled `std` half of the feature still identifies actors; whitening or mean+std standardisation is the version most likely to reach L <= 0.15. A: `centre_mean` is already in the grid. Whatever the outcome it is the right reference for any group-aware release, and "adapter-then-normalise" is a cheap stacked encoder to test.
* **H-A-007 (0.55)**: the second clause (residualisation leaves L_worst >= 0.45) is 0.5; kill-the-learned-adapter probability ~0.25. Its outcome gates H-A-004: if L_desc < 0.5, centre is not colour and stain augmentation is aimed at the wrong thing.
* **H-C-002 / H-A-002 (0.60)**: core clause = "rate alone cuts leakage a lot": A d=8 >= 40% (0.6; KTH analogue was ~50% at ~15 numbers, but with U ~0.8, not 0.98), C posterior-6 <= 0.20 (0.7). The pivotal clause is H-A-002 falsifier (b), "some d reaches L_worst <= 0.20 at U >= 0.98": I put it at 0.45, i.e. nearly even odds that rate alone solves A. The generator's "never below 0.25" (prior 0.65) is the less likely direction in my view. H-C-002(c) (windowed release leaks >= 0.1 more) 0.4.
* **H-X-001 (0.45 core)**: core = "no adversary gain at d <= 4, gain >= 0.10 at d >= 32 in at least one focus". As written (4 clauses incl. d* within 4x and the fool-the-adversary gap in >= 30% of runs) ~0.2.
* **H-C-007**: its arms 0, 1, 2(partly), 4, 5 are cells of the builder's `c_rate_d64` / `c_adv_d64_l*` grid; the UAR effect is read from the same runs. Only the SISER entropy objective and the TDNN adversary are new. As written (+1-4 UAR *and* leakage reduction < 0.10 *and* gap >= 0.3) ~0.2.
* **H-B-001 (0.35)**: (a) 0.85; (b) that GT/SAM-masked LAMs are >= 1.5x oracle on corr_noisy is the risk: masks only enter the loss, the inverse model still sees sprites, so GT-mask and no-mask arms should be close; whether the shortcut forms at all is the real unknown (~0.5). Stage it: 12 runs first.
* **H-C-003 (0.40)**: reduction >= 0.15 vs CE alone is plausible (hard negatives share the actor, so the loss is directly anti-actor); beating the rate-matched bottleneck is the test that matters (0.2). Watch the sentence-probe leak the loss may create.
* **H-X-007 (0.25)**: placebo (a), labels permuted across samples, is null by construction under the KL/CE-to-uniform confusion losses both builders use (an adversary that cannot learn gives near-uniform output and ~zero gradient). Make placebo (b) (irrelevant real attribute) and the noise-injection control primary. Gated on X-001 showing a real gain (>= 0.10); if X-001's gain is < 0.05 the control is moot.
* **H-A-003 (0.35)**: three-way comparison on >= 2 of 3 datasets (vs marginal adversary, vs LEACE, with L_worst within 0.05) is demanding; published APD effects are tiny (Reinhard +1.1, DANN +0.2). High real-world relevance (clinical data are confounded), and it is the A-side analogue of "conditional on action" in B'.
* **H-X-003 (0.30)**: dCor against a one-hot with 22-56 classes in a batch of 512 has ~10 samples/class; drop the CLUB arm (high variance, needs a q-network, adds cost without a clear win condition).
* **H-X-005 (0.30; gap half 0.55)**: the seen-vs-unseen gap is a free read-out of builder-C runs (`seen` vs `unseen` groups) and of A (`centre` vs `oodcentre`, with chance-normalisation because the centre sets differ). Do it as soon as the first adversarial runs land; the dense-proxy half waits for X-003 and ECAPA.
* **H-C-004 (core 0.55; as written 0.30)**: "sex still decodable on the identity frontier" likely. "Removing sex costs >= 4 emotion points" is the weak clause (0.2): sex is a rank-1 direction, so LEACE on sex should cost ~0-2 points.
* **H-A-006 (0.35 as written)**: see disagreement 2; the premise "RI saturated" does not hold for the four encoders we extract.
* **H-X-002 (0.30)**: held-out-family gain >= 0.10 in *both* foci is a strong joint claim (~0.3; one focus ~0.45). Needs tree and kernel attackers in both suites first.
* **H-A-004 (0.30)**: Reinhard re-embedding is a hard cheap comparator; augmentation covers colour only. 3.5 of the 6 GPU-h are re-extraction.
* **H-X-004 (0.20)**: codebook dropout is the speculative part; builder VQ arms may already show whether occupancy skew exists (C logs `vq_perplexity`).
* **H-B-003 (0.10 as written)**: R1 failing is a replication of the authors' stated boundary (low information); R2 (label-free intervention slice) is the interesting part (0.40) and is blocked twice.
* **H-A-005**: floor half (per-centre standardisation brings OOD centres to L <= 0.30) ~0.55 and is the A part of H-X-008; the Tent half is a small-gain add-on under a known-centre-ID threat model, collapse-prone on 2-class Camelyon.

## 4. Controls that must run BEFORE any exciting result is believed

Run order matters only for belief, not for scheduling: exciting jobs may run in parallel, but nothing is reported until its gate has cleared.

| Gate | Covers | Pass condition | If it fails |
|---|---|---|---|
| Builder `ctrl_const / ctrl_noise / ctrl_shuffle` (A), `c_ctrl_constant / length / shuf_id / shuf_emo` (C) | pipeline sanity | all at chance | fix before anything |
| H-A-001, H-C-001(a) raw reference | any leakage claim | raw >> label-only floor | stop that focus (FOCUS kill rule) |
| H-C-006 | every C identity number | non-speech <= ~5x chance; else report raw and trimmed | C benchmark needs trimmed variant first |
| H-C-005 | every C learned-z claim | learned frontier beats 6-number prosody | report hand-built winner |
| H-A-007 | every A learned-adapter claim | best adapter beats ridge-residualisation by >= 0.10 | "learned = colour" |
| H-X-008 | any per-sample encoder claim | state the group-aware floor next to it | encoder claim is only valid for per-sample release |
| H-A-002 / H-C-002 rate-matched bottleneck, PCA-d, random projection | every adversary, contrastive, penalty, VQ claim | method beats rate-matched bottleneck at equal d / bits | no method claim |
| H-X-007 placebo (b) + noise control | every adversary gain | placebo share < 0.25 | "regulariser", not adversary |
| Seen vs unseen source gap (H-X-005 half) | every "removal" claim | report L_unseen, not L_seen | claim only for seen sources |
| Held-out attacker family incl. LightGBM / ExtraTrees / RBF-SVM | every L_worst | reported | L_worst is only a max over NN-ish families |
| 3 encoder seeds x attacker seeds, bootstrap over slides / actors / episodes | every claim | CI excludes comparison value | label "not replicated" |

## 5. (b) Recommended queue for the next 24 GPU-hours (4 GPUs)

"24 GPU-hours" read as the first ~6 wall-clock hours on four GPUs; wave 2 continues to the end of the day (~96 GPU-h) with gates.
Builder jobs are not duplicated: keep `a_extract_*` and `c_feats_*` (prio 10), `make_jobs.py ref` (prio 11-14), `sweep0` (prio 20-22).
Priority numbers: lower runs first. New code needed is marked [NEW]. Jobs are `name: / priority: / cmd:` YAML as in `jobs/specs`.

**Wave 0, CPU only, start now on the VM CPUs (do not occupy a GPU slot)**
* `x_cpu_c_controls` (H-C-005, H-C-006): per clip, 3- and 6-number prosody (duration, RMS, median F0 | + F0 sd, RMS sd, voiced fraction; pYIN) and, separately, log-mel mean+std of the first/last 150 ms non-speech (VAD at noise floor + 6 dB). Attackers LR / MLP / kNN / LightGBM; builder-C splits and T1/T2/T3; also per-clip-median-F0-normalised variant. Controls: shuffled actor labels (chance), speech-only segments (positive). Report in multiples of 1/91. ~1.5 CPU-h.
* `x_cpu_a_descriptors` (H-A-007 part 1): 24 colour/blur/noise/JPEG statistics per patch from the parquet pixels; MLP + LightGBM + LR probes, slide-disjoint; ridge residualisation of cached embeddings once `a_extract_phikonv2` lands. ~2 CPU-h.
* RI anchor (builder-A): `export_official.py` + official RI on raw phikonv2 / phikon vs `ri_compare.py`; required before H-A-006 means anything.

**Wave 1, ~23 GPU-h (about 15 mine + ~8 builder-C)**

| # | Name | Prio | Focus | Hyp | Config sketch | GPU-h |
|---|---|---:|---|---|---|---:|
| 1 | `q101_a_floors_phikonv2_{camelyon,tcga,tolkach_esca}` | 15 | A | H-A-001, X-008(A), A-005 floor, controls | `run_sweep.py --encoder phikonv2_clsmean --dataset D --seed 0 --grid raw,lda,centre_mean,leace,isp,ctrl_const,ctrl_noise,ctrl_shuffle`; [NEW] methods `centre_zscore`, `centre_whiten` (PCA-64, ridge 1e-2); [NEW] patch-disjoint probe flag; per-attacker output | 3 x 0.6 |
| 2 | `q102_a_floors_midnight_*` | 15 | A | same | same with `midnight_clsmean` | 3 x 0.6 |
| 3 | builder C `ref` jobs | 11-14 | C | H-C-001, controls, H-C-004 data | `make_jobs.py ref` (raw x3 seeds, ctrl x4, layers, mean, leace_tr, leace_oracle, smooth, pca) | ~3 |
| 4 | `q103_c_peractor_norm` | 16 | C | H-X-008(C) | [NEW] `run_exp.py --method peractor`: subtract each actor's mean over all their clips from pooled mean+std stats (PCA-64 first), variants mean-only / whitening / 5-20-80 clips per actor; attackers on T1/T2/T3, emotion on test fold | 0.3 |
| 5 | `q104_b_kill_stage1` | 16 | B' | H-B-001 stage 1 | reach-v3, kinds {none, indep, corr_noisy sigma 1.5, corr_det}, arms {no-mask, GT-mask, SAM-mask}, z-dim 8, seed 0 (12 runs, ~0.1-0.2 GPU-h each) + SAM masks for the 3 distractor kinds (~1 GPU-h if Builder-B has not made them); probes: action ridge on `indep` test, sprite-velocity ridge; [NEW] LAM trainer by Builder-B | 2.5 |
| 6 | `q105_a_rate_phikonv2_*` | 17 | A | H-A-002, X-001 lambda=0 | `--grid vib,vq,lda` + [NEW] `pca`, `randproj` at d in {2,4,8,16,32,64,128,256}; log `rate_bits_per_patch` | 3 x 0.8 |
| 7 | `q106_c_posterior_windows` | 17 | C | H-C-002 (a)(c) | [NEW] posterior-6 representation from the emotion head; `--cfg '{"dz":4,"pool":25,"bott":"vib","beta":0.0,"lam":0.0}'` (5 windows x 4 numbers) vs pooled d=20; [NEW] window-order-shuffle attacker | 1.0 |
| 8 | builder C `sweep0` | 20-22 | C | H-C-002(b), X-001/C-007 cells, X-004 plain VQ | `make_jobs.py sweep0` (28 configs); bundle several configs per job to amortise the ~7 GB frame load | ~4.7 |
| 9 | `q107_c_adv_slice` | 18 | C | H-X-001, C-007 | c_adv at dz in {4, 8, 32, 128} x lam in {1, 3} (missing cells of the builder grid), `bott=vib,beta=0.0`; [NEW] `--adv_labels perm|sentence` placebo flag (used in wave 2) | 1.4 |
| 10 | `q108_a_adv_slice_phikonv2_*` | 18 | A | H-X-001 | `adv_retrain` and `dann` at dl in {4, 32}, lam 3, vs the `vib` lambda=0 rows; cheap suite (linear, mlp, knn) for the grid, full suite only on candidates; unify adversary with C first | 3 x 0.5 |
| 11 | `q109_c_matched_seed0` | 19 | C | H-C-003 | [NEW] `--method matched`: sampler of (sentence, emotion) groups x 4 actors, InfoNCE with positives {same sentence + emotion, other actor}, hard negatives {same sentence + actor, other emotion}; arms CE, SupCon-any-emotion, matched, matched-with-same-actor-positives, matched+adv; d=64, 4 folds, seed 0; sentence-probe on z | 0.9 |
| 12 | `q110_a_reinhard_reembed` | 20 | A | H-A-007 arm 2 | Reinhard-normalise pixels to a pooled training-centre template, re-embed phikonv2 and phikon, run LA-0 grid | 1.0 |
| 13 | `q111_a_floors_phikon_dinov2L_*` (if slack) | 20 | A | H-A-001 encoders 3-4 | as #1 | 3.6 |

GPU lanes for readability, not binding (workers pull by priority): GPU0 = rows 1, 2, 12 (A floors + Reinhard, ~4.6); GPU1 = rows 6, 10 (A rate +
adversary slice, ~3.9; row 13 fills here if there is slack); GPU2 = rows 3, 4, 7, 9 (C ref, per-actor, posterior, adv slice, ~5.7); GPU3 = rows 5, 8, 11
(B' stage 1, C sweep0, matched-content, ~8.1). Wave 1 total ~22 GPU-h, first results from rows 1-4 inside ~1.5 wall-hours. Decision gates after wave 1:

* **G1 (A premise)**: L_worst(LEACE) < 0.15 in >= 5/9 cells -> stop A, move its GPU share to C and B'/next field. Else A continues.
* **G2 (C validity)**: H-C-006 >= 10x chance -> queue trimmed-audio features (`c_feats_*_trim`) before any C headline; H-C-005 6-number set on the learned frontier -> report it as the hand-built winner.
* **G3 (rate vs adversary)**: from rows 6, 8, 9, 10: if the adversary adds < 0.05 at all d in both foci, park H-X-002 / X-003 / X-005 / X-007 / A-003 and spend the freed GPU-h on G4 and replication; if >= 0.10 at d >= 32, run wave-2 items 1-4.
* **G4 (B')**: GT/SAM-masked NMSE ratio on corr_noisy >= 1.3 x oracle -> stage 2; < 1.25 -> B' dead, free its slot.

**Wave 2, conditional, ~50 GPU-h**

| # | Name | Prio | Trigger | Config sketch | GPU-h |
|---:|---|---:|---|---|---:|
| 1 | `q201_x_adv_full_{A,C}` | 23 | G3 adversary useful | full d x lambda grid of H-X-001 on phikonv2 (3 ds x 3 seeds) and on C (seeds 1, 2 of builder grid + d={2,4,8,32,128}); report in bits/nats | A 6, C 4 |
| 2 | `q202_x_placebo` | 24 | after 1 | best cell per focus; arms real, label-permuted, irrelevant attribute (A: random 50-way slide hash; C: sentence), zero-gradient adversary, noise injection; 5 encoder seeds x 3 attacker seeds | 1.5 |
| 3 | `q203_x_dcor` then `q204_x_proxy` | 25 | G3 adversary gain >= 0.05 | `--pen dcor|hsic` class-balanced sampler (A 8 centres x 64, C 32 actors x 16) vs adversary and rate-matched; proxy: A 24-d descriptor, C ECAPA 192-d; wall-clock logged | 4 + 3 |
| 4 | `q205_x_population` | 25 | G3 gain >= 0.10 and tree/kernel attackers added | K=6 diverse adversaries, log-sum-exp, 0.1 re-init/epoch, held-out families LightGBM, RBF-SVM, 1-NN, ExtraTrees, transformer/TDNN | 6 |
| 5 | `q206_c_matched_seeds12` + ablations | 24 | row 11 of wave 1 beats CE by >= 0.08 | seeds 1, 2 and the rate-matched comparison at equal d | 2.7 |
| 6 | `q207_a_condadv` | 25 | G3 and V-split generator done | V in {0, 0.5, 0.8}; M0 CE, M1 marginal adversary, M2 conditional adversary, M3 LEACE, M5 ComBat; balanced-ID and OOD test | 3 |
| 7 | `q208_a_stainpair` | 26 | H-A-007 L_desc >= 0.5 | 4 augmented views x phikonv2/phikon re-extraction, stain-pair InfoNCE + cross-centre SupCon, vs Reinhard re-embedding | 6 |
| 8 | `q209_a_rate_midnight_phikon` + seeds 1, 2 of the A Pareto-relevant configs | 23 | G1 passes | replicate the configs that matter across encoders (claims need >= 2 of 3 encoders and 3 seeds) | ~8 |
| 9 | builder C `seeds12` | 30 | after wave 1 | `make_jobs.py seeds12` | 9.4 |
| 10 | `q210_b_stage2` | 24 | G4 passes | H-B-002: rho_eff {0.2, 0.6, 0.9, 0.97} x z-dim {3, 8} x {no-mask, GT-mask} x 3 seeds, push-v3; then H-B-003 remedies (R2 needs `drive_from="command"`) | 5 (+8) |

Sum: wave 1 ~23 + wave 2 ~50 = ~73 GPU-h of the ~96 available; the remainder is slack for the unmeasured A-grid overrun and reruns.
Nothing in wave 2 should be enqueued until its trigger has been read from aggregated results, not from training logs.

## 6. (c) The three hypotheses whose falsification would most change the programme

1. **H-A-001 (headroom premise).** P(falsified) ~0.22. If LEACE already drives the retrained worst-case probe near chance, Focus A has
   no job (it reduces to chasing a saturated RI), FOCUS rule 6 sends A's GPU share elsewhere, and about half of the X-hypotheses lose their A arm. Cheapest high-stakes check in the programme.
2. **H-X-001 together with its rate rows H-A-002 / H-C-002 (rate versus adversary).** P(not "rate first, adversary only at high rate") ~0.55;
   P(rate alone already solves A, falsifier (b) of H-A-002) ~0.45. Either the adversary is useless everywhere (then H-X-002, X-003, X-005, X-007, A-003, C-007 are moot and the learned contribution must be rate-limited,
   contrastive or matched-pair encoders) or it matters already at low rate (the KTH regularity does not transfer and "rate does the work" is off the table). This decides which encoder family we train next in both foci.
3. **H-B-001 (B' kill test).** P(B' dies) ~0.65. A falsified B' returns its slot to the unscouted fields in FOCUS (egocentric wearer privacy, embedding privacy for retrieval, gaze, EEG), which need a
   benchmark build and a scout before any GPU use; if B' survives, an entire third focus plus the action-conditional analogue of H-A-003 and the twin-render version of the matched-pair contrastive loss become live.

Near misses: H-X-007 (adversary is a regulariser, P ~0.25 conditional on a real gain) and H-C-006 (channel confound, would invalidate every C identity number until retaken on trimmed audio).

## 7. (d) Disagreements with the generator

1. **Priors score whole conjunctions, which makes them inconsistent across files.** H-A-001 carries five clauses at 0.75; H-X-001 four clauses at 0.45;
   H-C-007 three at 0.55. I rank on the core clause and give the as-written value where it differs (H-A-001 ~0.3, H-X-001 ~0.2, H-C-007 ~0.2, H-C-004 ~0.3, H-X-006 ~0.1).
2. **H-A-006(b), "RI is saturated", is false for the encoders we extract.** `src/a_pathorob/ri_compare.py` (builder-A's transcription of the PathoROB README) lists phikon RI 0.623 / 0.011 / 0.795 and phikon-v2
   0.619 / 0.019 / 0.768 on TCGA / Camelyon / Tolkach, and kaiko-vitb8 0.763 / 0.147 / 0.896. Saturation (0.87-0.93) holds only for gated top models. The PathoROB paper's own ComBat (+27% RI) and Reinhard (+16%) show RI moves for such models. So RI can separate methods here and the statement "RI spread < 0.08" is ~0.2.
   Consequences: (i) FOCUS.md's "headroom is NOT RI" holds for leaderboard-top models, not for ours; amend the wording when this is confirmed (verify the transcription against the README first); (ii) a gain in RI on phikon is not evidence for a new method; (iii) H-A-006 should be re-scoped to "does the retrained probe rank methods differently from RI, and do RI gains survive a retrained probe", which is cheaper and still informative.
3. **H-A-002's pivotal clause.** The generator puts "no rate reaches L_worst <= 0.25 at U >= 0.98" at 0.65. With 2-6 tissue classes and 4-8 numbers I think rate alone reaching it is close to even (0.45 for reaching <= 0.20), and that outcome moots every adversary hypothesis in A.
4. **H-C-004's "sex removal costs >= 4 emotion points" (0.55) is too high.** Sex is a rank-1 direction; LEACE-on-sex should cost ~0-2 points. Probability ~0.2.
5. **H-X-007's placebo (a) cannot work as designed** under confusion-to-uniform losses (no learnable signal -> no gradient). The informative arms are the irrelevant-attribute placebo and noise injection; prior down to 0.25.
6. **H-C-007 duplicates the H-X-001 C arm.** Merge; only the SISER entropy objective and the TDNN adversary are new. H-A-005 floor duplicates the A part of H-X-008; merge. H-B-001 and H-B-002 share a sweep (rho x arm x z-dim) and are costed 15 + 5 separately; merged and staged they cost ~10, with 2.5 up front.
7. **Cost estimates are throughput-based and double count shared grids.** Builders already queue the base arms of 8+ hypotheses (A GRID, C ref / sweep0 / seeds12). Marginal cost is ~61 GPU-h plus builder grids, but the A grid itself is unmeasured and could dominate: 45 configs per (encoder, dataset, seed) x 36 cells at ~3 min each is ~80 GPU-h. Fix: grid with the cheap suite (linear, mlp, knn), full suite (tfm, trees) only on candidates; 1 encoder x 3 datasets x seed 0 first; rescale from `a_smoke_tolkach` timings.
8. **Suite gaps the hypotheses silently rely on.** A-SUITE promises LightGBM + RFF-SVM; builder-A implemented linear / mlp / knn / tfm only. C-SUITE promises LightGBM and a TDNN; builder-C has linear / mlp / knn / transformer. Held-out-family claims (X-002, X-003) and "worst case" in general need tree and kernel families; add before wave 2.
9. **Threshold units**: absolute accuracies written for 23-way enrolment (C-001, C-005, C-006) do not match the 91-way builder protocol; restate as multiples of chance.
10. **Order of the generator's "first batch".** I agree with controls + premises first, and add two items that are free once C runs exist: the seen-vs-unseen gap (X-005 half) and the secondary-secret measurement (C-004), both pure aggregation. I put H-A-004, H-X-004 and Obliviator behind controls, because each needs heavy new code (pixel pipeline, Sinkhorn-VQ, kernel erasure) and each has a hard cheap comparator (Reinhard re-embedding, plain VQ at equal bits, ISP).

## 8. What would re-rank this list

* `a_smoke_tolkach` timings -> rescale all A costs; if a cell exceeds ~1 GPU-h, cut the A grid before wave 1 row 6.
* First `leace` / `raw` numbers: L_worst(LEACE) < 0.15 -> H-A-001 falsified, A rows drop to PARK.
* H-C-006 or H-C-005 results change the C benchmark definition or the floor every C row is compared to.
* Any adversary Delta < 0.05 at d >= 32 in both foci -> ranks 14-17 and 19 drop to PARK; H-C-003 and H-A-004 move up.
* Rate-only bottleneck reaching L_worst <= 0.20 at U >= 0.98 in A -> A becomes a benchmark-and-controls contribution; G2 rows lose their A arm.
* B' stage 1: ratio on corr_noisy >= 1.3 -> promote H-B-002 to rank ~12; < 1.25 -> mark rows 20 and 24 KILL.
