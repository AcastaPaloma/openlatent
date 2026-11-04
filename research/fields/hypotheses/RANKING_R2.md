# Hypothesis ranking, ranker pass 2 (2026-11-04; written after reading the VM queue at 12:52Z)

Inputs read in full: CHARTER, FOCUS, journal 2026-11-04, results_notes A/B/C/D, RANKING (round 1), INDEX, all 19 `H2-*.md`, `literature/round2_task_agnostic_fields.md`,
plus the code that decides how evidence is read (`a_pathorob/{splits,adapters,suite,aggregate_p2}.py`, `c_cremad/{ae,run_exp,dither_report}.py`, `scripts/{worker,normalize_queue}.py`).

## 0. Evidence boundary (read this before the numbers)

* **What the notes contain.** A.md: protocol 2, **seed 0 only**. B.md: closed. C.md: section 2d (the pre-registered AE round, 3 seeds x 4 folds) is complete and says FAIL;
  **section 2e (trimmed AE, dithered-code control) does not exist yet** (2d ends "pending at the time of writing this block"). D.md: headroom stage final (NO-GO); section 5.3 (post-hoc learned arms) still says "(pending)".
* **[PEEK] = ranker's own look at finished job outputs on the VM** (read-only `scp` of `*.jsonl`, `preds/`, `result_s0.json` into my scratchpad; I ran the repo's own `aggregate_p2.py` and C's `dither_report.py`
  and a 40-line script that reproduces the A.md L and held-class numbers exactly, e.g. midnight/Tolkach `ae dl32 lam10` L 0.27, held 0.993, retention 0.618). Point estimates, partial seeds, no intervals unless the
  official aggregator printed them. **Not an aggregate and not a verdict: when the notes are written, the notes win.**
* **Disclosure.** I have seen midnight/TCGA seed 1 (and 0), dinov2-L/TCGA and /Tolkach seeds 0-2, phikon/TCGA seed 0, midnight/Tolkach seed 0. I have NOT seen midnight/Tolkach seeds 1-2 or midnight/TCGA seed 2
  (running/pending). The kill margins in sections 5 and 7 were written after the peek for the cells I saw and are blind only for those three jobs.
* "GPU-h" below = worker-hours. A-type jobs are CPU-bound (LightGBM/RFF attackers, 700-1400% CPU, GPU 11-40%): the real ceiling is 56 cores, not 4 GPUs. Keep at most 3 A-type attacker jobs concurrent.

## 1. What the evidence says now (the five facts that move the ranking)

**F1. The lead is shrinking at its first out-of-sample look. [PEEK]** Config `ae beta1e-3 dl32 lam10` (the seed-0 post hoc pick), held-class utility minus the best floor with L <= L_AE + 0.05:

| cell | seed | L | held acc | Delta vs LDA family (A.md matching) | Delta vs any inductive floor (LDA, LDA+PCA, PCA, rand. proj.) | retention Delta |
|---|---|---|---|---|---|---|
| midnight/TCGA | 0 | 0.35 | 0.800 | +0.164 (vs lda_pca d8) | +0.065 (vs randproj d4) | +0.016 |
| midnight/TCGA | 1 | 0.32 | 0.778 | +0.011 (vs plain lda 0.767) | +0.011 | -0.025 |
| midnight/Tolkach | 0 | 0.27 | 0.993 | +0.083 (vs lda) | +0.083 | +0.216 vs PCA-4 |
| midnight/Tolkach | 1, 2 | | | running / pending | | |

The seed-0 TCGA gain was mostly a weak comparator draw (LDA-family held acc at that leakage 0.636 at seed 0, 0.767 at seed 1) and a reference set that excluded random projection.
Under the official 2-seed aggregate for midnight/TCGA the **V-selected** AE is `dl32 lam0` (no adversary, L 0.58) with Delta -0.058 [-0.115, -0.007] against the matched-dimension LDA arm: it does not win.
Dinov2-L, 3 seeds (V-selected, Tolkach): Delta -0.004 [-0.017, 0.006]. Best-of-10 AE minus best inductive floor on this "negative" encoder: +0.031, +0.032, +0.019 (TCGA s0-2), **+0.190** (Tolkach s0, L 0.08),
+0.028, -0.021 (Tolkach s1-2): typical selection inflation about +0.03, worst case +0.19. Phikon/TCGA s0: best-of-10 +0.071 vs inductive floors, but +0.10 to +0.16 for several configs if only the LDA family is the reference.

**F2. The comparator set decides the answer (floor-curve density).** The same AE point gains +0.16 against the LDA family and +0.01 to +0.07 against all inductive floors; the +0.19 on dinov2-L sits at L 0.08 where the only floor is plain LDA.
Retention "+0.24" is against PCA-4 (4 numbers vs a 32-number code). No floor family currently populates L 0.15-0.40 at high dimension; noise-dithered PCA / LDA+PCA (H2-014) would. Until it exists, no Delta in A is interpretable.

**F3. Seeds do not replicate the test set.** `splits.make_splits` draws the sealed C from `RandomState(1234)`, independent of the run seed. Seeds 1-2 therefore re-draw A/B/V and the model init, **not** the 12 (Tolkach), 24 held (TCGA) test cases that
dominate every interval. "3 seeds" is the right replication for training noise and no replication of test-unit noise. The held-class set is also fixed (TCGA: breast+colon seen, the two lung subtypes held). Cross-fitting with a rotating C (H2-016 step 3) is the replication that matters.
V is ~7 cases in Tolkach and ~29 in TCGA, so V-selection inside a cell is close to random among 10 AE configs (this alone makes H2-001's "same family selected in >= 2 of 3 seeds" a ~coin flip).

**F4. C is decided against the AE; the dither control, run by C on its own, points the same way. [PEEK, seed 0, post hoc; interpolation is mine]** C.md 2d: selected AE (d32, lam3) U_HE 0.326 [0.307, 0.345] at 6.6x chance identity; every adversarial AE loses 0.19-0.24 U_HE to the width-matched LDA-4 comparator;
the only positive differences are lam 0 (leaks 14x at d16, 35x at d32). Raw outputs of `c_dith_*`: dithered **label-free, adversary-free** AE (d32, lam0) r=1 U_HE 0.413 at 10.1x, r=2 0.300 at 3.7x, i.e. about 0.35 at 6.6x (>= the adversarial AE at equal leakage); LDA-4 padded to 16 dims, undithered: 0.450 at 6.5x
(dominates both). So on C the adversary adds nothing over rate, and a label-supervised projection beats every label-free code even on the held-out emotions. The trimmed variant (`c_aetrim_*`, s0 running) can only worsen criterion 3.

**F5. VM state at 12:52Z.** Running: `a_p2h_midnight_tolkach_esca_s1`, `a_p2h_midnight_tcga_s2`, `c_aetrim_d32_l3_s0`, `d_ph_lrn_gtr_s1` (D is closed; post hoc). Pending (13): `a_p2h_midnight_tolkach_esca_s2` (decisive), 3 Camelyon `a_p2x_*` (no held-class cell: cannot bear on the lead),
`c_aetrim_s1-2`, 7 label-supervised C seeds (`c_adv/c_rate/c_he_*`, family already closed). Done: all six dinov2-L p2h cells, midnight/TCGA s0-1, midnight/Tolkach s0, phikon/TCGA s0, C dithered controls s0. **No phikon-v2 holdout job exists**; phikon has TCGA s0 only.
Parked: 28 `a_gate_*`, 9 `b_*`. Load average 29 on 56 cores.

## 2. Axes (unchanged from round 1)

Likelihood = my probability that the decision-relevant core clause holds, after F1-F5 (generator prior in brackets). App (1-5) = D (0-2, changes which encoder family or field we train next) + U (0-1, prunes or unlocks >= 3 follow-ups) + T (0-1, transfers across fields) + L (0-1, itself a learned method).
Info/GPU-h = H2(likelihood) x (App/5) / max(cost, 0.25 GPU-h); H2 = binary entropy in bits. It is a heuristic; the 0.25 floor makes every free analysis score 1.3-3.7, so **tiers decide order, the number breaks ties inside a tier**:
Tier 1 = can kill the live lead or changes how all evidence is read; Tier 2 = opens/closes the next field or a floor; Tier 3 = only worth running if the lead survives (a gate decides); Tier 4 = park.

## 3. Ranking table

| Rank | ID | Tier | Likelihood [gen] | App | Info/GPU-h | Cost GPU-h (CPU) | Status / blocker | Verdict |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | H2-016 power audit (steps 1-2; step 3 = X1) | 1 | 0.85 [0.70] | 4 | 1.95 | 0 (1.5 CPU-h) | preds on disk; **plus F3** | RUN FIRST, commit before any new verdict |
| 2 | H2-001 core, re-scoped to midnight x {TCGA, Tolkach}, V-selected | 1 | 0.12 survives [0.30]; as written (3 of 6 cells, 4 clauses) 0.03 | 4 | 0.85 | 0.5 left (in flight) | s1, s2 running/pending | READ WHEN DONE; promote pending s2 |
| 3 | H2-014 dithered label-free code | 1 | 0.55 kills [0.35] | 4 | 0.79 | 1.6 incl. null cells (3 CPU-h) | ~100 lines in `adapters.run_floor` (C already has `dither_r`) | BUILD NOW, RUN NEXT |
| 4 | H2-002 half (a): adversary adds <= 0.02 over lam0 | 1 | 0.65 [0.35 conj.; conj. 0.20] | 5 | 3.7 | 0 (1 CPU-h) | lam0 and lam>0 rows already in A p2 grid and C `c_ae_*` | RUN NOW on existing outputs; new arms R2-R4 -> tier 3 |
| 5 | H2-015 selection-inflation null | 1 | 0.40 kills [0.35] | 2 | 1.55 | 0 (2 CPU-h) | natural null (dinov2-L) now; dithered null after rank 3 | RUN NOW (natural), redo with dither null |
| 6 | H2-010 stage 1 = **SC-0** single-cell floors-only | 2 | 0.35 passes [0.45]; full claim 0.08 [0.15] | 4 | 1.49 (HVG-PCA encoder) / 0.30 (+scFM) | 0.5 (3-4 CPU-h) + ~1 builder-day | loader, data fetch, licence check | OPEN AS CHECK-ONLY (section 6) |
| 7 | H2-017 per-actor normalisation floor (C; A has `centre_whiten`) | 2 | 0.40 wins [0.45] | 2 | 1.29 | 0.3 | cached WavLM features | RUN NOW (CPU) |
| 8 | H2-008 gates G1/G2 (sentence probe on cached features) | 2 | gate fails 0.50; full claim 0.08 [0.20] | 2 | 1.6 gates / 0.11 full | 0.05 gates / 1.5 full | sentence probe | GATES NOW, full only if both pass |
| 9 | H-A-003 floors-only under centre-class confound 0.8 (round-1 carry-over) | 2 | LDA leaks L >= 0.4: 0.50 | 2 | 1.33 | 0.3 | spec edit of `a_sw2conf_*` (floors, no learned) | FILLER; outside H2-019 scope, counts neither way |
| 10 | H2-018 pre-commit h per cell (procedure) / the "law" | 2 | law holds 0.12 [0.30] | 3 | 1.27 | 0 | script | ADOPT PROCEDURE NOW; judge the law at DP |
| 11 | H2-011 JUMP: metadata gate then floors | 2 | check passes 0.25 [0.70] | 4 | 1.30 | 0.5 (1 CPU-h) | replicate structure unverified | SECOND CHOICE (section 6) |
| 12 | H2-009 D digits/entity reframe, check only | 2 | check passes 0.30 [0.20] | 2 | 0.70 | 0.5 | oracle variants (~1 builder-h) | RUN AS SIDE CHECK (inside D, not a new focus) |
| 13 | X1 cross-fit remedy (H2-016 step 3, rotating C) | 2/3 | decides only if mixed | 4 | 0.19 | 2-4 | 1 flag in `make_splits` | RUN ONLY IF IN-FLIGHT + DITHER ARE MIXED |
| 14 | H2-005 slide fingerprint (second secret) | 3 | 0.60 [0.60] | 2 | 0.39 (x P(lead alive) ~0.1) | 1 (6 CPU-h) | needs embeddings -> rerun | GATED |
| 15 | H2-012 EEG (REVE only) | 4 | 0.20 [0.35] | 3 | 0.29 | 1.5 | weights/licence unverified | PARK |
| 16 | H2-004 dose-response | 3 | 0.30 [0.40] | 3 | 0.26 | 2 (10 CPU-h) | lead alive | GATED |
| 17 | H2-007 conditional-decoder VAE | 3 | 0.15 [0.30] | 3 | 0.18 | 2 | lead alive | GATED |
| 18 | H2-003 label-free contrastive | 3 | 0.25 [0.35] | 3 | 0.16 | 3 | lead alive | GATED |
| 19 | H2-001 widened (phikon-v2, phikon) | 3 | 0.03 [0.30] | 4 | 0.06 | 2.5 | no phikon-v2 holdout job exists | GATED (filler only) |
| 20 | H2-013 astronomy twin render | 4 | 0.40 [0.65] | 2 | 0.13 | 3 (+1 builder-day) | self-built simulator can be tuned to pass | PARK |
| 21 | H2-006 universal adapter | 4 | 0.05 [0.20] | 3 | 0.06 | 3 | needs the AE lead | PARK |
| - | H2-019 stop rule | gov. | stop condition met 0.42 [0.35] | - | - | - | needs amendment (section 7) | ADOPT WITH AMENDMENTS |

### Row notes (only where I differ or the reason is not in section 1)

* **H2-016.** Agree and promote. Add: C has no power problem (U_HE paired SE 0.012-0.015, MDE80 ~0.03-0.04, so the +0.05 criterion was resolvable); the problem is specific to A (12 / 24 held / 48 units) and to D's 323 queries. Add F3 (fixed C, fixed held-class set) to the audit text: it changes what "more seeds" can buy.
* **H2-001.** Core clause = both midnight cells, V-selected, vs the full inductive floor envelope including dither. Phikon-v2 and phikon cannot be required for "survives" because the jobs do not exist and phikon/TCGA s0 already shows no gain against inductive floors beyond best-of-10 noise.
* **H2-014.** The AE (beta 1e-3) releases a sample but its learned posterior variance is small (79-150 bits), so it is effectively a nonlinear deterministic bottleneck with an adversarial regulariser. "PCA-32 + Gaussian channel" is the matching floor class. Port C's fixed-noise-per-item dither (`run_exp.py`, `dither_r`) to A; noise must be present for attacker training, probe training and test alike.
* **H2-002.** Half (a) is nearly settled by C (dithered lam0 AE >= adversarial AE at equal leakage). Half (b) (nonlinear decoder beats PCA-d by >= 0.03 AUC in >= 3 of 4 cells) is ~0.25 and only A can answer it. Run (a) on existing grids today.
* **H2-017.** Prosody-six with oracle per-actor z-scoring reached 2.0x chance; a 1,536-d vector leaves LDA room for residual structure, so I keep it near even. Whatever it shows, it states the deployable baseline (transductive per-source normalisation) that any learned latent must beat.
* **H2-008.** Mean+std pooling may not resolve 12 sentences (G1) and the selected AE already loses 0.19-0.24 U_HE; the gates cost minutes, so run them and stop there unless both pass.
* **H2-009.** I am more optimistic than the generator: digits are not what topical queries use, so oracle retention near 1.0 is plausible (0.7); H1 (GTR leaks digit tokens above their strong frequency prior) is the coin flip (0.5). All D machinery exists.
* **H2-010 / H2-011 / H2-012 / H2-013.** See disagreements D9-D12. Only 010 and 011 pass a "learned latent + cheap + licence-clean" screen.
* **H2-018.** The procedure (commit h before any learned arm) costs nothing and would have saved the A and D detours; adopt it. As a law it already looks weak: midnight/TCGA s1 h = 0.856 - 0.767 = 0.089, g = +0.011 (12%); dinov2-L/Tolkach s1 h ~ 0.07, g <= 0.03.

### Ranker-added protocol items (not hypotheses; attach to every A claim from now on)

* **P1.** Replace "more seeds" by cross-fit: rotate C over K=4 unit-disjoint folds (`rng_c` fixed, chunk `fold` of the shuffled stratum), refit and V-select inside each fold.
* **P2.** The comparator is the lower envelope of ALL inductive floors plus dithered codes, fixed before reading; named-family comparisons are secondary and labelled.
* **P3.** A learned point with fewer than 2 floor points within +-0.05 of its L has "no comparator" and cannot be called a win.
* **P4.** Select family and configuration on seed-pooled V (as C does), never per seed; report the seed-matched comparator spread (0.636 vs 0.767 in F1) next to every Delta.

## 4. Round-1 carry-over (what is still open)

| ID | Disposition |
|---|---|
| H-A-001, H-C-001, H-C-005, H-C-006, H-X-008 (A part) | answered (gate G1 not met; raw measurable; prosody and channel floors measured; `centre_whiten` measured). H-X-008 C-part = H2-017 |
| H-A-002, H-C-002, H-C-007, H-X-001 | answered enough: rate ~ LDA, adversary adds nothing measurable (C), A's label-supervised families lose to matched LDA |
| H-A-003 | OPEN, floors-only check as rank 9; specs `a_sw2conf_*` were never queued and contain no AE/dither arms |
| H-A-006 | premise ("RI saturated") false for our encoders; remainder (do RI gains survive a retrained probe) PARK, 10 CPU-h |
| H-A-007 colour statistics | OPEN, CPU-only filler (2 CPU-h); matters only if learned adapters in A survive |
| H-C-004 attributes | free analysis on existing C runs that kept secrets (AE runs used `--no_secrets`); write-up by-product, no GPU |
| H-X-002, X-003, X-005, X-007, H-A-004, H-A-005, H-C-003 | PARK (no adversary gain to refine; C closed) |
| H-X-004, H-X-006 | PARK |
| H-B-001..003 | CLOSED (B' NO-GO) |

## 5. Decision output 1: queue for the next ~12 GPU-hours

Expected spend is 4-7 GPU-h, not 12. The 12 is an envelope covering two mutually exclusive continuations (tier 3 if the lead lives, stage 2 of a new field if SC-0 passes); if both die, **leave the GPUs idle and write up**.

**Do now on the VM (no new code):**
* `sed -i 's/^priority:.*/priority: 3/' ~/queue/pending/a_p2h_midnight_tolkach_esca_s2.yaml` (decisive job; reserved priority per the short-decision-batch rule).
* Park (not delete): `a_p2x_{dinov2L,midnight,phikon}_camelyon_s0` (Camelyon has no held-class cell). Demote to lowest priority: the 7 `c_adv/c_rate/c_he_*` seeds. Keep `c_aetrim_s1-2` (3 GPU-min each, completes criterion 3).
* Do not requeue `d_ph_lrn_gtr_s2`; let s1 finish.
* `normalize_queue.py` groups by name prefix, so an `e_` focus needs no code change; its first jobs go to priority <5 on submission (standing rule).

| Order | Item | Hypotheses | GPU-h | Gate / what it decides |
|---:|---|---|---:|---|
| 0a | Power audit steps 1-2 on A preds (all done cells) and C/D intervals; add F3 | H2-016 | 0 (CPU) | Commit the MDE80 table before any aggregate is read as a verdict |
| 0b | Attribution on existing grids: AUC of (L, U_held) frontier for lam0 vs lam>0 vs PCA/randproj, A cells + C `c_ae_*` | H2-002(a) | 0 | If lam>0 - lam0 <= 0.02 in >= 3 of 4 cells: drop the word "adversarial" from the lead |
| 0c | Natural-null selection inflation (dinov2-L, 6 cell-seeds, best-of-10) | H2-015 | 0 | Null band (+0.02..+0.03 typical, +0.19 worst) vs observed |
| 0d | G1/G2 sentence gates; per-actor normalisation arms; commit h per cell | H2-008, H2-017, H2-018 | 0.3 | Closes C's loose ends; fixes the deployable floor |
| 0e | JUMP replicate-structure metadata look (20 min, no profile download); SC-0 licence/size verification | H2-011, H2-010 | 0 | Section 6 |
| 1 | **In flight + promote:** `a_p2h_midnight_tolkach_esca_s1` (GPU0), `..._s2` (pending), `a_p2h_midnight_tcga_s2` (GPU2) | H2-001 core | 0.5 | Aggregate `aggregate_p2.py --compact --runs a_p2h_midnight_*`; apply section 7 criteria |
| 2 | **`a_dith_midnight_{tcga,tolkach_esca}_s{0,1,2}` + `a_dith_dinov2L_{tcga,tolkach_esca}_s0` (null)** | H2-014, H2-015 | 1.6 | AE within 0.03 of dithered floor in both midnight cells = lead dead (E0) |
| 3 | `e_sc0_{immune,pancreas,hlca100k}` floors-only | H2-010 stage 1 | 0.5 (+3-4 CPU-h) | SC-K0..K3 (section 6) |
| 4 | `d_dig_check_gtr_s{0,1,2}` (digit tier + `oracle_mask_digits`) | H2-009 | 0.5 | H1/H2 of the pre-registration; K1-K4 only if both pass |
| 5 | `a_conf_floors_midnight_{camelyon,tcga,tolkach_esca}_s0` | H-A-003 | 0.3 | LDA L >= 0.4 under rho 0.8 -> a second, label-supervised regime where floors are weak |
| 6 | **If mixed:** `a_xf_midnight_{tcga,tolkach_esca}_f{0..3}` (+ `--seen_swap` variant) | H2-016 step 3, P1 | 2-4 | The only replication that resamples test units |
| 7 | **If lead alive (all of section 7 "ALIVE"):** `a_p2h_phikonv2_{tcga,tolkach_esca}_s0-2` (1.5), H2-004 (2), H2-005 (1), H2-002 new arms (1), H2-003 (3) | tier 3 | ~8.5 | Generalisation and attribution; not before ALIVE |
| 8 | **If SC-0 passes:** `e_sc1_*` AE+adv, cVAE, dither on HVG-PCA and one scFM | H2-010 stage 2 | 4-5 | Pre-registered margin +0.05 macro-F1 over best inductive floor |

Spec sketches (new code marked):
```yaml
# NEW CODE: AD.DITH = [("dither", dict(base=b, d=d, r=r)) for b,d in (("pca",16),("pca",32),("pca",64),("lda_pca",32),("lda_pca",64)) for r in (0.25,0.5,1,2,4)]
name: a_dith_midnight_tolkach_esca_s0        # x {tcga, tolkach_esca} x seeds 0-2; dinov2L s0 as null
priority: 3
cmd: USE_TF=0 $HOME/venvs/a/bin/python research/fields/src/a_pathorob/run_sweep.py --encoder midnight_clsmean --dataset tolkach_esca --seed 0 --grid dith --holdout
hf_path: runs/a_dith_midnight_tolkach_esca_s0
# NEW CODE: named grid p2dith = AD.P2 + AD.DITH + AD.CONTROLS; --fold K --nfold 4 (rotating sealed C); --seen_swap (held-class set = the other half)
name: a_xf_midnight_tcga_f0
cmd: USE_TF=0 $HOME/venvs/a/bin/python research/fields/src/a_pathorob/run_sweep.py --encoder midnight_clsmean --dataset tcga --seed 0 --grid p2dith --holdout --fold 0 --nfold 4
# NO NEW CODE: floors only, confounded training
name: a_conf_floors_midnight_tcga_s0
cmd: USE_TF=0 $HOME/venvs/a/bin/python research/fields/src/a_pathorob/run_sweep.py --encoder midnight_clsmean --dataset tcga --seed 0 --grid raw,lda,lda_pca,pca,centre_whiten,leace,isp --confound 0.8
```
Concurrency: at most 3 A-type jobs at once; SC-0, D and C jobs fill the fourth slot. Builder lead times (start in parallel now, they dominate wall-clock): A dither + fold flags ~2 h; SC-0 loader ~0.5-1 day; D digit tier ~1 h.

## 6. Decision output 2: open ONE new focus now?

**Recommendation: open single-cell batch integration as a CHECK-ONLY focus ("Focus E, SC-0"), not locked; do not open JUMP; do not choose "none".** It is not written into FOCUS.md as a locked focus unless SC-K0..K3 clear (charter rule 6).

Why now rather than never: the live lead has ~0.12 survival probability, C is decided, and D/B' are closed. Without a pre-warmed candidate the programme either idles or stops on a lead-dies reading while cheaper checks exist. SC-0 costs <= 0.5 GPU-h and <= 1 builder-day and can run in parallel with the kill tests.
Why single-cell over JUMP: (1) the field itself ships the label-only control (`embed_cell_types`, one-hot of labels), so the "task implies the secret" trap is pre-registered by construction; (2) frozen scFM embeddings are the closest analogue of "a learned latent released for unknown future use" and the opened scFM paper lists invariance objectives as untested; (3) the held-out-class and retention protocols of A map one-to-one (held cell types, kNN-structure retention); (4) published signal that label supervision costs intra-type structure (scIB-E) is the same shape as the lead;
(5) JUMP's strongest inductive floor is nearly the method itself: LDA on train-compound identity estimates the nuisance covariance from replicates, which is what Sphering/TVN do; profiles are tabular CellProfiler features, not a learned latent; and the decisive precondition (cross-source replicates for held-out compounds) is unverified.
Why not none: see above; the cost of the check is below the cost of one more A job wave.

**SC-0 pre-registration (to be committed before any model number):**
* *Data.* scIB Immune (33,506 cells, 10 batches, 16 types) and Pancreas (16,382, 9 batches, 14 types) as small sanity sets [scout numbers, unverified]; **HLCA** (CELLxGENE Census, 14 studies, ~107 donors, 5 annotation levels) subsampled to <= 100k cells as the instrument that has donors and a hierarchy. Unit = donor where annotated; otherwise batch, and then intervals are descriptive.
* *Encoder.* FM-free first: 2,000 HVG, log-normalised, PCA-50 ("encoder 0"). scFM embeddings (Geneformer, UCE, scGPT; 1-2.5 GPU-h each) only after SC-0 clears.
* *Protocol.* Leave-batch-group-out folds (>= 2 held-out batches per fold, every batch held out once; this is the cross-fit by construction). Adapter/floors fitted on training batches only. Utility = macro-F1 of a probe trained on training batches, scored on held-out batches, for (a) held-out cell types (A's held-class analogue), (b) fine labels when only coarse labels (levels 1-2, <= 10 classes) were used, (c) kNN-structure retention. Leakage = batch attackers (linear, MLP, kNN, LightGBM) retrained on half the held-out batches' cells (donor-disjoint if possible) and scored on the rest; L normalised to raw and chance.
* *Arms (floors and references only).* Inductive floors: PCA-d, **dithered PCA-d (H2-014)**, coarse-LDA + PCA padding, random projection, LEACE(batch) fitted on training batches. Transductive references (never counted as wins): per-batch whitening (`centre_whiten`), Harmony, label-free scVI with batch covariate. Controls: constant, noise, orthogonal batch shuffle at chance; `embed_cell_types`-style one-hot label code as the label-only ceiling for label metrics.
* *Kill conditions (any one stops SC-0 and Focus E):*
  SC-K0 instrument: HLCA subsample has < 8 studies or < 40 donors in held-out folds, or the floor-vs-floor permutation MDE80 on macro-F1 is > 0.05. (A's lesson: 12-48 units cannot adjudicate +0.04.)
  SC-K1 measurability: raw batch attacker on held-out batches < 3x chance or < 0.15 above chance, or any control > chance + 0.03.
  SC-K2 under-determination: coarse-LDA + PCA-16 keeps > 70% of raw's above-chance fine-label / held-type macro-F1 (label-only floor holds, the A and C pattern).
  SC-K3 headroom: the V-selected best inductive floor at batch leakage <= L(best transductive reference) + 0.05 is within 0.05 macro-F1 of that reference on >= 2 of the 3 datasets (one of them HLCA).
* *Stage-2 claim (only after GO):* label-free AE+adversary / cVAE / dither exceed the best inductive floor by >= +0.05 macro-F1 at matched L, 95% CI above 0 (bootstrap over donors, folds as repeats), >= 2 of 3 seeds, V-selected, power audit first. Budget <= 5 GPU-h and 1 builder-day; one attempt.
* *Expected cost.* GPU: <= 0.5 GPU-h (attackers); CPU: 3-4 h (PCA, Harmony, scVI references); builder: ~1 day (loader, folds, metric wrappers).
* *Licences to verify before download (none verified by me):* per-dataset `license` field of every HLCA/CELLxGENE dataset (CC BY 4.0 is typical, some may be NC); reuse terms of the GEO/ArrayExpress studies behind the scIB Immune/Pancreas objects; weight licences of Geneformer (Apache-2.0 per HF card, verify), scGPT (code MIT, checkpoint terms), UCE (MIT, figshare weights). Code: scib-metrics BSD-3 (use), scvi-tools BSD-3 (use), scIB-E is AGPL-3.0 (do not copy), harmonypy is GPL-3.0 as I recall (run as a reference, do not redistribute). Per rule 2, derived embeddings go to HF only if the data licence allows; raw datasets never.
* *Reusable builder code.* `a_pathorob/splits.py` (`PData`, unit-disjoint A/B/V/C + OOD halves; needs the hard-coded `ID_CENTRES/OOD_CENTRES` generalised and a `--fold` flag), `adapters.py` (`train_ae`, `train_vib/vq/dann/adv_retrain/supcon`, floors `lda`, `lda_pca`, `pca`, `randproj`, `centre_whiten` = per-batch whitening, `leace`, `isp`; `P2` grid), `attackers.py` + `suite.py` (six-attacker retrained suite, `knn_retention` = raw-neighbourhood retention ~ scIB-E kNN-Jaccard, `oodcentre` = unseen-batch attacker), `aggregate_p2.py` (unit bootstrap, matched-LDA arm, tally), `c_cremad/ae.py` and the `dither_r` hook in `c_cremad/run_exp.py`. New: AnnData -> `PData` loader, leave-batch-group-out folds, Harmony/scVI wrappers, scib-metrics calls.
* *JUMP, if SC-0 is killed:* first the 20-minute metadata look (do non-Target2 compounds have wells in >= 2 sources? from memory the production library was screened mostly in single sources: unverified, decisive). If yes, use Target2-style compounds split 50/50 with cross-source replicate mAP, floors = PCA-d, dithered PCA, LDA on train-compound identity (expect ~Sphering), reference = Harmony/Sphering; kill if the best inductive floor is within 0.03 mAP of Harmony at matched plate leakage. CC0 data, ~1 GPU-h. Note H2-011 labels "scenarios 4 and 5" as 302 and ~82k compounds while the scout lists 302 as the smallest scenario: verify before use.

## 7. Decision output 3: STOP versus pivot within the next 2-3 hours

Margins fixed here (cell-seed means, paired, vs the lower envelope of all inductive floors including dither; V-selected configuration; the generator's +0.02 / +0.04 are kept):

* **AE-DEAD** (stop spending on the AE line in A) if ANY of: (a) 3-seed Delta_held < +0.02 in **either** midnight cell (the generator's falsifier says "and"; a lead that lives in one 12-case cell is indistinguishable from F1-F3, so I stop on "either"); (b) dithered floor within 0.03 of the AE in U_held at L +-0.05 in both midnight cells (E0); (c) observed gains below the 95th percentile of the null maximum (H2-015) in both cells.
* **AE-ALIVE** only if ALL: Delta_held >= +0.04 with lower bound > 0 in both midnight cells (cross-fit or >= 2 of 3 seeds with pooled V selection), no dithered family within 0.03, H2-015 p < 0.1, dinov2-L (V-selected) <= +0.02. Anything between = **MIXED**: run X1 (2-4 GPU-h), do not claim, do not widen.
* **C-DEAD**: already true on criteria 1-2 (2d). H2-008 gates are the last loose end; criterion 3 (trimmed) cannot rescue it.

| Outcome at ~+3 h | Probability (mine) | Action |
|---|---|---|
| AE-DEAD, C-DEAD, SC-0 and D-digits and JUMP-metadata all fail their gates | 0.75 x 0.34 = **~0.25** | **STOP the whole line** (H2-019 S1 and S2 and S3 met): write up benchmark + controls + negative results, release the VM. Deployable recommendation: transductive per-source normalisation plus a rate cap |
| AE-DEAD, C-DEAD, at least one new check clears | **~0.50** | **PIVOT**: exactly one stage-2 attempt on the field that cleared (cap 5 GPU-h, 1 builder-day, margins pre-registered above), then STOP if it fails. A and C become the write-up |
| AE-ALIVE | **~0.10** | CONTINUE: tier 3 wave (phikon-v2 first), H2-004/-005; still only the SC-0 check for the new field |
| MIXED | **~0.15** | X1 cross-fit before any decision; SC-0 keeps running |

Amendments to H2-019 as written: (1) S3 names five candidates but only three are cheap (SC-0, D digits, JUMP metadata); evaluate DP2a on those at ~+3 h and run EEG/astronomy (DP2b, ~4.5 GPU-h plus builder-days, pass priors 0.2 and 0.4) only if the owner values breadth. (2) A stage-1 pass buys one stage-2 attempt and does not by itself cancel the stop, exactly as H2-019 says: only a learned arm beating its floor does. (3) The "eight independent failures" arithmetic overstates independence (one mechanism, one dataset family for three of them), so use it as a lower bound only.
What does NOT count as stop evidence: D's rare-word NO-GO, B' NO-GO, anything from dinov2-L, and A in-distribution (all already priced in).

## 8. Disagreements with the round-2 generator's priors

| # | Item | Generator | Mine | Reason |
|---|---|---|---|---|
| D1 | H2-001 survives | 0.30 | 0.12 core, 0.03 as written | F1: TCGA s1 gain collapses; phikon/TCGA s0 no gain against inductive floors; dinov2-L none; 3-of-6 cells needs jobs that do not exist; V-selection in Tolkach (~7 cases) is nearly random |
| D2 | "3 seeds" as replication | assumed | not replication of test units | F3: sealed C is `RandomState(1234)`, identical across seeds; cross-fit is the replication |
| D3 | H2-014 kills | 0.35 | 0.55 | F4: C's dithered lam0 AE >= the adversarial AE at equal leakage; F2: floor curve has holes at L 0.15-0.40 |
| D4 | H2-015 kills | 0.35 | 0.40 | dinov2-L null band +0.02..+0.03 typical, +0.19 at sparse L; but TCGA s1 already fails without it |
| D5 | H2-002 conjunction | 0.35 | 0.20 (half a 0.65, half b 0.25) | C evidence for (a); (b) needs the AE to beat PCA/randproj, which F1 does not show |
| D6 | H2-008 | 0.20 | 0.08 | selected AE loses 0.19-0.24 U_HE to LDA; G1 uncertain on mean+std pooling |
| D7 | H2-006 | 0.20 | 0.05 | requires the AE lead; dataset shift; every closed-form eraser we measured failed out of sample |
| D8 | H2-018 law | 0.30 | 0.12 | F1 points contradict g/h >= 0.4; keep the procedure |
| D9 | H2-010 check / full | 0.45 / 0.15 | 0.35 / 0.08 | donor/batch counts (power), inductive vs transductive confusion, hierarchy needs HLCA, strong FM-free baseline |
| D10 | H2-011 check | 0.70 | 0.25 | replicate structure for held-out compounds unverified; LDA on compound identity ~ Sphering; tabular features not a learned latent |
| D11 | H2-012 check | 0.35 | 0.20 | audit shows LEACE sufficient for LaBraM/CBraMod; weights and licences unverified |
| D12 | H2-013 check | 0.65 | 0.40 | G2 (clean-twin gap >= 0.10) is the B' trap; the simulator is ours to tune; scout said SKIP |
| D13 | H2-009 check | 0.20 | 0.30 | digits are not used by topical queries; H1 is the open part |
| D14 | H2-016 | 0.70 | 0.85 | printed intervals already imply it; add F3 |
| D15 | H2-019 stop condition | 0.35 | 0.42 | S1 ~0.78 (my stricter "either"), S2 ~0.95, S3 over five candidates ~0.55 (P(at most one passes) from my pass priors) |
| D16 | Falsifier wording of H2-001 | "and" over two midnight cells | "either" | see section 7 |

## 9. Dependencies on unfinished jobs, and what re-ranks this list

| Dependency | State | Consequence |
|---|---|---|
| A protocol 2 seeds 1-2 | midnight/TCGA s2 running; midnight/Tolkach s1 running, s2 pending; A.md still shows seed 0 | All [PEEK] numbers in F1 are provisional. Re-rank on the aggregate: if midnight/Tolkach s1-2 both reproduce >= +0.04, H2-001 core -> 0.35 and tier 3 opens |
| C section 2e | trimmed AE s0 running (s1-2 pending); dithered control s0 done but unwritten; F4 numbers are raw outputs read through C's own `dither_report.py`, interpolation mine | Trimmed variant cannot reverse 2d; if C writes a different dither reading, C's wins |
| C H2-008 sentence probe | not implemented | S2 is "met in spirit" by 2d; H2-008 gates close it formally |
| D post-hoc learned arms | s0 done, s1 running, unread by me | Cannot flip the pre-registered NO-GO |
| phikon-v2 holdout cells | no jobs | H2-001 widened and H2-014 on phikon-v2 cannot run until created |
| A dither/fold code | does not exist | Gates ranks 2-3, 6, 13 |
| SC-0 loader and licences | not started | Gates the whole new-field branch; start the builder now |

Triggers: midnight/Tolkach s1-2 both < +0.02 -> AE-DEAD (a), move H2-004/-003/-005/-006/-007 to PARK. Dither within 0.03 in both midnight cells -> E0, stop AE work in A at once. Power audit MDE80 > 0.08 in every A cell and cross-fit unaffordable -> suspend claims (S0). SC-K2 or K3 fires -> Focus E dies, pivot rule goes to D-digits/JUMP. Any learned arm beating its floor class by its margin with CI > 0 in two independent fields is the first real result and re-opens everything.
