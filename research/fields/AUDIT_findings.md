# AUDIT_findings.md: independent audit of programme 1 (fields A, B', C, D, E, F)

Auditor: AUDITOR-1 (did not build any of it). Date: 2026-11-04, VM clock about 17:45Z. Scope: CHARTER, FOCUS, journal, results_notes A-F and JUMP0, RANKING and RANKING_R2, all code under `research/fields/src/*` and `research/fields/scripts/`, raw runs on the VM (`~/work/runs`, `~/queue`) and the HF repo `KuanKuanKuan/openlatent-fields` (file listing only).

Method and limits. No git commands were run; commit questions were answered from `job_result.json`, `*.split.json` and file mtimes only. Re-aggregation was done on the VM CPU with the repo's own aggregators (read-only on `~/work/runs`; scratch output in the VM's `/tmp`). Two small synthetic/real-data sanity scripts were run on the VM CPU (Appendix B). Nothing was written to `~/queue` or `~/work`, no GPU job was started. Where a re-aggregation used fewer bootstrap resamples than the notes (B=20-300 instead of 1000), only point estimates are compared and interval ends are quoted as reproduced.

## 0. Bottom line

1. **All six headline numbers re-derive from raw files** (Section 2). I found no arithmetic error and no number in FOCUS.md or the journal that cannot be traced, apart from stale values listed in F-18.
2. **Two supporting claims that the programme uses as evidence are not valid**: (a) "the adversary adds nothing beyond a bottleneck or noise" is unsupported, because D's adversarial arm never trained (retrieval loss stuck at ln N, F-01) and A's adversaries never reach their own objective (F-02); (b) "A's AE lead is DEAD" is true for held-class label utility but the same analysis shows a replicated positive on the kNN-retention axis that FOCUS A itself names as criterion (ii) (F-03).
3. **Three of the five NO-GO verdicts are gate outcomes, not head-to-head nulls**: D (oracle headroom at one operating point, adversarial arm collapsed, digit attackers too weak), E (K2, a utility-only gate), F (a control clause the builder calls a construction flaw, then a headroom rule that counts "floor beats the published reference" as "no room"). None of E, F, D tested a valid learned arm. They should be reported as "not tested / inconclusive for learned adapters", not "NO-GO for learned adapters".
4. **Verdicts that stand**: C (strong), A for the specific claim "task-agnostic AE does not beat the cheap-floor envelope on held-class label utility in midnight" (resolved null, but ceiling-limited), A gate G1 and its replications (closed-form erasure fitted on seen units fails out of sample), B' for the one learned latent arm tested (bounded).
5. **Reopen**: F/JUMP (highest value: a 16-d linear compound-supervised map gives 3.6x raw AP on held-out compounds, so the room for a nonlinear supervised learned map is untested), D (cheap: rerun the adversarial arm with a sane lambda), A-retention (rescope, no new compute needed), E (needs a new pre-registration, not a continuation). The programme-level STOP is justified only for "label-free adapters vs cheap floors in A and C"; the stop rule's own S3 clause was evaluated on 3 of 5 candidate fields (F-05).

## 1. Severity key

high: changes how a verdict or a programme-level claim must be worded, or invalidates evidence used for it. medium: weakens a claim, or a protocol/bookkeeping defect that could bias results. low: hygiene, staleness, wording.

## 2. Re-derivation of headline numbers from raw result files

| Focus | Headline in FOCUS/notes | Source re-run | Reproduced | Note |
|---|---|---|---|---|
| A cross-fit midnight/TCGA | -0.028 [-0.063, +0.001], MDE80 0.048 | `aggregate_p2.py` on `a_p3cv_midnight_tcga`, B=300 | -0.028 [-0.059, +0.000], MDE 0.046 | same sign, same size |
| A cross-fit midnight/Tolkach | +0.003 [-0.007, +0.012], MDE80 0.014 | same, `..._tolkach_esca` | +0.003 [-0.006, +0.012], MDE 0.013 | |
| A 3-seed fixed-C midnight/TCGA, Tolkach | -0.056 [-0.108, -0.010]; -0.002 [-0.011, +0.010] | `a_p2h_midnight_*_s0-2` | -0.056 [-0.104, -0.008]; -0.002 [-0.011, +0.010] | in TCGA the V-selected AE is `dl32 lam0`, i.e. NO adversary (F-05) |
| B' corr_det, 3 seeds | raw 0.815, SAM 0.812, adv 0.802; SAM-adv per seed +0.023, +0.009, -0.002; exact-mask headroom 0.048 | `analyze.py` on `b_m_corr_det_s*`, `b_msam_corr_det_s*`, `b_m_clean_s*` | identical (also clean 0.705, rate4 0.839, gtmask 0.764) | per-seed headroom is 0.050, 0.055, 0.039 |
| C AE round | selected d32 lam3 U_HE 0.326, dU_HE -0.208 [-0.240, -0.179]; trimmed -0.231 [-0.261, -0.201] | `ae_report.py`, `select_ae.py` on `~/work/runs` | identical to the printed digits | selection script picks d32 lam3 |
| C rate/adv d16, seeds 1-2 (now finished, not in notes) | notes: "queued, cannot change verdict" | ad hoc with `aggregate.py` helpers | rate d16 UAR 0.728, dUAR vs LDA-5 -0.015 [-0.025, -0.007]; adv d16 -0.015 [-0.024, -0.007], dID vs LDA-5 -0.020 [-0.037, -0.004]; adv minus rate ID -0.009 [-0.019, +0.001] | consistent with seed 0 |
| D headroom | H1 0.179 [0.170, 0.189]; H2 0.039 [-0.003, 0.077] at tau 0.70 | `aggregate.py` on `d_hr1/2_gtr_s0-2` | identical; at tau 0.90 headroom 0.155 [0.109, 0.198], at 0.97 0.416 [0.376, 0.451] | FOCUS D still shows the 2-seed 0.041 [0.001, 0.081] |
| D digits | H1 0.010 [0.002, 0.018] | records in `d_dig_check_gtr_s*` | raw 0.2147 vs const 0.2047 (n1); n1s tier raw 0.0408 vs const 0.0163 | see F-10 |
| E | K2 r_fine HLCA 0.77, Immune 0.97, Pancreas 0.91; K3 gap +0.116 [0.099, 0.136] | `aggregate.py` HLCA (B=60) and Pancreas (B=200); JSON in git for all three | HLCA r_fine 0.7675, gap 0.116 [0.099, 0.133]; Pancreas r_fine 0.909 | |
| F | R0 0.9007/0.4096; JK1 shuffled 0.231/0.224/0.213 vs noise16 0.276/0.266/0.255; JK3 dU -0.106 | `f_jump/s4.json`, `r0_s2.json`; `aggregate.py --scenario 5` (B=20) | identical; JK3 fires 3/3, JK2 fires 0/3 | |

Numbers I could not re-derive from raw files: JUMP0 metadata counts (81,735 of 82,276 compounds; no metadata kept); literature-scout scores; ranker survival probabilities (priors, not measurements).

## 3. Findings

### High

**F-01 (high). D's post-hoc "adv" arm never trained, so "the adversary adds nothing and costs utility" is not evidence.**
Evidence: `~/work/runs/d_ph_lrn_gtr_s{0,1}/records.jsonl`, key `train_log`. For all 24 adv fits (d 16..512, lambda 0.3 and 3, seeds 0 and 1) the retrieval loss is 7.913-7.915 at the last logged step and within 0.025 of that value from step 250 on; ln(2,739 docs outside C) = 7.9153. That is the uniform-softmax value, i.e. the retrieval objective did not train. At step 0 (d16-d128, lambda 0.3) the confusion term is 101-154 nats against a retrieval loss of 8.6-12.2, so even lambda 0.3 makes the adversarial term dominate. Dev nDCG: adv d16 0.017-0.048 (chance 0.012) vs bott_mlp d16 0.147-0.150; d512 0.244 vs 0.283-0.286 (bott_mlp retrieval loss reaches 4.47-4.59). B' correctly flagged its own collapsed arms as a failed implementation; D did not. Texts affected: D.md 5.3 and 6.5 ("adv adds nothing over the bottleneck and costs utility"), FOCUS D "post-hoc learned arms ... cannot flip", journal 12:35Z and 14:15Z programme reading (4 settings).
Changes a verdict? Not D's pre-registered gate result, but removes D's contribution to the programme-level claim "adversary adds nothing".
Amendment: mark the arm "failed implementation (collapsed)". Rerun `d_ph_lrn` with lambda in {1e-3, 1e-2, 0.1} (confusion divided by its initial value or by H(pi)), add a collapse check (reject if retrieval loss > ln N - 0.5), 2 seeds, about 5 GPU-min per config. Only then state anything about adversarial D arms.

**F-02 (high). A's adversarial training does not reach its own objective; attribution "adversary adds nothing beyond rate/noise" is unsupported for A.**
Evidence: `extra.adv_train_acc_per_round` in `a_p2h_midnight_*_s0`, `a_p2h_dinov2L_tcga_s0`, `a_p3cv_midnight_tcga` jsonl. The retrained-from-scratch in-sample adversary ends at 0.75-0.94 accuracy (chance 0.25 TCGA, 0.5 Tolkach) in every `ae` lam>0 and `adv_retrain` run, and in AE it often rises over rounds (midnight/Tolkach `ae dl32 lam1`: 0.69 -> 0.94; TCGA `ae dl32 lam10`: 0.62 -> 0.86). Grid maximum lam is 10 with 12 rounds x 60 encoder steps = 720 steps. Toy check (Appendix B1, A's own `train_ae`, 2 centres, a 4-dimensional linear nuisance of 6000 patches): held-out linear attacker stays at 0.995 (lam 1), 0.938 (lam 3), 0.716 (lam 10), 0.659 (lam 30), 0.559 (lam 100). So with the shipped lam range the adversary removes almost nothing from a trivially removable nuisance. In real data lam 0 -> 10 does lower the held-out worst-attacker accuracy (midnight/TCGA cross-fit fold 0: 0.59 -> 0.43, chance 0.25) but also raises the rate (71 -> 187 bits). C's adversary is effective on unseen actors (identity 34.8x -> 6.6x at lam 3) but its in-sample accuracy was not logged (`ainfo` is discarded in `c_cremad/run_exp.py` line 111).
Changes a verdict? No for the A and C AE-versus-floor nulls (lam 0 AEs and all 12 configs also fail). Yes for the explanatory sentences in the journal (13:00Z, 14:15Z), FOCUS A ("adversary added nothing") and FOCUS C ("the adversary adds nothing").
Amendment: strike the attribution or restate as "with lam <= 10 and 720 steps the adversary did not converge". If the question matters: lam in {30, 100, 300}, 3x more steps, report in-sample adversary accuracy and require it within 0.05 of chance as an "adapter trained" criterion; log `adv_acc_per_round` in C.

**F-03 (high). A's "DEAD" verdict covers held-class label utility only; the retention axis (criterion ii of the revised FOCUS A rule) shows a replicated positive in one of two cross-fit cells and was left out of the DEAD criteria.**
Evidence (leakage-matched table, `aggregate_p2.py` without `--compact`): midnight/Tolkach, V-selected AE `dl32 lam10`, retention minus best inductive floor at L <= L_AE+0.05: cross-fit **+0.095 [+0.072, +0.124] (MDE80 0.039)** vs `lda_pca__d8` (retention 0.514 at L 0.30; AE 0.609 at L 0.37), fixed-C 3 seeds **+0.146 [+0.113, +0.180] (MDE 0.048)** vs `pca__d4`. midnight/TCGA: fixed-C `dl32 lam1` +0.092 [+0.058, +0.127], `lam3` +0.078 [+0.037, +0.111]; cross-fit `lam1` +0.013 [-0.007, +0.033] (not a win). The same row has held-class utility Delta of +0.003 (Tolkach), so the AE keeps neighbourhood structure at equal leakage and equal label utility. A.md "VERDICT ... DEAD" quotes only utility and says retention is "not more than undithered floors (0.487 vs 0.40-0.47 on TCGA)", which is the TCGA cell only. RANKING_R2 section 7's AE-DEAD rules (a)-(c) were written after the ranker's [PEEK] at 10 cell-seeds (3 of the 6 midnight ones; RANKING_R2 section 0) and contain no retention clause. Check against a metric artefact: per-dimension standardisation of PCA floors in `knn_retention` lowers their retention by only 0.01-0.04 (Appendix B2), so the gap survives. Caveat: retention is measured against raw kNN structure, which an MSE-reconstructing AE optimises directly, so its value as "utility" is arguable.
Changes a verdict? The held-class utility verdict stands. The label "A lead DEAD" must become "dead on held-class label utility; mixed on kNN retention (positive in midnight/Tolkach cross-fit, null in midnight/TCGA cross-fit, dinov2-L cross-fit not run)".
Amendment: add a retention row to the AE verdict; if the user cares about non-label structure, run dinov2-L cross-fit and phikon-v2 holdout (about 1 GPU-h each, extraction cached) and the H2-015 selection null before closing.

**F-04 (high). F/JUMP: the NO-GO is a gate artefact plus a headroom rule that does not measure what stage 2 needed; learned supervised adapters were never tested and the evidence suggests large room.**
Evidence: (i) literal gate failure is the two-sided control clause |U(shuffled compounds, raw) - U(noise16)| <= 0.01: 0.231 vs 0.276, 0.224 vs 0.266, 0.213 vs 0.255 (`results/f_jump/s4.json`), which F.md D1 attributes to a construction flaw (a source-carrying embedding ranks same-source negatives above cross-source positives, so any raw-based control reads below noise16). All substantive clauses pass (raw attacker 0.885/0.874/0.885 vs 0.60; controls <= 0.224). (ii) S5 JK3 fires 3/3 because the 16-d compound-supervised LDA `ldaP_16` (U_H 0.182, L_H 0.39) beats every transductive reference (Harmony U_H 0.076, L_H 0.99; dU -0.106 [-0.107, -0.104]). Rule JK3 counts "floor >= reference" as "no headroom", but the registered stage-2 target (F.md 2.6) is the envelope that already contains `ldaP_16`; the rule therefore measures that published methods are weak, not that a learned map has no room. (iii) `ldaP_16` is itself a learned supervised map: U_H 0.182 vs raw 0.051 (3.6x; +0.136 [0.132, 0.140] query-weighted in F.md 5.3) on 40k unseen compounds, MDE80 0.001. A nonlinear (MLP/contrastive) compound-supervised adapter, which is the programme's own thesis, was not run. (iv) JK2 was not decidable: the grid's smallest d is 16 and `ldaP_16` has L_H 0.39 > 0.25; d <= 8 was not run (F.md 5.5).
Reproduced: R0, S4 JK1, S5 JK3/JK2 verdicts, MDE80 0.0010.
Changes a verdict? "NO-GO for JUMP" must be reworded to "NO-GO for a label-free adapter versus an envelope that includes supervised LDA, gate clause flawed; supervised learned adapters untested". This is the verdict I would reopen first.
Amendment: new pre-registration on the existing S5 outputs: (a) d in {2,4,8} LDA arms to settle JK2; (b) one supervised nonlinear adapter (MLP, compound-contrastive, trained on T' compounds only, same fit set) against `ldaP_16` at matched L, 3 splits; est. under 2 GPU-h. Replace the two-sided shuffle clause by an analytic or same-embedding null (shuffle on `ctrl_noise16`).

**F-05 (high, wording). The programme-level STOP and "four settings" claims outrun the evidence.**
Evidence: H2-019 S3 ("of the five candidates H2-009..H2-013 at most one passes its stage-1 check"). Only three were run (D-digits, SC-0, JUMP); EEG (H2-012, ranker pass prior 0.20) and astronomy (H2-013, 0.40) were never run (RANKING_R2 section 7 amendment 1; journal 15:05Z "parked"). Of the three that were run, D-digits failed H1 on a metric with a saturated prior (F-10), SC-0 failed K2 on a utility-only gate (F-06), JUMP failed on a flawed clause (F-04); the one gate that passed (JUMP metadata) was escalated. Journal 14:15Z "in four settings ... learned adapters did not beat cheap floors" counts D, whose learned arm collapsed (F-01), and B', whose only learned latent arm was bounded. Journal 13:11Z records the GPUs idle by design while cheap discriminating controls were not run (H2-015, dinov2-L cross-fit, phikon-v2 holdout, B corr_noisy/glued adv, D adv relaunch).
Amendment: state the stop as "programme 1 stopped for A and C (label-free adapter vs cheap floors, resolved), and for B', D, E, F as gate outcomes without a valid learned arm; EEG and astronomy untested". Spend the idle budget on the F-01/F-03/F-04 reruns before writing SUMMARY.md.

**F-06 (high). E: the NO-GO is decided by K2, a utility-only gate whose object does not hide the batch; the headroom question was left open by the builder himself.**
Evidence: `results/e_sc0/*_final.json`, reproduced: K2 r_fine 0.77/0.97/0.91 for `ldaC_pca16`, but `ldaC_pca16` has L 0.84 [0.78, 0.87] on HLCA (it hides little of the batch) and label-free `pca16` keeps the same share (0.78/0.98/0.93; E.md 5.2 point 2: "fires because 16 dimensions suffice"). K3 finds a real gap on HLCA: scVI U 0.816 at L 0.80 vs the V-selected inductive floor 0.700 (+0.116 [0.099, 0.136]); Harmony (U 0.668, L 0.37) has no floor within its leakage in the registered grid and +0.354 [0.334, 0.371] against the post-hoc dense grid. scVI is transductive (batch covariates, all cells, one seed, 82 epochs); an inductive map was never trained. E.md 5.6 says this explicitly; FOCUS E ("headroom is TRANSDUCTIVE ... does not need a task-agnostic learned adapter beyond what scVI/Harmony already do") reads as a finding, which it is not.
Amendment: FOCUS E wording "gate K2 fired; stage 2 not run; inductive map to scVI/Harmony point untested". A new pre-registration (E.md 5.6 already sketches it) is required, not a continuation.

### Medium

**F-07 (medium). Ceiling: A's margins were unattainable by construction in Tolkach and nearly so in TCGA (the B' headroom lesson not applied).**
Evidence (cross-fit, `aggregate_p2.py`): midnight/Tolkach raw held-class accuracy 0.998 [0.996, 0.999], best floor at equal leakage 0.969, AE 0.971; maximum possible gain about +0.03 against an AE-ALIVE margin of +0.04. midnight/TCGA raw 0.891, floor 0.839, AE 0.810: maximum +0.052. The "resolved null (MDE80 0.014)" is therefore partly a ceiling null. Point estimates are at or below zero anyway (+0.003, -0.028), so the verdict stands.
Amendment: report "headroom to raw" next to every Delta; restate Tolkach as "no gain, at most +0.03 available".

**F-08 (medium). Comparator hygiene in A: transductive floor in the "inductive" envelope, floors chosen on sealed C, V-selection rule changed, pooled chance inflated.**
(a) `FLOORS` in `aggregate_p2.py` includes `centre_whiten` (needs the centre id and the test split's own statistics; A.md calls it "a reference, not an encoder") and it is the best "inductive" floor in the 3-seed midnight/Tolkach contrast (-0.002 vs `centre_whiten`) and TCGA envelopes. (b) The best floor at L <= L_AE+0.05 is chosen by point estimate on sealed C (winner's curse against the AE; the code comment says so); L itself is resolved to +-0.11 (TCGA) / +-0.22-0.26 (Tolkach fixed-C) against the 0.05 tolerance. (c) A.md ROUND 2 states the V-selection was "simplified" (max V held-class accuracy among AE configs with V_L <= median V_L of the grid) instead of the pre-registered "V_L within 0.05 of the best floor's V_L"; V has about 7 Tolkach cases. (d) Pooled cross-fit Tolkach `ctrl_const` scores 0.60 (chance 0.50), so normalised L of near-chance codes is inflated by about +0.2 in the pooled cell (TCGA constant is 0.25, fine).
All four are conservative or neutral for the AE; the cross-fit result against inductive dither floors (+0.003) agrees with (a)-(b), and over all 12 AE configs no Delta is positive with CI above 0 (best +0.003 Tolkach, -0.015 TCGA), which also neutralises (c). Amendment: drop `centre_whiten` from the "inductive" label, compute pooled L per fold, state V size.

**F-09 (medium). One class partition and shared test units.**
`seen_held_classes` fixes seen = first half of the sorted class names (TCGA: breast+colon seen, lung adeno+squamous held out; Tolkach 3 of 6); the ranker's `--seen_swap` was not built. Fixed-C "3 seeds" share the same 12/48 sealed units (acknowledged); cross-fit fixes this for midnight only. dinov2-L (fixed-C MDE80 0.093 TCGA, 0.061 Tolkach), phikon (1 seed, TCGA MDE 0.14-0.23) and the gated top encoders (UNI, Virchow2, ...) are inconclusive or untested.

**F-10 (medium). D's digit check "fails H1" through a scale-blind threshold; D's headroom verdict hinges on one operating point.**
(a) n1 (primary) includes ubiquitous tokens ("1", "2", "3", "95"), so the frequency prior already scores 0.204 and the 0.05 absolute threshold cannot be met. On the descriptive rare-number tier n1s (3 <= df <= 30) raw 0.0408 vs prior 0.0163 (2.5x; ridge on V 0.055 vs 0.008), not tested with a CI and not part of pass/fail. The attackers were trained on 3,716 docs, no Vec2Text. Reading: measurement failure, not "no leakage". (b) H2 passed at tau 0.97 (headroom 0.416 [0.376, 0.451]) and tau 0.90 (0.155 [0.109, 0.198]) and failed only at the oracle's own retention 0.70 (0.039 [-0.003, 0.077]); the pre-registered fallback ("judge at the oracle's own retention") is legitimate but the oracle is a reference, not a bound (D.md says so). (c) FOCUS D still quotes the 2-seed 0.041 [0.001, 0.081]. Status: undecidable margin, not a resolved null.

**F-11 (medium). B': scope of the NO-GO.**
Bounded: SAM minus adv per seed +0.023, +0.009, -0.002 gives mean 0.010, SD 0.0125, so a 3-seed t interval is about [-0.02, +0.04], below the 0.05 margin. Not tested: learned masks (lmask/advmask collapsed to 1.00, correctly reported as failed implementation), `adv` on corr_glued and corr_noisy (the kinds with an independent residual; 9 jobs parked at 12:11Z although the pre-registered rule listed all correlated kinds; B.md argues from "no leakage into z" for adv on corr_glued, an extrapolation), `indep` SAM arm. corr_det is degenerate as a leakage test (nuisance state is a deterministic function of the action; reference leak 1.00). One fixed distractor realisation per split. The utility-only rule ignores that adv leaks 0.065 vs SAM 0.204 at equal utility. Exact-mask headroom over SAM is 0.050/0.055/0.039 per seed (mean 0.048), so "undecidable for mask methods" holds on the mean only.

**F-12 (medium). C claim (1) is not leakage-matched on the full-emotion protocol.**
FOCUS C: "no label-supervised learned encoder beats LDA-5". Re-derived: UAR is lower by 0.015-0.022 (significant) but identity is also lower: VQ 4.1x (UAR 0.725), rate d4 4.8x (0.721), adv d16 5.1x (0.728) vs LDA-5 7.0x (0.743); identity differences -0.020 [-0.037, -0.004] (adv d16, 3 seeds), -0.031 [-0.053, -0.008] (VQ, 1 seed). These points are Pareto-incomparable with LDA-5; a noise-dithered or truncated LDA at 4-5x was not run in that protocol. The decisive C negative is the held-out-emotion protocol, where matched comparators exist (LDA-4 k3 4.4x, k5 7.2x, k16 6.5x) and the AE loses by -0.044 to -0.132 at equal leakage and -0.208 at matched width. Also: probe family and worst attacker are chosen per run on the sealed test actors (`uhe`, `worst_id`), symmetric across arms but optimistic.

**F-13 (medium). Reproducibility: code drifted under running jobs; recorded commit is HEAD after the run.**
`worker.py` pulls the shared checkout before each job and writes `git rev-parse HEAD` after the run (so the commit in `job_result.json` is not the commit the job started on: `a_p2h_midnight_tcga_s0` started at 537ff3f per its `split.json` and records 2c5485c). The four cross-fit folds of `a_p3cv_midnight_tcga` start on four commits (db653cf, 0d0b054, fd689db, b5a6748; Tolkach 2b383a8, 4708734, fbe3600, d76125e); `a_p2h_midnight_tcga_s0/s1/s2` start at 537ff3f, ad3ff41, 2acc1d0; `c_ae_d16_l0` seeds record two commits. Mitigation I could verify from VM mtimes: `a_pathorob/{adapters,run_sweep,splits,seed_audit}.py` last changed at 13:15:12Z, before the cross-fit jobs started (13:16-13:18Z, finished 14:06 and 14:24Z), so the cross-fit runs that carry the A verdict saw stable A code; `suite.py` and `attackers.py` (11:06 and 11:01Z) were stable for all p2h runs. Not verifiable without git: `adapters.py` between the p2h seeds (earlier edits possible between 11:11Z and 13:15Z) and `c_cremad/evaluate.py` (changed 11:56:22Z, after the `c_lda4_*` comparators ran at about 11:35-11:48Z and before the AE jobs). Amendment: pin each job to a `git worktree` checked out at a commit recorded before launch; the main agent should run `git diff <first>..<last> -- research/fields/src/a_pathorob research/fields/src/c_cremad` for the commits above and record "no functional change".

**F-14 (medium). Unreported or selectively summarised results.**
A.md contains no row for `a_p2h_phikon_tcga_s0` (phikon is the encoder with the worst Robustness Index and the most centre headroom): V-selected AE -0.113 [-0.207, -0.017] vs `lda_pca__d32`, MDE80 0.14, no gain; also `a_lrn_midnight_*`, `a_p2x_dinov2L_camelyon_s0` jsonl exist without a write-up, and gate cell `a_gate_phikon_tcga_s0` (finished 11:01Z, before the 11:15Z tally of 7) is missing from G1; recomputed, its LEACE L_worst is 0.72 [0.62, 0.82], so G1 is 0 of 8 and the conclusion is unchanged. The ranker's best-of-10 AE minus floor values (+0.03, +0.03, +0.02 TCGA; +0.19 Tolkach s0 at L 0.08; +0.03; -0.02) show positive best-of-N cells, and the selection null H2-015 that would calibrate them was never run. A's seed-0 summary was selective (self-corrected 13:30Z, credit to A).

**F-15 (medium). Queue priority and parking decisions shaped the order, not the outcomes, but they favoured late-arriving focuses.**
Reserved priority 4 (`results/*/job_result.json`): C 22 jobs, D 10, E 6, F 3, A 0 (one priority-5), B 0 (one priority-5). A's gate cells (they decide whether A exists) were still starved at 09:41Z and 10:11Z behind B and C jobs and only 7 of 36 had run by 11:15Z (journal); fair-share round robin by filename prefix gives equal shares regardless of decision value. Parked: 28 `a_gate_*` seed 1-2 and phikon-v2 cells (G1 decided on 8 cells at seed 0; harmless because G1 can no longer be met), 9 `b_*` (see F-11, not harmless), `~/queue/failed` still lists 3 `b_sam_*` specs although their reruns finished rc=0 and uploaded (the first, killed attempt's `job_result.json` was overwritten). The charter's "judge subagent" step was announced (journal 10:26Z) and never run for B'.

### Low

**F-16 (low). Charter rule 1 vs 2 in practice.** HF `KuanKuanKuan/openlatent-fields` holds 149 run folders, every one with `job_result.json`; all queue `done` jobs except `smoke` are there (journal "146 of 146" is now 149). 32 C CPU-run folders (`c_lda*`, `c_lda4*`, `c_pros*`, `c_chan*`, `c_fixcpu*`) are in git (`result_s0.json`) but not on HF, contradicting C.md ("raw per-actor counts: HF `runs/c_*`"). Conversely A (jsonl, preds), B (res_*.json), D (records.jsonl) and E/F raw predictions are on HF/VM only; git holds `job_result.json` and aggregates, against rule 1 ("results JSON on GitHub"). A re-derivation therefore needs the HF repo.
**F-17 (low). Manual interventions.** `d_dig_extract_gtr` crashed at its meta line; the completion marker was written by hand after verification (D.md 6.4; the job stays rc=1 in `~/queue/failed`). Six C specs with an unquoted `: ` quarantined/requeued (08:55Z). Watchdog cron installed 16:55Z. Journal time labels were ahead of the VM clock three times (relabelled), and the 16:55Z statement that two independent auditors had been launched was false when written (corrected 17:40Z).
**F-18 (low). Stale or imprecise wording in FOCUS.md.** D block uses the 2-seed headroom and "post-hoc learned arms queued" (finished; see F-01); C block "seeds 1-2 of rate/adv d16 still queued" (finished, consistent); A block says the seed-0 envelope included "dithered codes" (dithers exist only in the cross-fit grid; seed-0 envelope = LDA family + whitening + PCA + random projection); A block "validation-selected AE" is `lam0` (no adversary) in midnight/TCGA; "~27 gate seeds parked" is 28; "G1 0/7" omits the 8th finished cell. The stray file `ubuntu@192.222.52.237` in the repo root is a copy of `watchdog.sh` (untracked; delete).
**F-19 (low). Bootstrap units.** Correct for A (cases/slides), C (91 actors, paired resamples), E (donors; scIB sets descriptive on 9-10 batches), F (plates for L; compounds for U is anti-conservative, F.md reports the plate-cluster check, about 2x wider). B' intervals condition on one distractor draw; D resamples docs (894) and queries (323) though NFCorpus relevance judgments overlap across queries. Seeds replicate training noise only in A, C, D, E, B'; only F's three salts change the held-out units.
**F-20 (low). F absolute AP.** U_V(T_harmony) 0.194 vs U_H 0.076 on the same split: AP depends on the number of same-plate negatives in the evaluated universe (8.2k V vs 40.4k H compounds). Selection within V and reporting within H are each internally consistent; absolute values are not comparable with published mAP (R0 is validated only on S2). R0 passes if any of three candidate statistics matches (statistic shopping, but `mean_average_precision` matched both columns to three decimals).

## 4. Post-hoc change ledger (direction relative to a learned-adapter win)

C = conservative (raises the bar or shrinks the chance of a learned win), P = permissive, N = neutral/unknown.

| # | Change | When | Direction |
|---|---|---|---|
| 1 | A kill rule LEACE -> label-supervised LDA, held-class utility or retention | 11:15Z, after 7 of 36 gate cells (seed 0) | C |
| 2 | A seed-0 "leakage-matched gain" vs LDA family only | ~11:50Z | P, self-corrected 13:30Z |
| 3 | Floor chosen by point estimate on sealed C | aggregate_p2 | C |
| 4 | Floor envelope extended: per-centre whitening (transductive), PCA, randproj, later dither | 13:15Z after [PEEK] | C |
| 5 | AE-DEAD/ALIVE margins ("either" cell, +0.02/+0.04) set after peeking at 10 cell-seeds (3 of 6 midnight); no retention clause | 13:15Z | C |
| 6 | V-selection "simplified" | A.md ROUND 2 | N |
| 7 | 28 gate seeds parked; G1 closed at 7-8 cells | 12:50Z | N |
| 8 | B' weak -> strong distractor level after regime check | 08:50Z | P |
| 9 | B' lmask/advmask dropped after collapse; extras labelled post hoc | 10:05Z | N |
| 10 | B' decided before corr_noisy/glued; 9 jobs parked | 11:30Z / 12:11Z | C |
| 11 | C held-out-emotion protocol added after full-emotion negatives (pre-registered before any c_he number) | 11:35Z | P |
| 12 | C task-agnostic AE round granted after A's lead (pre-registered, validation-selected) | 11:58Z | P |
| 13 | C dither control added after reading the AE grid (1 seed) | after 12:45Z | N |
| 14 | D strongest floor must reach tau in every seed | after the 3-seed table | C for the "3% gives no privacy" sentence, N for H2 |
| 15 | D H2 judged at oracle's own retention (pre-registered fallback) | v0 text | C (removes the passing readings at 0.97/0.90) |
| 16 | D post-hoc adv arm with lambda 0.3/3 only | 12:30Z | spurious (collapse, F-01) |
| 17 | E interpretations I1-I8, amendments A1-A3 (before the first real number); K3 reference-rule sensitivities after seeing T_centre | 13:37-13:59Z | N |
| 18 | F tiers I1, interpretations I1-I9 (pre-registered); A2 relaxation ruled a deviation, literal NO-GO kept | 15:36Z / 16:26Z | C (kept the flawed clause) |
| 19 | JUMP metadata PASS escalated to a floors-only stage 1 | 15:40Z | P |
| 20 | Stop rule S3 evaluated on 3 of 5 candidates | 16:26Z | C |

Almost every post-hoc step raised the bar for the learned arm. That is defensible for a negative result, and it is why the A and C head-to-head nulls are credible. It is also why gate-type NO-GOs (D, E, F) need the wording changes above: each also stopped further tests of a learned arm.

Multiple comparisons: the programme tests many arms but its claims are nulls; multiplicity inflates false positives, not these negatives. The one place it matters is the original AE lead (best of 10 configs x 4 cells on seed-0 test numbers), which the replication correctly dissolved. The H2-015 selection null was not run, so selection inflation (+0.03 typical, +0.19 worst per the ranker) is unquantified.

## 5. Strength of each negative

Resolved nulls (enough power to exclude the stated margin):
- Closed-form linear erasure (LEACE/ISP) fitted on seen units leaves most of the source signal on unseen units: A gate L_worst 0.63-0.89 (0 of 7 cells below 0.15, one seed, CIs exclude 0.15); C 53x chance on unseen actors; D weakest floor; E L 0.95/0.90/0.98. Replicated in four fields.
- C: validation-selected task-agnostic adversarial AE vs width-matched and leakage-matched LDA-4 (3 seeds x 4 folds x 91 actors): dU_HE -0.208 [-0.240, -0.179] (matched width), -0.044 to -0.132 (matched leakage).
- A: task-agnostic AE (all 12 configs) does not beat the inductive-floor envelope on held-class label utility in midnight/TCGA and midnight/Tolkach, cross-fit (-0.028 [-0.059, +0.000], +0.003 [-0.006, +0.012]); ceiling-limited (F-07); retention is a separate, mixed result (F-03).
- F: the benchmark's published transductive methods are dominated on both axes by a 16-d compound-supervised LDA (dU -0.106, MDE80 0.001, 3 splits); E: per-batch centring/whitening raises nonlinear leakage (L 1.54 HLCA; independent random-forest check; reproduces on JUMP at L 0.99-1.00).
- C by-products: 300 ms of non-speech identify actors at 56x chance; LDA is the strongest identity attacker.

Bounded (an upper limit on the effect, below the margin but not tight, or a single arm):
- B' `adv` vs zero-shot SAM on corr_det: gain <= about +0.04 (3 seeds); exact-mask headroom 0.048.
- A label-supervised families (vib, vq, dann, adv_retrain, supcon_xc) vs matched LDA: below it in all four fixed-C cells, MDE80 0.05-0.14.
- C label-supervised encoders vs LDA-5: UAR lower by 0.015-0.022, identity not matched (F-12).

Inconclusive or not tested:
- D (all of it): margin undecidable at one point, adversarial arm collapsed (F-01), digit attackers too weak (F-10).
- E for any learned arm: none run (F-06). F for any learned arm: none run (F-04).
- B' learned masks; `adv` on corr_glued/corr_noisy.
- A on dinov2-L/TCGA (MDE80 0.07-0.13), phikon (1 seed), phikon-v2, gated top encoders, other class partitions, retention on TCGA beyond midnight, adversary attribution (F-02).
- EEG and astronomy.

Does any verdict depend on one weak instrument? The shared instrument is "max over a small attacker suite, normalised L, compared with the envelope of cheap floors at matched L". Its known weaknesses: L resolved to +-0.1 to +-0.26 against a 0.05 tolerance in A; pooled chance inflation (F-08d); attackers trained on a few units (A), 3,716 docs (D); adversaries trained against two attacker families but scored against up to six. This does not threaten the A and C head-to-head nulls (the same instrument scores the learned arm and the floors, and the cross-fit and 12-config checks hold). It does threaten the NO-GOs that came from single-metric gates (E K2, F JK1 clause and JK3, D oracle headroom and digits H1): those could be false NO-GOs for learned adapters.

## 6. Scoped claims table for a future SUMMARY.md

| Focus | What was measured | Number (interval) | Strength | Limits |
|---|---|---|---|---|
| A, gate | Retrained-attacker centre leakage after closed-form erasure, PathoROB, 7 cells | LEACE L_worst 0.63-0.89, every CI excludes 0.15 (one seed) | Resolved | 8 of 36 planned cells; seed 0 only |
| A, floor | Label-supervised LDA (C-1 dims) vs raw | label acc 0.983 vs 0.986 (midnight/Camelyon), 0.696 vs 0.702 (dinov2-L/TCGA) at L 0.00-0.20 | Resolved in the gate cells | in-distribution only; 2-4 classes |
| A, learned adapter | Task-agnostic AE (12 configs) minus cheap-floor envelope, held-class accuracy, midnight, cross-fit | TCGA -0.028 [-0.059, +0.000] (MDE80 0.046); Tolkach +0.003 [-0.006, +0.012] (MDE80 0.013) | Resolved null, ceiling-limited | one class partition; headroom to raw 0.05 / 0.03; midnight only; adversary not converged (F-02) |
| A, retention | AE minus best floor at equal L, kNN retention | Tolkach cross-fit +0.095 [+0.072, +0.124]; TCGA cross-fit +0.013 [-0.007, +0.033] | Mixed, unresolved | metric partly the AE's own objective; dinov2-L not cross-fitted |
| A, supervised families | vib, vq, dann, adv_retrain, supcon_xc vs matched LDA | below LDA in 4 of 4 fixed-C cells | Bounded | TCGA cells underpowered; 3 seeds share test units |
| B' | Learned latent (adv) minus zero-shot SAM, NMSE, corr_det | +0.010 (per seed +0.023, +0.009, -0.002); t-bound about +0.04 | Bounded | synthetic sprites, one distractor draw, 64x64 LAM; learned masks collapsed; glued/noisy adv untested |
| C, identity floors | Non-speech-only actor identification | 0.618 [0.579, 0.656] = 56x chance | Resolved | CREMA-D |
| C, label-supervised | Rate/adv/VQ encoders vs LDA-5, UAR and identity | UAR -0.015 [-0.025, -0.007]; identity 4.1-5.1x vs 7.0x | Bounded, not leakage-matched | WavLM-Base+ only |
| C, learned | Task-agnostic adversarial AE vs width- and leakage-matched LDA-4, held-out emotions | -0.208 [-0.240, -0.179]; -0.044..-0.132 | Resolved (for this AE, lam 3/10) | adversary effect on unseen actors is real but not shown to beat noise (one-seed dither control) |
| D | Retrieval-vector privacy, GTR-base x NFCorpus | H1 0.179 [0.170, 0.189]; H2 0.039 [-0.003, 0.077] at tau 0.70, 0.155 / 0.416 at 0.90 / 0.97 | Inconclusive | weak attackers; adv arm collapsed; digit secret not measurable |
| E | Single-cell batch integration, floors only | K2 r_fine 0.77 [0.75, 0.79] HLCA; scVI vs inductive floor +0.116 [0.099, 0.136] | Gate fired; learned adapters untested | one encoder, scVI one seed |
| F | JUMP S5, floors only | 16-d compound LDA U 0.182 vs Harmony 0.076, dU -0.106 [-0.107, -0.104]; MDE80 0.001 | Resolved for published methods; learned adapters untested | 5 sources; front end transductive; d < 16 untested |
| Cross-field | Per-source normalisation vs retrained nonlinear attackers | E L 1.54; F L 0.99-1.00 | Resolved (two fields) | PathoROB differs (centre_whiten was a strong floor there) |

## 7. Amendments, in order of value per cost

1. Reword FOCUS.md D, E, F and the programme-level STOP (F-04, F-05, F-06, F-10, F-11); strike "adversary adds nothing" (F-01, F-02). Zero compute.
2. Rerun D adversarial arm with lambda in {1e-3, 1e-2, 0.1}, collapse check, 2 seeds (minutes).
3. Add the retention row to A's verdict; run H2-015 and dinov2-L cross-fit (about 1-2 GPU-h) only if the retention question is wanted.
4. JUMP follow-up on existing S5 outputs: LDA d in {2, 4, 8}; one supervised nonlinear adapter vs `ldaP_16` (under 2 GPU-h).
5. Pin jobs to a worktree at a recorded commit; record commit before launch; reconcile `~/queue/failed`.
6. Commit raw A/B/D result JSON (or a pointer file with HF revision hashes) to git; move C's CPU runs to HF.
7. Delete `ubuntu@192.222.52.237`; correct stale numbers (F-18).

## Appendix A. Reproduction commands (VM, CPU)

`~/venvs/a/bin/python research/fields/src/a_pathorob/aggregate_p2.py --runs ~/work/runs/a_p3cv_midnight_tcga ~/work/runs/a_p3cv_midnight_tolkach_esca --B 300` (drop `--compact` to print the leakage-matched table with retention); same with the twelve `a_p2h_*` dirs; `~/venvs/b/bin/python research/fields/src/b_robotics/analyze.py ~/work/runs/b_m_corr_det_s{0,1,2} ~/work/runs/b_msam_corr_det_s{0,1,2} ~/work/runs/b_m_clean_s{0,1,2}`; `cd research/fields/src/c_cremad && python ae_report.py ~/work/runs && python select_ae.py ~/work/runs c_ae_`; `python research/fields/src/d_embedpriv/aggregate.py ~/work/runs/d_hr{1,2}_gtr_s{0,1,2}`; `bash research/fields/src/e_singlecell/run_final.sh pancreas 200`; `python research/fields/src/f_jump/aggregate.py --scenario 5 --dirs ~/work/runs/f_s5_eval_s0,..._s1,..._s2 --B 20 --out /tmp/x`.

## Appendix B. Sanity scripts (VM `/tmp`, not committed)

B1, adversary power (`/tmp/ae_adv_toy.py`): `from a_pathorob import adapters as AD`; 6000 patches, 64 dims, centre c in {0,1} adds +-2.0 to dims 0-3, 4 classes add 1.5 to dims 4-7; `AD.train_ae(P, sp, dl=8, lam, beta=1e-3, seed=0)` with A = 3000 patches; attacker = logistic regression trained on B (1500) and tested on C (1000). Output: lam 0 -> 0.996; 1 -> 0.995; 3 -> 0.938; 10 -> 0.716; 30 -> 0.659; 100 -> 0.559 (in-sample adversary 0.55 first round, 0.60-0.98 last three).

B2, retention metric (`/tmp/ret_check.py`): midnight/Tolkach cross-fit fold 0, PCA-d fitted on A, retention xc with `suite.knn_retention` (per-dimension standardised) vs a variant keeping PC variances (one global scale): d=4 0.402 vs 0.427; d=8 0.474 vs 0.515; d=16 0.564 vs 0.586; d=32 0.656 vs 0.670.
