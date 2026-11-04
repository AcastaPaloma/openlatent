# Contrarian stress test of the provisional shortlist (A, B, C)

Date 2026-11-04. Author: stress-test scout. Goal: try to break A (source/batch/centre-invariant embeddings), B (robotics distractor/embodiment-free latent actions) and C (identity-free expression / speech emotion on CREMA-D) before GPU-days are spent.

Evidence labels used below: **[opened]** I read the page/paper text; **[api]** I queried the host's API or file listing from the VM or laptop; **[probe]** I fetched bytes from the VM; **[search]** only a search snippet (not relied on); **[judgement]** my inference. "Not found" means my searches did not find it, not that it does not exist.

Budget used: VM downloads were two CREMA-D sample files (338 KB) and two 50 MB range reads that were deleted; no GPU job, nothing under ~/queue, the VM's HF token file was not read or used. Local: one 36 MB PDF in the session scratchpad (unreadable here, no pdftotext).

---

## 0. Verdicts and lock-in recommendation

| Field | Verdict | One-line reason |
|---|---|---|
| A. Source/centre-invariant embeddings | **GO-WITH-CHANGES** | PathoROB is runnable end to end for about 1.9 GB and about 2 GPU-hours, but RI is saturating (top 0.89-0.93), so only a retrained-centre-probe / unseen-centre protocol is a contribution. Drop BatchBench from the loop (sealed, server-scored, manual, 30-257 GB). |
| B. Robotics distractor/embodiment-free latent actions | **NO-GO as stated** (conditional re-entry only) | Probe metric is already at the floor with free masks (MaskLAM on Distracting Meta-World: NMSE 0.090 with SAM masks versus 0.142 for the label-supervised reference), at least 8 competing methods in 12 months, no reference code released, 49 GB per task. |
| C. Identity-free expression on CREMA-D (+RAVDESS) | **GO-WITH-CHANGES** | Data is 2.5 GB, ODbL, reachable from the VM, harness reuses the identity work. The two prior papers (WACV 2025, ICME 2026) use frozen/pretrained validators and shared actors, which is exactly the gap. But we must build the benchmark ourselves; the published pipeline cannot be replicated (needs gated DFEW, no checkpoints, no licence). |

**Recommended lock-in**
- **Exactly 2: A (PathoROB only) + C (CREMA-D, RAVDESS as transfer).** Both run in days on one GPU from ungated, VM-reachable data; both have a documented protocol flaw that is our contribution; they are complementary (frozen-embedding site invariance versus video/audio identity removal) so one harness core plus two adapters covers both.
- **Exactly 3: A + C + B reframed**, and only after B passes a one-day kill test (section 2.6). The reframing: action-correlated distractors and a retrained distractor probe, not "better probe MSE on DAVIS backgrounds". If the kill test fails, the third slot should go to the next-ranked field from the scouts (egocentric, gaze, EEG); I did not stress-test those.
- First benchmark to replicate: A -> PathoROB (about 2 GPU-h); C -> our own CREMA-D harness re-measuring the WACV baselines (blur, optical flow) under retrained attackers (about 8 GPU-h); B -> Distracting Meta-World, one task, LAPO vs GT-mask LAM (about 15-25 GPU-h, about 49 GB).

---

## 1. Field A: source / batch / centre invariance

### 1.1 Saturated or scooped? (Nov 2026)

**PathoROB (pathology).** Live, crowded, partly saturated.
- Published in Nature Communications 17:5218 (2026); repo last pushed 2026-10-07 with external leaderboard rows (RudolfV 2, Atlas 2) [api][opened].
- Current RI leaderboard (README): Atlas 2 0.928 (TCGA 2x2 0.879 / Camelyon 0.940 / Tolkach 0.964), RudolfV 2-S 0.917, Virchow2 0.861, UNI2-h 0.757, Phikon-v2 0.469 (0.619 / 0.019 / 0.768), Phikon 0.476, Kang-DINO 0.512. Atlas 2 and RudolfV rows are marked as not validated by the authors [opened].
- Newer Aug-Oct 2026 work, all of it pulling the same direction: HERO (arXiv 2609.35943 v2, 1 Oct 2026, ViT-G, PathoROB RI 0.892, TCGA 2x2 0.884, Camelyon 0.836; robustness from data curation and recipe, not a post-hoc head) [opened]; Filiot et al. fine-tuning recipe (2607.22861, Jul 2026): average RI 0.72 -> 0.87 on 10 models, weights on HF (wearewaiv/phaet, gated manual) [opened]; CRoMa (2607.25497 v4, 21 Aug 2026) argues the Robustness Index is unreliable for cross-model comparison and proposes a per-sample margin [opened]; "Do Center Biases Propagate?" (2609.28231, 23 Sep 2026): ComBat gives no consistent slide-level gain, new AUCC metric, no code [opened].
- APD is saturated for good models: best in-domain about -0.83% relative (CONCHv1.5), best out-of-domain about 0% (Atlas); only weak models reach -7.8% / -9.4% [opened, PathoROB HTML].
- Headroom that survives: (i) linear centre probes on frozen embeddings reach 88-98% accuracy across FMs (PathoROB Fig 1d), i.e. a retrained-probe protocol has large room; (ii) weak or older FMs (Phikon family) have RI near 0.02-0.6 on Camelyon; (iii) unseen-centre generalisation. Headroom that does not: RI of top models (ceiling 1.0, best 0.93), APD of top models.
- Not found: any paper applying LEACE/linear erasure, a rate bottleneck, or a retrained nonlinear centre probe to PathoROB [search]. That is our gap, but it also means there is no published number to validate those baselines against.

**BatchBench / Cell Painting.** Open but unusable as a loop metric.
- BatchBench (arXiv 2610.30104, 21 Oct 2026, Topf, Sanchez-Fernandez, Klambauer) is two weeks old with 7 baseline entries all from the authors: ResNet18 0.175, DANN 0.170, deepCORAL 0.201, AdaBN 0.244, TENT 0.249, CS-ARM-BN 0.256, ARM-BN 0.260 overall accuracy (3 seeds); TARGET and REST subsets stay at 0.003-0.035 and 0.001-0.007 [opened]. Not saturated, wide headroom.
- But: test sources 4, 6, 9, 13 are never released; scoring is server-side in a Docker container with no network, "scores are published after manual evaluation"; the results/requests datasets are private [opened][api]. There is no local deterministic metric.
- The scouts' pattern repeats here: training-free adaptation (AdaBN +7 points over ResNet18) beats adversarial removal (DANN -0.5 points).
- Other recent Cell Painting items: CP-BG-Bench (2609.12761, 11 Sep 2026) shows four standard protocols rank the same encoders differently [opened]; BALANS (2601.22331) and CellPainTR (2509.06986) exist [search]. Arevalo et al. (Nat Commun 2024) is the local scripted suite, but I could not locate its code repo (three guessed URLs returned 404), so treat it as unverified.

**Single-cell.** Mature and noisy: 2026 benchmarks of single-cell foundation models disagree with each other on batch integration (Kedzierska zero-shot benchmark Aug 2026 says no consistent win over scVI; CellBench-LS says FMs win; Advanced Science 2026 says no win) [search]. The scouts' own evidence: scIB-E finds gains within one s.d.; batch is often confounded with biology. No privacy framing, no retrained-attacker evaluation found.

### 1.2 Is the benchmark deterministic, downloadable, licensed?

PathoROB [opened][api][probe]:

| Item | Value |
|---|---|
| Code | https://github.com/bifold-pathomics/PathoROB, BSD-3-Clause, `pip install -e .` (python 3.10), `random.seed(1000)` then sampled per-repetition seeds in `apd.py`; `tolkach_splits.json` and `data/metadata` ship in the repo; 20 repetitions for APD, 50 k-means inits for clustering |
| Data (3 HF datasets, all ungated) | `bifold-pathomics/PathoROB-camelyon` 0.466 GB (CC0 1.0); `PathoROB-tcga` 1.121 GB in 3 parquet shards (CC-BY-NC-SA 4.0); `PathoROB-tolkach_esca` 0.316 GB (CC-BY-SA 4.0). Total 1.90 GB. Note: the dataset id `bifold-pathomics/pathorob` (used in the scout file) returns 401; use the three ids above |
| VM reachability | huggingface.co 200; 50 MB ranged read of the Camelyon shard ran at 28 MB/s |
| Models, ungated | `owkin/phikon`, `owkin/phikon-v2` (licence "other"), `kaiko-ai/midnight` (MIT) |
| Models, click-through gated (needs the account behind the HF token to accept terms) | `MahmoodLab/UNI`, `UNI2-h`, `CONCH` (cc-by-nc-nd-4.0), `paige-ai/Virchow2` (cc-by-nc-nd-4.0), `bioptimus/H-optimus-0` (apache-2.0), `prov-gigapath/prov-gigapath` (apache-2.0), `histai/hibou-L` |
| Models, manual approval | `bioptimus/H0-mini`, `H-optimus-1`, `wearewaiv/phaet` |
| Determinism | deterministic given frozen features and the shipped seeds; features must be extracted per model (no precomputed embeddings released); splits fixed. RI and clustering are computed on the benchmark's own patches |
| Offline | yes after one download of data plus weights |
| Caveats | `ND` licences: do not upload embeddings of UNI2-h/Virchow2 to HF (rule 2 applies to our mask weights only). TCGA is NC-SA, so derived artefacts must carry a compatible licence. Independent unit is the slide/case, not the patch: bootstrap over slides |

BatchBench [opened][api]:
- HF `ai-lab-jku/jumpcp-challenge-datasets`: ungated, no licence field, no dataset card; `2k/500x500` 119.0 GB images, `negcons/500x500` 88.1 GB, `2k/200x200` 29.9 GB, `negcons/200x200` 14.1 GB, `mnist/200x200` 5.8 GB (total about 257 GB). Baselines: https://github.com/ml-jku/batchbench-baselines (Docker submission). Leaderboard Space: https://huggingface.co/spaces/ai-lab-jku/jumpcp-challenge. Upstream Cell Painting Gallery cpg0016 S3 reachable from the VM (200).
- Not deterministic for us: sealed test, manual evaluation. Only a leave-source-out proxy on the 9 public sources is possible locally, and it would not be comparable to the leaderboard.

Single-cell: CELLxGENE curation API reachable from the VM (200); I did not download any dataset, so size and licence per dataset are unverified.

### 1.3 Convincing and falsifying results; published baselines to verify against

Convincing (PathoROB, at least 3 FMs spanning the RI range, e.g. Phikon-v2, UNI2-h, one top model):
- Mask/projection fitted on training centres only; evaluated on held-out slides from unseen centres.
- Retrained centre probes (linear and a nonlinear MLP, retrained on the transformed training embeddings) drop to chance-normalised leakage <= 0.2, while biology accuracy on unseen centres stays within 1 point of raw.
- APD and RI at least as good as Reinhard + ComBat, beating LEACE, a rate bottleneck and a task-label-only projection with non-overlapping slide-level bootstrap CIs.

Falsifying:
- LEACE or ComBat + Reinhard ties the learned method inside the CI (the EEG audit already shows linear erasure takes a linear subject probe to chance), or
- a retrained MLP probe recovers more than half of the leakage after our mask (linear-only guarantees), or
- the gain is only on RI of weak models.

Published numbers usable as replication anchors [opened]: raw RI per model and dataset (leaderboard above); Reinhard +16.2% (preprint) / +19.1% (Nat Commun version) average RI; ComBat +27.4%; Reinhard + ComBat up to 0.92 (Virchow2, Atlas, UNI2-h); downstream APD gains Reinhard +1.11 points, DANN +0.23 points (12 of 20 models improved), Reinhard + DANN +1.30; Phikon-v2 Camelyon clustering score -0.99 -> +0.61 with ComBat; Filiot fine-tuning 0.72 -> 0.87 average RI; centre linear-probe accuracy 88-98%. Replication check: reproduce the leaderboard RI for Phikon, Phikon-v2 and UNI2-h within about 0.01-0.02 per dataset before anything else.
Task-label-only, LEACE, rate bottleneck: no published PathoROB numbers found; they are ours to produce and cannot be cross-checked against literature.

### 1.4 Most likely way to waste two weeks
Sweeping 20 foundation models (gated approvals, 99k patches each) and optimising RI/clustering in-sample, then landing at "ties Reinhard + ComBat (0.92) at a 0.93 ceiling" because (a) RI is a kNN statistic now disputed by CRoMa, (b) the fit and the score use the same patches, and (c) ComBat/Reinhard are the strong, cheap, published floor. Guard: pick 3 models, fit on training centres, report only held-out-centre retrained probes and APD.

### 1.5 Synthetic mechanism check (CPU, numpy, seconds; mechanism only, not evidence about real FMs)
I expected a task-label-only supervised projection to inflate RI without removing the centre signal. Script: `/private/tmp/claude-501/-Users-kuanw-Me-code-openlatent/7467a5d1-83aa-4aea-a58e-08e14af19e26/scratchpad/ri_gaming_sim.py` (6 classes, 4 centres, d=64, 5 seeds, test decorrelated). Result, honestly: **the gaming hypothesis was not confirmed.** In the main setting (centre offsets half-shared with class directions, train correlation 0.5) raw RI 0.655, centre-mean standardisation 0.665, linear centre erasure 0.582, class-LDA 0.629. Findings that did hold:
- RI and a retrained probe can rank methods differently: LDA beats linear erasure on RI (0.629 vs 0.582) while its retrained centre accuracy stays at 0.49 against 0.25 chance, versus 0.25 for linear erasure. In the other two settings the two metrics agreed, so this is a weak, setting-dependent effect.
- Linear erasure fools a linear probe (0.25) but a retrained nonlinear probe still gets 0.55 (2.2x chance); centre-mean standardisation was the only transform that also brought the nonlinear probe near chance (0.27-0.33). Same lesson as the EEG audit: linear-only guarantees need a nonlinear retrained attacker next to them.
- With high train-time correlation (0.8) class-LDA RI collapses to 0.43: label-only projection picks up the shortcut.

---

## 2. Field B: robotics latent actions free of distractor / embodiment

### 2.1 Saturated or scooped? (Nov 2026)

Crowded, and saturated on the metric that is cheap to compute.
- Methods published in the last 12 months aimed at exactly this: Object-centric LAPO (2502.09680, AAAI-26 oral; "mitigates distractor effects by 50%"), LAOF optical-flow constraint (2511.16407), VLM-targeted LAM (2601.22714, up to 6x success on Distracting MetaWorld), MaskLAM / Segment to Focus (2602.02259), Why Latent Actions Fail (2605.20223, v3 14 Oct 2026, theory on exogenous state), OTF-LAM (2606.30544, v2 31 Jul 2026), Motion-Focused LAM (2606.18955), Dueling World Models (2608.06706, training-free), CLAW adversarial latent regularisation (2606.04130), DLAM (2607.27138), Latent Policy Steering for cross-embodiment (2609.22521, Sep 2026, Robomimic + real) [opened]. Two searches found no new distractor-LAM paper dated Oct-Nov 2026 (absence of evidence only).
- Current best numbers on Distracting Meta-World (MaskLAM table, normalised linear-probe NMSE, lower is better; NSR = normalised success rate) [opened, project page]:

| Method | DMW NMSE | DMW NSR | DCS NMSE | DCS return (NR) | DROID NMSE |
|---|---|---|---|---|---|
| LAPO | 0.309 +- .009 | 0.636 +- .015 | 0.511 | 0.076 | 0.848 |
| LAOM-Labels (ground-truth actions) | 0.142 +- .008 | 0.781 +- .033 | 0.216 | 0.566 | 0.882 |
| MaskLAM@SAM | 0.090 +- .005 | 0.832 +- .031 | 0.233 | 0.358 | 0.798 |
| MaskLAM@GT mask | 0.088 +- .004 | 0.843 +- .021 | 0.206 | 0.378 | n/a |

- Reading: on DMW a zero-shot SAM mask already beats the label-supervised reference on both probe error and success, and a ground-truth mask is only 0.002 better. A learned mask cannot beat the free baseline there; the best case is a tie. About 2k action labels are enough for near-peak DMW success (same page).
- Where headroom remains [judgement]: DCS return (0.36-0.38 vs 0.57 with labels); real video (DROID probe NMSE 0.80-0.88 is close to uninformative, but there is no ground truth for what the nuisance is and no closed-loop metric); action-correlated distractors, where the authors of Dueling World Models state their method fails and UniVLA is reported not to help; and the nuisance-side measurement: no paper reports a retrained probe that decodes the distractor (DAVIS clip/frame) from z.
- Also: the Distracting Meta-World distractors are DAVIS videos that are independent of the actions (the dataset card says "easy" dynamic setting), so the hardest known failure mode is not in the dataset.

### 2.2 Determinism, download, licences, VM

- Dataset (third party, "independently generated, not an official Meta-World, DAVIS, or Meta release") [opened][api]: https://huggingface.co/datasets/EpicPinkPenguin/visual_distracting_metaworld. Ungated, **no licence field**. 50 MT50-v3 tasks, 1,000,000 train + 100,000 test transitions per task, 128x128, 4-D actions, 39-D states, columns `observation`, `observation_distracted`, `mask` (GT arm), `pred_mask`, `pred_mask_distracted` (SAM 2.1), scripted-expert trajectories with success 1.0, DAVIS-2017 (90 videos). About 49 GB per task, **2,463 GB total**; an MT10 subset would be about 490 GB. Twin renders (clean vs distracted, same state) are in the data, which gives the exact invariance ground truth. DAVIS terms not verified; davischallenge.org and the ETH zip mirror return 200 from the VM.
- Sibling: `EpicPinkPenguin/visual_distracting_control_suite` 875 GB; `droid_dataset_segmentation_mask` 18.6 GB mask-only sidecar.
- VM: huggingface.co 200, 50 MB ranged read at 32 MB/s (about 26 minutes per task on one stream). Disk free 967 GB.
- Code: MaskLAM repo "coming soon" (only the project-page repo exists); VLM-LAM (2601.22714) lists no code; the Object-Centric LAPO repo named on its arXiv page (github.com/dunnolab/object-centric-lapo) returns **404**; the only released relevant code is `dunnolab/laom` (Apache-2.0, last updated Jul 2025, Distracting Control Suite only, datasets about 60 GB each, about 1.1 TB total on S3). So a replication means reimplementing LAPO/LAOM/MaskLAM on the EpicPinkPenguin data.
- Deterministic: probe-based metrics are deterministic given the fixed parquet splits and seeds. Closed-loop success needs MuJoCo + Meta-World v3 and is not bitwise deterministic (nobody claims it is).
- Metric-validity caveat [judgement]: data come from scripted experts, so actions are a near-deterministic function of state and goal; a linear probe of action from z can be low partly because z carries state. Nikulin et al. themselves say the probe does not show whether exogenous noise is also encoded.

### 2.3 Convincing and falsifying results; published baselines

Convincing (as reframed):
- A learned mask/latent with no SAM, no GT mask and no VLM matches MaskLAM@SAM within one SE on DMW NMSE (about 0.09) and NSR (about 0.83) over at least 5 MT10 tasks x 3 seeds, AND a freshly trained distractor probe (DAVIS clip id / frame index from z) is near chance, AND this holds on a held-out distractor type (procedural noise or an action-correlated distractor), where SAM-style masks are expected to fail.

Falsifying: GT-mask LAM ties or beats the learned method (already 0.088), or the retrained distractor probe shows LAPO's z does not leak distractor identity any more than the masked model's (then the leakage axis adds nothing).

Published baselines to verify against [opened]: the table above; LAPO near-zero success under DAVIS distractors and "Twin target" (clean next frame) restoring it (2601.22714); LAPO + best VLM probes nearly match LAPO without distractors; UniVLA no gain with distractors; object-centric LAPO halves the penalty; latent-dimension sweep in 2601.22714. Chance for NMSE is 1.0 by normalisation (my reading, check the paper's definition). Task-label-only = LAOM-Labels with about 2k labels (published). LEACE, rate bottleneck (only the latent-dim sweep), DANN: no published numbers on DMW.

### 2.4 Most likely way to waste two weeks
Re-implementing a LAPO/MaskLAM stack with no reference code and downloading hundreds of GB, then finding that free masks already put the probe at the floor, so the learned method can only tie. Secondary trap: calling a lower probe NMSE a win when the probe is not the same quantity as policy return (DCS shows NMSE parity with labels, 0.233 vs 0.216, but return 0.358 vs 0.566).

### 2.5 If kept at all
Only as an add-on focus with a restricted claim: action-correlated distractors plus retrained distractor probes (the unpublished axis).

### 2.6 One-day kill test for B (about 49 GB, 15-25 GPU-h)
One task, 3 seeds each of LAPO, LAOM-Labels (2k labels), MaskLAM@GT mask on the shipped data, plus a retrained distractor probe on each z. Proceed only if (a) LAPO z leaks the distractor measurably more than the masked z (probe accuracy gap with CI), and (b) the probe gap tracks policy success. Otherwise stop.

---

## 3. Field C: identity-free expression and speech emotion on CREMA-D (+RAVDESS)

### 3.1 Saturated or scooped? (Nov 2026)

Video side: partially scooped, with a hole exactly where we are strongest.
- WACV 2025 (Xu et al., arXiv 2412.00277) [opened, HTML]. CREMA-D: unprotected 87.2% accuracy / 100% identity (91-way); proposed 78.8% / 2.016% (chance 1.099%); Gaussian blur 1 20.97% / 2.15%; blur 2 77.78% / 88.98%; trade-off framework (SPAct) 71.83% / 33.02%; optical flow 16.31% / 1.03% (emotion destroyed); image swapping 64.96% / 3.67%. Recovery attack (U-Net retrained on protected frames) leaks 5.65% for the proposed method.
- The validator is a ResNet50 trained on raw CREMA-D train and frozen; it "is not retrained on protected outputs". Identity classifiers reach 100% on the test set, which implies the same 91 actors are in train and test; the paper does not say the split is actor-disjoint.
- ICME 2026 follow-up by the same group (arXiv 2603.21387, Mar 2026) [opened, HTML]: CREMA-D (30% held out) and RAVDESS used for transfer; accuracy on CREMA-D 80.9 / 82.1 (R(2+1)D / I3D) vs 89.2 / 88.3 unprotected; privacy preservation ratio P_pre 0.907 CREMA-D, 0.936 RAVDESS, with a pretrained ArcFace binary matcher whose training is not described; baselines blur P_pre 0.22, adversarial 0.64, face swap 0.59. No code link, no leakage in accuracy units.
- Code for WACV 2025 is public but unlicensed (https://github.com/fengxxu/ppfer), needs DFEW (application-gated) and ImageNet U-ViT pre-training, and ships no checkpoints or splits. The project page still says "under construction". So the published pipeline cannot be replicated.
- So the utility-privacy frontier on CREMA-D is claimed, but it is measured with frozen or pretrained validators and shared-actor splits. Nobody has measured it with retrained attackers on actor-disjoint splits.

Speech side: crowded, different corpus.
- VoicePrivacy 2026 (workshop 26 Sep 2026 at Interspeech): 13 teams; English track uses LibriSpeech + IEMOCAP (gated), UAR from emotion2vec, attackers "stronger, domain-aware" [opened]. One system description (PSST1): EER 8.53% at UAR 58.4% [search]. VPC 2024 reference points [from other_modalities.md, opened by that scout]: unprotected EER 4.59 / WER 1.85 / UAR 71.1; best overall 40.4 / 3.2 / 60.8.
- SISER (2609.02941, Sep 2026): speaker-invariant SER on IEMOCAP, test UA 60.63 vs 56.46 vanilla and 56.95 GRL; **no speaker-probe accuracy reported, speaker removal shown only by t-SNE**; internal inconsistencies in the tables [opened]. Code: github.com/slp-lab-research/siser.
- No CREMA-D speaker-removal paper found in Aug-Nov 2026 [search]. A z-level, retrained speaker-probe protocol on CREMA-D audio is not scooped, but it is not part of the VoicePrivacy benchmark either, so it is a benchmark we define.

Headroom for a retrained-attacker protocol: large on the identity side (the only published validators are frozen), unknown on the utility side because no actor-disjoint accuracy for a strong encoder is published (one 2026 Scientific Reports paper reports about 53% speaker-independent audio-visual with a hybrid fusion model [search]; human audio-visual recognition of the intended emotion is 63.6%, audio-only 40.9%, video-only 58.2% [search, Cao 2014 figures confirmed by a secondary page]). The utility axis may be label-noise-limited.

### 3.2 Determinism, download, licences, VM

CREMA-D [opened][api][probe]:
- Source: https://github.com/CheyneyComputerScience/CREMA-D (git-lfs) and mirror https://gitlab.com/cs-cooper-lab/crema-d-mirror (200 from the VM). Raw bytes work from the VM via `https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/<file>.wav` and `.../VideoFlash/<file>.flv`: sample files 72,862 B (wav) and 265,922 B (flv) downloaded fine. Extrapolation for 7,442 clips: about 0.54 GB audio + about 1.98 GB video = about 2.5 GB (the README's 7.55 GB is the full lfs clone with mp3 and .git). VM has ffmpeg and torch 2.7.0.
- Licence: ODbL 1.0 for the database, DbCL 1.0 for contents (LICENSE.txt). ODbL share-alike applies to derived databases, so do not upload derived clips; upload splits and weights only. README asks users to fill in a Google registration form (not enforced technically). No login.
- Structure: 91 actors, 12 sentences x 6 emotions (4 intensities, some unspecified); filename `actor_sentence_emotion_level`. Same text in every emotion, so lexical content carries no emotion.
- Splits: not shipped. We define them (actor-disjoint for utility, sentence-disjoint for the identity attacker).

RAVDESS [api][probe]: Zenodo 1188976, licence cc-by-nc-sa-4.0 per the Zenodo API, 25.6 GB in 49 zips (speech video about 0.5-0.6 GB per actor, audio-speech zip 208 MB), reachable from the VM (HEAD 200). 24 actors, so only a transfer set.

IEMOCAP (needed for VoicePrivacy): gated (USC request, 7-9 days per scout); not needed for our protocol.

### 3.3 Convincing and falsifying results; published baselines

Convincing:
- Actor-disjoint 6-class accuracy within 10 points of the unprotected model trained with the same recipe, AND retrained identity attackers (linear, MLP, video/temporal net; trained on protected outputs of enrolled actors on some sentences, tested on held-out sentences) at <= about 3x chance in 91-way closed-set (<= about 3.3%) and open-set verification EER >= 40%, with 95% CIs from bootstrap over actors/clips, AND a frontier that beats LEACE and a rate-matched bottleneck, AND the RAVDESS transfer holds.
- Cross-modal check: attack video-protected output with the audio channel (and vice versa) so the claim is about the clip, not one stream.

Falsifying:
- A rate-matched bottleneck or LEACE reproduces the learned frontier (what our KTH results found).
- No operating point exists because identity lives in expression dynamics (published: FLAME expression + jaw coefficients identify 1,429 speakers at 61.1% top-1 on CANDOR, 2510.11223; speaker verification trained on emotion embeddings alone partly works, 2409.03655) [from affect scout, opened by that scout].
- Utility collapses to the optical-flow regime (16.3% accuracy, chance 16.7%).

Published numbers to verify our harness against [opened]: WACV and ICME tables above (use blur sigma, optical flow, face swap rows as sanity checks; ours should reproduce their ordering, not their absolute values, since splits differ); chance 1/91 = 1.099% and emotion chance 1/6 (class prior must be measured from the label file; from memory about 0.17 for five classes and 0.15 for neutral, unverified); VoicePrivacy anchors above. Not published: LEACE or rate bottleneck on CREMA-D; any retrained-attacker number on protected CREMA-D output.

### 3.4 Most likely way to waste two weeks
Building an adversarial or diffusion identity-suppressor and scoring it with the published protocol (shared actors, frozen validator): every method looks private, utility numbers (80-89%) are not actor-disjoint and not comparable, and the first retrained attacker recovers identity. That reproduces our Finding 1 on a new dataset and nothing else. Guard: freeze the split file and the attacker suite first, and run the cheap floors (blur, optical flow, LEACE, rate bottleneck) before any learned method.

---

## 4. Choosing exactly 2 and exactly 3 focuses

Ranking after stress test (feasibility x headroom x distance from prior art), my judgement:
1. **C (CREMA-D)**: highest reuse of existing harness, cheapest data, a documented protocol hole. Risk: acted, label-noisy data and we define the benchmark ourselves.
2. **A (PathoROB)**: best deterministic scripted benchmark and largest nuisance (34 centres); risk: crowded, RI saturating. Value comes from the retrained-probe protocol, not from RI.
3. **B (reframed)**: only with a surviving kill test.

**Best 2:** A + C. Reasons: (i) both executable from ungated data on the VM in days; (ii) each has a specific, evidenced protocol flaw in prior work that our harness fixes; (iii) A is frozen-embedding and C is end-to-end video/audio, so together they test whether the method conclusions transfer (rate bottleneck and LEACE as floors, adversary as optional) without sharing failure modes; (iv) no dependence on gated approvals except optionally UNI2-h/Virchow2 for A.

**Best 3:** A + C + B reframed (action-correlated distractors, retrained distractor probe), gated by the section 2.6 kill test. Reason: B is the only shortlist field with an exact leakage ground truth (twin renders), and the one place a hand-built mask (SAM) is not guaranteed to work. If the kill test fails, replace B by the next scout-ranked field rather than keeping a saturated one.

**First benchmark per focus and estimated cost** (my estimates, assumptions stated; one RTX A6000 unless noted):

| Focus | First benchmark | Data / disk | Estimated cost | Replication target |
|---|---|---|---|---|
| A | PathoROB (3 datasets, 3 models: Phikon-v2, Phikon, UNI2-h after click-through; Midnight optional) | 1.9 GB | about 2 GPU-h (feature extraction of 99,392 patches at roughly 10-20 min per ViT) plus about 3 CPU-h for APD (20 repetitions) | Leaderboard RI within 0.01-0.02 per dataset; Reinhard +16-19% and ComBat +27% relative RI deltas |
| C | Own CREMA-D harness: unprotected utility/identity, then blur sigma, optical flow, LEACE, rate bottleneck, each with 3 retrained attackers | about 2.5 GB (+ RAVDESS speech about 13 GB video if used) | about 8 GPU-h (video encoder fine-tune about 0.5 GPU-h per run; attackers minutes to 1 GPU-h) | Ordering of WACV Table 1 baselines (blur 1 low leakage and low accuracy, blur 2 high both, flow near chance accuracy) |
| B | Distracting Meta-World, one task, LAPO vs LAOM-Labels vs MaskLAM@GT, 3 seeds, plus retrained distractor probe | about 49 GB, about 26 min download | about 15-25 GPU-h (12 runs at about 1-2 h each; reimplementation time dominates); full MT10 would be about 490 GB and 150-250 GPU-h | Published DMW table above (note: it aggregates tasks, so a single task will not match) |

---

## 5. What I could not verify (so the reader can discount)

- PathoROB per-dataset centre counts and the APD correlation levels (not visible in the text I received); the PathoROB paper PDF could not be read locally and I used the arXiv HTML (first 100k of 241k characters only).
- Which model wrappers the repo supports beyond `uni2h_clsmean`; the licence text of Phikon (HF says "other").
- Arevalo et al. benchmark repo location and the JUMP profile sizes; BatchBench training source IDs; single-cell dataset licences.
- MaskLAM and VLM-LAM beyond the project page / HTML (seeds count, exact normalisation of NMSE, action-correlated distractor evidence).
- DAVIS-2017 redistribution terms for the HF Distracting Meta-World dataset (which has no licence).
- VoicePrivacy 2026 official results table (the overview deck was not opened); ICME 2026 paper's identity-matcher training.
- GPU-hour figures are estimates from throughput assumptions, not measurements. The synthetic simulation in 1.5 is a mechanism check on random Gaussians.

## 6. Sources opened (key URLs)

PathoROB: arxiv.org/abs/2507.17845 (HTML), github.com/bifold-pathomics/PathoROB, huggingface.co/datasets/bifold-pathomics/PathoROB-{camelyon,tcga,tolkach_esca}; HERO arxiv.org/abs/2609.35943; Filiot arxiv.org/abs/2607.22861; CRoMa arxiv.org/abs/2607.25497; Center Biases arxiv.org/abs/2609.28231; BatchBench arxiv.org/abs/2610.30104, github.com/ml-jku/batchbench-baselines, huggingface.co/datasets/ai-lab-jku/jumpcp-challenge-datasets; CP-BG-Bench arxiv.org/abs/2609.12761.
Robotics: arxiv.org/abs/2601.22714, 2602.02259 (+ segment-to-focus.github.io), 2605.20223, 2606.30544, 2606.04130, 2607.27138, 2609.22521, 2502.09680; github.com/dunnolab/laom; huggingface.co/datasets/EpicPinkPenguin/visual_distracting_metaworld.
Affect: arxiv.org/abs/2412.00277, 2603.21387, 2609.02941, 2603.17567; github.com/fengxxu/ppfer; github.com/CheyneyComputerScience/CREMA-D; zenodo.org/records/1188976; voiceprivacychallenge.org/vp2026/.
