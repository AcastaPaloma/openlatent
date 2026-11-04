# Literature scout: CT and medical-imaging privacy attacks and defences (2024-2026 first)

Scout: CT / medical imaging cluster, 2026-11-04, programme 2 (adversary suite). Nothing was run; every number is an author claim unless marked "my arithmetic".
Read-depth tags: **FULL/PART** = HTML or full text opened (PART = targeted extraction of method, protocol, tables, not every section); **ABS** = abstract or listing page only;
**SNIP** = seen only in a search snippet, NOT opened, not used for any claim. Several publisher pages (Nature, Lancet, RSNA, ScienceDirect) returned 403 or a cookie wall; those are SNIP.
arXiv IDs run to 2610 because today is 2026-11-04. Many 2026 items are unreviewed preprints.

## 0. Verdict in one paragraph

Medical-imaging privacy has the same evaluation flaw as programme 1's HAR results, plus one new feature: the field's reference task (patient linkage across timepoints, Packhauser-style)
works on open, ungated data for **chest radiographs** but **not for CT**. For CXR, three things are now on record: (i) the main learned defence (PriCheXy-Net) was scored with a single Siamese
re-trained on the defended images and no held-out attacker family; (ii) the one paper that rescored it with three re-trained families plus a non-learned structural attacker (LAPIS, 29 Oct 2026, ChestX-ray14)
reports a much weaker defence than the original paper (cross-study AUC 0.826 and structural Rank-1 22.4%, versus 0.577 originally; protocols differ, so not like-for-like) and itself omits blur, downsampling, PCA and random-warp floors;
(iii) the 2026 critique of the biggest foundation-model "re-identification" claim shows a frozen ResNet never trained on retinal images beats RETFound and 16x16 raw pixels get 42% patient R@1 on the same data, i.e. the trivial floor is the story.
For **CT** itself I found no open longitudinal re-ID benchmark at useful scale: LIDC-IDRI has 8 patients with two CT timepoints, NLST (26,254 subjects) is CDAS-gated, CT-RATE is form-gated and non-commercial,
and the only large open multi-study CT set is DeepLesion (4,427 patients, 10,594 studies; licence unverified). So "CT and medical imaging" in practice means **CXR as the primary testbed and CT as a small, flagged confirmatory arm**.
Best candidate for the programme: a floors-first, retrained-attacker-suite re-audit of CXR linkage defences, then (only if headroom exists) a defender trained on a population of attackers and tested on held-out attacker families.

## 1. Protocol audit: who retrained the attacker?

| Paper | Secret | Attacker(s) | Retrained on defended data? | Held-out family? | Floors reported |
|---|---|---|---|---|---|
| PriCheXy-Net (2209.11531, PART) | patient identity, ChestX-ray14 | one Siamese (2x ResNet-50), 10 runs | yes, "re-trained ... for each model configuration by using deformed images only" | no; paper states adaptive attackers not evaluated | Privacy-Net, DP-Pix only; no blur, no random warp of equal magnitude |
| LAPIS (2610.40722, PART) | patient identity, ChestX-ray14 | ResNet-50, ConvNeXt-T, ViT-S verification nets, worst case, plus SSIM/NCC structural retrieval | yes (orig-orig, orig-anon, anon-anon pairs) | no; but a non-learned structural attacker is reported separately | full-latent noise, random orthonormal replacement, donor swap; no blur/downsample/PCA/label-only |
| Keyed Nonlinear Transform (2605.14123, ABS) | patient identity in split-inference features | not stated | not stated | not stated | none; re-ID AUC 0.635 -> 0.586, i.e. the undefended attacker was already near chance |
| Shadow defence (2506.15711, ABS), learnable perturbation (2503.16542, ABS) | image reconstruction from gradients | "various GIAs", not named in abstract | not stated | not stated | non-targeted DP noise, two unnamed baselines |
| GradInvDiff (MICCAI 2025, PART incl. reviews) | image reconstruction (attack paper) | its own diffusion-guided attack | n/a | n/a | defences tested = Gaussian noise sigma 0.01 and batch 8 only; reviewers flag no clipping or noise multiplier for the "DP" claim |
| Brain-MRI FM linkage (2610.20494, PART) | linkage, sex | Canberra-distance 1-NN (no training), RF for sex | n/a, no defence tested | n/a | FreeSurfer baseline 0.813/0.980 linkability (OASIS-2/MIRIAD), so a handcrafted baseline beats 4 of 5 encoders on OASIS-2 and all 5 on MIRIAD (my arithmetic on their table) |

Take-away: the only medical paper with more than one retrained family is a week old, unreviewed, has no code, and has no cheap-floor row.

## 2. Key papers (20)

| # | Paper (ID/venue, date, tag) | What it gives us | Sceptical note |
|---|---|---|---|
| 1 | Packhauser et al., Deep learning-based patient re-identification ... chest X-ray, arXiv 2103.08562, Sci Rep 2022 [ABS] | Founding result. ChestX-ray14 (112,120 images, 30,805 patients): verification AUC 0.9940, retrieval P@1 0.9963; transfer to CheXpert and COVID set AUC up to 0.987. Matches ">10 years apart" | 2021 anchor. Abstract gives no code link and no gap breakdown; the abstract does not mention the "near-identical" confound (row 5) |
| 2 | Packhauser et al., PriCheXy-Net, arXiv 2209.11531, MICCAI 2023 [PART via ar5iv] | Learned deformation-field anonymiser (U-Net, flow bounded by tanh, scaled by mu in {0.001, 0.005, 0.01}, Gaussian-smoothed). ChestX-ray14 256 px, 70:10:20 patient split, 5,000 test pairs. Re-ID AUC/utility: real 81.8/80.5; mu=0.01 57.7/76.2; Privacy-Net 49.8/57.5; DP-Pix 50.0-52.5/50.0-52.9 | No random-flow control of equal mu, no held-out family, utility scored with the training-time auxiliary classifier. Smoothing and mu chosen by hand |
| 3 | Paul, Shovo, Sidibe, LAPIS, arXiv 2610.40722 (29 Oct 2026) [PART, HTML] | Closest prior art for D1/D2. Splits: model 18,510 patients, release 7,588 patients / 27,962 images, test 4,707. Release set, worst of 3 retrained families: Original AUC_cross 0.974, reconstruction 0.961, PriCheXy-Net 0.826 (Rank-1 22.4%), LAPIS 0.791 (Rank-1 34.7%), full-latent noise 0.897. Acquisition-matched negatives: LAPIS 0.742. Authors say PriCheXy's own 0.577 is not comparable | Unreviewed, no code or weights, single dataset. PriCheXy-Net beats LAPIS on the structural and same-image attacks (Rank-1 22.4% vs 34.7%) while LAPIS wins on cross-study AUC, so the ranking depends on the attacker. Part of the "leak" is acquisition/device signal. No label-only, PCA, blur or downsample floor |
| 4 | Nebbia et al., Re-identification of patients from imaging features extracted by foundation models, npj Digit Med 8 (2025), PMC12283961 [PART via Europe PMC full text] | Frozen CXR Foundation embeddings, L2 NN, same-day pairs excluded. MIDRC: 106,473 images, 39,749 patients, 16,234 with >=2 timepoints; image R@1 50.1%, trained contrastive head 74.0%. R@1 rises with timepoints (MIDRC 18.6% at 2 -> 70.3% at 7+). Demographics: no extra advantage for re-ID'd images on MIDRC. Code github.com/QTIM-Lab/re_identification | No formal threat model, text/table inconsistencies (e.g. identical gender rows for both datasets). MIDRC source row appears to include NIH and CheXpert images, so it may overlap ChestX-ray14. CXR Foundation used via API |
| 5 | Engelmann, Wagner, Zhou, Keane, Matters Arising: Near-identical images, not foundation models, explain purported re-identification, npj Digit Med 9:159 (2026), PMC12905310 [PART via Europe PMC] | On GRAPE (fundus): frozen ResNet10t never trained on retinal images, patient R@1 82.1% vs RETFound 54.7% (copied from Nebbia, not re-run); raw 16x16 pixels 42.5%. Short follow-up (mean 18 months), consistent fixation. Demographic predictability confounded by image quality. Calls for pixel baselines and an explicit threat model | One dataset, different modality (retina). Direction of the lesson transfers to CXR but is not shown there. "Retrieval of other images of a known person" is not identification: use that wording |
| 6 | Bajar et al., Quantifying privacy risks in foundation model representations of brain MRI, arXiv 2610.20494 (14 Oct 2026) [PART, HTML] | OASIS-2 (150 subjects) and MIRIAD (69 subjects, 708 scans). Five frozen encoders. Linkability up to 0.97, ReID@1 up to 99.7% (MIRIAD); sex balanced accuracy up to 95.5% (RF). **Canberra distance beat cosine for every encoder.** SAM-Med3D (supervised segmentation encoder) leaks least: 13.9% ReID@1 on OASIS-2 | Tiny cohorts, T1 brain only, no defence tested. The metric finding (Canberra > cosine) matters for our cosine-kNN scorer. Licence/access not stated |
| 7 | Lee and Kim, Keyed Nonlinear Transform, arXiv 2605.14123 (May 2026) [ABS] | Drop-in keyed obfuscation of split-inference features: re-ID AUC 0.635 -> 0.586, classification within 1 pt, segmentation -4.4 Dice | Abstract names no dataset, no attacker detail; baseline attacker barely above chance. Inversion resistance argued analytically. Treat as "unverified claim" |
| 8 | Trunk-CT scout re-ID with deep metric learning, J Digit Imaging 2024, PMC11169436 [PART via Europe PMC] | Only CT linkage study I found. 5,540 patients, 8,828 scout images, test = 1,291 follow-ups against 4,808 baselines, unseen patients: ACC1 0.907, ACC10 0.969; cross-vendor 0.859 vs same-vendor 0.929. EfficientNetV2-L + AdaCos, code public | Data private (Yamaguchi hospital), scout/surview images not volumes, time gap not reported. Shows CT surview is a biometric but cannot be reproduced by us |
| 9 | Gao et al., Pitfalls of defacing whole-head MRI, arXiv 2501.18834, Comput Biol Med 2025 [ABS] | Cascaded diffusion "refacing" from defaced MRI: generated faces closer to the original than a population-average face (p<0.05), 180 train / 484 test subjects; defacing also removes anatomical signal (muscle radiodensity correlation lost). A defender-vs-adaptive-attacker story | Dataset names not on the abstract page. Skull-stripped input defeats it. Ground truth = the same scan's surface, not an external photo gallery |
| 10 | Aerni, Zhang, Tramer, Evaluations of ML privacy defenses are misleading, arXiv 2404.17399, CCS 2024 [ABS] | Prior empirical-defence evaluations underestimate leakage "by an order of magnitude" (5 case studies); weak attacks, average rather than worst-case samples, no DP baseline | Not medical; membership inference. Gives the template and the vocabulary for "strengthen the suite and the verdict moves" |
| 11 | Santos et al., Persistent demographic information in X-ray foundation embeddings, ICML 2025 workshop [ABS] | MIMIC-CXR and CheXpert, CXR Foundation and BiomedCLIP: age, sex, ethnicity, insurance predictable; removing the most predictive dimensions has "minimal impact on task performance" but does not remove the signal | Abstract has no numbers; MIMIC-CXR is credentialed. Supports "closed-form dimension removal fails" (same as programme 1 LEACE finding) |
| 12 | Glykos et al., Reconstructing training images from foundation model parameters in healthcare, ICML 2026 workshop (poster 78898) [ABS] | 16 FMs (12 medical), 7 datasets, 3 domains. Current inversion yields no recognisable images -> "risk low"; X-ray most vulnerable; linear probing blocks the attack | Workshop abstract, dataset and attack names missing. A SIIM 2025 abstract (SNIP) reports chest X-ray reconstruction from FM embeddings, so the two disagree; in-domain trained decoders are a different, stronger attacker than their pipeline |
| 13 | Muthyala et al., Frozen FM embeddings discard small-lesion signal in chest radiography, arXiv 2606.11606 (Jun 2026) [ABS] | 5 frozen ViTs on 492,724 CXR (NIH-CXR14, MIMIC, Emory) + ChestX-Det10 (3,543 images, 1,462 boxes): CLS at chance for small perturbations (AUC 0.500-0.524) while patch-local probes reach ~1.0 | Utility paper, not privacy. Matters for us: (a) which token the attacker sees is part of the threat model; (b) it gives a utility that a label-only code cannot hold |
| 14 | Dutt et al., CheXGenBench, arXiv 2505.10496, TMLR 2026 [ABS] | Unified fidelity/privacy/utility benchmark for synthetic CXR, 11 generators, 20+ metrics, code on GitHub (Raman1121/CheXGenBench); "high privacy risk regardless of fidelity" | Dataset not stated on abstract page; likely MIMIC-CXR (credentialed). Code licence not stated |
| 15 | Dutt, The Devil is in the Prompts, arXiv 2502.07516 (Feb 2025) [ABS] | De-identification traces in MIMIC-CXR prompts are the most memorised tokens; inference-time mitigations ineffective | MIMIC-CXR (credentialed, DUA forbids re-identification attempts); abstract-only |
| 16 | Wang et al., GradInvDiff, MICCAI 2025 (LNCS 15973), github R00TSEN1650/GradInvDiff [PART incl. reviews] | Diffusion-guided gradient inversion on ChestX-ray8, Acevedo-20, LiTS 2D slices; ResNet18/LeNet. Reported SSIM 0.716 vs 0.487 (ChestX-ray8, ResNet) | Reviewers: two weak rejects, one accept; code repo was README-only at review; noise test is sigma 0.01 only; 3D volumes not tested |
| 17 | Jiang, Ma, Yang, Shadow defence 2506.15711 (Med Image Anal 2025) and Jiang, Firouzi, Chakrabarty 2503.16542 [ABS both] | Targeted noise at sensitive regions (ChestXRay, EyePACS: +3.73/+2.78 PSNR gap); learnable latent perturbation with minimax, "12.5% reduction in attacker accuracy" | Neither abstract names attackers or says adaptive; no code links |
| 18 | Pei et al., Medical Image De-Identification Benchmark (MIDI-B), arXiv 2507.23608 (Jul 2025) [ABS] | Deterministic DICOM header + pixel PHI benchmark on TCIA data: validation 216 subjects, test 322; ten teams, scores 97.91-99.93%; answer keys and validation script public; validation set CC BY 4.0 (via search result) | Saturated; rule-based tools; synthetic identifiers may look unlike real ones |
| 19 | Sharma et al., Cross-dataset linkage of brain MRI using image similarity, arXiv 2602.10043 (Feb 2026) [ABS] | Non-trained image similarity links skull-stripped T1 scans across timepoints, scanners and protocols with near-perfect accuracy | Abstract has no datasets or numbers. Another instance of "no learning needed" |
| 20 | Tian et al., Towards all-in-one medical image re-identification (MedReID), arXiv 2503.08173, CVPR 2025 [ABS] | Benchmark of 25 foundation models and 8 multimodal LLMs on 11 datasets; code github tianyuan168326/All-in-One-MedReID-Pytorch | Dataset release not mentioned on the abstract page; framed as clinical matching as much as privacy |

Adjacent, opened but not medical, not used for claims: Bissias et al., Contrastive Privacy, 2605.02977 [ABS] (CLIP-based semantic audit of sanitisers; finds residual leakage in all four PII tools tested);
TrustCLIP, 2607.04484 [ABS] (trains a feature transform against a generative reconstruction attacker; the abstract does not say whether held-out attackers were tested). Also opened [ABS]: Ziller et al. 2312.04590 (very large DP budgets block reconstruction),
Packhauser et al. 2211.01323 (latent-diffusion anonymous CXR with patient-matching filter), PPCMI-SF 2603.05541 (keyed latent transform for CT-angiography segmentation, claims resistance to inversion and membership, attack details missing).

## 3. Open data: what an unattended rented VM may use

"Ungated" = downloadable without a credentialing step or a form that a human must approve. Licence facts are only as good as the page I opened.

| Dataset | Size / patients / longitudinal | Access and licence | Verdict for us |
|---|---|---|---|
| ChestX-ray14 (NIH) | 112,120 frontal images, 30,805 patients, about 3.6 images per patient (Packhauser); our repo's reference set | NIH Box, no login. Google Cloud page: "no restrictions on the use", asks for attribution and citation (Wang et al. CVPR 2017). HF mirror card lists licence "unknown" | **Yes.** Already wired into `openlatent`. Tabulate time gap and view position per query first (near-identical confound) |
| MIDRC CXR subset (Nebbia) | 106,473 images, 39,749 patients, 16,234 with >=2 timepoints, mean 5.1 timepoints | Public per the paper; terms not verified; table suggests NIH and CheXpert sources, so overlap with ChestX-ray14 is likely | Candidate external set only after an overlap check; do not call it independent until verified |
| MIMIC-CXR-JPG | 377,110 images, ~227k studies | Credentialed PhysioNet (CITI + DUA); DUA says users must not try to re-identify individuals | **No.** Both gated and the DUA conflicts with linkage research |
| CheXpert | not verified (page did not load) | Registration/research agreement from memory, not opened | Treat as gated until checked |
| CT-RATE | 25,692 volumes (50,188 with reconstructions), 21,304 patients; file names encode patient index and scan letter | HF dataset, cc-by-nc-sa-4.0, form-gated (name, institution, email); academic use only | Flag. Needs the owner to click through; cannot be fetched blind |
| LIDC-IDRI (TCIA) | 1,010 patients, 1,308 studies incl. DX/CR, 244,527 images, 133 GB; **only 8 patients have two CT timepoints** | CC BY 3.0, public, TCIA Data Retriever | Useful for single-scan attribute probes only. Useless for longitudinal re-ID |
| LUNA16 (Zenodo 3723295) | 7 subset zips of ~6.3-7.3 GB (+ part 2), derived from LIDC | CC BY 4.0 (LIDC under CC BY 3.0), open | Same limitation as LIDC |
| TotalSegmentator v2.0.1 (Zenodo 10047292) | 1,228 CT, 23.6 GB zip | CC BY 4.0, open | Yes for attribute/site probes and 3D encoders. Per-scan metadata columns (age, sex, institute, scanner) are from memory, not verified; patient-level repeats unknown |
| AMOS22 (Zenodo 7155725) | 500 CT + 100 MRI, 24.2 GB | CC BY 4.0, open | Yes, multi-centre/vendor; no patient IDs seen |
| DeepLesion (NIH) | 32,735 lesions, 32,120 slices, 10,594 studies, 4,427 patients (from papers via snippets; readme v6 not machine-readable here); 16-bit PNG in 56 zips with +-30 mm context | NIH Box download; the NIH licence text was not verifiable; an HF mirror's CSV card says cc-by-sa-4.0 | **Best open multi-study CT set**; confirm licence and per-patient/study IDs on download |
| CT Colonography, ACRIN 6664 (TCIA) | 825 patients, 836 studies, 3,451 series, 941,771 images, 496.67 GB | CC BY 3.0 images, CC BY 4.0 clinical data | Same-session prone vs supine pairs (positional change), not longitudinal |
| NLST | 26,254 CT subjects, 11.3 TB | CDAS approval required; no redistribution | **No** (a 119-subject "NLST-New-lesion-LongCT" analysis result exists on TCIA, licence line ambiguous; not verified) |
| MedMNIST v2 | 12 2D + 6 3D sets, 28x28 up to 224 (MedMNIST+) | Mostly CC BY 4.0 (DermaMNIST NC) | No patient IDs mentioned; good only for membership/attribute work, not linkage |
| MIDI-B (TCIA) | validation 216 subjects, test 322 | CC BY 4.0 validation set, answer keys and script | Deterministic but saturated header/pixel-PHI benchmark |
| Head CT with faces + photos | none verified | n/a | Dropped, as the lead noted |

Ethics note: use pseudonymous within-dataset IDs only, never link to outside identities, report aggregates, and say "linkage/retrieval" not "re-identification" (Engelmann et al. warn the latter misleads lay readers).

## 4. Deterministic benchmarks that exist

- `openlatent score` (ours): cosine kNN, ties by expected rank, byte-identical on same machine. Add Canberra and L1 as second deterministic scorers (row 6: Canberra beat cosine on all five encoders), plus a raw-pixel NCC/SSIM scorer (rows 3, 5).
- Nebbia's code (QTIM-Lab/re_identification): deterministic L2 NN with same-day exclusion; trained arm is InfoNCE, seed-deterministic only.
- MIDI-B answer-key scoring (header/pixel PHI, TCIA).
- CheXGenBench (synthetic CXR, 20+ metrics) and MedReID (CVPR 2025) have code; their data are partly credentialed or unspecified.
- No deterministic CT linkage benchmark found. We would have to build one from DeepLesion (and optionally CT-RATE if approved).

## 5. Candidate directions (score 1-5, 5 best; total /30)

Criteria: **V** value for the robust-suite goal; **D** data availability and licence (ungated only); **Det** deterministic verification; **C** compute on 4x A6000 in days (5 = under one GPU-day);
**L** fit for learned methods; **N** novelty/gap.

| ID | Direction | V | D | Det | C | L | N | Total |
|---|---|---|---|---|---|---|---|---|
| D1 | Floors-first re-audit of CXR linkage defences under a retrained-attacker suite (ChestX-ray14) | 5 | 5 | 4 | 5 | 3 | 4 | **26** |
| D2 | Defender trained on a population of attackers, tested on held-out attacker families (CXR learned anonymiser; stage 2 of D1) | 5 | 5 | 4 | 4 | 5 | 4 | **27** (conditional on D1 headroom) |
| D4 | Learned/adaptive inverters vs published medical FL and split-feature defences, scored by reconstruct-then-re-ID | 3 | 5 | 3 | 3 | 4 | 4 | **22** |
| D3 | CT-native linkage benchmark from open multi-study CT (DeepLesion; CT Colonography supine/prone as a side test) | 4 | 2 | 3 | 3 | 3 | 5 | **20** |
| D5 | Dense-utility scanner/site/sex hiding on open CT (TotalSegmentator, AMOS) | 3 | 4 | 4 | 3 | 4 | 2 | **20** |

### D1. Floors-first re-audit of CXR linkage defences under a retrained-attacker suite

- Idea: build the standardised attacker suite on the one place medical linkage is measurable with open data. Attackers: (a) frozen-FM kNN with cosine, L2 and Canberra (DINOv2 and RAD-DINO are the open candidates, licences not checked; CXR Foundation is an API with terms), (b) re-trained Siamese ResNet-50 (Packhauser), (c) InfoNCE head on frozen FM (Nebbia), (d) ConvNeXt/ViT re-trained (LAPIS), (e) non-learned structural NCC/SSIM with affine registration, (f) reconstruct-then-re-ID (attacker-trained decoder on the released z; the README's own headline requirement), with patch-token versus CLS access as an explicit threat-model axis (row 13).
  Defences to rescore: PriCheXy-Net (re-implement from the paper), LAPIS-style donor residual swap (no code), and every floor.
- Why it is high value: it is the cleanest test of the programme's falsifiable core outside HAR. LAPIS already shows the verdict on PriCheXy moves by 0.2+ AUC when the suite is strengthened; nobody has run the cheap floors.
- Data: ChestX-ray14 only (ungated, in the repo). Release/eval split by patient, ~7.6k release patients as in LAPIS gives workable power; compute the real MDE from the number of independent patients with >=2 images.
- Trivial floors that would kill it: (1) **random smooth deformation of equal mu** (PriCheXy's own mechanism without learning; the paper has no such control), (2) Gaussian blur and x4-x16 downsample, (3) JPEG/quantisation and dither, (4) PCA-k or random projection of FM features (rate bottleneck) for the latent-release variant, (5) label-supervised LDA on the 14 findings (a 14-dim code that carries utility and little identity). If any floor matches the learned defences at equal worst-case leakage, D1 becomes a negative-result paper with a benchmark; that is still a valid programme outcome.
- Biggest risk: the measured "identity" is partly acquisition/device (LAPIS: 0.742 with acquisition-matched negatives) and near-identical visits (Engelmann). Mitigate by stratifying queries by time gap and view position, excluding same-day pairs (Nebbia), and reporting the acquisition-matched number alongside the headline.
- Compute: roughly 150 attacker fits at 256 px, 10-30 min each, about 40 GPU-hours, under one wall-clock day on 4 GPUs. Uses cheap deterministic tier first.

### D2. Defender trained against an attacker population, scored on held-out families (CXR)

- Idea: train a learned anonymiser or adapter against 3-4 re-trained attackers (e.g. ResNet-50 Siamese, ConvNeXt, InfoNCE head), hold out a whole family (the structural NCC/SSIM attacker, a frozen-DINOv2 kNN, or a different-objective model) and an entire second dataset if overlap-checked. Compare with the lower envelope of D1's floors. This is exactly the programme's "does a suite-trained defender generalise to unseen attackers" question, on a task with real stakes.
- Utility must not be the 14 labels alone: a label-only LDA code already holds those (programme 1 lesson, Focus A). Use utility that LDA cannot hold: held-out findings and region-level probes (ChestX-Det10 boxes, patch-token probes; row 13 shows CLS embeddings destroy small-lesion signal), so there is room above the label-only floor.
- Trivial floors: as D1, plus the label-only LDA arm and a rate bottleneck without an adversary (programme 1 found the adversary added nothing beyond a bottleneck or noise in four settings).
- Biggest risk: the held-out structural attacker already defeats learned latent swaps (LAPIS: 22.4% vs 34.7% Rank-1 favouring the deformation method, the reverse for AUC_cross), so generalisation may fail in a way the floors also fail, giving a negative result; also PriCheXy and LAPIS have no released code, so baselines must be re-implemented.
- Compute: adversarial training with 3-4 attackers in the loop, 3 seeds, about 3-5 GPU-days. Gate it behind D1's headroom check and an oracle arm.

### D4. Adaptive inverters against published medical FL / split-feature defences

- Idea: take Shadow defence, learnable perturbation, KNT and plain Gaussian-noise gradient/feature sharing, train an attacker that knows the defence (a denoiser or a learned inverter on defended gradients/features, plus GradInvDiff-style guidance), and score by reconstruct-then-re-ID rather than PSNR. Aerni et al. is the template (row 10).
- Data: ChestX-ray14 and MedMNIST, ungated. Learned fit is good (learned defences versus learned inverters).
- Trivial floors: gradient clipping plus Gaussian noise, larger batches and multi-step FedAvg updates (a snippet-only 2025 detection paper reports training-mode client updates resisting inversion, SNIP), DP-SGD. The risk is that realistic FL settings already defeat inversion, leaving no headroom to defend.
- Biggest risk: PSNR/SSIM are not identity; deterministic seeding of optimisation attacks is fragile; much of the evidence is abstract-only; the fit with "latents z = E(x)" is indirect. Compute is 4 GPU-days or more for a few hundred images per condition.

### D3. CT-native linkage benchmark

- Idea: Packhauser-style Siamese and cosine/NCC scorers on DeepLesion slices linked across the 10,594 studies of 4,427 patients, a frozen 3D CT encoder (if an open one is permitted) on CT Colonography prone/supine pairs, and trunk-CT scout results (row 8) as the published reference. Novelty is the highest because I found no public CT FM linkage study (CT-FM papers evaluate retrieval for organ field-of-view, not privacy; SNIP of 2501.09001).
- Floors that would kill it: raw-pixel NCC on body outline and table position (rows 5, 8, 19 all say no learning needed), downsample x4, and, on the other side, the possibility that linkage is so easy there is nothing to defend.
- Biggest risk: ground truth. DeepLesion key slices from different studies may sit at different z-levels, so "same patient" pairs are not anatomically matched; the licence is unverified; CT Colonography is same-session, not longitudinal; CT-RATE would need the owner to click through the gate. Treat as a confirmatory arm of D1/D2, not a standalone focus.

### D5. Dense-utility scanner/site/sex hiding on open CT

- Idea: keep segmentation-grade (dense) features, hide scanner/site/sex, so label-only LDA cannot hold the utility. Data are clean (TotalSegmentator CC BY 4.0, AMOS CC BY 4.0).
- Floors: per-site intensity normalisation, resampling/downsample, HU windowing, noise, PCA. Programme 1 closed Focus A and Focus E with exactly this shape of result (adversary adds nothing beyond a bottleneck or noise; per-source whitening is a strong floor on PathoROB but raised leakage on single-cell).
- Biggest risk: repeating a negative; the harmonisation literature is crowded; the "task implies the secret" trap for sex (organ presence) and scanner (texture is task-relevant for segmentation). Low prior; only if D1 is done and the owner wants breadth.

### Recommended order

1. D1 (about 1 day, mostly deterministic, answers whether the suite moves conclusions). 2. D2 only if D1 shows headroom against the floor envelope and an oracle arm; pre-register margin, comparator and stop rule first (CHARTER lessons). 3. D3 as a small confirmatory CT arm if DeepLesion's licence and IDs check out. D4 and D5 are reserve.

## 6. What I would NOT pursue

- **Head CT / MRI defacing and face reconstruction.** Already dropped for lack of open ground truth; I found nothing that changes this. Gao et al. (row 9) uses the same scan's surface as truth, a population-average surface comparison, not a photo-gallery identification; the 2024 MRI-render face-search result and the head-CT de-id numbers (85% -> 64%) are SNIP only. Also ethically the heaviest topic.
- **DICOM metadata and burned-in PHI (MIDI-B).** Saturated at 97.9-99.9% for rule-based tools on synthetic identifiers; no learned-method headroom, and it is not a latent-sharing problem.
- **Report leakage on CT-RATE or MIMIC-CXR.** CT-RATE is form-gated and non-commercial; MIMIC-CXR is credentialed and its DUA prohibits re-identification attempts. The memorisation papers (rows 14, 15) rely on MIMIC.
- **Patient-level membership inference suites for diagnostic models.** The 2026 Nature audit of chest-radiograph models is occupied ground (SNIP, secondary coverage only: RMIA on TorchXrayVision models, minority patients over-represented in the top-1% risk tail) and its defence is DP, not a learned adapter.
- **Embedding inversion as a headline attacker.** Two abstracts disagree (row 12 vs the SIIM snippet); keep it as component (f) of the D1 suite, not a focus.
- **MedMNIST for linkage.** No patient IDs mentioned and 28x28 destroys the thoracic geometry that makes CXR a biometric.
- **LIDC-IDRI / LUNA16 for longitudinal re-ID.** Eight patients with two CT timepoints.
- **A mask-release direction for CT (organ masks instead of images).** Tempting as the analogue of the canonical stick figure, but sex and age are implied by which organs and which shapes exist; "the task implies the secret".
- **Scanner/site invariance as a stand-alone CT focus** (D5 aside): crowded, and two of our own foci already ended negative on the same pattern.

## 7. Cross-cutting observations for the suite design

1. **Metric choice is part of the attacker.** Canberra beat cosine for all five brain-MRI encoders (row 6); the repo's cosine kNN is therefore a weak deterministic scorer. Add Canberra, L1 and a pixel-NCC scorer to the deterministic tier.
2. **More gallery images per patient raises R@1 sharply** (row 4: 18.6% at 2 timepoints to 70.3% at 7+), so the number of timepoints is a suite parameter, not a footnote.
3. **A supervised segmentation encoder leaked least** (SAM-Med3D 13.9% ReID@1 on OASIS-2 vs 92% for AnatCL); task-supervised encoders are a natural "floor" for latent release and cost nothing to include.
4. **The token the attacker sees matters** (CLS vs patch, row 13). Declare it in the threat model.
5. **The attacker suite should include at least one non-learned family** (NCC/SSIM with registration) because it is the one learned defences are least often trained against (row 3).
6. **Datasets overlap**: MIDRC's CXR subset appears to contain NIH and CheXpert images; the brain-MRI linkage paper (row 19) warns about cross-dataset reuse. Run a near-duplicate check across any pair of sets before calling one "external".

## 8. Seen only in a search snippet (not opened; do not cite numbers)

- Knolle et al., Disparate privacy risks from medical AI, Nature 2026 (doi 10.1038/s41586-026-10688-0): paywalled; secondary coverage only.
- Steeg et al., re-identification of anonymised MRI head images with PimEyes/FaceCheck.ID, eClinicalMedicine 2024: 403.
- Head-CT facial de-identification (J Digit Imaging 2026; SSRN preprint) and Int J CARS 2025 masking study.
- Seibel et al., OCT linkage with foundation models, BVM 2026 (RETFound 78% Rank-1).
- Annotated test-retest lung CT set (Sci Data 2024, 32 NSCLC patients, same-day repeats).
- Vision Foundation Models for Computed Tomography (CT-FM), arXiv 2501.09001; Practical feasibility of gradient inversion, arXiv 2508.19819; Levy, The Security of Deep Learning Defenses in Medical Imaging (adaptive evasion, not privacy); Revisiting LiRA under realistic assumptions, arXiv 2603.07567.
- SIIM 2025 abstract on chest-X-ray reconstruction from FM embeddings.
