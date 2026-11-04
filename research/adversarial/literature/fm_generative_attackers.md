# Scout: foundation-model and generative attackers (2024-2026)

Programme 2 (adversary suite), literature cluster "FM / generative attackers", written 2026-11-04. Scout stance: sceptic. Everything below was opened with
WebFetch unless marked; the fetch tool summarises pages with a small model, so any number that ends up in a paper must be re-read from the source. "Read" column:
F = full text (or the first ~100k characters of it), P = partial, A = abstract page only. Nothing was run, no git commands were used.

## 1. Bottom line

1. **There is almost no published work that tests foundation-model (FM) attackers head-to-head against obfuscated HAR video, skeleton or IMU outputs.** I found no
   LoRA-versus-frozen-versus-full-fine-tune attacker comparison in vision-privacy at all, no IMU/time-series-FM identity-leakage paper, no skeleton-FM identity paper and no
   Video-LLM identity-from-obfuscated-video paper. Absence from a noisy US-only search is not proof, but the gap is real enough to be a contribution.
2. **The evidence that "FM attackers break what old attackers could not" is mixed and is the sceptic's best friend.** Positive: FaceLinkGen breaks three 2023-2025 face-template
   defences with a distillation attacker that costs under 2 GPU-hours on one A6000 (2602.02914); white-box inversion of a 2026 latent video anonymiser pushes attribute cMAP back
   from 70.5 to 73.7 (2511.08666); frozen CXR-FM features re-identify 26-50% of images at rank 1 and a trained head doubles that (Nebbia 2025). Negative: a 10-layer ResNet that never
   saw retinal images beats RETFound at retina re-identification (72.0 vs 38.9 image-level R@1) and a 16x16 raw-pixel vector gets 25.1 (Engelmann 2026); 15 hand-made FreeSurfer
   numbers beat three of five brain-MRI encoders at subject linkage on OASIS-2 (0.75 vs 0.14-0.66); and in our own programme DINOv2 was below the R3D video attacker on every BDQ and
   canonical-stick-figure output.
3. **Working hypothesis from our own data (not from the literature): an FM attacker only beats a from-scratch attacker when the defended artefact still looks like natural
   images or video** (raw, blur, x16 downsample, person-box blur), because that is where pretraining transfers. On difference maps, stick figures and 1-2 pixel maps it is
   not stronger. If true, the FM family is a "natural-likeness" test, not a universal upper bound, and the suite must say so.
4. **Generative reconstruction is bounded by the data-processing floor.** Any reconstruction is a function of the released artefact, so a classifier trained directly on that artefact
   can in principle recover whatever the reconstruction recovers. Reconstruct-then-re-identify (RTR) adds something only through an *external prior* (a pretrained identity model,
   a diffusion prior, a raw gallery) at a small labelled-data budget. That is a legitimate and under-tested attacker (label-free, gallery-linkage), but not a stronger one in the
   information sense.
5. **Weak attackers are the norm.** The 2026 survey of privacy-preserving action recognition (2608.04501) says under 10% of papers test an adaptive attacker. In my reads, Aslam 2026
   uses one ImageNet ResNet-50, one seed; ReGenHuman 2026 uses one off-the-shelf re-ID checkpoint with no raw baseline and no retraining; Fioresi 2026 names no unseen attacker
   architecture. A standardised suite has room to matter.
6. **Top three directions** (details in section 6): D1 pretrained-versus-scratch x frozen/LoRA/full-FT attacker ladder as the held-out family (24/30); D6 attacker-budget curves and a
   saturation stopping rule, sharing D1's runs (23/30); D3 longitudinal gallery re-identification on CT/CXR with learned-metric attackers and brutal floors (22/30). D2 (RTR) and
   D4 (cross-encoder bridge) are 22 and 21; D5 (zero-shot Video-LLM) is 18.

## 2. What our own data already says (programme 1, `research/privacy-obfuscation/docs/journal.md`, `results/SUMMARY.md`)

- DINOv2 (probe / fine-tune) is the strongest attacker on raw, blur and x16-downsampled KTH (identity 99.3/99.2 raw, 87.8/94.0 at x16 vs ResNet/R3D 92.6) and on VP-UCF101 attributes
  (cMAP 78.9 vs ResNet-50 76.6, R3D 75.4 on raw; 1-2 points ahead on blur8).
- On learned and canonical outputs it is weaker than the R3D video attacker: BDQ a5 ens 18.6/35.1 vs R3D 37.0; scenario-split BDQ 10.3/12.8 vs 24.7; canonical one-posture
  stick figures 23.1/25.8 vs R3D 29.1; canonical x16 8.4/4.0 vs best of ResNet/R3D 13.7.
- The one place DINOv2 changed a conclusion: harness on VP-HMDB51, recommended person-box blur sigma 16: privacy 52.3 with the standard suite, 43.4 after the DINOv2 re-run
  (certified privacy now the worst over re-runs). On UTD-MHAD and both KTH splits the DINOv2 re-run was weaker or equal.
- The harness lists "no reconstruction attacker" and "no cross-dataset protocol" as known limits (`docs/harness.md`), and R5 already treats ResNet-50 and DINOv2 as
  attackers the learned obfuscator never trained against. No V-JEPA/VideoMAE, no LoRA tier, no random-init-ViT control, no linkage-to-raw-gallery attacker exists yet.

## 3. Key papers (20)

| # | Paper (ID / URL) | Read | What it shows | Numbers worth keeping | Sceptic's note |
|---|---|---|---|---|---|
| 1 | Nebbia et al., "Re-identification of patients from imaging features extracted by foundation models", npj Digit Med 8:469, 2025 (doi 10.1038/s41746-025-01801-0; full text via Europe PMC PMC12283961) | F | Zero-shot L2 search on frozen FM features re-identifies patients across time; a trained head roughly doubles it | CXR Foundation v1 (API, frozen): MGH image R@1 25.9, MIDRC 50.1 (patient-level 30.7 / 48.5). Frozen vs trained linear layer on held-out test: MGH 32.6 -> 63.7, MIDRC 59.1 -> 74.0. More timepoints: MGH 18.2 (2) -> 53.8 (5+); MIDRC 18.6 -> 70.3 (7+). RETFound full fine-tune: abstract gives 82.3 / 93.9 / 63.7 image-level (fundus / OCT / CXR) | Query pool = whole dataset, same-day excluded; no raw-pixel or scratch-CNN floor (see #2). The "fine-tuned" CXR arm is only a linear layer because the model is API-only. Frozen baselines differ between Table 2 and Table 5 (32.6 vs 25.9) and the paper does not reconcile them |
| 2 | Engelmann et al., "Matters Arising: near-identical images, not foundation models, explain purported re-identification" (doi 10.1038/s41746-026-02440-9; Europe PMC PMC12905310) | F | Replication on open GRAPE: non-FM baselines match or beat the FM | GRAPE image/patient-level R@1: RETFound 38.94 / 54.72; ResNet10t (never saw retinal images, described as frozen/untrained) 72.04 / 82.08; raw 16x16 pixels 25.13 / 42.45. Mean follow-up 18 months | The best single argument for floors-first. Ambiguity: "untrained" vs "never saw retinal images" is not resolved in what I read. Recommends spelling out the threat model |
| 3 | Bajar et al., "Quantifying privacy risks in foundation model representations of brain MRI", arXiv 2610.20494 (Oct 2026) | F | Five frozen encoders, linkage of subjects across scans by Canberra/cosine nearest neighbour; no training | ReID@1 OASIS-2 / MIRIAD: AnatCL 0.921 / 0.997; DINO-ViT-B/8 (natural images) 0.824 / 0.987; 3D-Neuro-SimCLR 0.665 / 0.979; Brain-MRI-SSL 0.576 / 0.982; SAM-Med3D 0.139 / 0.896; FreeSurfer 15 features 0.750 / 0.999; chance 0.007 / 0.015 | Only baseline is FreeSurfer; no raw-pixel or untrained-net floor. Natural-image DINO reaches 0.82, so medical pretraining is not required. Scan interval and preprocessing confound OASIS vs MIRIAD |
| 4 | Tai, "Pretrained, Frozen, Still Leaking: auditing cross-encoder attribute transfer in EEG foundation models", arXiv 2606.09189 | P (first 100k of 150k chars) | Frozen BIOT / LaBraM / EEGPT leak spectral attributes on subject-disjoint splits; a linear bridge transfers an attacker across encoders | Subject-disjoint gain 0.284-0.451 (all CI lower bounds > 0); cross-encoder gain 0.244-0.435, CI lower bounds >= 0.08; reference-set identity BPID@1 0.950 (BIOT), 0.760 (LaBraM), 0.473 (EEGPT), 0.200 Sleep-EDF; Gaussian noise, bottleneck, DP-SGD head did not close the attribute channel | Closest thing to a wearable-FM identity audit. No raw-feature or untrained-encoder baseline visible in what I read. 8-23 subjects per set |
| 5 | Fioresi, Dave, Shah, "Privacy Beyond Pixels: latent anonymization for privacy-preserving video understanding", arXiv 2511.08666 (ICLR 2026) | P (first 100k of 131k) | Adapter on frozen VideoMAE / V-JEPA / I3D features; evaluates black-box and white-box inversion | VISPR cMAP: VideoMAE-B 70.47 -> 49.92, V-JEPA-H 72.44 -> 51.42. VP-HMDB51 cMAP: raw 76.4, adapter 70.5, black-box inversion 58.8 (chance 58.3), white-box inversion 73.7; VP-UCF101: raw 75.9, adapter 69.6, black-box 58.4, white-box 72.6. CASIA-B gait retrieval 69.7 -> 53.5 (26.7 without the consistency loss) | Attacker architectures not stated; no unseen attacker architecture tested; static-clip classifiers for VISPR; white-box leak 73.7 is the real number. "35% leakage reduction" is relative to a weak attacker |
| 6 | Guo et al., "FaceLinkGen: rethinking identity leakage in privacy-preserving face recognition with identity extraction", arXiv 2602.02914 | F | Distil protected templates into a frozen ArcFace-family space, regenerate faces with Arc2Face (SD1.5), link by cosine | Breaks PartialFace (ICCV23), MinusFace (CVPR24), FracFace (NeurIPS25): LFW 1:1 acc 0.988-0.992; Face++ Success@5 0.979-1.000; near-zero-knowledge (30 paired samples used for validation only, generic blur-style proxy for the unknown conversion): Face++ Success@5 0.946-0.963; distillation under 2 h on one A6000, regeneration ~3 s per 5 images | Best template for our RTR direction and for compute. Authors withhold code. Defences were evaluated by their authors with pixel-loss attackers, which the paper argues is the wrong yardstick |
| 7 | Rubab, Tong, Ross, "Unmasking face embeddings: reading, rendering and naming with foundation models", arXiv 2609.00411 | A | A precomputed linear map from face-recogniser embeddings into off-the-shelf FM spaces gives text search, rendering via an unmodified diffusion decoder, and naming; no training of either model | No numbers in the abstract | Cross-encoder bridge idea (D4) in its cheapest form; unverified beyond the abstract |
| 8 | Zhang et al., "Unlocking visual secrets: inverting features with diffusion priors", arXiv 2412.10448 (TMLR 2025) | F (HTML) | LDM-guided inversion of intermediate features of ResNet-18/50, ViT-base, YOLO and CLIP ViT-B/32 | Black-box, deepest layer, ViT-base: SSIM 0.71 (diffusion) vs 0.55 (decoder only), PSNR 19.5 vs 11.5; trained on 4,096 images, tested on 1,024; CLIP end-to-end black-box SSIM 0.46; white-box ViT-base SSIM 0.91 | No identity metric, no defence evaluated, CLIP/ViT-B only (no DINOv2 in what I read). Measures image similarity, not leakage of a secret |
| 9 | Zheng, Ma, Tong, Xie, "Diffusion transformers with representation autoencoders", arXiv 2510.11690 (ICLR 2026) | F (first 100k of 133k) | Frozen DINOv2 / SigLIP2 / MAE encoders plus a trained ViT decoder reconstruct ImageNet at rFID ~0.5 | rFID: DINOv2-S/B/L 0.52 / 0.49 / 0.52, SigLIP2-B 0.53, MAE-B 0.16, SD-VAE 0.62; decoder ViT-XL ~107 GFLOPs | Not a privacy paper. Establishes that frozen DINOv2/SigLIP2 token grids keep enough pixel detail for near-faithful decoding, and that encoder size barely matters. rFID is distribution-level, says nothing about per-image identity |
| 10 | Oz et al., "Reconstructing training data from real-world models trained with transfer learning", arXiv 2407.15845 | A | Embedding-space reconstruction of training images for classifiers trained on DINO-ViT / CLIP embeddings | Abstract only | Attacks the downstream model, not the released embedding; relevant only as a prior that FM embeddings are invertible enough to matter |
| 11 | Ye et al., "BiggerGait", arXiv 2505.18132 (NeurIPS 2025) | F (HTML) | Layer-wise features of frozen-or-not DINOv2 / CLIP / SAM give state-of-the-art RGB gait recognition | CCPG rank-1 (CL/UP/DN/BG): DINOv2-S 89.8/92.1/93.7/97.5 vs silhouette DeepGaitV2 78.6/84.8/80.7/89.2; DINOv2-L 92.8/94.6/95.7/98.2; CLIP-S 81.0/86.2/89.7/96.9; cross-domain avg DINOv2-S 67.0 vs DeepGaitV2 47.4; DINOv2-S 8.8 h on 8 x 24 GB RTX 6000 | Gives a scale answer: S -> L adds 0.7-3.0 points for DINOv2 (CLIP up to +6.5, SAM up to +3.2 on CL/UP/DN/BG); backbone choice at size S (DINOv2 89.8 vs CLIP 81.0 on CL) is 8.8 points. Frozen/not frozen unclear. CC BY-NC-ND on the arXiv text; code "will be released" |
| 12 | Aslam et al., "From pixels to privacy: temporally consistent video anonymization via token pruning", arXiv 2603.26336 | F (HTML) | Typical 2026 evaluation: one ResNet-50 (ImageNet init) privacy attacker retrained on anonymised output, I3D utility model, single seed 42 | VPUCF cMAP raw 76.62 -> 60.30; VPHMDB 76.62 -> 70.32; PAHMDB 70.20 -> 65.92 | No frozen/fine-tuned variants, no FM attacker, no adaptive attack, no CIs. Evidence for "the field's attackers are weak" |
| 13 | ReGenHuman, "Re-generating human appearances for realistic full-body video anonymization", arXiv 2606.14972 | F (HTML) | Text-to-video generative anonymiser, evaluated with a single off-the-shelf clothes-changing re-ID checkpoint (CSCI-V, EVA02 hybrid) and Qwen3-VL-8B utility QA | Standard re-ID R-1: Gaussian blur 50.0, pixelation 100.0, blackout 7.6, DeepPrivacy2 body 7.0, theirs 3.2 (no raw-footage row) | Gait deliberately preserved (source pose reused) and not tested; no retraining of the attacker; "near chance" not quantified |
| 14 | "Privacy beyond the face: assessing gait privacy through realistic anonymization in industrial monitoring", Sensors 26(1):187, 2026 (PMC12788357) | F | Pose-guided full-body anonymiser (DeepPrivacy2 FB_cse) leaves gait identity intact; skeleton gait model GPGait++ on YOLOv11m-Pose keypoints | 10 participants; rank-1 69-96% depending on protocol; original-trained model on anonymised frames loses only 2-6 points | Tiny private dataset, one anonymiser, one gait model. Supports "skeleton survives generative anonymisation" |
| 15 | Tömekçe, Vero, Staab, Vechev, "Private attribute inference from images with vision-language models", arXiv 2404.10618 (NeurIPS 2024) | A | Seven VLMs infer owner attributes from ordinary photos, accuracy scales with general capability | Up to 77.6% | Zero-shot semantic attacker; exploits context and the action-implies-attribute prior, so the floor is the task-label attacker |
| 16 | Zheng, Jacobs, Parekh, "Demographic predictability in 3D CT foundation embeddings", arXiv 2412.00110 | A | Self-supervised 3D CT embeddings on NLST carry demographics | Age RMSE 3.8 years, sex AUC 0.998, race AUC 0.878 | Attribute leakage, not identity; model names not in the abstract |
| 17 | Pai et al., "Vision foundation models for computed tomography" (CT-FM), arXiv 2501.09001 | A + HF card | Label-agnostic contrastive SegResNet pretrained on 148k IDC CT scans, open weights, includes retrieval tasks | HF card: Apache-2.0, 77.8M parameters, 512-d pooled embedding | Pretraining data came from the Imaging Data Commons, which hosts NLST: possible overlap with any NLST attack set (unverified) |
| 18 | Ueda and Ishida, patient verification across CXR and CT scout images, medRxiv 2026.08.11.26360177 | F (prose; tables did not render) | From-scratch EfficientNetV2-S + AdaCos metric learning verifies patients across CXR and CT scout | AUC 0.9996-0.9998 (CXR, CT scout, cross-modality); 13,664 test patients; private data | No FM used and none needed; test is balanced-negative verification only; validation shares patients with training. The scratch-CNN floor is very strong at 2D |
| 19 | Packhaeuser et al., Sci Rep 2022, arXiv 2103.08562 (older pointer from #1 and #18) | A | Deep verification on ChestX-ray14 re-identifies patients across a decade | AUC 0.9940, retrieval P@1 0.9963; external up to AUC 0.987 | Pre-2024, kept because every newer CXR paper points back to it |
| 20 | "Privacy-preserving action recognition: taxonomy, methods, and privacy-utility trade-offs", arXiv 2608.04501 | P (first 100k of 312k) | Survey; threat-model taxonomy; says adaptive attackers are tested in under 10% of papers (abstract per the HTML), only STPrivacy and MPPAR retrain attackers | Section on foundation-model privacy was headings only in what I read | Use as the citation for "attackers are weak"; do not rely on its FM claims |

Also opened, minor: DiffMI (2504.18015, A: training-free diffusion inversion of face recognition, 84-93% success on "inversion-resilient" systems); LeakyCLIP (2508.00756, A: CLIP
inversion with SD refinement, +258% SSIM on ViT-B-16, membership inferable even from low-fidelity reconstructions); Schwethelm et al. (2403.07588, A: diffusion-prior
reconstruction as a DP auditing tool, data priors change success a lot); PrivHAR-Bench (2604.00761, A: 1,932 videos, 15 classes, nine graduated tiers, pose keypoints and eval code
released; R3D-18 88.8% clear -> 53.5% at the strongest tier; attackers not named in the abstract); "Pervasive blind spot" (2511.02367, A: 508 personal clips, VLMs beat humans at
sensitive-attribute inference); "Who can see through you" (2512.18264, A: input perturbation against VLM attribute inference, no adaptive attacker mentioned).

## 4. Findings by question

**Frozen vs fine-tuned vs LoRA/adapter attackers.** One clean quantitative data point in vision: trained head on frozen CXR-FM features doubles MGH rank-1 (32.6 -> 63.7) and full
fine-tuning of RETFound reaches 82-94% image-level on retina (Nebbia). Nothing compares frozen, LoRA and full fine-tuning on the same backbone and defended data. The 2026
survey warns fine-tuning may erode earlier privacy protection (search snippet, not verified in the section I read). For us a frozen raw-trained attacker already understated
identity by a median 47 KTH points, so "frozen" must be a *floor of the ladder*, never the headline.

**Pretraining, data scale, compute.** Evidence points both ways and is thin. Capability helps on semantic attribute inference (Tömekçe: "scales with general capabilities").
Encoder size barely matters for invertibility (RAE: DINOv2 S/B/L rFID 0.52/0.49/0.52) and adds 0.7-3 points for DINOv2 gait (BiggerGait S -> L; CLIP gains up to 6.5), while the
choice of pretraining objective moves 8.8 points at size S (DINOv2 vs CLIP). Domain pretraining is not needed (natural-image DINO 0.82 ReID@1 on brain MRI; a small CNN that never saw retinal images beat RETFound). Data
quantity on the attacker side matters most through *labelled pairs per identity* (Nebbia timepoints 18 -> 54-70% R@1; FaceLinkGen's near-zero-knowledge setting needs only 30 validation pairs plus a generic proxy transform). No paper plots leakage
against attacker budget; I found no "scaling law of attacks" for vision.

**Do FM attackers break defences older attackers could not?** Yes for face templates (FaceLinkGen vs three published defences), plausibly for latent video anonymisers under white-box
access (Fioresi), no consistent yes for obfuscated HAR video (our data: only person-box blur on VP-HMDB51). The sceptic's reading of the medical papers is that most
"FM re-identification" is near-duplicate and acquisition matching that cheap baselines also do (Engelmann; Bajar's FreeSurfer row; Ueda's scratch CNN).

**Diffusion / inversion reconstruction.** Mature for faces (FaceLinkGen, DiffMI, Rubab) and for generic ImageNet-style features (Zhang, RAE), absent for HAR obfuscations,
skeletons, motion maps or CT/CXR latents. The key unknown is whether low-information HAR outputs (1x2 px motion map, canonical figure x16) are invertible to anything identity-bearing;
my prior is no, which makes RTR a test of mid-strength defenders and of released *latents*, where the black-box/white-box gap in Fioresi (58.8 vs 73.7 cMAP) is the interesting thing.

**Retrieval / gallery attackers.** Strong, cheap and mostly training-free: Nebbia (L2 on frozen features), Bajar (Canberra/cosine, subject linkage), Tai (centroid BPID@1 up to 0.95).
Our harness has no linkage-to-raw-gallery attacker, only classifiers retrained on defended outputs; those are different threat models and both belong in the suite.

**HAR.** Video: Fioresi (VideoMAE/V-JEPA features, gait retrieval on CASIA-B survives the adapter unless a temporal loss is added), BiggerGait (DINOv2 RGB gait beats silhouettes),
Sensors 2026 (appearance hidden, gait intact), ReGenHuman (gait explicitly retained). Skeleton: no FM-based skeleton identity attack found. IMU: nothing for IMU FMs; HARNet is
academic-licence only; closest is the EEG audit (#4).

**CT / CXR.** CXR re-identification is well documented (#1, #2, #18, #19); CT has demographics-from-embeddings (#16) and a CT-scout/CXR cross-modality verifier (#18) but no
published re-identification study on CT FM embeddings that I could find. NLST has ~2.8 studies per patient (73,118 studies / 26,254 patients) so longitudinal CT linkage is
feasible on open CC BY 4.0 data.

## 5. Open checkpoints and datasets (licence, gating, size, 4x RTX A6000 = 48 GB each)

VRAM remarks are my estimates (unverified); licences and sizes come from the pages I opened.

| Item | Licence (as shown) | Gated | Size | Use on 4x A6000 |
|---|---|---|---|---|
| DINOv2-L (facebook/dinov2-large) | Apache-2.0 | no | 0.3B | frozen trivial; LoRA/full FT at 224 px, bs 32 bf16 should fit one GPU |
| DINOv3 ViT-L/16 (LVD-1689M) | "DINOv3 License" (Meta; text not opened) | **yes** (HF login, share info) | 0.3B | fails the ungated requirement; use DINOv2 |
| SigLIP2 so400m-patch14-384 | Apache-2.0 | no | 1B | frozen and LoRA fine; full FT needs checkpointing |
| VideoMAE-L (MCG-NJU/videomae-large) | **CC BY-NC 4.0** | no | 0.3B | research-only; keep derived weights off any commercial path |
| V-JEPA 2 ViT-L fpc64-256 | MIT | no | 0.3B | 64 frames at 256 px is long-sequence; use fewer frames or an attentive probe; larger variants exist (not opened) |
| RAD-DINO | MIT ("research only, not for clinical use") | no | 86.6M (ViT-B/14, from dinov2-base) | trivial; trained on 882,775 CXR from MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX |
| BiomedCLIP | MIT (research-only wording, "deployed use out of scope") | no | n/s | trivial |
| CT-FM (project-lighter/ct_fm_feature_extractor) | Apache-2.0 | not stated | 77.8M SegResNet | whole-volume embedding in seconds; bottleneck is DICOM I/O |
| Merlin (stanfordmimi/Merlin) | MIT tag | not stated; one secondary report says gated | n/s | unresolved, check before planning |
| Qwen3-VL-8B-Instruct | Apache-2.0 | not stated | 9B bf16 | inference ~one GPU (est. 18-20 GB weights); LoRA with checkpointing is tight |
| HARNet / ssl-wearables | academic-use licence (Oxford), UK Biobank pretraining | no | n/s | licence conflicts with "weights to HF" if derived weights are shared |
| CXR Foundation (Google) | API only (per Nebbia) | n/a | n/a | cannot be fine-tuned |
| NLST CT (TCIA) | **CC BY 4.0**, unrestricted download | no | 26,254 patients, 73,118 studies, 11.3 TB DICOM, chunks of ~2,500 participants | open; I/O heavy; check TCIA usage policy for a re-identification clause (not opened) |
| CT-RATE | **CC BY-NC-SA 4.0**, terms forbid re-identification attempts and redistribution | **yes** | 25,692 volumes / 21,304 patients, 21.3 TB | avoid for a patient-linkage study |
| MIDRC, GRAPE, OASIS-2, MIRIAD | described as public in the papers | not checked | CXR 39,749 patients (MIDRC) | licences not opened |
| HAR sets in our tree | KTH, VP-UCF101, VP-HMDB51, Charades, UTD-MHAD | already local | small | licences unchanged from programme 1 |

## 6. Candidate directions (score 1-5; sceptic)

Criteria: **V** value for the robust-suite goal, **D** data availability and licence, **Dt** deterministic verification, **C** compute on 4x A6000 in days, **L** fit for learned
methods, **N** novelty / gap. Total /30.

| Dir | Name | V | D | Dt | C | L | N | Total |
|---|---|---|---|---|---|---|---|---|
| D1 | Pretrain x adaptation ladder as the held-out family (HAR first, then CXR/CT) | 5 | 4 | 4 | 4 | 4 | 3 | **24** |
| D6 | Attacker-budget curves and a saturation stopping rule (shares D1 runs) | 4 | 5 | 4 | 4 | 2 | 4 | **23** |
| D3 | Longitudinal gallery re-ID on CT/CXR latents with learned-metric attackers and hard floors | 4 | 3 | 5 | 3 | 4 | 3 | **22** |
| D2 | Reconstruct-then-re-identify (decoder or diffusion + external recogniser) | 3 | 4 | 3 | 3 | 5 | 4 | **22** |
| D4 | Cross-encoder bridge attacker for released latents | 2 | 4 | 4 | 5 | 2 | 4 | **21** |
| D5 | Zero-shot Video-LLM / VLM semantic attacker | 2 | 5 | 3 | 4 | 1 | 3 | **18** |

Ranking logic: weight V first; D3 beats D2 on the tie because CT is half the programme and D3 is the only CT-native direction; D4 is a variant of "retrained attacker plus an
unnatural assumption".

### D1. Pretrain x adaptation ladder as the unseen family   (24/30)
- Design: for each defended output, attackers = {scratch R3D-18 and ViT, Kinetics/ImageNet-pretrained R3D (pretraining without an FM), **random-init copy of the FM architecture**
  (the control that isolates pretraining), DINOv2-L frozen linear probe, frozen attentive probe, LoRA (rank 8/16), full fine-tune, V-JEPA 2 ViT-L probe and LoRA, SigLIP2 frame}
  x 3 seeds, scenario-split protocols from programme 1. For CXR: RAD-DINO / BiomedCLIP / CXR-Foundation / scratch CNN; for CT: CT-FM / scratch 3D CNN. Report the worst case, the
  rank correlation of defenders across families, and Delta_FM = worst FM minus worst old suite per defender.
- Stage 0 (about 1 builder-day, an estimated 50-100 GPU-hours): 7 existing HAR defenders spanning natural-likeness (raw, blur8, x16, person-box blur on VP-HMDB51, BDQ->x16, canonical
  stick x16, motion map x64) on KTH and VP-HMDB51; pre-register margin from a person-cluster bootstrap power audit (KTH has 25 persons, so expect an MDE of several points; compute it,
  do not assume).
- Biggest risk: unit counts. With 25 persons and 2-4 attackers per family the family-level contrast may be below MDE, and pretraining contamination (VideoMAE and V-JEPA saw Kinetics
  and web video; UCF101/HMDB51 actions overlap) can inflate FM attackers on action-implied attributes.
- Trivial floor that kills it: (a) the random-init copy of the same architecture, or a scratch R3D at 2x budget, matches the FM attacker within MDE on every cell (pretraining is not the
  source of strength, so there is no "FM family"); (b) Delta_FM <= MDE on all defenders (then record it as the result: the old suite was not blind to FMs at these operating points);
  (c) FM gains vanish under the unseen-scenario/unseen-action protocols (shortcut gains, not identity gains).
- Why it fits the programme: it is the cheapest direct test of "does the worst case move programme 1's conclusions", reuses all prepared data, and produces the held-out family for the
  later "defender generalises to unseen attackers" claim. It also tests hypothesis 3 above (natural-likeness).

### D6. Attacker-budget curves and a stopping rule   (23/30)
- Design: on 3 defenders, vary labelled clips per identity (1, 2, 5, all), epochs (12/24/48), parameters (S/B/L), pretraining (none / supervised / FM); fit L(n) = L_inf + a n^-b with
  bootstrap over persons; define "certified privacy at budget B" and a rule "the suite is complete when adding a family or budget moves the worst case by less than MDE".
- Evidence for the premise: label budget matters (Nebbia timepoints, FaceLinkGen near-zero-knowledge with 30 validation pairs), model size does little for DINOv2 (RAE, BiggerGait S -> L; CLIP gains more), objective matters (DINOv2 vs CLIP). No
  published curve exists, which is both the gap and the warning.
- Biggest risk: not enough independent units to tell a saturated curve from noise; extrapolated asymptotes are fragile.
- Floor that kills it: a flat curve from the smallest budget (budget is not an axis worth reporting), or a curve still rising at the largest affordable budget (only a lower bound is
  possible, report that and stop).

### D3. Longitudinal gallery re-identification on CT / CXR latents   (22/30)
- Design: NLST subset (CC BY 4.0, ~2.8 studies per patient) and a public CXR set; threat model = attacker holds earlier scans (gallery), sees a defended latent or image of a later
  scan. Attackers: zero-shot kNN, contrastive metric head on frozen features (Nebbia style), fine-tuned FM, scratch metric-learned CNN (Ueda style). Defenders: the programme's learned
  adapters. Report R@1 / mAP by number of timepoints and time gap, bootstrap over patients.
- Stage 0 (cheap, mostly CPU): header-only audit of DICOM geometry and scanner fields as a metadata attacker, raw downsampled-volume kNN, and organ-volume descriptors as
  FreeSurfer-style floors; then CT-FM zero-shot retrieval on a ~500-patient slice. Check CT-FM / IDC overlap with NLST first.
- Biggest risk: the Engelmann problem (near-identical images, scanner and protocol signatures, short intervals) so that "re-identification" is acquisition matching; plus "task implies
  the secret" (anatomy is both the patient signature and the diagnostic signal); CT-RATE is off the table by its terms; DICOM I/O at 1 TB per 2,500 participants.
- Floor that kills it: metadata-only or raw 16x16 / downsampled kNN reaching about half of the best FM attacker's R@1 at matched gallery, or a scratch metric CNN matching the FM
  (Ueda-like). Then the FM story is dead and only the floors remain, which is still a clean negative result.

### D2. Reconstruct-then-re-identify   (22/30)
- Design: train a decoder from the released artefact to frame(s) on paired (original, output) data the attacker can make by running the public defender on public videos (RAE-style ViT
  decoder for latents; conditional U-Net or SD-refined decoder for pixel obfuscations; template: FaceLinkGen, under 2 GPU-hours per defender on an A6000 as reported). Apply frozen
  external recognisers (clothes-changing re-ID, RGB gait such as BiggerGait-style, face when visible) against a raw gallery, with *no identity labels from the target set*.
  Compare against direct retrained classifiers and direct kNN at matched label budgets (1, 5, 20 clips per identity).
- Biggest risk: the data-processing floor. On the defenders that matter (canonical stick x16, motion map x64) the output holds almost nothing to decode; on weak defenders (blur,
  x16) retrained classifiers already read 95%. The real target is *released latents* (adapters on FM features, CT/CXR codes) and the label-free gallery threat model.
- Floor that kills it: RTR no better than kNN on the released output at matched labels; or RTR equals chance on canonical/motion-map outputs (expected), leaving only a redundant
  confirmation at weak defenders. Positive controls (blur8, x16) must succeed, negative control (all-black) must sit at chance.

### D4. Cross-encoder bridge attacker   (21/30)
- Design: Tai's linear-bridge protocol and Rubab's linear maps, applied to latent-sharing defenders (CXR RAD-DINO vs BiomedCLIP, EEG, WavLM vs others): an attacker with a *different*
  encoder and a small paired set attacks the defended code. Cheap (linear fits).
- Biggest risk: needing paired data means the attacker already owns the inputs; threat model is weak for HAR pixels, plausible only for shared embeddings. Mostly a stricter version of the
  retrained attacker.
- Floor: a direct retrained attacker on the defended code with the same pairs; if bridge <= direct, drop. Keep as an arm of D3, not a stand-alone focus.

### D5. Zero-shot Video-LLM / VLM semantic attacker   (18/30)
- Design: Qwen3-VL-8B (Apache-2.0) at greedy decoding on defended clips for VP-UCF101/HMDB51 attributes and Charades scene secrets; few-shot in-context identity on KTH is not
  realistic (25 identities, long context).
- Biggest risk: on heavy obfuscation a zero-shot VLM is simply out of distribution and at chance; on light obfuscation classical attackers are already at ~95%. It mainly rediscovers
  the task-label prior (VP attributes are largely implied by the action in our results).
- Floor that kills it: the task-label attacker P(secret | action) and the constant predictor. Worth one half-day as a "semantic prior" sanity member, not a pillar.

## 7. Recommendation and what would change my mind

- Run D1 Stage 0 first, with D6's budget factor added at no extra data cost, and D3 Stage 0 (metadata and raw-pixel floors) in parallel because it is mostly CPU. Put D2 behind a gate:
  only if D1 shows the FM family is *not* redundant, or if a latent-releasing defender (adapter on FM features) enters the programme.
- The robust-suite claim should be framed as "the worst case over families and budgets, with the scratch-architecture control and the cheap floors reported next to it". The
  literature gives two reasons for that framing: FM re-identification claims in medicine collapsed under cheap baselines (#2, #3, #18), and in our own data FM attackers are weaker than R3D
  on the defenders that work.
- I would raise D2 if Stage 0 shows FM attackers winning only on natural-image-like outputs (then generative decoding of low-information outputs is the missing check), and drop D1's
  FM members to a single sanity attacker if Delta_FM is within MDE everywhere.
- Gaps to state in any write-up: no IMU-FM or skeleton-FM leakage evidence found; no Video-LLM identity-from-obfuscation evidence found; no frozen/LoRA/full-FT comparison in vision
  privacy found; no attacker-scaling curve found. Two parts of the evidence base (#4, #5) were read only partially.

## 8. Not relied on (seen only in search snippets, not opened)

DRAG / "DRAG ICML 2025" feature-inversion claim (secondary GitHub list only; not located in the paper); LoRA-Leak and "LoRA vs full fine-tuning membership inference" language results
(membership inference on LMs, not attribute inference in vision); MultiPriv (2511.16940); "Encoding versus linear use of patient characteristics in chest X-ray foundation models"
(Diagnostics 2026, RAD-DINO and BiomedCLIP, says no re-identification attempted); CVPR 2026 "What your features reveal" black-box feature inversion (page returned 403);
Membership inference against video LLMs (2604.27002); ICCV 2025 skeleton motion retargeting (Carr et al.); BigGait CVPR 2024 (only through #11).
