# Literature scout: human-activity-recognition privacy attacks and defenses (2024-2026)

Scout: HAR cluster, 2026-11-04. Programme 2 (adversary suite). Everything below was opened unless marked. Read depth is stated per paper:
**FULL** = arXiv HTML/README read in the sections that matter; **PART** = HTML read via targeted extraction (method, protocol, tables), not every section; **ABS** = abstract/listing page only; **SNIP** = seen only in a search-result snippet, NOT opened (do not cite numbers from these without opening them).
Dates in the arXiv IDs run to 2608/2609 because today is 2026-11-04. Older work is cited only where a recent paper points back to it.

## 0. One-paragraph verdict

Across video, skeleton, IMU, radar and gait, the 2024-26 defense papers I opened nearly all report privacy against **one attacker family that was trained on the raw data and then pointed at the obfuscated data** (PrivDiffuser, ImmCOGNITO, PMR, NTU120-Privacy-21K, GaitProtector, PATN) or a single retrained architecture (SPLAVU, From Pixels to Privacy). The only IMU paper I found that retrains several attacker families on the released data (C-AAE) shows a 2x spread in user-identification F1 between a CNN and LightGBM on the same PAMAP2 release (0.157 vs 0.323), and its own ADPCM-only ablation (a plain quantiser, no learning) delivers about a third to four-fifths of the claimed drop depending on the family (RF 32%, LGBM 41%, LR 83%; my arithmetic on the table as relayed by a fetch summary, to be re-checked against the PDF). A 2026 survey (arXiv 2608.04501, ABS) counts under 10% of privacy-preserving-action-recognition papers testing adaptive attackers. So the gap is real and is largest in **wearable IMU and skeleton**, where data are tiny (cheap suite), licences are mostly open, and no retrained-attacker benchmark exists. The programme-1 floors (label-only LDA, per-window normalisation, rate bottleneck, canonical skeleton) are the main way each direction below can die.

## 1. Protocol audit: who retrained the attacker?

| Paper (venue/ID) | Secret | Attacker(s) | Trained on | Retrained on defended data? | Adaptive/white-box? |
|---|---|---|---|---|---|
| PrivDiffuser (PoPETs 2025, 2412.14499, PART) | gender, weight, ID; MotionSense, MobiAct, WiFi-HAR | one CNN (4 conv + 3 FC) | RAW data ("we train this model on raw sensor data") | no | no; threat model says adversary does not know data was obfuscated |
| CLOAK (2512.12086, PART) | same | same CNN design | not verified | not verified | no |
| C-AAE (2507.18072, FULL via fetch) | user ID; MotionSense (24), PAMAP2 (9) | CNN, LightGBM, RF, LR | anonymised data, 4:1 random split | **yes**, 4 families | no |
| PATN (2511.07242, PART) | gender (MotionSense), age (ChildShield) | fixed CNNs, jointly attacked as 3 archs | raw | no | white-box perturbation vs fixed models |
| ImmCOGNITO (2602.07139, PART) | ID, mmWave gestures (PantoRad, MHomeGes) | Tesla-ID, Pantomime-ID, PointNet++-ID | fine-tuned on original point clouds | no (the text says "same classifier" is applied to original and generated) | no; "stronger adversaries" listed as future work |
| PMR (2405.05428, PART) | actor ID, NTU RGB+D 60 | SGN relabelled to actor (85% top-1 on raw) | original skeletons | not stated | no; utility = MSE only, no action accuracy |
| GaitProtector (2605.12431, PART) | ID, CASIA-B | GaitBase, DeepGaitV2, SwinGait; surrogate ensemble then held-out backbone | pretrained on CASIA-B | no | "adaptive" = re-binarisation only |
| NTU120-Privacy-21K (GitHub README, FULL) | ID, NTU RGB+D 120, 38 subjects | ArcFace, OSNet, YOLOv11 pose MLP/TCN, GaitBase | clean gallery | **no** ("no identity model is trained on anonymized probes") | no |
| SPLAVU (ICLR 2026, 2511.08666, PART) | VISPR / VP-UCF101 / VP-HMDB51 attributes | linear/ResNet attribute classifier; black- and white-box inversion | anonymised static clips | yes (one family) | white-box inversion restores cMAP 73.7 vs 70.5 (raw 76.4) on VP-HMDB51 |
| From Pixels to Privacy (2603.26336, PART) | VP-UCF/VP-HMDB/PA-HMDB attributes | ResNet-50 | anonymised train set | yes (one family) | no |
| Wearable-VLM TGAP (2608.28691, PART; images, not HAR) | sex, income, location | fresh attackers on filtered tokens, 4 frozen LLM families | filtered tokens | yes, multi-family | no |
| Programme 1 (our SUMMARY.md) | KTH/Charades identity | linear, MLP, kNN, LightGBM, R3D-18, ResNet-50, DINOv2 | defended data | yes | frozen understated KTH identity by a median 47 points |

Reading: a protocol that fixes one raw-trained attacker is the norm. Programme 1 already measured that this is the dominant source of apparent privacy on video. The same check has not been run on IMU or skeleton defenders in any paper I opened.

## 2. Key papers (20, with read depth)

### Video / latent / benchmarks
1. **SPLAVU / Privacy Beyond Pixels**, Fioresi, Dave, Shah, ICLR 2026, arXiv 2511.08666. PART. Frozen video encoder plus 3-layer anonymising adapter; self-supervised clip-level privacy term, latent-consistency loss. 35% relative privacy reduction (VISPR cMAP) with utility within 1-2%. VP-HMDB51 (top-1 / cMAP): raw 72.6 / 76.4, SPLAVU 72.1 / 70.5, black-box inversion 67.3 / 58.8, white-box inversion 68.0 / 73.7; chance cMAP 58.3. Gait on CASIA-B is NOT suppressed by the latent-consistency loss (the paper says so). Code MIT. Relevance: the method our video harness already follows; note that "chance" cMAP (58-59) is the same blind floor as in our SUMMARY.md.
2. **From Pixels to Privacy** (token-pruning anonymiser), 2603.26336. PART. Action CLS plus privacy CLS tokens, prune tubelets. Evaluates by retraining an I3D action model and a ResNet-50 privacy classifier on the anonymised train set (so single-family retrained). VP-HMDB51 cMAP 76.6 (no pruning) to 70.3 at 79.6 top-1; harder pruning 60.3 cMAP at 69.0 top-1.
3. **NTU120-Privacy-21K** (Nabeel, Alshiha, Behzad; ACCV 2026; GitHub safwanx/anonymized-not-anonymous, paper "coming soon"). FULL README. 21,600 NTU RGB+D 120 videos, 38 subjects. A multi-cue attacker suite (ArcFace face, OSNet appearance re-ID, YOLOv11-pose MLP, GaitBase silhouette). Identity R1 on cross-view C2: clean 83.1 face / 85.3 re-ID / 10.8 pose / 18.0 silhouette; face blur drops face to 3.4 but re-ID stays 84.7; DeepPrivacy2 body anonymisation leaves re-ID 14.2, pose MLP 9.0 (above the 4.86% action-conditioned null). Single seed, closed-set 38/18/3 identities, frozen attackers. Closest existing thing to a suite; the obvious upgrade is retraining each attacker on anonymised data.
4. **PrivHAR-Bench**, 2604.00761, CC BY 4.0 paper. PART. 1,932 UCF101 clips, 15 classes, 9 parallel privacy tiers (blur, Canny, AES block permutation at B=4/8/16, background-removed), lossless frames, pose keypoints, group splits, eval.py; Zenodo DOI 10.5281/zenodo.19352048. R3D-18 accuracy 88.8 (clear) down to 53.5 (encrypted, no background). No identity labels (UCF101 has none) and the "privacy audit" is ArcFace face detection only. Useful as a data-format template, not as a privacy benchmark.
5. **Privacy-Preserving Action Recognition: Taxonomy, Methods, and Privacy-Utility Trade-offs**, 2608.04501. ABS. PRISMA review of 32 papers 2018-2026: adversarial learning 52%, skeleton 20%, crypto 12%, DP 8%; 10% use a formal privacy definition, 65% ad-hoc metrics, 40% report cMAP inconsistently, cross-dataset generalisation checked in under 15%, adaptive attackers in under 10%. Cite for the gap, not for numbers.
6. **Video-DPRP**, 2503.02132. ABS plus intro. Per-video differentially private random projection (random projection, noise, right singular vectors), tested on UCF101/HMDB51 plus PA-HMDB/VISPR attributes. Relevance: it is a published instance of our "random projection + noise" floor; treat as floor reference, not a direction.
7. **The Pervasive Blind Spot**, 2511.02367. ABS. 508 everyday personal videos from 58 people; VLMs outperform human evaluators at inferring personal attributes, driven by temporal behaviour; explanations are unreliable. Evidence that a VLM belongs in the attacker suite as a zero-shot family.

### Egocentric
8. **EgoPrivacy**, Li, Zhang, ..., Vasconcelos, ICML 2025, arXiv 2506.12258, code MIT, annotations on HF `williamium/EgoPrivacy`. FULL. 5,625 Ego-Exo4D clips (839 participants, 131 scenes) plus 4,000 Charades-Ego clips (112 participants). Seven tasks: gender/race/age, ego-to-ego and ego-to-exo identity retrieval, scene retrieval, moment retrieval. Threat model with four capability levels (zero-shot, fine-tuned, retrieval-augmented, identity-level). Zero-shot foundation models: 73.15 / 65.36 / 79.64 % for gender / race / age. Fine-tuned (1 A100, batch 8, 8 frames, MLP head on CLIP-H/14, VideoMAE, EgoVLPv2, LLaVA-1.5, VideoLLaMA2): race 72.01, age 80.72; best top-1 hit rates 81.2% ego-to-ego identity, 50.31% ego-to-exo identity, 89.21% scene, 15.43% moment. Retrieval-augmented attack adds accuracy by pulling exo clips of the same person. This is an attack-only benchmark: **no learned defender is evaluated**. Note scene retrieval (89%) is easier than identity (81%): scene and identity are entangled, the Charades lesson from our SUMMARY.
9. **Person Identification from Egocentric Human-Object Interactions using 3D Hand Pose**, 2509.16557, CC BY 4.0. ABS-level only (PDF body unreadable by the fetch tool; I could confirm metadata and that Ego4D/MECCANO/FPHA/ENIGMA-51 are cited, nothing else). Do not cite numbers.

### Skeleton / gait
10. **Adversary-Guided Motion Retargeting for Skeleton Anonymization (PMR)**, Carr, Xu, Lu, 2405.05428. PART. Motion/privacy embedding split, privacy embedding swapped with a dummy skeleton; NTU RGB+D 60 (40 actors, single-actor classes, camera 1 for test). SGN relabelled to actor: 85.0 top-1 / 95.87 top-5 on raw. PMR constant dummy 6.48 top-1 at MSE 0.0120; PMR random dummy 12.02; DMR 19.3-20.1; UNet 3.03 at MSE 0.083 (destroys motion). Utility is reconstruction MSE only. Moon et al. (AAAI 2023, 2111.15129, ABS, 2021 background): ID classifier about 80% and gender about 87% on raw skeletons. A PAKDD 2025 follow-up (Carr, Zhao, Xu, Lu, "Explanation-Based Anonymization Methods for Motion Privacy") is SNIP only (dblp listing; a README snippet claimed 81.6 to 21.8 re-ID; not verified).
11. **Probing Identity-Specific Motion Signatures (Baller120)**, 2607.03633, CC BY 4.0 paper. PART. 4,583 NBA free-throw clips, 120 players. MViTv2, VideoMAEv2, UniFormerV2 fully fine-tuned for identity under three regimes (appearance / silhouette / skeleton): silhouette-only and skeleton-only inputs stay competitive with RGB and are more robust to appearance shifts. Evidence that a skeleton or silhouette release is not an identity release; matches our stick-figure result (53% KTH identity before canonicalisation).
12. **GaitProtector**, 2605.12431. PART. Training-free diffusion latent optimisation on silhouette sequences with an ensemble of surrogate gait models and a held-out backbone. CASIA-B (124 subjects): Rank-1 89.6 to 15.0 with 56.7% impersonation success under black-box; downstream utility on Scoliosis1K diagnosis 91.4 to 74.2. The only paper I opened whose design is "train against an ensemble, test on a held-out recogniser family". Attackers are pretrained recognisers, not retrained on protected data, and per-sample optimisation is expensive.

### Wearable IMU
13. **C-AAE: Compressively Anonymizing Autoencoders**, Fujimoto, Nakamura, Arakawa, 2507.18072. FULL (fetch summary). MotionSense (24 users, 50 Hz iPhone) and PAMAP2 (9 users, hand/chest/ankle IMUs, 12 activities). AAE then ADPCM quantiser. Attackers (CNN, LightGBM, RF, LR) are trained and tested on anonymised data. Mean user F1 baseline -> AAE -> C-AAE: MotionSense CNN 0.532 -> 0.259 -> 0.073 (chance about 0.042, privacy criterion missed); PAMAP2 all three IMUs: CNN 0.990 -> 0.250 -> 0.157; LGBM 0.717 -> 0.460 -> 0.323; RF 0.689 -> 0.454 -> 0.321; LR 0.558 -> 0.240 -> 0.193. ADPCM alone: LGBM 0.717 -> 0.555, RF 0.689 -> 0.572, LR 0.558 -> 0.254. Activity F1 loss up to 16 points in the hand+chest case. Two lessons: (a) attacker family changes the privacy conclusion by about 2x; (b) the quantiser, a rate bottleneck, is itself a strong floor (it supplies 32% / 41% / 83% of the baseline-to-C-AAE F1 drop for RF / LGBM / LR, my arithmetic). Caveat: the ADPCM-only rows are my reading of a fetch summary of the table.
14. **PrivDiffuser**, Yang and Ardakanian, PoPETs 2025, arXiv 2412.14499. PART. Guided diffusion with classifier-free guidance and mutual-information regulariser. MotionSense gender: raw attacker 93.52% acc, ObscureNet 53.46, plain diffusion 51.34, PrivDiffuser 49.96; HAR accuracy 96.32 vs ObscureNet 94.96. Attacker is one CNN trained on raw data. Code at github.com/sustainable-computing/PrivDiffuser (artifact badged available/functional/reproduced). **CLOAK** (2512.12086, PART) is the latent-diffusion follow-up, four time-series datasets plus faces, claims up to 7.21% lower utility loss and 5.76% lower privacy loss than SOTA; same style of single-CNN privacy metric.
15. **PATN**, 2511.07242, PART. Predictive adversarial perturbations on MotionSense and ChildShield; attack success 40.11% / 44.65%, EER 8.30 -> 41.65. It is an adversarial-example defence against fixed models, which retraining defeats by construction.
16. **WristPrint**, CCS 2021, SNIP only (search snippets): 353 users, 10 weeks wrist accelerometry, about 96% re-identification risk from a day of data; background for the IMU attack side. Also SNIP: Perelli et al. HCAIep 2024 (HAR features used for person identification from smartphone motion), Capture-24 (dataset record opened, see section 5).

### Radio (mmWave radar, Wi-Fi CSI)
17. **ImmCOGNITO**, 2602.07139. PART. Graph autoencoder (Temporal Graph KNN) for mmWave gesture point clouds. PantoRad (7,208 samples, 21 gestures, 41 participants) and MHomeGes (21,487 samples, 25 volunteers, 10 gestures), 77 GHz IWR1443. ID attack (fine-tuned PointNet++) reaches 84.4% on PantoRad (91.4% best subset); after de-identification Tesla-ID 57.2 -> 4.1, Pantomime-ID 52.2 -> 19.3, PointNet++-ID 84.4 -> 24.9; Tesla-GR gesture accuracy 96.5 -> 87.2. The three ID models are different architectures, so there is a spread (4.1 vs 24.9 on the same release); still frozen.
18. **Towards Trustworthy Wi-Fi CSI-based Sensing**, 2511.20456. PART. Adversarial-robustness benchmark, five CSI architectures, NTU-Fi (HAR, HID) and CSI-Bench (HAR, HID; five environments, commercial IoT devices), with open-source framework. HAR is easy to attack, HID resistant. This is about attacking classifiers, not privacy, but its dataset and tooling inventory (NTU-Fi, CSI-Bench) is reusable.
19. **DP-for-CSI-and-radar** (not directions): Protecting Human Activity Signatures in Compressed 802.11 CSI Feedback (2512.18529, ABS; epsilon-DP stochastic quantiser, simulations, no numbers in abstract); DP Feature Release for Wireless Sensing (2512.20323, ABS; WiMANS, Person-in-WiFi 3D, Resp-CSI); IDG-DP radar (2411.02099, PART; membership-inference threat, not identity from released representations). Also SNIP: SnoopFi ("Defeating CSI obfuscation mechanisms", ScienceDirect 2025) reconstructs sensing signals from obfuscated CSI with limited training samples, i.e. the adaptive-attacker precedent in this modality.

### VLM-based attackers (cross-cutting)
20. **Defending Wearable VLMs Against Private Attribute Inference (TGAP)**, 2608.28691. PART. 3,221 image-question records, attributes location/income/sex/interests; privacy accuracy 56.7 -> 7.4 at relaxed utility 74.4 vs 76.6; freshly trained attackers on filtered tokens over four frozen LLM families. Images from Reddit and VQA sets, not video HAR, but it is the best example of a multi-family retrained evaluation in the 2026 wearable literature.

Not opened but seen: MultiPriv (2511.16940), Through Their Eyes (2508.07658), Thapar ECCV 2020 / ICCV 2021 (egocentric wearer identification from gait in head-motion optical flow, reported 92.5% closed set, defence reaching 12%), Hanisch PoPETs 2023 gait privacy (2203.04179), RelCon (2411.18822), SlotFM (2509.21673), HAR-FM, MASTER (IMU foundation models: encoders exist, I did not verify weights or any identity probe).

## 3. Strongest attacks by modality (what we would put in a suite)

| Modality | Strongest known attacks (numbers as reported) | Family diversity available | Foundation-model attacker? |
|---|---|---|---|
| Video | identity: fine-tuned R3D/ResNet/DINOv2 (programme 1: 95-99% KTH identity through blur/pixelation up to x16; Charades 91% with people removed); ArcFace/OSNet re-ID 85.3% on NTU120 cross-view; attributes: ResNet-50/linear on anonymised features | frame models, 3D video models, face/re-ID/pose/gait specialists (NTU120-Privacy-21K), VLM zero-shot (Pervasive Blind Spot) | VLMs, VideoMAE-v2, DINOv2 (programme 1), EgoVLPv2 |
| Skeleton | SGN actor re-ID 85% top-1 / 95.9% top-5 (NTU60, 40 actors); ID about 80%, gender about 87% (Moon); fine-tuned video backbones on silhouettes/skeletons competitive with RGB (Baller120) | GCN family (ST-GCN, 2s-AGCN, MS-G3D, Shift-GCN, CTR-GCN), SGN, PoseC3D, anthropometric (bone-length) MLP, gait CNNs | none for skeleton found |
| Wearable IMU | CNN user ID PAMAP2 0.99 F1 (all IMUs), LightGBM 0.72, MotionSense CNN 0.53 (C-AAE); gender 93.5% MotionSense (PrivDiffuser); WristPrint 96% day-level re-ID (SNIP) | CNN, LightGBM, RF, LR, kNN/DTW, transformers | IMU motion foundation models exist (RelCon etc., SNIP); no identity-leakage study found |
| Radar / CSI | PointNet++ ID 84-91% on gesture point clouds (ImmCOGNITO); clean CSI HAR/HID models are near-perfect in Trustworthy-WiFi's table (all but BiLSTM and ResNet18 on CSI-Bench) | point-cloud nets, CSI CNN/LSTM/ResNet | none |
| Egocentric | EgoPrivacy fine-tuned: 81.2% ego-ego ID, 50.3% ego-exo ID, 89.2% scene; zero-shot gender 73% | CLIP-H/14, VideoMAE, EgoVLPv2, LLaVA-1.5, VideoLLaMA2, retrieval-augmented attack | yes (zero-shot and fine-tuned) |

## 4. Strongest defenses by modality, and whether anyone attacked them adaptively

| Modality | Defenses (2024-26) | Best reported | Against retrained or adaptive attackers? |
|---|---|---|---|
| Video | SPLAVU latent adapter; token pruning; penalty-driven anonymiser (2504.14301, ABS); DP random projection; block permutation | cMAP about 70 vs raw 76 vs blind about 58-59 | SPLAVU: retrained single family plus inversion; white-box inversion restores cMAP. No suite. |
| Skeleton | PMR retargeting; adversarial frame anonymiser (Moon); explanation-guided DP noise (PAKDD 2025, SNIP) | SGN top-1 85 -> 6.5 | No. Attacker not shown to be retrained. |
| IMU | Olympus/ObscureNet GANs; PrivDiffuser; CLOAK; C-AAE; PATN; PrivCLIP/CFD-HAR user-controllable (2508.03989 PART, 2603.11526 PART; both about hiding user-chosen sensitive activities or attributes; CFD-HAR has small MotionSense/DSADS experiments and measures "identity" from predicted attributes, not with a retrained identity attacker) | gender 93.5 -> 50.0 (PrivDiffuser) | PrivDiffuser/CLOAK/PATN no; C-AAE yes, multi-family, still a single random split |
| Radar | ImmCOGNITO; mmFilter (cited in it) | ID 84.4 -> 24.9 | No |
| CSI | randomised/obfuscated CSI; key-based fuzzers; DP quantiser | n/a | SnoopFi (SNIP) broke obfuscators |
| Egocentric | Thapar 2021 optical-flow perturbation (SNIP); bystander face obfuscation | wearer ID down to 12% | Not against fine-tuned foundation models; EgoPrivacy evaluates no defense |
| Gait silhouette | GaitProtector (ensemble surrogates, held-out backbone) | Rank-1 89.6 -> 15.0 | held-out backbone yes, retraining no |

Programme-1 mechanisms to carry over: the "task implies the secret" (action implies VP attributes; in IMU the activity is a gait cycle, so gait carries identity; in Charades-Ego the scene is the identity), and a deterministic floor (canonical stick figure, rate bottleneck) that beats or ties learned adapters.

## 5. Datasets: licence, size, fit

All numbers from pages I opened unless marked.

| Dataset | Modality | Units | Size | Licence / access | Notes for our use |
|---|---|---|---|---|---|
| KTH | video | 25 people, 6 actions, 4 scenarios, 2,391 sequences | about 1.1 GB | "publicly available for non-commercial use" (no named licence) | in harness |
| UCF101 | video | 13,320 clips, 101 classes | - | not re-opened | no person ID; PrivHAR-Bench subset in 15 classes |
| HMDB51 / VP-UCF101 / VP-HMDB51 / PA-HMDB | video | clip-level privacy attributes | - | not re-opened | in harness; attributes mostly action-implied (SUMMARY finding 6) |
| UTD-MHAD | RGB-D + IMU + skeleton | 8 subjects (from SUMMARY) | - | page 404, not verified | tiny identity set; each subject a separate session |
| Charades / Charades-Ego | video / ego+exo | about 90 people; Ego: 4,000 clips, 112 participants | - | AI2 terms, not re-opened | scene = identity (SUMMARY) |
| Ego-Exo4D (EgoPrivacy split) | ego+exo | 5,625 clips, 839 participants, 131 scenes | large | licence agreement, not opened | annotations on HF (`williamium/EgoPrivacy`), repo MIT; code files absent when I fetched it |
| NTU RGB+D 60 / 120 | RGB, depth, IR, 3D skeleton | 56,880 / 114,480 samples, 60 / 120 classes; 40 actors in NTU60 (PMR paper), NTU120 subject count not on the page I opened | large | **academic research only; no redistribution, no derived datasets, no commercial use without ROSE Lab permission; account + request form** | cannot upload derived data to HF; weights probably fine but ask the owner before releasing anything derived |
| PAMAP2 | IMU x3 + HR | 9 subjects, 18 activities, 100 Hz | 656 MB | CC BY 4.0 | very few subjects |
| UCI-HAR | waist smartphone | 30 volunteers, 10,299 windows, 50 Hz | small | CC BY 4.0 | official subject-disjoint split |
| WISDM (smartphone+watch, biometrics) | phone+watch IMU | 51 subjects, 18 activities, 15.6M rows, 20 Hz | - | CC BY 4.0 | UCI page says usable for behavioural biometrics: best open identity set |
| HHAR | 8 phones + 4 watches | 9 users, 43.9M rows | - | CC BY 4.0 | device heterogeneity = device-fingerprint confound |
| MotionSense | iPhone pocket IMU | 24 subjects, 6 activities, 15 trials each, 50 Hz | small | MIT | per-subject gender/age/weight/height table; 14 M / 10 F |
| Capture-24 | wrist Axivity AX3 | 151 participants, about 4,000 h (2,500 h labelled) | 6.4 GB zip | CC BY, **record asks users not to try to identify individuals** | licence/ethics conflict for a re-ID attack; use for utility/scale only or ask the owner |
| PantoRad / MHomeGes | 77 GHz radar point clouds | 7,208 / 21,487 samples; 41 / 25 people | - | licence not verified | from ImmCOGNITO |
| NTU-Fi (HAR, HID), CSI-Bench | Wi-Fi CSI | see 2511.20456 Table 1 | - | licence not verified | CSI-Bench has 5 environments |
| CASIA-B | silhouettes | 124 subjects, 11 views | - | request, not verified | for gait only |
| Baller120 | video | 4,583 clips, 120 players | - | NBA broadcast; not verified | likely not redistributable |

HF rule check: public data are never re-uploaded; only NTU RGB+D adds a derivative-data restriction.

Deterministic privacy-utility benchmarks that exist: PA-HMDB, VISPR, VP-UCF101, VP-HMDB51 (video attributes, no identity), PrivHAR-Bench (2026, deterministic transforms, no identity), EgoPrivacy (ICML 2025, attacks only), NTU120-Privacy-21K (ACCV 2026, frozen multi-cue attackers, 38 subjects), TVRID ICPR 2026 competition (2605.04977, ABS: 86 identities, top-view RGB-D re-ID, server-side scoring; no defender framing). None for wearable IMU, skeleton or radar with a retrained-attacker protocol, which is the gap.

## 6. Candidate directions (scored 1-5; 5 is best; "compute" 5 = cheapest)

Columns: V = value for the robust-suite goal; D = data availability and licence; Det = deterministic verification; C = compute on 4x RTX A6000 in days; L = fit for learned methods; N = novelty/gap.

| # | Direction | V | D | Det | C | L | N | Sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H-A | Wearable IMU: retrained-suite audit of 2024-25 sensor obfuscators, then leave-one-attacker-family-out defender | 5 | 5 | 5 | 5 | 3 | 4 | 27 |
| H-B | Re-attack programme-1 video/pose releases and a SPLAVU latent adapter with a stronger suite (video FM, VLM, pose-sequence, gait, retrained) | 5 | 5 | 5 | 4 | 2 | 3 | 24 |
| H-C | Skeleton identity on NTU RGB+D 60/120: learned retargeting vs canonical-skeleton floor under a retrained GCN suite | 4 | 3 | 5 | 5 | 4 | 3 | 24 |
| H-D | Radar and CSI identity obfuscation (ImmCOGNITO-style) with retrained and family-diverse attackers | 4 | 3 | 4 | 5 | 4 | 4 | 24 |
| H-E | Egocentric wearer-identity defender against the EgoPrivacy suite (including retrieval-augmented attack) | 4 | 2 | 4 | 3 | 4 | 5 | 22 |

### H-A: wearable IMU audit plus held-out-family defender (recommended first)
- What: take the published obfuscators with code or simple re-implementations (ObscureNet/Olympus-style GAN, PrivDiffuser [code public], CLOAK, C-AAE, PATN) plus programme floors, release the obfuscated windows, and attack with a suite retrained on the release: 1D-CNN/ResNet, transformer, LightGBM on statistical + FFT features, kNN/DTW, linear/MLP probes on pretrained IMU encoders, plus a denoise/inversion attacker (SnoopFi analogue). Report worst-of-suite per defender. Second stage: train a defender against K-1 families and test on the held-out family, rotating (leave-one-family-out), which is the charter's "generalises to unseen attackers" test.
- Secrets: closed-set user identity on held-out trials/sessions (PAMAP2, MotionSense, UCI-HAR, WISDM, HHAR); gender/age as user-level attributes (bootstrap over users, group splits).
- Data and licence: CC BY / MIT, small (hours to tens of GB), no access friction. Capture-24 only for scale and with the owner's no-identification clause respected.
- Determinism: hashed user/trial splits, fixed windows, LightGBM and CNN seeds; each config one result.json. IMU compute: the whole audit is on the order of 20 GPU-hours (hundreds of fits of a few minutes).
- Why it could change conclusions: C-AAE already shows a 2x spread between CNN and LightGBM on one release and that quantisation alone gives about half the gain; PrivDiffuser/CLOAK/PATN use a single raw-trained CNN and gender with 24 users.
- Biggest risk: power and "gender with 24 users": there are 24 independent units, 14/10 imbalance, and window-level splits can let the "gender" classifier memorise users. Second: device or session offsets (gravity orientation, sensor bias) can make "identity" an acquisition artefact (HHAR heterogeneity, PAMAP2 per-subject placement), the CREMA-D channel lesson.
- Trivial floors that would kill a learned defender: label-supervised LDA to C-1 dims (6-12 activity classes means 5-11 dimensions that carry little identity); per-window de-meaning and accelerometer-magnitude only; low-pass plus 5x downsample; noise/dither; PCA; the ADPCM quantiser alone. The learned arm is only interesting on utility the floors cannot give (held-out activity classes, finer gait features, step counting) at matched worst-suite leakage, as in programme 1 Focus A.

### H-B: strengthen the attack on what we already released (internal, no new data)
- What: run the charter's falsifiable core on programme-1 representations: KTH/UTD/Charades stick figures (canonical, one posture per clip, x16/x32), BDQ->pool x16..x64, plus a SPLAVU-style adapter on VideoMAE/V-JEPA features with identity labels (KTH, Charades). Add attacker families that programme 1 did not use: pose-sequence GCN/transformers on the redrawn skeletons (timing and proportion leakage), video FMs fine-tuned (VideoMAEv2/UniFormerV2/MViTv2 as in Baller120), a gait recogniser on silhouettes rendered from the stick figure, face-free appearance re-ID (OSNet), and a VLM prompted zero-shot for attributes (Qwen/InternVL class). Report worst case and the delta against the programme-1 best-of-two.
- Value: directly answers "does the suite move programme 1's headline (canonical stick figure 13.7 identity at x16; 8% unseen scenario+actions at x32)?". Even a null ("no move") is a robust claim.
- Risk: attackers saturate at chance on the coarsest releases (1-2 px), so only the middle of the curve can move. VLM nondeterminism: fix temperature 0 and a fixed prompt set.
- Floor/kill: this is attack-side; the controls are black-box 4.0% identity, shuffled labels, brightness-only 22.4%. Attribute leakage is bounded below by the action-only predictor (blind cMAP about 58-59).

### H-C: skeleton privacy on NTU RGB+D
- What: NTU60 (40 actors) and NTU120 with an unseen-action identity split (train on 60 actions, test on the other 60), attackers SGN, CTR-GCN, ST-GCN/2s-AGCN, PoseC3D, bone-length MLP, trained on the released skeletons; defenders PMR, Moon adversarial frame anonymiser, and a learned retargeter versus the canonical-skeleton floor from programme 1.
- Risk: the canonical redraw (template bone lengths, one posture per clip, temporal quantisation) is deterministic and was the best KTH floor; a learned retargeter has to beat it, and PMR reports only MSE as utility (needs action accuracy). NTU licence: academic only, derived datasets prohibited, so no HF release of derived skeletons; weights would need owner confirmation.
- Floor: canonical skeleton + class-LDA on joints + PCA + time quantisation.

### H-D: radar / CSI
- What: ImmCOGNITO-style obfuscation on PantoRad/MHomeGes and NTU-Fi/CSI-Bench HID, with at least point-cloud nets (PointNet++, Tesla, Pantomime), kNN on handcrafted micro-Doppler/point statistics, and retrained versions; plus a de-obfuscation attacker for CSI obfuscators.
- Risk: licences not verified, ImmCOGNITO code availability unknown, room/session confound (identity = environment), little domain knowledge in the team; HID in NTU-Fi is near-saturated in the clean setting.
- Floor: voxel quantisation / point subsampling / jitter plus gesture-LDA.

### H-E: egocentric wearer privacy
- What: first learned defender on EgoPrivacy (no defenses evaluated in the paper): transformation on the ego clip (stabilise head motion, hand mask, learned pool), trained against zero-shot and fine-tuned CLIP/VideoMAE/EgoVLPv2 families plus the retrieval-augmented attack, test on Charades-Ego as OOD and on a held-out family.
- Risk: access to Ego-Exo4D, heavy compute (VideoMAE fine-tunes on about 9.6k clips per attacker, EgoPrivacy used one A100 each), and scene = identity (scene retrieval 89% vs identity 81%) means hiding the wearer can reduce to hiding the room, as in our Charades finding; the retrieval-augmented attack uses exo videos the defender cannot touch.
- Floor: ego-motion stabilisation, hand mask, downsample or frame difference, scene-label-only.

## 7. Recommended order and what a first week looks like
1. H-A stage 1 (audit only, floors first, no defender training): MotionSense + PAMAP2 + UCI-HAR + WISDM. Gates: raw identity attacker well above chance and above activity-label-only; constant and shuffled controls at chance; MDE from the real number of users; run per-window de-meaning, magnitude-only, downsample, LDA floors before any published defender. If the retrained suite leaves the published defenders at or below floor leakage, stop and write the negative.
2. H-B in parallel on cached representations (it is cheap attack-side work and uses existing result files).
3. H-C only if H-A shows the suite moves conclusions, because skeleton attacks and floors are the same machinery on a bigger dataset.
H-D and H-E stay as parked options (data friction).

## 8. What I would NOT pursue
- Membership-inference and DP-accountant work for HAR (IDG-DP radar, CSI DP quantiser, DP feature release): the threat is not identity from the released representation and the abstracts I opened report no attack numbers.
- Encryption/steganography families (PrivHAR-Bench AES permutation, StegaVAR 2512.12586 [ABS]): threat model is a key holder, not a retrained attacker; PrivHAR-Bench has no identity labels.
- Sensitive-activity hiding (PrivCLIP, CFD-HAR, replacement autoencoders): the secret is a subset of the task label, so deleting classes from the label space is the trivial floor.
- Adversarial-example robustness of HAR classifiers (Trustworthy Wi-Fi, skeleton query attacks 2501.16843 [title only]): security of the classifier, not leakage.
- Federated label leakage (2505.20924, title only) and event-camera re-ID (AnonyNoise 2411.16440, title only): different threat or modality outside the harness.
- Another new single-attacker video anonymiser (token pruning, penalty-driven, frequency perturbation): the literature has 10+ and the comparison protocol, not the method, is the missing piece. Training a bigger GAN/diffusion obfuscator on IMU before the audit also qualifies.
- Capture-24 re-identification: licence record asks users not to identify individuals.

## 9. Caveats on this scout
- About 30 searches; WebFetch could not parse PDFs, so I read arXiv HTML where it existed. PrivDiffuser/CLOAK/PATN/PMR/ImmCOGNITO/GaitProtector were read through targeted extraction, not line by line; the protocol facts in section 1 come from those extracts and should be re-checked against the PDFs before being cited as quotes.
- Dataset licences for UCF101, HMDB51, UTD-MHAD, Charades, Ego-Exo4D, CASIA-B, PantoRad, MHomeGes, CSI datasets were not opened.
- I did not find "Exposing and Erasing Identity in Skeleton Motion" (a title surfaced on a GitHub list) or any IMU foundation-model identity probe; absence in my searches is not evidence of absence.
- ArXiv IDs in the 26xx range are accurate for this timeline (today 2026-11-04) but I could not independently confirm acceptance venues beyond what each page states.
