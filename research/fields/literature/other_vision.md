# Literature scout: other vision-centric fields for a learned mask / latent

Scout round 1, 2026-11-04. Scope: everything outside robotics and emotion. Every paper below was opened (abstract page, HTML, or PMC full text); items seen only in search snippets are listed separately at the end and are not evidence. Where a page showed only the abstract, the takeaway says so. Scores are mine and are judgments, not measurements.

Frame used throughout: a field qualifies if a learned `z = E(x)` or mask must keep a task label `y` and drop or isolate a nuisance `s`, and a fixed script can score both (task metric plus an attacker for `s`, retrained on the released `z`).

## 0. Headline

The strongest fields I found are not privacy fields in the legal sense. They are **source-nuisance fields in biomedical imaging** (pathology medical center, Cell Painting batch/lab). They have the same maths as the KTH/Charades work (keep y, hide s), public data, and fixed benchmarks with sealed test splits. Both 2025-2026 benchmarks independently report that adversarial removal (DANN) barely helps, which matches our finding that adversaries buy little. The best field that stays inside privacy is **egocentric wearer privacy**: a fresh 2025 benchmark, no defence evaluated, and it reuses our video pipeline (Charades-Ego is one of its test sets).

## 1. Key papers (all opened)

| # | Paper | Link | One-line takeaway |
|---|---|---|---|
| 1 | EgoPrivacy: What Your First-Person Camera Says About You? (Li et al., ICML 2025) | arXiv 2506.12258 | Seven attack tasks on Ego-Exo4D (5,625 clips) and Charades-Ego (4,000 clips); fine-tuned ego-to-ego identity HR@1 79-81%, zero-shot race/gender 65-73%; **no defence is evaluated**. |
| 2 | MIRAGE: full-body bystander privacy for smart glasses (Umair et al., Sep 2026) | arXiv 2609.24537 | Box-masking the body drops silhouette ReID to 10.86% (chance 11.12%); an attacker retrained on the sanitised pose signals still gets 26.2% gait Rank-1 (from 90.25%). Abstract only; hand-built mask, not learned. |
| 3 | BatchBench (Oct 2026) | arXiv 2610.30104 | Public leaderboard on JUMP-CP-2k with sealed test sources 4, 6, 9, 13; ResNet18 0.175, DANN 0.170, best test-time adaptation (ARM-BN) 0.260 overall accuracy; TARGET/REST subsets stay near floor for every entry. |
| 4 | Arevalo et al., Evaluating batch correction methods for image-based cell profiling (Nat Commun 2024) | PMC11297288 (doi 10.1038/s41467-024-50613-5) | Ten methods on five JUMP scenarios, 4 batch + 6 biology metrics (weights 0.4/0.6, includes replicate-retrieval mAP); Harmony and Seurat RPCA top; scVI helps in some scenarios; **in the hardest scenario no method removes batch**. Data CC0. |
| 5 | CellPainTR (Caruzzo, Ye, Sep 2025) | arXiv 2509.06986 | Transformer with source-specific context tokens claims to beat ComBat and Harmony on JUMP and an unseen dataset; abstract only, code on GitHub. |
| 6 | de Jong et al., Current pathology foundation models are unrobust to medical center differences | arXiv 2501.18055 | All 10 public pathology FMs encode the medical center strongly; only one has Robustness Index above 1 (barely). |
| 7 | Kömen et al., Towards Robust Foundation Models for Digital Pathology (PathoROB) | arXiv 2507.17845 | 20 FMs, 4 datasets, 34 centers, 99,392 patches; Robustness Index 0.463-0.877; ComBat +27.4% RI but not consistent downstream; DANN +0.23 pp on average; stain-norm + ComBat reaches RI up to 0.92. |
| 8 | Tömekçe et al., Private attribute inference from images with VLMs (NeurIPS 2024) | arXiv 2404.10618 | Seven VLMs infer an image owner's attributes at up to 77.6% from benign photos; no defences given. |
| 9 | Fan et al., Who can see through you? (Dec 2025) | arXiv 2512.18264 | Adversarial shielding against VLM attribute inference; VPI-COCO (522 images); reports privacy attack rate below 25% with non-private-query accuracy above 88%. Abstract only; attackers not retrained or adaptive per the abstract. |
| 10 | Privacy Blur: quantifying privacy and utility for image data release (Meta FAIR, Dec 2025) | arXiv 2512.16086 | Gaussian blur is reversible enough to break; pixelization/DP-Pix are better; the "discrimination" attack is trained on obfuscated data (the only attack that matches our protocol); privacy tested only on digits/plates as face proxies; utility cost of 1x1 pixelization on ImageNet is -0.49 pt. |
| 11 | Gao et al., Pitfalls of defacing whole-head MRI (Comput Biol Med 2025) | arXiv 2501.18834 | Cascaded diffusion models rebuild faces from defaced MRI closer to the original than a population-average face (484 unseen subjects); defacing also weakens muscle-radiodensity prediction. |
| 12 | Eimon et al., Feature Coding for Scalable Machine Vision (FCTM) | arXiv 2512.10209 | MPEG Feature Coding for Machines test model: about 85% average bitrate cut at preserved task accuracy; privacy appears only as a stated trade-off, never measured. |
| 13 | Eimon et al., Next-generation consumer experience with FCM | arXiv 2512.09232 | 75.90% bitrate reduction versus remote inference; privacy not addressed (abstract only). |
| 14 | InfoDecom (AAAI 2026) | arXiv 2511.13365 | Split-inference defence: strip redundant information from smashed data, then add calibrated noise; claims better utility-privacy than baselines (abstract only; attacks unnamed). |
| 15 | SafeOR: privacy-preserving OR workflow analysis with digital twins (MIDL 2025) | arXiv 2504.12552 | Depth + segmentation twin of OR video matches raw RGB on 38 simulated trials; **no re-identification test**, internal data. |
| 16 | PhysioLatent (Zhou et al., CVPR 2026 workshop) | arXiv 2509.25348 | Latent-space editing of heart-rate in face video; HR error 10.0 bpm, PSNR 38.96; a rare latent (not pixel) method in telehealth video. |
| 17 | Bhutani et al., Preserving privacy and video quality through remote physiological signal removal (Commun Eng 2025) | PMC11977227 | Eleven hand-built edits to hide pulse; best sliding-window averaging gives about 22 bpm error; only 6 of 25 LGI-PPGI subjects are public. |
| 18 | Mercader-Perez et al., Learning what's real: disentangling signal and measurement artifacts (ICLR 2026 FM4Science workshop) | arXiv 2604.09787 | Two-encoder separation of galaxy vs instrument on DESI Legacy and HSC images; abstract gives no disentanglement metric. |
| 19 | DiffSLVA: diffusion models for sign language video anonymization (older, 2023) | arXiv 2311.16060 | Zero-shot diffusion anonymization of signers; abstract reports no recognition or identity metric. Included only because it is the field's reference point. |

Regulation and industry drivers (opened):
- **EU AI Act Art. 5(1)(f)**: emotion inference in workplace and education prohibited, except medical or safety reasons; **5(1)(g)** bans biometric categorisation to infer race, political opinions, etc.; both apply since 2 Feb 2025 (artificialintelligenceact.eu/article/5/).
- **Digital Omnibus**: provisional political agreement 7 May 2026 moves Annex III high-risk obligations from 2 Aug 2026 to 2 Dec 2027 and Annex I to 2 Aug 2028; still needs formal adoption per the source; Art. 5 bans are not delayed (hlc.com/en/publications/eu-legislators-agree-to-delay-for-highrisk-ai-rules).
- **EDPB Opinion 28/2024** (adopted 17 Dec 2024): an AI model is anonymous only if identification and extraction via queries are insignificant; controllers must document it; **no test method is prescribed** (cms.law summary). That is the gap a certified, deterministic evaluator fills.
- HIPAA: I could not open a primary source. The Security Rule NPRM (6 Jan 2025) and the Safe Harbor "full-face photographs" identifier come from search snippets and background knowledge only; treat as unverified.

## 2. Candidate fields (scores 1-5; total out of 30)

Columns: V = mask/latent value, D = open data and licence, B = deterministic benchmark, C = compute on 4x A6000 in days, F = fit for learned methods, N = novelty/gap.

| Field | V | D | B | C | F | N | Total |
|---|---|---|---|---|---|---|---|
| A. Cell Painting / high-content screening: source-invariant latents | 5 | 5 | 4 | 4 | 4 | 4 | **26** |
| B. Computational pathology: center/stain-invariant latents | 5 | 4 | 5 | 5 | 4 | 3 | **26** |
| C. Egocentric / AR glasses: wearer and bystander privacy | 5 | 3 | 4 | 4 | 5 | 5 | **26** |
| D. VLM attribute and location inference from shared images | 4 | 4 | 3 | 4 | 4 | 5 | 24 |
| E. Latent as shareable interface (split inference, FCM) with a privacy axis | 5 | 4 | 4 | 4 | 5 | 2 | 24 |
| F. Sign-language video anonymisation | 5 | 3 | 2 | 3 | 5 | 5 | 23 |
| G. Medical image de-identification (head MRI defacing) | 4 | 3 | 3 | 3 | 3 | 4 | 20 |
| H. Telehealth face video (keep pulse, hide identity) | 3 | 3 | 3 | 5 | 2 | 3 | 19 |

### A. Cell Painting / high-content screening (batch, lab and microscope as the nuisance)
- **Case.** z must keep compound or mechanism of action and drop source, batch and plate. Arevalo et al. (PMC11297288) give a local, scripted metric suite (10 metrics plus replicate-retrieval mAP) on CC0 JUMP data, and BatchBench (arXiv 2610.30104) adds a sealed-source leaderboard that fits the charter's sealed-split rule.
- **Evidence it is open.** In the hardest JUMP scenario no method removed batch; scVI beat linear methods only in some scenarios; DANN did not improve on an unadapted ResNet18 in BatchBench (0.170 vs 0.175). Test-time adaptation helps most, so a learned, label-free adapter is the live question.
- **Scores.** B is 4 not 5 because the BatchBench private test is server-scored with manual evaluation (slow loop); the Arevalo metrics are fully local. Raw JUMP is very large, so we would work from the BatchBench subset or well-level profiles.
- **Biggest risk.** BatchBench TARGET and REST accuracies are near floor (about 1-3% and below 1%) for every method, so only POSCON moves; our gains could sit inside seed noise. It is also not a privacy field, so a reader may call it a different paper.
- **Fit with our findings.** "Where the secret lives decides what hides it": source signal in Cell Painting is spread over intensity statistics and morphology, so a purely linear fix (ComBat, sphering) stalls in the hard scenario; this predicts learned fixes should help there and not in the easy one.

### B. Computational pathology (medical center and stain as the nuisance)
- **Case.** Frozen FM embeddings carry the center more than the biology (de Jong et al., arXiv 2501.18055). PathoROB (arXiv 2507.17845) ships three fixed metrics (Robustness Index, average performance drop, clustering score), so `E(x)` plus a deterministic kNN metric is a complete loop.
- **Evidence.** Hand-built fixes are partial: Reinhard +16.2% RI, ComBat +27.4% RI with mixed downstream effect, DANN only +0.23 pp. A learned supervised projection or concept-erasure-style adapter is not in the table I could read.
- **Scores.** C is 5 because everything can run on frozen embeddings or 99k patches. D is 4 because I verified the benchmark uses three public sources (Camelyon, TCGA-UT, Tolkach ESCA) but did not read their licences.
- **Biggest risk.** Crowded and moving fast (Klaus-Robert Müller is a co-author; a Nature Communications version and industry adoption were reported by search aggregators, which I did not verify). A win on RI could also be "overcorrection" that hurts biology; the downstream drop metric must be part of the headline.
- **Why it keeps the programme's thread.** The nuisance (center) is a retrained-attacker target: train a center classifier on released z and report accuracy over chance, exactly as for identity.

### C. Egocentric / AR glasses (wearer identity and demographics; bystanders)
- **Case.** EgoPrivacy (arXiv 2506.12258) shows fine-tuned attackers recover wearer identity at HR@1 about 80% and demographics from first-person video, and evaluates no defence. MIRAGE (arXiv 2609.24537, Sept 2026) is the first full-body bystander mask I found but it is box-masking, not learned, and I read only its abstract.
- **Why learned.** The wearer's body is barely visible, so the "canonical stick figure" route that won on KTH and Charades is unavailable; the leaking signal is hands, gait-induced camera motion and scene. That is a genuinely new case of "where the secret lives".
- **Scores.** D is 3: Ego-Exo4D and Charades-Ego are gated and I did not confirm their licences; we already hold Charades. B is 4: the attack tasks, splits and metrics exist, the defence protocol (retrained attacker, keep egocentric action) is ours to fix.
- **Biggest risk.** Wearer identity in EgoPrivacy is largely scene identity (scene retrieval is 89% when fine-tuned; the paper's Table 3 as I read it). From the Charades result in `SUMMARY.md`, removing the scene removes the action. Also the keep-task choice (which Ego-Exo4D task) is not decided yet.

### D. VLM inference of private attributes and location from shared images
- **Case.** VLMs infer attributes at up to 77.6% (arXiv 2404.10618); a first defence reports attack rate below 25% at over 88% non-private accuracy on 522 images (arXiv 2512.18264). Attackers there are fixed VLMs, not VLMs fine-tuned on the shielded images, which is the same flaw we found in frozen-attacker protocols.
- **Scores.** B is 3: a VLM attacker is deterministic only if weights are pinned and decoding is greedy; the benchmark is small. N is 5 because the retrained/adaptive-attacker version is untested.
- **Biggest risk.** The defence may be an adversarial perturbation that fits one surrogate VLM and dies under JPEG or a newer model; that is a robustness paper, not a latent paper.

### E. Latent as shareable interface (split inference, MPEG Feature Coding for Machines)
- **Case.** FCM compresses intermediate features for edge-cloud inference; the test model reports about 85% bitrate savings (arXiv 2512.10209) and privacy is only asserted. Adding a measured identity-leak axis (retrained attacker on the bitstream's decoded features, reconstruction attack) to a rate-accuracy benchmark is a clean OpenLatent contribution: score z by bits, task and leakage.
- **Scores.** N is 2: MPEG and industrial groups own the rate-accuracy axis; the privacy attacks are generic (InfoDecom arXiv 2511.13365 claims a better utility-privacy trade-off, but I read only its abstract).
- **Biggest risk.** Standard-driven field with entrenched baselines; the privacy axis may be received as a footnote. Best used as an extra metric in other fields rather than a field of its own.

### F. Sign-language video anonymisation
- **Case.** Hands and face are both linguistic and identifying (arXiv 2311.16060), the sharpest "same pixels carry task and secret" conflict I found. The 2025 comparison of face swap vs avatars proposes a combined expression/identity score (Electronics 2025; I could not open it).
- **Scores.** B is 2: published evaluation is qualitative per the abstract I read; recognition accuracy plus a signer attacker would have to be built.
- **Biggest risk.** Few signers per public corpus. Identity chance is high and the metric is low-powered, like UTD-MHAD (8 people) in our own results. Licences of Phoenix14T and How2Sign unverified.

### G. Medical image de-identification (head MRI defacing)
- **Case.** Diffusion models re-face defaced MRI (arXiv 2501.18834), so deterministic defacing fails to hide the face and also removes anatomy. A learned face-safe latent that keeps non-facial anatomy is plausible.
- **Biggest risk.** For most brain tasks full skull-stripping already works, so the "keep" side is thin; the cited keep-task is muscle radiodensity from head MRI. 3D diffusion attackers are also heavy.

### H. Telehealth face video (pulse kept, identity hidden)
- **Case.** Works exist on both directions (arXiv 2509.25348 edits pulse by latent; PMC11977227 hides pulse with hand-built edits), but data are small (6 public LGI-PPGI subjects in that study).
- **Biggest risk.** My own reasoning, unverified: classical rPPG already averages skin pixels spatially, which removes most spatial identity; if so there is no learned-method gap.

## 3. Recommendation and what I would not pursue

**Top 3 by total (all three at 26; ordering is a tie-break by fit with our existing code and findings):**
1. **C. Egocentric wearer privacy** (V5 D3 B4 C4 F5 N5 = 26): highest novelty, no defence in the benchmark, reuses our video stack. Gate risk: licences.
2. **B. Pathology center-invariant latents** (V5 D4 B5 C5 F4 N3 = 26): best deterministic loop, cheapest compute; crowded.
3. **A. Cell Painting source-invariant latents** (V5 D5 B4 C4 F4 N4 = 26): best data licence (CC0), same code as B; low-signal subsets.

A and B share one codebase (label vs source nuisance); running both costs little more than one.

**I would NOT pursue:**
- **H (rPPG).** Probable hand-built solution plus tiny data.
- **G (MRI defacing).** Weak "keep" signal; skull-stripping is the obvious baseline; heavy 3D attackers.
- **E as a standalone field.** Crowded and standards-driven; use it as an added "bits" and "leakage" metric instead.
- **F (sign language) as a main bet.** Highest-risk, side bet only: benchmark must be built and the signer pool is tiny.

**Dismissed without scoring (reason):**
- **Autonomous-vehicle and dataset-release face/plate masking.** The mask is a detector, so the failure is detector recall, not learning a latent. Privacy Blur (arXiv 2512.16086) shows 1x1 pixelization costs under 1 pt on ImageNet, so hand-built obfuscation is nearly free in utility. CoopScenes (arXiv 2407.08261) was opened but its abstract gives no recall or detection numbers.
- **Fall detection / ambient assisted living.** Search found privacy handled by sensor choice (radar, thermal, LiDAR), unreleased datasets, no vision mask problem. Not opened beyond search results.
- **Operating-room video.** Real data are not public; SafeOR (arXiv 2504.12552) is internal data with no re-ID test.
- **Astronomy.** arXiv 2604.09787 is a workshop paper whose abstract gives no disentanglement metric; community and benchmark are thin.
- **Classroom, workplace, retail, sports.** Drivers are emotion bans under Art. 5(1)(f), which belongs to the emotion scout; I found no benchmark with a fixed keep-label and nuisance label. Document/ID images, satellite and dashcam: one or two searches, nothing worth opening.

## 4. Search-only leads (not opened; not evidence)
GaitGuard arXiv 2312.04470; P2MFDS arXiv 2506.17332 (radar fall dataset, "will release"); SIMBA split-inference benchmark and FSInfo arXiv 2504.10016; HACA3+ arXiv 2604.19474 and NeuroHarm-Kit (MRI harmonisation; a possible fourth source-nuisance field); DefaceQA; EgoBlur arXiv 2308.13093; BMVC 2023 de-identification that keeps rPPG (site did not resolve); Electronics 2025 sign-language IAEP paper (MDPI blocked); Nature 2026 smart-glasses feature article; HIPAA Security Rule NPRM commentary.
