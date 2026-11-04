# Literature scout: affect, emotion and human behaviour

Scout run 2026-11-04 for the fields programme (cluster: facial expression / speech emotion / multimodal affect,
depression-stress-pain, engagement, driver monitoring, autism/child behaviour, gaze, sign language, movement quality,
workplace safety). Recent (2024-2026) sources first; older ones only where a recent paper points back.

Reading status. "Opened" = I fetched the arXiv abstract page, HTML or PDF and read it. Most arXiv fetches returned only the
abstract, so numbers below are abstract-level unless stated. Items seen only in a search snippet are listed in section 5 and
are not relied on. Licence statements come from the dataset's own page where I could open it; otherwise marked "secondary".

## 0. One-paragraph picture

The recurring structure in this cluster is exactly the privacy-obfuscation set-up: a signal that carries a task (emotion,
sign content, gaze, movement quality) and a nuisance that must go (speaker, face identity, iris, body shape). The
literature is thick on methods and thin on honest evaluation. Three facts matter for us:

1. Frozen or pre-trained identity validators are still the norm. The one CREMA-D face-expression paper I could read in full
   (WACV 2025) validates with a ResNet50 pre-trained on the raw training split. A 2026 survey of 32 privacy-preserving
   action-recognition papers finds fewer than 10% test adaptive attackers.
2. "Expression-only" does not mean identity-free. FLAME expression and jaw coefficients alone identify 1,429 speakers at
   61.1% top-1 (458x chance). Speaker verification trained on emotion embeddings alone still works. This is our
   "where the secret lives" finding in a new domain: identity lives in dynamics and style, not only in appearance.
3. Speech is the only sub-field with an official, deterministic, retrained-attacker benchmark (VoicePrivacy). Everywhere
   else the benchmark is either a one-off paper protocol or has to be built.

## 1. Key recent papers (all opened; one-line takeaway)

| # | Paper (arXiv / URL) | Year | Takeaway |
|---|---|---|---|
| 1 | Investigating Identity Signals in Conversational Facial Dynamics via Disentangled Expression Features, arXiv 2510.11223 | 2025 | FLAME expression+jaw coefficients only, Conformer + supervised contrastive: 61.14% on 1,429-way ID (CANDOR). Facial dynamics are an identity signature; an "expression latent" must be attacked, not assumed clean. |
| 2 | Xu et al., Facial Expression Recognition with Controlled Privacy Preservation and Feature Compensation, WACV 2025 (CVF open access; no arXiv found) | 2025 | Closest prior art for field B. CREMA-D: 78.84% expression acc, 2.01% identity leakage (chance 1/91 = 1.10%), R3D utility net, wavelet low/high split. Validator = ResNet50 pre-trained on raw CREMA-D train; text I read does not retrain it on outputs. Limitation they state: needs identity labels, so no in-the-wild sets. |
| 3 | Anon-NET, expression-consistent anonymisation of talking-head video, arXiv 2601.11635 (ICCVW 2025) | 2025-26 | Diffusion inpaint + video-driven animation to keep age/gender/race/pose/expression. Abstract names no unlinkability metric: the evaluation question is open. |
| 4 | NullFace: training-free localised face anonymisation, arXiv 2503.08478 (FG 2026) | 2025-26 | Identity-embedding perturbation in a diffusion inversion; localised so expression regions can be left intact. Abstract only; claims attribute preservation without numbers. |
| 5 | Privacy Beyond Pixels: Latent Anonymization, arXiv 2511.08666 | 2025 | Adapter on a frozen video encoder removes skin colour/gender/clothing from features; near-baseline action/detection utility, 35% less leakage. The only paper I found that is literally z = E(x) with an anonymising adapter. |
| 6 | Privacy-Preserving Action Recognition: Taxonomy, Methods, Trade-offs (survey), arXiv 2608.04501 | 2026 | 32 papers: 65% ad-hoc metrics, under 10% test adaptive attackers, under 15% cross-dataset. Independent confirmation of our frozen-attacker finding. |
| 7 | VoicePrivacy 2024 Challenge Evaluation Plan v2.0, arXiv 2404.02677 | 2024 | The deterministic benchmark: ASV attacker re-trained on anonymised LibriSpeech-360 (semi-informed); emotion = wav2vec2 SER trained per fold on original IEMOCAP, 4-class UAR. Original UAR 69-71; baselines B1-B6 sit at 36-56 UAR with EER 4.5-34. Warns a badly trained ASV "gives a false sense of privacy". |
| 8 | SEF-MK: speaker-embedding-free anonymisation via multi-k-means quantisation, arXiv 2508.07086 | 2025 | WavLM features -> k-means pool -> resynthesis. EER 38.0, WER 3.31, UAR 57.6 (original 70.07). Attacker that knows the k-means pool drops EER to 14-26: pool secrecy matters. |
| 9 | StreamVoiceAnon+, arXiv 2603.06079 | 2026 | Frame-level emotion distillation: UAR 39.7 -> 49.2, EER 49.0, WER 5.77. Keeps about 70% of raw emotion at near-chance speaker verification (SEF-MK in row 8: 82%). |
| 10 | Why disentanglement-based speaker anonymisation fails at preserving emotions, arXiv 2501.13000 (ICASSP 2025) | 2025 | Emotion is lost because the intermediate representation is sanitised too hard; UAR alone is a poor emotion metric. |
| 11 | Privacy vs Emotion Preservation Trade-offs in Emotion-Preserving Speaker Anonymization, arXiv 2409.03655 | 2024 | Systems "excel at one but not both"; speaker verification trained on emotion representations alone partially works: speaker info sits inside emotion embeddings. |
| 12 | Evaluating Identity Leakage in Speaker De-Identification Systems, arXiv 2508.14012 (targeted at ICASSP 2026) | 2025 | EER + CMC + CCA/Procrustes: all SOTA systems leak; best is only slightly better than random; weakest has 45% top-50 hit rate. |
| 13 | SISER: speaker-invariant SER with entropy-based adversarial training, arXiv 2609.02941 | 2026 | ECAPA-TDNN adversary on wav2vec2; IEMOCAP UA 60.63 vs 56.46 without. Speaker removal here improves utility, not privacy; whether a retrained probe confirms removal is not shown on the abstract page. |
| 14 | Trade-offs in Privacy-Preserving Eye Tracking through Iris Obfuscation, arXiv 2504.10267 | 2025 | OpenEDS2019/2020. Blur and noise leave iris-ID at 95-96% (raw 96-98%); rubber-sheet cuts ID to 7/2.5% but gaze (ResNet50) error 3.2 -> 19.6 deg; style transfer 70-84 / 6-18% ID with gaze ~3.2-3.3 deg. Classifiers "frozen" after training; retraining on obfuscated data not stated. |
| 15 | Lensless Gaze Is Not Private by Default, arXiv 2609.09188 | 2026 | 36 subjects. Identity from eye crops 97.7%; released 128-way gaze token 38.1% single-frame but continuous gaze output 72.6% and repeated-frame summaries rise. What you release decides what leaks. |
| 16 | PrivateGaze, arXiv 2408.00950 | 2024 | Learned "privacy preserver" turns full-face images into obfuscated ones usable by a black-box gaze estimator; claims identity and gender protected (no numbers on abstract page). |
| 17 | Towards Privacy-Aware Sign Language Translation at Scale, arXiv 2402.09611 (ACL 2024) | 2024 | Face blur costs 0.6-1.3 BLEU on DailyMoth-70h (28.8 -> 27.5). Identity is NOT measured: the privacy claim is argued, not tested. Face landmarks flagged as possible biometric but also untested. |
| 18 | Faces of Fairness, arXiv 2502.11049; Biased Heritage, arXiv 2503.03446 | 2025 | AffectNet most demographically biased FER set; stereotypical bias propagates more than representational bias; accuracy does not imply fairness. Motivates demographic removal but gives no mask/latent method. |
| 19 | Rehab-Pile: Standardized Benchmark for Skeleton-Based Rehabilitation Assessment, arXiv 2507.21018 (v2 Jul 2026) | 2025-26 | Unified archive of rehab datasets, classification and regression of movement quality; no identity/privacy angle. A ready deterministic benchmark for the utility side. |
| 20 | Adversary-Guided Motion Retargeting for Skeleton Anonymization (PMR), arXiv 2405.05428; MMASD+, arXiv 2408.15077 | 2024 | Skeleton anonymiser via dummy-skeleton retargeting plus adversaries; MMASD+ releases skeleton/mesh/optical-flow of children with ASD as "privacy-preserving" with no re-identification test stated. |

Also opened, supporting: LISTEN, arXiv 2510.10444 (audio LLMs lean on lexical content; relevant as a control that CREMA-D's
fixed 12 sentences removes); CV4Edu 2026 "Do We Need Faces?" (non-archival: body-CLIP 57.3% vs face 57.8% on DAiSEE, 7,919 clips);
Narula et al., arXiv 2001.06103 (2020, Yale/JAFFE adversarial emotion/identity; says datasets with both labels are the missing benchmark);
EASY, Interspeech 2025 (sequential speaker/content/emotion disentanglement; abstract only); arXiv 2602.09904 (federated engagement, no numbers).

## 2. Data availability and licences (affect side)

| Dataset | Access as verified | Identity labels? | Notes |
|---|---|---|---|
| CREMA-D | Open. GitHub (git-lfs, about 7.55 GB), ODbL 1.0 + DbCL; asks for a tracking form. 7,442 clips, 91 actors, 6 emotions x 4 intensities, 12 sentences, audio + video | yes (91) | Same 12 sentences in every emotion, so text carries no emotion signal. Derived datasets inherit ODbL share-alike. |
| RAVDESS | Open. Zenodo 1188976, CC BY-NC-SA 4.0 (page text inconsistently says CC BY once), 25.6 GB, 24 actors, audio + video + song | yes (24) | Small for identity attacks; good as a second set. |
| IEMOCAP | Gated: signed academic release form to USC SAIL (secondary sources) | yes (10) | Needed for VoicePrivacy emotion scoring. |
| DFEW, RAF-DB, AffectNet, Aff-Wild2 | Gated: emailed application from university address / PI-signed licence / signed EULA (about 14 days for Aff-Wild2); all non-commercial (secondary summaries of official pages) | not verified; WACV paper notes in-the-wild sets lack identity labels | Fine for utility-only checks, unusable for a retrained identity attacker unless identities are recoverable. |
| DAiSEE | Request form (Google Form), academic only; arXiv page gives homepage, no licence stated; 9,068 clips, 112 users | users, yes | 4-level labels per state; class prior not on abstract page, so the label-only baseline must be measured. |
| DAIC-WOZ / E-DAIC | Data-use agreement, academics only (USC ICT portal). Portal lists transcripts, audio, "facial features", i.e. derived features, not raw video | n/a | Owners already withhold pixels; nothing for a pixel mask to do. |
| OpenEDS2019 | "Upon request" via challenge page; no licence text found; 152 users | yes | Utility labels: segmentation; OpenEDS2020: gaze. |
| ETH-XGaze | 110 participants; code CC BY-NC-SA; dataset licence agreement not read | yes | Unverified. |
| ASL Citizen | Direct download from Microsoft (secondary: non-commercial); about 83k clips, 2,731 signs, 52 signers | yes (52) | Signer-disjoint official split means identity attack needs our own split. |
| How2Sign / YouTube-ASL | Cited in 2402.09611 without terms; H2S 9 signers; YT-ASL 2,500+ signers | H2S yes, YT-ASL no | |
| UI-PRMD | Open, PDDL 1.0 (project page), 10 subjects, Vicon + Kinect | yes (10) | Small. KIMORE licence not found. |
| MEAD | Public Google/Baidu Drive, no licence text found, 60 actors x 8 emotions x 7 views | yes (60) | Candidate second video set if terms allow. |

## 3. Candidate fields (scores 1-5; higher is better for us)

Columns: V = mask/latent value, D = open data and licence, B = deterministic benchmark available or buildable cheaply,
C = compute on 4x A6000 in days, F = fit for learned methods, N = novelty/gap.

| Field | V | D | B | C | F | N | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| B. Identity-free facial/audio-visual expression (CREMA-D, RAVDESS) | 5 | 4 | 3 | 4 | 5 | 4 | 25 |
| A. Speaker-free speech emotion (VoicePrivacy protocol + CREMA-D) | 5 | 4 | 5 | 4 | 5 | 2 | 25 |
| D. Signer-anonymised sign language (ASL Citizen) | 4 | 4 | 3 | 4 | 4 | 4 | 23 |
| C. Gaze / eye: gaze kept, iris identity removed (OpenEDS, ETH-XGaze) | 4 | 3 | 3 | 5 | 4 | 3 | 22 |
| E. Movement-quality skeleton, subject-free (Rehab-Pile, UI-PRMD) | 3 | 4 | 4 | 5 | 3 | 2 | 21 |
| F. Engagement / depression / pain / autism from face or child video | 4 | 1 | 2 | 4 | 2 | 3 | 16 |

### B. Identity-free facial expression, video (and audio-visual) on CREMA-D / RAVDESS
- Case. A face carries identity and emotion at once, and CREMA-D gives 91 actors x 6 emotions x 12 fixed sentences with
  audio and video, so we can train an attacker that is retrained on the obfuscated output and compare against the
  1/91 identity chance and a "label-only" floor. The only CREMA-D prior (WACV 2025) validates with a pre-trained ResNet50, and
  expression-only coefficients are known to leak 61% identity at 1,429 speakers, so a retrained-attacker re-run of its
  outputs is a cheap, high-information first experiment.
- Mask/latent angle. Candidates to compare: landmark/AU/FLAME masks, temporal-difference + learned pool (our BDQ -> pool
  recipe from KTH), frequency split (their method), adversarial latent with an adapter (paper 5).
- Where the secret lives (hypothesis). Static face structure and speaking-style dynamics (mouth motion across the 12
  sentences) identify; emotion lives in coarse temporal deformation. KTH analogue: style leaks across unseen sentences, so
  test identity on unseen sentences.
- Biggest risk. CREMA-D is acted and exaggerated, so utility is easy to keep and results may not transfer to spontaneous
  affect; no in-the-wild set with both labels is open. Mitigation: RAVDESS (and MEAD if terms allow) as transfer, and report
  the claim as "acted-corpus" only.

### A. Speaker-free speech emotion
- Case. Speech is the one place where a deterministic, retrained-attacker benchmark already exists (VoicePrivacy: EER with a
  semi-informed ASV attacker plus per-fold SER UAR), and emotion is still the weakest axis (best systems keep 49-58 UAR
  vs 70 raw). A z-level view (speaker-free emotion embedding with a retrained speaker probe, no resynthesis) is cheap on
  CREMA-D and can be scored against chance (1/91) and an emotion-label-only floor.
- Where the secret lives (hypothesis). Timbre is global and static; prosody is dynamic. Paper 11 suggests speaker identity
  sits inside emotion embeddings, which predicts adversarial removal alone will under-remove.
- Biggest risk. Crowded and community-owned; VoicePrivacy uses a frozen SER and gated IEMOCAP, and 2026 edition rules may
  differ (plan seen only in search snippets). Novelty would have to come from the normalisation and z-level protocol, not
  from beating baselines on EER.

### D. Signer-anonymised sign language
- Case. Non-manual markers (face) carry grammar, so sign language has the same "face carries both" conflict as emotion, and
  the leading paper blurs faces and never measures identity. ASL Citizen gives 52 signers and 2,731 signs with a downloadable
  baseline (top-1 about 63%; counts and accuracy are from search-result summaries of arXiv 2304.05934, which I did not open), and the keypoint/canonical-skeleton recipe from KTH transfers directly.
- Where the secret lives (hypothesis). In hand shape and size proportions and signing rhythm rather than the face alone, as
  for the canonical stick figure on KTH; face landmarks also identify.
- Biggest risk. 52 signers is small for a retrained attacker (overfit memorisation), the official split is signer-disjoint
  (identity needs a new split), and the Deaf community's priorities on face information (non-manuals) may make a mask that
  removes the face unusable for the actual purpose.

### C. Gaze / eye: keep gaze, remove iris identity
- Case. Cheap (small near-eye images), continuous deterministic utility (angular error), and a recent benchmark paper showing
  the trade-off exists. Paper 15 shows what is released matters (token 38% vs continuous output 73% identity).
- Where the secret lives (hypothesis). Iris texture is separable from pupil position, so a learned mask should do very
  well (rubber-sheet hurts gaze because it destroys geometry, not because it is hard).
- Biggest risk. Data access (OpenEDS "upon request", ETH-XGaze licence unread), several recent competing papers (14-16),
  and frozen classifiers in the benchmark may already look strong; the field is adjacent to affect rather than inside it.

### E. Movement-quality skeletons, subject-free
- Case. Rehab-Pile provides the utility benchmark ready-made and UI-PRMD is public domain; canonical skeletons from our KTH work
  suggest a mask that removes body proportions while keeping movement deviation.
- Biggest risk. Quality scores legitimately depend on body proportions and subject, the datasets are tiny (UI-PRMD has 10
  subjects), and it is mostly a re-run of existing skeleton-anonymisation work (paper 20 and its 2022 predecessor).

### F. Engagement, depression, pain, autism (the clinical/education bucket)
- Case on paper is strong (sensitive, regulated), but see below.

## 4. Sceptic's verdict

- I would NOT pursue F. Every dataset is gated or derived-features-only (DAIC-WOZ ships facial features, not video), sample
  sizes are tiny, labels are noisy (DAiSEE engagement: body CLIP 57.3% vs face 57.8% suggests the signal is weak or the class
  prior dominates; the prior must be measured), and a retrained identity attacker has few subjects to learn from. A result there
  would be unverifiable by anyone else.
- E is the weakest of the pursuable ones: low novelty and the identity/task coupling (body size drives joint angles) may make
  "subject-free" meaningless for quality scores. Keep it only as a cheap utility benchmark spin-off.
- A is attractive because the benchmark exists, but that is also why it is the most contested; do not lock it unless the
  z-level, chance-normalised, label-only-floor protocol is itself the contribution. Combine it with B on CREMA-D so one harness
  yields both modalities and a cross-modal check (remove identity in video, attack with audio).
- Dismissed with no candidate: driver monitoring (no verified open set with licence and identities; DMD has 37 drivers per a
  2026 paper snippet, terms not checked); workplace safety (my searches found no skeleton or pose anonymisation work
  specific to workers; adjacent only: Video-DPRP, ECML PKDD 2025); EEG cross-subject emotion (subject removal there serves
  generalisation, the data are gated, and it is not mask-like); FER demographic-fairness removal (no open FER set I could verify
  has demographic labels; fairness criteria are a different objective than hiding a secret).

## 5. Seen only in search snippets (NOT opened; do not cite)

DICE-FER (CVPRW 2025, Aquib et al.: MI-based identity decoupling, CVF returned 403); Face Video Steganography for Depression
Assessment (ICCVW 2025, CVF 403); VoicePrivacy 2026 evaluation plan (Eurecom listing: stronger attacker, cross-gender trials,
multilingual track); KFAAR (NDSS 2025); GANonymization and AnonEmoFace (landmark GANs, 2023-24); Miao et al. emotion-preserving
disentanglement anonymiser (Computer Speech & Language 2025); Balancing cardiac privacy with quality in video (Commun. Med. 2025) and BMVC 2023 rPPG de-ID; PrivateEyes
(CVPR 2026, third-party note); DiffSLVA 2311.16060 and DiffSign 2412.03878 (sign-language anonymisation by diffusion);
DSFDA 2503.20771; MPDRM depression model (Neurocomputing 2025).

## 6. Suggested next step if B+A are locked

A CREMA-D "identity-free affect" harness: fixed actor-wise and sentence-wise splits; emotion utility from a frozen-then-
retrained suite on the obfuscated data; identity from retrained linear, MLP and video/audio nets (91-way closed set, plus
unseen-sentence and open-set verification EER); normalise as (score - chance) / (raw - chance) and report the label-only
floor; sealed test split. First experiment: re-run the WACV-2025 pipeline (project page fengxxu.github.io/ppfer, linked from
the paper) and measure leakage with a retrained attacker; our prior is that 2.01% rises sharply.
