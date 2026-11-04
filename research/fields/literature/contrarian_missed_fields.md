# Contrarian scout: fields none of the six scouts listed (2026-11-04)

Scope: find a field better than A (source/batch/centre-invariant embeddings, 26), B (robotics distractor/embodiment-free latent actions, 27)
and C (identity-free affect on CREMA-D, 25). Qualifying test: a learned mask or latent `z = E(x)` keeps a task signal and hides/isolates a
nuisance or sensitive property; a deterministic benchmark exists or is cheap to build; data openly downloadable; fits 4x RTX A6000 for days.
Scores are 1-5 on six axes (latent value, open data+licence, deterministic benchmark, compute, fit for learned methods, novelty/gap), total
/30, same rubric as the scouts. Evidence level matches the other scout files: abstract pages and project pages were opened, full texts mostly
not, so numbers are authors' self-reported and the scores are priors (error bar about +/-2).

## 0. Verdict (read this first)

**No candidate clearly beats A-C. Two tie the weakest (26 vs C's 25), none beats A (26) or B (27).** I am about 85-90% confident no
decisively better field exists in what I could reach. Two findings are still actionable:

1. **Embedding-privacy for retrieval (text embeddings in vector DBs; keep retrieval utility, defeat inversion / PII-token recovery), 26.**
   A real contender to replace C, not a clear winner. Reason it is attractive: the latent literally leaves the owner's trust boundary,
   compute is hours not days, data and attackers are open, and the 2024-26 defence papers share the exact flaw our Finding 1 names
   (fixed attackers; adaptive evaluation is absent or one task and three models). Reason it is not decisive: very crowded since 2025,
   and C reuses the existing identity harness whereas this needs a new text harness.
2. **Cross-instrument disentanglement in astronomy (physics latent vs instrument latent, paired galaxy images), 26.** Same shape as A, so
   treat it as a **second domain for field A**, not a new field. Its distinctive asset is exact ground truth: cross-matched pairs of the
   same galaxy plus continuous, measured nuisance variables (PSF size, depth) and a GalSim-simulable tier. That fixes A's stated weakness
   (batch confounded with biology, no ground truth for "what should be invariant").

Everything else scored 24 or lower. The strongest pattern in this round: almost every new "hide X keep Y" field I found is source/instrument
invariance in another costume (astronomy, remote sensing, iBCI sessions, mouse labs, multilingual/source embeddings). That supports A as
the right family and argues for one invariance codebase with several domains rather than more separate fields.

How hard I looked: 34 WebSearch queries (about 20 in extended mode) and about 30 page opens (arXiv abstract pages, project pages, GitHub and
HuggingFace dataset cards, two arXiv HTML full texts). Fields checked: cross-site neuroimaging (fMRI/dMRI/MRI), cryo-EM, high-energy physics
(pileup masks, mass decorrelation), optical and radio astronomy, seismology, climate, remote sensing, molecules/assay bias, materials,
genomics/ancestry, proteins, recommenders, LLM concept erasure, text anonymisation, text-embedding inversion, multilingual/language-identity
erasure, dialect/accent, document AI/PII redaction, spurious correlation and CLIP debiasing, face-template/soft-biometric privacy, iBCI neural
decoding, cross-subject fMRI decoding, animal behaviour (MABe/DooM), industrial anomaly detection, causal representation learning, VAE/
tokeniser latents. **Not checked** (so a miss is possible): federated learning, dataset distillation, sports tracking, world models beyond
robotics, time-series foundation models, EHR, non-speech audio/music, deepfake detection, visual place recognition, person re-ID.

## 1. Scored candidates

| # | Candidate | Latent value | Open data+licence | Det. benchmark | Compute | Fit learned | Novelty/gap | Total |
|---|---|---|---|---|---|---|---|---|
| E1 | **Embedding privacy for retrieval**: keep retrieval utility, defeat inversion and PII/attribute recovery from text embeddings | 5 | 4 | 4 | 5 | 5 | 3 | **26** |
| E2 | **Astronomy physics/instrument latent** (cross-matched galaxy images, HSC x Legacy; extend to Euclid/JWST) | 4 | 4 | 3 | 5 | 5 | 5 | **26** |
| E3 | Mouse behaviour, lab-invariant (DooM, 15 labs, 33 behaviours) | 3 | 4 | 4 | 5 | 4 | 4 | 24 |
| E4 | iBCI session-invariant decoding (FALCON) | 3 | 5 | 4 | 5 | 3 | 4 | 24 |
| E5 | HEP per-particle pileup mask (PhyGHT dataset) | 4 | 3 | 3 | 5 | 4 | 3 | 22 |
| E6 | Cryo-EM conformation latent vs pose/noise (CryoBench) | 4 | 4 | 5 | 3 | 3 | 3 | 22 |
| E7 | Language/source confound erasure in multilingual retrieval | 4 | 5 | 4 | 5 | 3 | 2 | 23 |
| E8 | Spurious-correlation / CLIP debiasing (Waterbirds, CelebA) | 3 | 5 | 5 | 5 | 4 | 1 | 23 |
| E9 | Industrial anomaly detection under lighting shift (MVTec AD 2) | 3 | 3 | 4 | 5 | 3 | 4 | 22 |
| E10 | Climate forced response vs internal variability (ForceSMIP) | 3 | 3 | 5 | 4 | 3 | 3 | 21 |
| E11 | LLM text anonymisation vs LLM attribute-inference (SynthPAI) | 4 | 4 | 3 | 3 | 3 | 2 | 19 |
| E12 | Document-image PII redaction (LeakageBench) | 3 | 2 | 3 | 4 | 3 | 4 | 19 |

Reference: A 26, B 27, C 25 (SYNTHESIS.md).

## 2. The two candidates at 26

### E1. Embedding privacy for retrieval (text embeddings in vector databases / RAG)

**Why it is a bottleneck, not a convenience.** Embeddings are stored and queried by third parties, and inversion recovers text: multi-step
correction recovers 92% of 32-token inputs exactly and full names from clinical notes (Vec2Text, arXiv 2310.06816). Few-shot attacks need
one to 1k aligned pairs and, per the authors, beat every defence they tried (ALGEN, 2502.11308). Image embeddings leak semantics without
reconstruction (SLImE, 2601.22929). The task signal (neighbourhood structure for retrieval) and the leak (semantics) are the same object, so
this is a true utility/privacy trade-off with a rate-like knob (dimension, quantisation, noise, learned projection).

**State of defences (opened).**
- Eguard (2411.05034, v2 Nov 2025): transformer projection + text mutual-information optimisation; reports attack F1 from roughly 82-99% to
  3-6% with utility near unchanged (SST2 94.3 vs 93.8, question retrieval 96.9 vs 96.7). Adaptive test: attacker retrained on defended vectors,
  but only SST2 and three embedders. Ran on 2x RTX A6000.
- SPARSE (2602.07090, Feb 2026): learned differentiable mask over embedding dimensions per user-defined concept + elliptical noise. Six
  benchmarks (STS12, FiQA, STSB, STS14, Quora, NFCorpus), GTR-base/Sentence-T5/SBERT, attacks Vec2Text/GEIA/MLC, PII-Masking-300K (stated
  Apache 2.0) and MIMIC-III. At eps=10 STS12 leakage 60.1 -> 19.3, utility 65.3. **Attackers are fixed; no adaptive or retrained attacker.**
  It is a learned mask on a latent, i.e. our exact object.
- Gaussian noise is the common baseline; Denoising-Aware Inversion (2608.18610, Aug 2026) shows an attacker who models the noise recovers
  text (about +154% relative BLEU over the standard generative inverter). Zero2Text (2602.01757, snippet only) says DP-style defences fail
  against an adaptive threat.

**Gap.** No unified benchmark with (i) retrained/adaptive attackers, (ii) retrieval utility on BEIR/MTEB, (iii) rate controls and a closed-form
floor (LEACE/PCA), (iv) cross-embedder transfer. This is our Finding 1 (frozen attackers understate leakage) plus Finding 2 (rate does most of
the work), in text.

**Deterministic benchmark we can build.** Frozen embedders (GTR-base, bge/e5-small); utility = nDCG@10 on BEIR subsets (SciFact, NFCorpus, FiQA)
and STS Pearson; secret = PII-Masking-300K entity tokens and sensitive attributes (ground-truth annotations, no human judge); attacker suite
retrained on the defended embedding: token-bag MLP (MLC-style), ALGEN-style few-shot alignment + generation, kNN-reconstruction from an
external corpus, and a Vec2Text corrector retrained for the headline number (public code; MSMARCO 8.8M docs, "even a few days on a single
GPU" for hypothesis generation, DDP supported). Report worst-case over the suite, normalised by chance and by the PCA/noise-at-equal-utility
baseline, with bootstrap over documents.

**Risks.** (1) Crowded and moving fast (6+ papers in 2025-26); expect scoop pressure. (2) Because retrieval utility *is* neighbourhood
structure, topic-level semantic leakage cannot be removed (ALGEN, SLImE); the benchmark must define the secret as entities/attributes, not
topic. (3) BEIR sets carry mixed (often non-commercial) licences; MIMIC-III is gated, so avoid it. (4) Eguard already claims 95%+ token
protection, so headroom may sit in stronger attackers and cross-embedder transfer rather than in the headline number.

**Kill test (hours).** Train a linear/MLP projection with an adversarial inversion decoder on GTR-base + NFCorpus; if a retrained token-bag
MLP and ALGEN-style attacker still recover more than 50% of PII tokens at nDCG@10 loss under 3%, or if it cannot beat Gaussian noise at equal
utility against the retrained attacker, drop it.

### E2. Astronomy: physics latent vs instrument latent

**Case.** Joint analysis across DESI Legacy, HSC, Euclid, LSST needs features independent of PSF, depth and pixel scale. "Learning What's
Real" (2604.09787, ICLR 2026 FM-for-Science workshop; Freeman, Hogg et al.) trains a dual encoder with a counterfactual flow-matching decoder on
about 100k cross-matched Legacy/HSC galaxy images: physics latent predicts ellipticity R2 0.934, redshift 0.801, stellar mass 0.755; instrument
latent predicts galaxy depth 0.682, PSF depth 0.667. **Training took about 8 h on 4 H100.** Authors state the split is imperfect (instrument
latent still carries redshift/morphology, physics latent some observing-condition information), several claims rest on visual inspection,
only two instruments, and an untrained ResNet beats the encoders on some properties. This is exactly the gap a rate-bottleneck + retrained-
attacker protocol fills.

**Data.** Multimodal Universe (2412.02527, NeurIPS D&B): collection CC BY 4.0 with per-subset terms to respect; HF org MultimodalUniverse;
HSC subset 88.9 GB, about 73k rows, with per-band PSF shape moments, no redshift field; Legacy Survey DR10 is 124M images hosted at Flatiron
(Globus), so the HSC x Legacy pairing must be built locally (the GitHub page names no cross-match). AION-1 (2510.17960) shows frozen
multi-survey embeddings already transfer morphology across Legacy/HSC at 84-86%, so the headroom is in the *instrument leak*, not in
morphology accuracy.

**Deterministic benchmark we can build (three tiers).** T1 synthetic: GalSim galaxies with known Sersic/ellipticity parameters rendered with
random PSF, noise, pixel scale (exact truth for both physics and nuisance). T2 real pairs: HSC x Legacy cross-match; physics probes
(spectroscopic redshift, ellipticity) and *continuous* nuisance probes (PSF size, depth) retrained as attackers; paired-consistency distance of
the physics latent. T3 transfer to a third instrument (Euclid Q1 / JWST, availability not verified by me).

**Risks.** Requires paired observations (authors list unpaired as future work); physical labels come from SED fitting and are only partly
recoverable from imaging; licence per subset unconfirmed; the cross-match build is real engineering (TB-scale hosting). Ceiling worry is
smaller than in pathology (RI 0.89-0.93) because no strong baseline suite exists yet.

**Recommendation.** Add as T2-style domain to field A so A has one dataset with exact invariance ground truth; do not spend a lock slot on it.

## 3. Checked and scored below 25 (one line each, with the reason)

- **E3 Mouse behaviour, lab-invariant.** DooM (bioRxiv 10.64898/2026.10.18.758876, Oct 2026): hand-labelled video and pose from 15 labs, 33
  behaviours, community challenge, HiDRA "generalizes to labs unseen during training", data and classifiers released as open resources
  (licence not named on the page; full PDF unreadable to me). Pose-based so compute is trivial, but it is A on keypoints, and the authors
  already report cross-lab generalisation, so the gap is unclear. 24; worth a look as a cheap A-domain if the licence is clean.
- **E4 iBCI session invariance (FALCON).** NeurIPS 2024 D&B; five datasets (2 monkeys M1, monkey M2, human H1, human H2 handwriting, finger
  B1) on DANDI, CC BY 4.0; hidden held-out sessions scored on EvalAI (single T4); few-shot methods only marginally beat a static Wiener filter
  on M1, so headroom is real. SPINT (2507.08402) uses dynamic channel dropout (a literal learned/forced mask) and permutation invariance. But
  the nuisance is covariate drift, not a sensitive property, one participant per dataset, and the "hide" framing fits loosely. 24.
- **E5 HEP pileup mask.** PhyGHT (2602.20475, KDD 2026): "Pileup Suppression Gate" is a soft mask over tracks; dataset on Zenodo
  (18746388), two 10k-event files at mu=60 and mu=200, MIT code, dataset licence and per-particle label scheme not stated on the README. Also a
  mass-decorrelation literature (adversarial pivots, 2017 onwards) with no 2025 benchmark found. Small, domain-specific, labels supervised so no
  adversary-vs-rate puzzle. 22.
- **E6 Cryo-EM heterogeneity (CryoBench, 2408.05526, NeurIPS 2024 spotlight).** Five synthetic datasets with exact ground-truth ensembles
  (IgG-1D 100 models/100k images, IgG-RL, Spike-MD 46k structures, Ribosembly 16 states, Tomotwin-100) on Zenodo; ten methods benchmarked. Best
  deterministic ground truth of the lot, but pose and CTF are handled by the physical forward model rather than learned removal, setup is
  heavy (reconstruction pipelines), and a 2025 critique says simulated data lack real-noise features. 22.
- **E7 Language/source confound erasure in retrieval.** LangSAE Editing (2601.04768, ACL 2026) uses a post-hoc SAE to suppress language-linked
  units; "The Medium Is Not the Message" (2507.01234, EMNLP 2025) uses linear concept erasure and reports large gains "often dramatically".
  Open data and exact bitext truth, but the closed-form linear erasure already works and two venues have taken it. 23.
- **E8 Spurious correlation / CLIP debias.** Evergreen but rankings on Waterbirds, CelebA and NICO++ barely correlate (2409.04188 snippet),
  CLIP debias methods report worst-group accuracy in the low 90s; no new gap. 23.
- **E9 MVTec AD 2 (2503.21622).** Lighting is the explicit nuisance, test lighting unseen at training, SOTA below 60% mean AU-PRO, hidden
  ground truth on an evaluation server; data CC BY-NC-SA 4.0 behind a form. Server-gated scoring is not purely local-deterministic. 22.
- **E10 Climate (ForceSMIP, J. Climate Apr 2026; CNN follow-up 2609.05359).** Perfect-model testbed, exact truth from ensemble means,
  "fraction of MSE removed". Data hosting and licence not stated on the project page; it is signal extraction/denoising, so the hide/keep
  framing and retrained-attacker protocol do not transfer. 21.
- **E11 LLM text anonymisation.** Active (Dou et al. 2402.13846, SEAL 2506.01420, GRASP 2608.06526, IntentAnony 2601.04265), evaluated with
  LLM adversaries on SynthPAI. Scores are not comparable across papers (different judges, attacker LLMs, attribute sets), RL on 7-8B models
  plus a large adversary is tight on 4x48 GB, output is rewritten text not `E(x)`. 19.
- **E12 Document-image PII redaction.** LeakageBench (EMNLP 2026 Findings, arXiv 2609.02207): 500 images, 11,954 GDPR-aligned annotations,
  even GPT-5.5 + code interpreter leaks critical PII on 96.8% of pages; but dataset source/licence not stated and no utility metric. 19.
- **Dropped without scoring.** Recommender attribute unlearning (ADV-MULTVAE 2206.04500, 2410.20965; snippets only; established, small
  headroom). Face-template soft-biometric privacy (Melzi et al.; training sets like MS1M are encumbered; snippets only). Concept erasure in
  LLM representations (LEACE guarantee is linear-only, 2503.20098 snippet; LIBERTy 2601.10700 has structural counterfactual ground truth but
  data openness not stated). Genomics/ancestry (real phenotypes gated, only simulation open). Molecules/assay bias (InfoAlign, MICON; no
  deterministic benchmark; assay overlap in ChEMBL below 0.5% of pairs). Materials, seismology, proteins, remote sensing (GeoCrossBench
  2511.02831 is cross-band generalisation costing 11,900 H100-hours, not hide/keep), cross-subject fMRI decoding, radio RFI flagging (mature,
  2019-22 benchmarks), causal representation learning (CausalVerse 2510.14049, synthetic and theory-heavy), VAE/tokeniser latents (analysis
  papers only; no hide/keep benchmark).

## 4. Implications for the lock-in decision

1. The contrarian round fails to find a clearly better field. Lock-in confidence can go up, with the caveat that unchecked areas (list in
   section 0) remain.
2. Keep A, B, C as the shortlist. Two cheap adjustments cost little: (a) add astronomy T1/T2 as an A domain for exact invariance ground
   truth; (b) hold E1 as the named substitute for C if the stress-test scout finds CREMA-D blocking (acted data, small cohort, ODbL
   share-alike) or if the existing identity harness proves hard to reuse.
3. E1 deserves a two-hour kill test regardless; it is the only candidate whose headline claim (learned mask on a latent, evaluated with
   fixed attackers) matches our central methodological finding.
4. Do not generalise the stress-test: all scores here come from abstract/project pages; E1's attacker costs and E2's cross-match build are
   the two numbers most likely to move a score.

## 5. Seen only in search snippets (not opened, not relied on)

Zero2Text arXiv 2602.01757; Platonic Universe 2509.19453; Panopticon 2503.10845; PANGAEA 2412.04204; Fundamental Limits of Perfect Concept
Erasure 2503.20098; Reassessing the Validity of Spurious Correlations Benchmarks 2409.04188; InfoAlign 2406.12056; MICON 2504.09544; AssayMatch
2511.16087; DisPred (ancestry) 2205.04673; ADV-MULTVAE 2206.04500; RedactionBench 2606.18782; "Shadow Queries for Private Retrieval in Vector
Databases" 2609.04767; "Ghost Vectors" 2606.18497; ML-SUPERB 2.0; accent-group fairness (Swain et al., Oct 2025).

## 6. Paper index (opened)

2602.07090 (SPARSE; abs + HTML), 2411.05034 (Eguard; abs + HTML), 2502.11308 (ALGEN), 2310.06816 (Vec2Text), 2608.18610 (Denoising-Aware
Inversion), 2601.22929 (SLImE), 2607.01276 (Embedding Inference Attack), github.com/vec2text/vec2text, 2604.09787 (Learning What's Real; abs +
HTML), 2510.17960 (AION-1), 2412.02527 (Multimodal Universe), github.com/MultimodalUniverse/MultimodalUniverse, huggingface.co/datasets/
MultimodalUniverse/hsc, 2408.05526 (CryoBench) + cryobench.cs.princeton.edu, FALCON (snel-repo.github.io/falcon, bioRxiv
10.1101/2024.09.15.613126), 2507.08402 (SPINT), 2602.20475 (PhyGHT) + github.com/rAIson-Lab/PhyGHT, DooM (sciety.org/articles/activity/
10.64898/2026.10.18.758876), 2601.04768 (LangSAE Editing), 2507.01234 (Medium Is Not the Message), 2503.21622 (MVTec AD 2) +
mvtec.com page, 2609.05359 (CNN forced/internal) + ForceSMIP site (sites.google.com/ethz.ch/forcesmip/about), 2609.02207/ACL
2026.findings-emnlp.2350 (LeakageBench), 2607.17890 (STACE concept-erasure stress test), 2601.10700 (LIBERTy), 2511.02831 (GeoCrossBench),
huggingface.co/datasets/ai4privacy/pii-masking-300k (licence file not read).
