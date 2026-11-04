# Methods line: what transfers across fields (scout report, 2026-11-04)

Scope: learned task-preserving / attribute-removing representations `z = E(x)`, 2024-2026, plus the strongest evaluation-protocol papers. Method: WebSearch to locate, WebFetch to open. **Evidence level: every paper below had its arXiv abstract page (or ISCA page) opened; I did not read full texts, so numbers are the authors' self-reported abstract/page numbers and are marked "self-reported" where used.** Papers whose pages gave no numbers are said to give none. Anything I only saw in a search snippet is excluded from the tables and listed under "leads, not opened" at the end. `docs/harness.md` does not exist in the repo (only `docs/design`, `docs/research`), so harness constraints were taken from the charter and `results/SUMMARY.md`.

Our own anchor results (from SUMMARY.md) that the literature is read against:
1. Frozen-attacker protocols understate leakage (KTH: median 47 points). Retrained attackers are the only honest ones.
2. Rate (number of numbers per frame) does most of the work: at 1-2 numbers/frame the adversary adds nothing (a0 == a-adv); at ~15 numbers/frame it buys ~15 identity points.
3. The best KTH result is a hand-built canonical stick figure, which cannot transfer outside articulated bodies.
4. Where identity is the scene (Charades), person-level obfuscation fails.

---

## (a) Map of method families and evidence of transfer

| Family | Fields with evidence (opened papers) | Transfers? | Notes |
|---|---|---|---|
| **Rate-limited bottleneck (low-dim / VQ / pooling), no adversary** | speech (Privacy PORCUPINE, Interspeech 2024), robotics/games latent actions (LAPA 2410.11758, discrete VQ over frame pairs), our video results; theory 2608.11003 | **Yes, structurally.** It is a loss-free architectural knob, applies to any modality. | Privacy-specific evidence is thin: PORCUPINE shows codebook-usage skew *leaks* (uneven information rate), so plain VQ is not automatically private; fix = space-filling VQ + occurrence normalisation. LAPA is utility-only (no leakage measured). |
| **Adversarial / minimax removal (gradient reversal, GAN)** | video (52% of 32 PPAR papers, survey 2608.04501), EEG (NeuroPriv 2609.00390), fair representations (FRG 2510.21017, NeurIPS 2025), speech eval (2508.09803), text (LLM adversary, 2402.13846) | **Yes in recipe, not in guarantee.** Same loop everywhere; protection only holds against the attacker family used. | Survey: <10% of papers test adaptive attackers; only 10% use a formal privacy definition. NeuroPriv (self-reported): task 0.788 -> 0.781 but identity 0.692 -> 0.206, where chance is 1/36 = 0.028, i.e. still ~7x chance. |
| **Closed-form linear erasure (LEACE, ISP)** | text/embeddings (LEACE 2306.03819), face-identity in CLIP/DINOv2 (ISP 2604.05296), speech (2605.01381 evaluates LEACE on HuBERT) | **Yes, best transfer evidence.** Needs only frozen features + labels; deterministic, no training loop. | Guarantee is for *linear* probes on the *fit* distribution. 2605.01381 (speech/text): LEACE "performs well on containment and disentanglement but struggles to generalise to unseen data". ISP (self-reported abstract): reduces linear identity access to near chance with small utility loss, transfers CelebA-20 <-> VGGFace2. |
| **Nonlinear erasure by distribution matching (MMD/orthogonal projection)** | text only (LEOPARD 2507.12341, ECAI 2025) | Plausible, unproven outside NLP. | Works on frozen embeddings, discrete attributes; cascaded after LEACE. Page gives no numbers. |
| **MI / conditional-MI / dependency suppression** | image fairness VAE (CAD-VAE 2503.07938, AAAI 2026), generic dependency suppression (2603.04224), latent video adapter (AAM 2511.08666) | **Loss-level yes; evidence mixed.** | None of these pages gave numbers except AAM ("35% reduction in privacy leakage", utility near baseline, ICLR 2026). Estimator quality (kNN density, CLUB-type bounds) is the failure point; none of these papers were verified against retrained attackers on my reading. |
| **Information-bottleneck with exact privacy constraint** | theory (2608.11003, cs.IT) | Not usable as a method yet. | ADMM with convergence proof; page reports no experiments. Useful as a statement that rate and relevance alone cannot guarantee independence. |
| **Latent adapter on frozen foundation encoder** | video foundation models, 5 tasks incl. detection and anomaly (AAM 2511.08666) | **Yes, by construction** (any frozen encoder). | Closest published sibling of what a field-agnostic harness would do. Attackers used are not described on the abstract page. |
| **Attention / token selection** | video ViT (2603.26336, CVPR) | Modality-specific recipe (needs a tokeniser and two heads). | No numbers on the page. |
| **Generative (diffusion) anonymisation** | faces (ID2Face 2510.24213) | Weak. Domain-specific generators, evaluated by fixed pretrained recognisers. | Page reports no numbers. |
| **Attacker-in-the-loop LLM rewrite** | text (Staab 2402.13846, ICLR 2024; SEAL 2506.01420, NeurIPS 2025) | Pattern transfers (anonymiser vs inference model), implementation does not. | SEAL (self-reported): 8B students match a GPT-4 anonymiser trade-off on SynthPAI. SynthPAI is LLM-judged, so not deterministic. |
| **Skeleton / hand-built transforms** | video (20% of PPAR papers; ours: canonical stick figure) | **No.** Needs articulated bodies. Survey: ~85% accuracy, appearance discarded. | Honest label: this is a trick, and it is our current best on KTH. |
| **Machine unlearning** | LLMs and classifiers (2503.06991, 2505.16831) | Contributes *evaluation* ideas, not an encoder. | Both say logit-level metrics give false security; forgetting is often reversible. |
| **Differential privacy** | video (PPAR survey only) | Survey reports DP methods often under 70% utility. | No DP-representation paper opened (gap). |
| **SAM2 / segmentation masks** | none found for privacy 2024-26 | Unknown. | Gap. |

**Transfer verdict.** Three families have evidence in 3+ unrelated fields: (i) adversarial removal (as a recipe), (ii) rate bottlenecks (as an architecture), (iii) frozen-feature linear erasure (as a closed form). Everything pixel-, face- or skeleton-specific is a one-dataset trick.

---

## (b) Ranked learned methods to implement as generic encoders E(x)

Ranking = (transfer evidence) x (value as a control or baseline) / (cost). All accept `x` of any shape and a label pair (task `y`, private `s`) and return `z` with an explicit bit/number budget.

| # | Method | Why it earns a slot | Paper(s) | Cost |
|---|---|---|---|---|
| 1 | **Rate-limited bottleneck, no adversary** (pooling / low-dim / VQ with occurrence-normalised codebook), rate as the swept axis | Our own finding says rate does most of the work; the harness must always carry this control or any adversarial gain is unattributed. | PORCUPINE (Interspeech 2024, [ISCA](https://www.isca-archive.org/interspeech_2024/vali24_interspeech.html)); LAPA [2410.11758](https://arxiv.org/abs/2410.11758); theory [2608.11003](https://arxiv.org/abs/2608.11003) | Low (~100 lines; VQ + usage normalisation is the only subtlety) |
| 2 | **Adversarial encoder with attacker restarts / retrained evaluation attackers** (gradient reversal or alternating minimax, adversary re-initialised every k steps, report only retrained attackers) | Most-published recipe in 4 fields; our harness already has it (BDQ a-weights). Need it as the field-agnostic default. | NeuroPriv [2609.00390](https://arxiv.org/abs/2609.00390); FRG [2510.21017](https://arxiv.org/abs/2510.21017); eval caveats [2508.09803](https://arxiv.org/abs/2508.09803) | Low-medium (already built for video; needs modality-agnostic attacker heads) |
| 3 | **Frozen-encoder privacy adapter** (small module on a foundation encoder; MI loss to remove `s`, co-training loss for `y`, latent-consistency loss for unseen tasks) | Makes the harness cheap per field: one frozen backbone + one adapter. Published, ICLR 2026. | AAM [2511.08666](https://arxiv.org/abs/2511.08666) | Medium (needs per-field backbone; losses are small) |
| 4 | **LEACE / ISP closed-form projection** on frozen embeddings | Deterministic, zero-training reference: any learned method must beat it. Also tells us how much leakage is linear. | LEACE [2306.03819](https://arxiv.org/abs/2306.03819); ISP [2604.05296](https://arxiv.org/abs/2604.05296) | Very low (a covariance solve) |
| 5 | **LEOPARD nonlinear projection** (MMD between class-conditional distributions, learned orthogonal projection, rank as knob) | The only nonlinear eraser with a stated rank knob that fits a "frozen z" harness; cascade after #4 shows what linear erasure missed. | [2507.12341](https://arxiv.org/abs/2507.12341) | Medium |
| 6 | **Dependency-suppression VAE / CMI** | Worth one trial only if a field has a correlated task and private attribute (CAD-VAE explicitly models the shared part). Estimator quality is a risk. | CAD-VAE [2503.07938](https://arxiv.org/abs/2503.07938); [2603.04224](https://arxiv.org/abs/2603.04224) | Medium-high |
| 7 | Token-selection head (two CLS tokens, keep top-k tubelets) | Only for ViT-tokenised fields; test later. | [2603.26336](https://arxiv.org/abs/2603.26336) | Medium |
| skip | Diffusion face anonymisers; LLM rewriters; skeleton transforms | Domain-bound; not field-agnostic. | [2510.24213](https://arxiv.org/abs/2510.24213), [2402.13846](https://arxiv.org/abs/2402.13846), [2506.01420](https://arxiv.org/abs/2506.01420) | High |

Minimum experimental design the literature supports: for each field, sweep rate on #1, add the adversary (#2) at fixed rates, report #4 as the floor, report #3 when a backbone exists. Gain from #2 over #1 at matched rate is the headline number.

---

## (c) Benchmarks (any field) with a leakage/utility trade-off OpenLatent could plug into

| Benchmark | Field | What is measured | Deterministic? | Fit |
|---|---|---|---|---|
| **VoicePrivacy 2024 Challenge** ([2404.02677](https://arxiv.org/abs/2404.02677)) | speech | Privacy: ASV equal error rate on anonymised speech; utility: ASR WER, emotion UAR (metric names from the evaluation-plan summary; page confirms evaluation scripts, baselines and data are provided). | Scripts and fixed datasets provided; attacker training seeds are the only free variable. | **Best match**: published trade-off, shipped evaluation scripts, an `E(x)` is a feature-level anonymiser. |
| **VoicePrivacy attack literature**: VoxGuard ([2509.18413](https://arxiv.org/abs/2509.18413)), Franzreb ([2508.09803](https://arxiv.org/abs/2508.09803)), content attack ([2506.09521](https://arxiv.org/abs/2506.09521)) | speech | Low-FPR membership-inference metric; target-speaker-robust evaluation; text-only BERT ASV attack. | Yes (scripted). | Add as extra attackers. VoxGuard: fine-tuned informed attackers are "orders of magnitude" stronger at low FPR despite similar EER; gender and accent recoverable at near-perfect accuracy after anonymisation (self-reported). Content attack reaches mean EER 35% from words alone. |
| **ISP identity-leakage benchmark** ([2604.05296](https://arxiv.org/abs/2604.05296)) | vision embeddings (CLIP, DINOv2/v3, SSCD) | Open-set verification at low false-accept rate, diffusion template-inversion check, face-context attribution; CelebA-20, VGGFace2. | Mostly (fixed probes, inversion is stochastic). | Directly the frozen-embedding setting; includes a reconstruction-attack leg. |
| **Concept-subspace framework** ([2605.01381](https://arxiv.org/abs/2605.01381)) | speech, text | Containment (concept fully in subspace, not outside) and disentanglement; HuBERT phone vs speaker. | Closed-form estimators, yes. | Cheap metric layer that works on any `(z, concept)` pair; explicitly measures generalisation to unseen data. |
| **FairMedFM** ([2407.00983](https://arxiv.org/abs/2407.00983), NeurIPS 2024) | medical imaging | 17 datasets, 20 foundation models, 9 fairness metrics, 6 mitigation methods; reports "limited effectiveness of existing unfairness mitigation methods". | Yes (open-source pipeline). | Fairness gap, not leakage, but frozen FM embeddings + sensitive attributes make leakage probes easy to add. Good for a field where `s` = sex/age/site. |
| **Unlearning evaluation** ([2503.06991](https://arxiv.org/abs/2503.06991), [2505.16831](https://arxiv.org/abs/2505.16831)) | classifiers, LLMs | Representation-level checks and reversibility after minimal fine-tuning; unlearned vs retrained-from-scratch. | Representation metrics yes (CKA, PCA shift). | Gives protocol, not data. Port the relearn test: fine-tune the attacker k steps on a small slice of obfuscated data and plot leakage vs k. |
| **NeuroPriv / EEGMAT** ([2609.00390](https://arxiv.org/abs/2609.00390)) | EEG | Gender, age, identity attackers (balanced acc.) vs cognitive-state task; chance 0.028 for 36-way identity. | Unknown; code availability not shown on the page. | Small and clean, good second field after speech; check dataset size and license first. |
| SynthPAI ([2402.13846](https://arxiv.org/abs/2402.13846), [2506.01420](https://arxiv.org/abs/2506.01420)) | text | LLM-inferred attributes vs rewrite utility. | **No** (LLM attacker/judge). | Out unless attacker is pinned to a fixed open model. |

**Protocol rules the literature supports (adopt in the harness).**
1. Retrain attackers on obfuscated training data; never report frozen ones (survey 2608.04501: <10% of PPAR papers test adaptive attackers; 65% use ad-hoc metrics; ~40% define cMAP inconsistently).
2. Report low-FPR operating points, not only averaged EER/accuracy (VoxGuard).
3. Control for the task-side confound that leaks privacy information: target-speaker choice (Franzreb), linguistic content (2506.09521). Equivalent in our harness: scene/session identity.
4. Include a held-out-distribution leg (unseen actions/scenario in our case); LEACE-style guarantees fail there (2605.01381).
5. Add a reversibility leg: minimal fine-tuning should not recover `s` ("suppressed, not erased", 2505.16831).
6. Add an inversion leg where `z` is image-like or generative: ISP's diffusion template inversion with matched budgets.

---

## Sceptic's paragraph

The overrated method is **adversarial minimax training as the source of privacy**, which is 52% of the PPAR literature. Everything in our own table says rate does the work: at 1-2 numbers per frame the adversary adds nothing, and even at 15 numbers it buys ~15 identity points on KTH, far less than moving the pool size. The published evidence is no better: NeuroPriv's headline "privacy-aware" encoder leaves 20.6% identity accuracy against a chance of 2.8% (self-reported, one dataset), FRG's guarantee is conditional on the adversary family it optimises against, and the speech field shows that EER-style scores understate leakage by orders of magnitude at low FPR once the attacker is fine-tuned (VoxGuard). A model that wins against its own co-trained critic has proved little; I would not believe any adversarial result in a new field until it beats a rate-matched no-adversary control and a retrained attacker. Runners-up: diffusion/generative face anonymisers (judged by fixed pretrained recognisers only) and "unlearning" claims judged by logit metrics (2503.06991 shows these give false security). Closed-form LEACE is under-rated by comparison: it is deterministic, has a real guarantee, and the only reported weakness (generalisation to unseen data, 2605.01381) is exactly what our eval already tests.

---

## Gaps and leads (not opened, therefore not cited as evidence)

- No 2024-26 paper found on SAM2-style learned masks for privacy; no DP-representation paper opened. Both are open niches for the contrarian scout round.
- Search snippets pointed to: DySan motion-sensor GAN (arXiv 2003.10325, 2020), an anonymising autoencoder for IMU (1810.11546, 2018), PA-HMDB51 adversarial action privacy (1906.05675, 2019), Genie (2402.15391), a latent-action survey (2507.01925), Stenger et al. IEEE Access 2025 on face-anonymisation re-identification, CLUB bottlenecks (2207.04895), a human-free Kinetics paper (2311.06231). Open these before relying on them.
- The abstract pages of 2511.08666, 2603.26336, 2604.05296 and 2608.04501 do not name the attacker protocols; check whether their attackers are retrained before treating their leakage numbers as comparable to ours.

## Paper index (25 opened)

2604.05296, 2605.01381, 2608.04501, 2511.08666, 2603.26336, 2503.06991, 2505.16831, 2507.12341, 2603.04224, 2503.07938, 2509.18413, 2506.09521, 2608.11003, 2609.00390, 2306.03819, 2410.11758, Privacy PORCUPINE (Interspeech 2024), 2510.21017, 2404.02677, 2508.09803, 2407.00983, 2402.13846, 2506.01420, 2510.24213, 2504.14301 (penalty-driven image anonymiser, CVPRW 2025; page gives no numbers, not used in tables).
