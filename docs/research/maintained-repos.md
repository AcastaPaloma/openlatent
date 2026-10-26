# Maintained repos and verifiable checkpoints

Snapshot: 2026-10-22. Activity numbers cover the 12 months before that date and come from
[`scripts/repo_health.py`](../../scripts/repo_health.py) run on [`repos.txt`](repos.txt):

```bash
python3 scripts/repo_health.py docs/research/repos.txt health.json
```

"Replied" counts issues opened in the window that got at least one comment. "Verified"
means we downloaded the artifact, checked its size against the release, recorded a
SHA-256, and ran a forward pass.

## Bottom line

No repo that implements a privacy-preserving action recognition or latent anonymization
*method* is maintained. Every one of them was released alongside its paper and then left
alone (table C). The trustworthy starting points are maintained libraries that supply
OpenLatent's parts: encoders with pinned checkpoints, the contrastive loss, an adversarial
training reference, and attack tooling (table A). We implement the method ourselves on top
of them, as the video design doc already plans.

Of the three tracks, chest X-ray has the strongest maintained base (torchxrayvision with
verified weights, including an autoencoder baseline). Video has a good one (V-JEPA 2 in
`transformers`, PyAV, lightly). Skeletons have none: pyskl and mmaction2 are close to
dormant.

## A. Build on these (maintained)

| Role in OpenLatent | Repo | ★ | Last commit | Commits / issues opened / replied / closed / PRs merged (12 mo) | License | Latest release |
|---|---|---|---|---|---|---|
| Chest X-ray encoder, utility model, AE baseline | [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) | 1.2k | 2026-10-18 | 26 / 7 / 5 / 6 / 16 | Apache-2.0 (main library; baseline models vary) | 1.5.7, 2026-10-17 |
| Medical imaging pipelines | [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) | 8.8k | 2026-10-22 | 288 / 126 / 66 / 161 / 283 | Apache-2.0 | 1.6.1, 2026-09-27 |
| Video encoders (VideoMAE, V-JEPA 2) | [huggingface/transformers](https://github.com/huggingface/transformers) | 168k | 2026-10-22 | 3531 / 1604 / 1409 / 2314 / 3537 | Apache-2.0 | v5.20.2, 2026-10-19 |
| V-JEPA 2 reference code | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 4.8k | 2026-03-23 | 4 / 58 / 19 / 10 / 4 | MIT | none |
| Video decoding | [PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | 3.3k | 2026-10-22 | 356 / 99 / 65 / 113 / 299 | BSD-3-Clause | v19.0.2, 2026-10-21 |
| NT-Xent / SimCLR privacy branch | [lightly-ai/lightly](https://github.com/lightly-ai/lightly) (`lightly/loss/ntx_ent_loss.py`) | 3.8k | 2026-10-15 | 101 / 46 / 35 / 38 / 101 | MIT | v1.5.26, 2026-07-27 |
| Adversarial (encoder vs adversary) training reference | [fairlearn/fairlearn](https://github.com/fairlearn/fairlearn) (`fairlearn/adversarial/`, PyTorch engine) | 2.3k | 2026-10-21 | 133 / 29 / 15 / 55 / 139 | MIT | v0.15.0, 2026-10-12 |
| Attribute inference, model inversion, reconstruction attacks | [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (`art/attacks/inference/`) | 6.3k | 2026-10-08 | 30 / 7 / 3 / 1 / 11 | MIT | 1.20.1, 2025-07-07 |
| Face-blur baseline for video | [facebookresearch/EgoBlur](https://github.com/facebookresearch/EgoBlur) | 300 | 2026-05-06 | 34 / 3 / 2 / 3 / 0 | Apache-2.0 | v2.0.1, 2026-04-14 |
| Formal DP route (training-data privacy, a different threat) | [meta-pytorch/opacus](https://github.com/meta-pytorch/opacus) | 2.0k | 2026-09-24 | 22 / 18 / 15 / 14 / 0 | Apache-2.0 | v1.6.0, 2026-05-05 |

Notes:
- fairlearn's adversarial mitigation and ART's attacks target tabular classifiers. Use them
  as tested reference implementations of the minimax loop and the attack interfaces, not as
  drop-in components for image or video latents.
- ART's activity is uneven: 30 commits and 11 merged PRs, but only 1 issue closed in 12 months.

## Pinned checkpoints

| Checkpoint | Source | Pin | Trained on | License | Verified |
|---|---|---|---|---|---|
| DenseNet-121, NIH (`densenet121-res224-nih`) | torchxrayvision release `v1` | sha256 `31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf`, 28,382,096 bytes | NIH ChestX-ray14 | Apache-2.0 | yes: 14 NIH findings out, 1024-d features |
| DenseNet-121, CheXpert only (`densenet121-res224-chex`) | torchxrayvision release `v1` | sha256 `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`, 28,381,998 bytes | CheXpert | Apache-2.0 | hash and size only |
| ResNet-101 autoencoder (`101-elastic`) | torchxrayvision release `v1` | sha256 `877ded430383a2bbce756737b316d76de42defed3079eebf2cce305f7be34df9`, 125,341,920 bytes | PadChest, NIH, CheXpert, MIMIC | Apache-2.0 | yes: 512×3×3 latent, 224×224 reconstruction |
| V-JEPA 2 ViT-L, 64 frames, 256px | Hugging Face `facebook/vjepa2-vitl-fpc64-256` | revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | self-supervised video | MIT | not loaded (1.2 GB); 318k downloads, 216 likes |
| V-JEPA 2 ViT-L, SSv2 classifier | Hugging Face `facebook/vjepa2-vitl-fpc16-256-ssv2` | revision `4aa02df83918538fc21cfaf576382fa20e489a80` | SSv2 | MIT | not loaded |
| VideoMAE-Base | Hugging Face `MCG-NJU/videomae-base` | revision `dc740ceda42fce44faed2ea03c6d447db72f6af9` | Kinetics-400 | **CC-BY-NC-4.0** | not loaded |

Verification ran torch 2.14.1 and torchxrayvision 1.5.7 on CPU. The torchxrayvision
release assets carry no published digest, so these hashes are our own record: check
downloads against them.

**Contamination risk on ChestX-ray14.** The NIH DenseNet and the autoencoder were trained
on NIH images, and nothing we found says they excluded the official test list. Using them
to embed or score that test split may overstate utility and distort leakage. Prefer the
CheXpert-only DenseNet as the frozen encoder for NIH evaluation, or train our own on the
`encoder_train` split.

**License risk on video.** VideoMAE weights are non-commercial. V-JEPA 2 (MIT) avoids this
and is natively supported in `transformers` (`models/vjepa2`). The video design doc should
default to V-JEPA 2.

## B. Protocol references (read, don't depend on)

| Repo | Why it matters | Activity | License |
|---|---|---|---|
| [Voice-Privacy-Challenge-2026](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026) | The most actively maintained anonymize-then-evaluate benchmark in any modality: `run_anonymization.py`, then `run_evaluation.py` with separate `evaluation/privacy` (attacker) and `evaluation/utility` trees. Cited in the proposal [7]. A model for how our harness should be laid out. | 229 commits in 12 months; last 2026-09-01; few issues | GPL-3.0 (copyleft, so copy the design, not the code) |
| [vec2text/vec2text](https://github.com/vec2text/vec2text) | The best-known embedding inversion attack (text). A reference for how strong inversion of shared embeddings can get. | 2 commits, 5 issues all replied; last 2025-12-27 | no standard license detected |
| [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching) | Privacy attacks on federated learning (gradient inversion). A nearby threat model. | 19 commits; last 2026-01-24 | MIT |

## C. Method repos (dormant, read for details only)

| Repo | Paper | Last commit | 12-mo activity | License | Status |
|---|---|---|---|---|---|
| [DAVEISHAN/SPAct](https://github.com/DAVEISHAN/SPAct) | SPAct, CVPR 2022 | 2022-06-05 | none | MIT | open issues about missing files since 2022; no weights |
| [UCF-CRCV/PrivacyBeyondPixels](https://github.com/UCF-CRCV/PrivacyBeyondPixels) | SPLAVU, ICLR 2026 | 2026-04-09 | 3 commits (initial), 1 issue, no reply | MIT | no weights; reference for adapter tokenization and NT-Xent head |
| [UCF-CRCV/TeD-SPAD](https://github.com/UCF-CRCV/TeD-SPAD) | TeD-SPAD, ICCV 2023 | 2025-02-18 | 1 issue, replied | MIT | anonymizer weights on Hugging Face |
| [ml-postech/Skeleton-anonymization](https://github.com/ml-postech/Skeleton-anonymization) | Moon et al., AAAI 2023 | 2022-12-29 | none | unclear | — |
| [kaipackhaeuser/PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) | PriCheXy-Net, MICCAI 2023 | 2023-10-05 | none | GPL-3.0 | — |
| [kaipackhaeuser/CXR-Patient-ReID](https://github.com/kaipackhaeuser/CXR-Patient-ReID) | Packhäuser et al., Sci. Rep. 2022 | 2022-12-19 | none | GPL-3.0 | the published re-ID attacker; reimplement or keep it isolated (copyleft) |
| [ffhibnese/Model-Inversion-Attack-ToolBox](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox) | model inversion toolbox | 2026-09-15 | 3 commits, 1 issue, no reply | **none** | cannot reuse code without a license |
| Token-pruning video anonymization ([Rabusi/…](https://github.com/Rabusi/From-Pixels-to-Privacy-Temporally-Consistent-Video-Anonymization-via-Token-Pruning-for-Privacy-Pres)) | arXiv 2603.26336 | 2026-03-30 | README edits only | **none** | cannot reuse code |

## Considered and set aside

| Repo | Reason |
|---|---|
| secretflow/secretflow (2.7k★) | Split learning, and its attacks and defenses, are no longer on `main`; the last sync was 2025-11-13 |
| open-mmlab/mmaction2 (5.2k★), kennymckormick/pyskl (1.3k★) | 1–2 commits in 12 months, issues unanswered, last releases 2023 and 2022. The skeleton track has no maintained base |
| vturrisi/solo-learn (1.6k★) | 1 commit, no issue replies in 12 months; lightly covers the same losses and is active |
| EleutherAI/concept-erasure (270★) | No commits since 2024-10. LEACE is closed-form and short, so implement it (with tests against the paper) instead of depending on it |
| hukkelas/deep_privacy2 (383★) | No commits since 2024-01 |
| hanweikung/face_anon_simple (216★) | Active, but AGPL-3.0 |
| deepinsight/insightface (30k★) | Active, but no license on the repository; check the model-zoo terms before using it as a face re-ID attacker |
| privacytrustlab/ml_privacy_meter (728★) | No commits since 2025-04; membership inference is a different threat model |
| MCG-NJU/VideoMAE (1.8k★) | No commits since 2023; the model is maintained inside `transformers` instead |
