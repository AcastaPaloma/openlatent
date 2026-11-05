# Data whitelist (licence steward: main agent). Identity or linkage attacks run ONLY on datasets listed ALLOWED here.
Source: Auditor-2 findings, research/adversarial/AUDIT_findings.md section 2 (read at the source 2026-11-04). A dataset not on this list is not cleared; edge cases go to the owner (the user) before download.
A licence snapshot (card text / LICENSE file, retrieval date, sha256) must be committed next to this file before the first download of an ALLOWED dataset.

| Dataset | Status | Identity/linkage attack | Derived artefacts | Conditions |
|---|---|---|---|---|
| ChestX-ray14 (HF mirror alkzar90/NIH-Chest-X-ray-dataset) | ALLOWED with conditions | nothing forbids it; precedent Packhauser 2022, PriCheXy, LAPIS | aggregate metrics, predictions keyed by hashed IDs, code | (a) licence snapshot committed; (b) attribution block (Wang et al. CVPR 2017, NIH Clinical Center) in every write-up; (c) download the zips, never run the repo loading script; (d) integrity check against an official copy; (e) aggregate reporting only, no cross-dataset linkage, NO public release of trained linkage attackers or per-patient embeddings; (f) images are not re-uploaded. Job specs for this dataset MUST set `hf_allow: ["*.json","*.log","*.md"]` |
| MotionSense | ALLOWED (MIT) | yes (identity defences are its own use case) | yes, keep the copyright notice | keep the per-subject demographic table out of any released artefact that carries subject IDs |
| WISDM biometrics (UCI 507) | ALLOWED (CC BY 4.0) | yes (explicit biometric use) | yes with attribution | check the phone-model confound before use |
| UCI-HAR / PAMAP2 / HHAR | ALLOWED (CC BY 4.0), low power | yes by licence | yes with attribution | 9-30 subjects only |
| KTH, Charades, UTD-MHAD (programme-1 holdings) | PREDICTIONS ONLY | no new downloads for identity work without the owner | aggregates/predictions only; non-commercial notice | no derived frames |
| Capture-24 | NOT CLEARED | the catalogue entry says users agree not to attempt to identify individuals (source: search result) | - | - |
| CT-RATE, NLST/TCIA, BOXRR-23, Nymeria, MIMIC-CXR | NOT CLEARED | forbidden or gated | - | - |
| UCF101 / HMDB51 | NOT CLEARED for derived release | no subject IDs, no identity attack possible | - | licence sources conflict |

Model/weight licences are also gated: PriCheXy-Net GPL-3.0 (run as a separate process, do not vendor into the repo), RAD-DINO MSR research licence, DINOv3 gated, VideoMAE CC BY-NC, HARNet academic-only; fine-tuned derivatives inherit these and may not be uploaded publicly.

## Owner decision 2026-11-05 (recorded by the main agent): ai4privacy/pii-masking-300k
| Dataset | Status | Identity/linkage attack | Derived artefacts | Conditions |
|---|---|---|---|---|
| ai4privacy/pii-masking-300k | APPROVED by the owner on 2026-11-05 for academic, non-commercial research use only (HF card `license: other`; `LICENSE.md` restricts to academic non-commercial use, requires acknowledging AI4Privacy, and requires a written licence for redistribution, uploading to databases, sharing through any medium, and derivative works; the data is watermarked; one literature scout had read it as Apache 2, which is wrong) | the data are synthetic text with annotated PII spans of fictitious people; it is used as a SECRET for embedding-privacy tests, not to identify anyone | aggregate metrics only (JSON numbers, counts, intervals); NO raw text or text samples in any committed or uploaded file; NO embeddings, caches, trained adapters, attackers or per-record outputs in git or on Hugging Face (they are derivative works); everything derived stays on the VM scratch disk | (a) download only to the VM scratch directory, never into the git checkout; (b) acknowledge AI4Privacy in every document that reports a number from it; (c) jobs that touch it must set `hf_allow: ["*.json","*.log","*.md"]` and the JSON must contain aggregates only; (d) non-commercial use only: if the owner's use ever becomes commercial this approval is void; (e) the pre-registration (including the splits that avoid template leakage between train and test) is committed before any number. |
