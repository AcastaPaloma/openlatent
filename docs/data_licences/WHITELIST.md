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
