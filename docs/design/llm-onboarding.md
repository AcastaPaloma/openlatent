# LLM dataset onboarding

Status: prototype (`src/openlatent/onboard.py`), step 1 of 5. Date: 2026-11-05.

## Goal

Point the harness at any raw dataset folder. An LLM decides what the dataset is (samples, the
person or patient, the task, the private attributes) and which obfuscation to try first. Code then
checks the LLM's answer against the data before anything trains.

```
raw folder -> 1. inspect + LLM spec -> 2. method plan -> 3. train (ppv workers) -> 4. score -> back to 2
                                                                         -> 5. sealed eval, once
```

This prototype covers steps 1 and 2. Steps 3 to 5 reuse `research/privacy-obfuscation` (workers,
anonymizers) and the CT experiment protocol (sealed patients, best of fresh attackers).

## What the LLM decides, and what it does not

| LLM decides | Code decides |
|---|---|
| sample source, ID field, regex for fields in file names | that every field exists and every regex matches |
| group (unit that must not cross splits) and identity | the split: SHA-256 of the group, fixed before any result |
| utility and privacy targets, and their kinds | that the kinds parse (binary has 2 values, ages parse) |
| pixel or latent space, methods from a fixed menu | that the methods exist; a raw baseline is always present |
| questions for a human | nothing passes without a human reading `dataset.json` |

The check that matters most: if one identity lands in more than one split, the spec is rejected.
This catches the common mistake of grouping by recording session (KTH `d1`/`d2`, UCF101 `gXX`)
when one person appears in several sessions.

## Usage

```bash
uv sync --extra llm
uv run openlatent onboard path/to/dataset --out runs/onboard/<name>          # rows sent to the LLM
uv run openlatent onboard path/to/dataset --out runs/onboard/<name> --headers-only  # gated data
```

Outputs: `summary.txt` (exactly what the LLM saw), `dataset.proposed.json` (always kept),
`dataset.json` and `manifest.csv` (only if every check passes).

Tests: `python src/openlatent/onboard.py` (stub LLM, fake ChestX-ray14, KTH and CT-RATE folders).

## Prior work (searched 2026-11-05)

- No system found that selects a dataset-level obfuscation method with an LLM. The nearest work is
  LLM text anonymization that searches a privacy-utility frontier:
  [Adaptive Text Anonymization (ACL 2026 Findings)](https://arxiv.org/html/2602.20743) and
  [AURA](https://www.alphaxiv.org/abs/2605.30848) (privacy checked against an agentic re-ID attacker).
- AutoML agents ([AIDE, DS-Agent, AutoML-Agent](https://arxiv.org/html/2508.10177v1),
  [MLZero](https://arxiv.org/pdf/2505.13941)) automate pipelines. MLZero has a "perception" step
  that reads the dataset first, which is the same role as step 1 here. None of them have a privacy
  objective or a held-out attacker.

## Limits of the prototype

- One table per dataset. CT-RATE labels live in a second CSV; joins are not supported yet.
- The live Claude call is not tested here (no credentials in the dev shell). The stub tests cover
  everything after the call.
- The plan is not yet converted to `ppv` run configs; that is step 3.
- Spec is JSON, not the YAML in `video-harness.md`, to avoid a new dependency.
