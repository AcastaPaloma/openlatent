# OpenLatent

Deterministic before/after privacy scoring for raw inputs `x` and learned latents `z = E(x)`.
The same feature file in gives a byte-identical report out, so the score works as a
regression test after every encoder change.

This is the cheap, repeatable layer. Every number here is a **lower bound** on leakage:
an encoder trained against an adversary can fool fixed scorers while identity is still
recoverable. Headline claims still need the best of 5+ freshly trained attackers and a
reconstruction-then-re-ID attack.

## Setup

```bash
uv sync
uv run pytest
```

## ChestX-ray14 workflow

Download `Data_Entry_2017_v2020.csv`, `test_list.txt` and the image folders from the
[NIH box](https://nihcc.app.box.com/v/ChestXray-NIHCC). Keep data under `data/` (git-ignored).

```bash
# 1. Patient-disjoint splits; eval = the official (patient-wise) test list
uv run openlatent splits --meta data/Data_Entry_2017_v2020.csv --test-list data/test_list.txt --out runs/splits.csv

# 2. "Before" features. Raw pixels are a weak baseline; better: embeddings from a
#    frozen re-ID network, saved with openlatent.features.save_features.
uv run openlatent pixels --splits runs/splits.csv --images data/images --size 64 --out runs/x.npy

# 3. "After" features: encode the eval images with your encoder, cache to disk once
#    (save_features(path, z, image_ids) writes z.npy + z.ids.txt).

# 4. Score both the same way, then compare
uv run openlatent score --meta data/Data_Entry_2017_v2020.csv --splits runs/splits.csv --features runs/x.npy --out runs/x.json
uv run openlatent score --meta data/Data_Entry_2017_v2020.csv --splits runs/splits.csv --features runs/z.npy --out runs/z.json
uv run openlatent compare --before runs/x.json --after runs/z.json \
  --x-features runs/x.npy --z-features runs/z.npy \
  --meta data/Data_Entry_2017_v2020.csv --splits runs/splits.csv --out runs/compare.json
```

## Splits (`data/chestxray14.py`)

| Column | Values | Rule |
|---|---|---|
| `split` | `encoder_train`, `encoder_val`, `eval` | eval = official test list (or a hash of patient ID); val = hash of patient ID |
| `probe_role` | `fit`, `test` | eval patients split 50/50 by hash; probes fit on one half, scored on the other |
| `reid_role` | `gallery`, `query` | eval patients with 2+ images; earliest follow-up is the gallery item, the rest are queries |

Hashes are SHA-256 of `salt:patient_id`, so assignments do not depend on row order,
`PYTHONHASHSEED` or which images are present.

## Scorers

| Report key | What | Chance | Deterministic because |
|---|---|---|---|
| `reid` | cosine kNN re-ID: top-k, MRR, verification AUC | 1/gallery, H(n)/n, 0.5 | no training; ties use their expected rank, so a constant `z` scores exactly chance; impostor pairs use a fixed seed |
| `sex_probe` | logistic probe, ROC AUC | 0.5 | convex; C picked by patient-grouped CV with fixed folds |
| `age_probe` | ridge probe, R² | 0 | closed form; alpha picked the same way |
| `identity_alignment` | normalized unbiased HSIC between a kernel on features and 1[same patient] | 0 | closed form on a hash-chosen subsample of whole patients |
| `findings_probe` | 14 logistic probes, mean disease AUC (utility) | 0.5 | as above |
| `x_vs_z` (compare) | linear CKA and distance correlation (NoPeek) between x and z | 0 | closed form |

`compare` reports `leakage_removed = (before - after) / (before - chance)` for privacy
(1 = reduced to guessing, negative = z leaks more) and `utility_kept = (after - chance) / (before - chance)`
for utility. Both are `null` when `before` is not at least 0.01 above chance.

Reports from different harness configs refuse to compare. Bitwise reproducibility is
guaranteed on the same machine and library versions; across BLAS builds expect agreement
to many decimal places, not bit-for-bit.

## Design docs

- [Video harness: dataset contract and pipeline](docs/design/video-harness.md) — SPLAVU-style
  latent anonymization for arbitrary labelled video datasets, reusing these scorers. Not implemented yet.
