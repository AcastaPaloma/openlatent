# Shutdown manifest, 2026-11-05 (owner order: prepare the training shutdown, push to GitHub and Hugging Face, clear the VM)

VM: ubuntu@192.222.52.237 (4x RTX A6000). Final state before clearing: 0 running, 0 pending, 365+ jobs done (every job_result.json in git), 11 early failures.

## Where everything went
| Artefact | Disposition |
|---|---|
| Code, job specs, notes, journals, audits, results (aggregates and job_result.json) | GitHub `AcastaPaloma/openlatent`, branch `fields-research` (tagged `fields-research-final-2026-11-05`) |
| Job outputs (metrics, logs, checkpoints, JSON) for all 395 successful jobs | Hugging Face `KuanKuanKuan/openlatent-fields`, `runs/<job>`; verified 395 of 395 present; README added. Chest X-ray (`t_*`) and PII (`d_pii_*`) jobs uploaded JSON/log/md only |
| Sealed-query ledgers and salts | in git: `research/adversarial/sealed/` (T ledger 92 entries, salts) and `results/N`, `results/N2` ledgers |
| Trained chest X-ray models (defenders, T7 generators, verifier, classifier) and the split metadata | PRIVATE local archive on the owner's machine: `~/Me/openlatent-vm-archive-2026-11-05/` (409 MB; not in git, not on HF by licence policy) |

## Deleted from the VM (regenerable from git + public data, or deliberately never published)
| Item | Size | How to regenerate |
|---|---|---|
| `~/work/data` (public datasets) | 30 GB | download scripts in `research/*/src`, dataset list in `docs/data_licences/WHITELIST.md` |
| `~/work/cache`, `~/work/feats`, `~/work/ext`, `~/work/scratch`, `~/work/f3_regen`, `~/work/stress_probe` (embeddings, features, scratch) | ~42 GB | the `*_extract*` / `*_embed*` job specs |
| `~/work/t_data/cxr` (ChestX-ray14 image bank, HF mirror) | 49 GB | `t_prep` job spec; mirror `alkzar90/NIH-Chest-X-ray-dataset` |
| `~/work/t_work`, `~/work/t_agg_*` (cached features, checkpoints for screens) | ~3.5 GB | the `t_*` job specs |
| `~/work/h_imu` (MotionSense windows, PrivDiffuser clone) | 4.6 GB | `research/adversarial/src/h_imu` |
| `~/n_releases`, `~/n2_releases`, `~/n2_preds`, `~/n_run*`, `~/n_*` (planted-leak releases, predictions) | ~17 GB | `research/adversarial/src/n_control` (certificates and manifests are in `results/N`, `results/N2`) |
| `~/work/runs` (local copies of outputs already on HF), `~/work/logs`, `~/queue` | ~4.5 GB | HF `runs/<job>` |
| `~/venvs/*` (a, b, d, e, f, t) | 15 GB | per-focus requirements in the job specs |
| Restricted artefacts: D3 per-document arrays and embeddings (PII-Masking-300K), chest X-ray bank and attackers | in the above | never uploaded by licence; PII data deleted per its conditions |

Not lost, but not preserved either: the intermediate per-image attacker models of the chest X-ray ladder (R1-R5), regenerable by the T3/T4/T5 specs (about 30 worker-hours).

## What stays on the VM
The operating system, the ssh login and the empty `~/openlatent` git clone only. The watchdog cron and all workers were removed. To resume: `git clone` branch `fields-research`, restore datasets, create venvs, start `research/fields/scripts/worker.py --gpu N`, and install the cron from `research/fields/scripts/watchdog.sh`.
