# OpenLatent fields programme: charter

Started 2026-11-04 on branch `fields-research`. Compute: 4x RTX A6000 48 GB (ubuntu@192.222.52.237, 993 GB disk).

## Question
The privacy-obfuscation programme (research/privacy-obfuscation) learned which masks and latents hide identity while
keeping an activity/CT-style task. Which *other fields* need a learned mask or latent `z = E(x)` that keeps a task
signal and removes (or isolates) another property, and where can the result be verified by a deterministic benchmark?
Lock two to three fields, then run a permanent hypothesis -> experiment -> adjust loop on them.

## Rules (from the user)
1. Logs, code, results JSON: GitHub (this branch), micro-commits, always pushed.
2. Weights and datasets: HuggingFace only (`KuanKuanKuan/openlatent-fields`), never duplicated in git.
3. Literature starts at 2024-2026; older sources only to resolve a question the recent ones point back to.
4. Prefer learned methods over hand-built deterministic transforms. Deterministic *evaluation* is required.
5. Every candidate field needs a benchmark that is existing, close to existing, or one we can build and fix.
6. Lock-in rule: fields are chosen when we are ~90% sure no better field exists (a contrarian scout round must fail
   to beat the shortlist). Then the focus list is frozen in `FOCUS.md` and the loop works only on it.
7. Honesty rules and evaluation protocol from the kuanresearch skill apply (worst-case evaluator suite, chance and
   trivial-baseline normalisation, bootstrap over independent units, sealed test split, controls before belief).

## Roles
- **Main agent** (this session): receives all comms, owns the queue, decides continue / kill / replicate.
- **Scouts**: literature. Phase 1 maps fields; after lock-in they keep reading, for the focus fields only.
- **Hypothesis generator -> ranker**: continuous. Generator writes `hypotheses/H-*.md`; ranker groups and scores
  them (likelihood x applicability to training) into `hypotheses/RANKING.md`.
- **Trainers/monitor**: the VM workers pull jobs from `jobs/pending`; the 15-minute tick checks every machine.
- **Judges**: short-lived subagents asked "is this experiment worth continuing?" with the partial curves.

## Layout
```
CHARTER.md  FOCUS.md  literature/  hypotheses/{H-*.md,RANKING.md}  jobs/{pending,running,done,failed}/*.yaml
results/<job>.json  journal/YYYY-MM-DD.md  scripts/
```
Job = one YAML config in git. Result file = config, git commit, metrics with intervals, timings; checkpoint to HF.
