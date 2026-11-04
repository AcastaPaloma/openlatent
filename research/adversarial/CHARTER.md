# Adversary-suite programme (OpenLatent, programme 2)

Started 2026-11-04 on branch `fields-research` (shared with programme 1, directory `research/adversarial/`). Same machine and queue:
ubuntu@192.222.52.237, 4x RTX A6000, workers + fair-share queue from `research/fields/scripts/` (`worker.py`, `tick.sh`, `normalize_queue.py`).
Job prefixes for this programme: `h_` (human activity recognition), `c3_`/`t_` (CT / medical imaging), `r_` (robust-suite infrastructure); the queue normaliser interleaves
every prefix with the programme-1 prefixes.

## User request (verbatim intent)
"Go through the literature of adversarial agents: training a robust suite of adversarial models seems interesting, back to human activity recognition and CT scans.
An autoresearch loop producing hypotheses for that, and queueing, everything we have discussed." Same agent groups as programme 1: literature scouts, a contrarian
round, lock-in of two to three focuses, a continuous hypothesis generator and ranker, builders per focus, 15-minute monitor ticks, micro-commits to GitHub,
weights and datasets to HuggingFace only (`KuanKuanKuan/openlatent-fields`, no redundancy), learned methods preferred, deterministic benchmarks.
Interpretation (stated to the user): "adversarial agents" = adversarial models (attackers that try to recover identity or attributes from masks and latents, and
defenders trained against suites of them); agent-driven red-teaming (LLM agents that search for attackers) is one literature cluster, not the whole programme.

## Question
Programme 1 measured, with retrained attackers, that learned adapters for "keep the label, hide the source/identity" did not beat cheap floors, and that
single-attacker numbers mislead. The open question here is the attacker side: what is a *robust suite of adversaries* (population, family diversity, adaptive
re-training, search over attacker architectures, foundation-model attackers, compute scaling), how much does the defender's measured privacy move when the suite
is strengthened, can a defender trained against such a suite hold up against *unseen* attackers, and does this change the picture for human activity recognition
(identity/attribute privacy in video, skeleton, wearables) and CT / medical imaging (patient re-identification, latent sharing)?
Falsifiable core: a standardised suite ("AutoAttack for privacy") whose worst case moves the conclusions of programme 1's HAR results, or a defender trained
against a suite that generalises to attacker families it never saw.

## Rules (user's, unchanged)
1. Logs and code to GitHub, micro-commits, always pushed. 2. Weights/datasets to HuggingFace only, never duplicated; public datasets never re-uploaded.
3. Literature starts at 2024-2026; older sources only when a recent one points back. 4. Learned methods preferred; deterministic evaluation required.
5. Lock-in rule: two to three focuses are chosen when ~90% sure nothing better exists (contrarian round must fail to beat the shortlist); freeze in `FOCUS.md`.
6. A main agent (this session) receives all comms; subagents decide whether an experiment is worth continuing; jobs queued and dequeued by the main agent.

## Lessons inherited from programme 1 (binding; each cost us hours)
- Pre-register the protocol, kill rule, margin and comparator BEFORE any model number; commit it; log amendments, none may relax a kill condition.
- Run FLOORS FIRST and name the strongest trivial floor: label-supervised linear projection (LDA), PCA, random projection, noise/dither, per-source normalisation,
  rate bottleneck without an adversary. A learned method counts only against the lower envelope of ALL floors at matched leakage, never against one comparator.
- Declare a headroom check with an oracle arm and a power audit (minimum detectable effect from the real number of independent units) before fixing any margin.
  Seeds that share test units replicate training noise only: use cross-fit / rotating test units.
- Attackers are always retrained on the released representation (linear, MLP, kNN, LightGBM, transformer, plus a held-out family); worst case reported; a frozen
  raw-trained attacker understates leakage. Controls: constant and shuffled-label inputs at chance; raw reference well above the label-only baseline.
- Beware "the task implies the secret", dataset confounds (CREMA-D full-clip identity was 76% recording-channel signal), and closed-form erasure fitted on seen units.
- Select on validation, never on the sealed test; disclose anything seen early; say "not yet replicated" for single seeds; correct yourself in the journal.
- Queue hygiene: validate YAML with yaml.safe_load before queueing; cap CPU threads <= 8; priorities are relative inside your own prefix; <5 is reserved for the main
  agent (a new focus's first jobs may use priority 4 once; short pre-registered batches < 1 GPU-h may be promoted on request); one job per GPU worker.
- Stop rule: if cheap gates (floors, headroom, power) fail, stop the focus and write the negative result with its by-products; do not spend GPU to fill the queue.
- Changing a test after data: a kill criterion may only be TIGHTENED after numbers are seen, and then both verdicts (original and tightened) are reported side by side;
  loosening after data is a deviation, is labelled as such, and does not change the verdict (programme 1, Focus A and F, 2026-11-04).
- Data-use gate (added at the 2026-11-04 audit): identity or re-identification experiments run only on datasets whose terms allow them (ChestX-ray14, which OpenLatent already uses;
  CC-BY / MIT IMU sets that ship user IDs; programme-1 artefacts). Datasets whose terms forbid identifying individuals (CT-RATE, NLST/TCIA, Capture-24 per its record, BOXRR-23) are excluded
  from identity work; borderline cases go to the owner before download. Journal time stamps use the VM clock.

## Layout
```
CHARTER.md  FOCUS.md  literature/  hypotheses/{H-*.md,INDEX.md,RANKING.md}  jobs/specs/*.yaml  results/  results_notes/<focus>.md  journal/YYYY-MM-DD.md  src/<focus>/
```

## Added after the 2026-11-04 audit
- Data-use gate: no identity or re-identification attack on any dataset whose terms forbid identifying individuals (flagged so far: CT-RATE, NLST/TCIA, Capture-24 per a snippet, BOXRR-23); defaults are ChestX-ray14 (already used by OpenLatent) and
  CC-BY / MIT IMU sets that ship user IDs; edge cases go to the owner before download.
- Kill criteria may be tightened after data only with the original verdict reported alongside; they are never loosened after data (programme-1 lesson, Focus A and Focus F).
- Every pre-registration names its seeds-vs-test-units design (cross-fit or rotating test units) so seeds replicate test noise, not only training noise.
- Literature numbers quoted in FOCUS.md or any write-up must be verified against the primary source (programme-2 scouts read mostly abstracts).
- CPU etiquette (2026-11-04, load average hit ~200 on 56 cores): the worker now sets OMP/MKL/OPENBLAS/NUMEXPR threads to 8 for every queued job; ad-hoc CPU work run directly over ssh must not exceed 3 concurrent processes x 8 threads per builder
  (use `taskset`/`--procs` accordingly); never start a job with unbounded BLAS/OpenMP threads.
