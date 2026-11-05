# Reading guide: what was done, in what order to read it (2026-11-04 to 2026-11-05)

Branch `fields-research`. Weights and job outputs: Hugging Face `KuanKuanKuan/openlatent-fields` (public datasets are never re-uploaded). Everything below is in the repo.

## 15 minutes: the whole picture
1. `research/SUMMARY.md` (44 lines). The scoped results of both programmes, with the process record and limits. Start here. Every row names its strength and what was withdrawn.
2. This file's "Five findings" below.

## 1 hour: how the claims were made and checked
3. `research/fields/CHARTER.md` and `research/adversarial/CHARTER.md`. The rules (pre-registration, floors first, retrained attackers, bootstrap over independent units, honesty rules, data-use gates).
4. `research/adversarial/FOCUS.md` (outcome blocks at the end). Programme 2: each result with its original and tightened verdicts.
5. `research/fields/FOCUS.md` (outcome blocks at the end). Programme 1, same format.
6. The four audits, which state what is and is not verified:
   - `research/fields/AUDIT_findings.md` (programme-1 headline numbers re-derived from raw files),
   - `research/adversarial/AUDIT_findings.md` (programme-2 literature, licences, design),
   - `research/adversarial/AUDIT_results.md` (programme-2 results re-derived: numbers reproduce, wording corrected),
   - `research/fields/AUDIT_results_2.md` (programme-1 later results re-derived: same).

## Half a day: the evidence and the reasoning
7. `research/adversarial/journal/2026-11-0[45].md` and `research/fields/journal/2026-11-0[45].md`. Chronological, including every correction, retraction and mistake (swapped agents, wrong clock stamps, over-read results).
8. Per-focus notes: `research/fields/results_notes/{A,B,C,D,E,F,JUMP0}.md`, `research/adversarial/results_notes/{H,T,S,N}.md`. Builders' full write-ups with pre-registrations (the section numbers are cited from the journals).
9. `research/*/hypotheses/` (`RANKING*.md` ranks the hypotheses; `H3-018` is the pre-registered stop table) and `research/*/literature/` (the scouts' surveys).
10. `docs/data_licences/WHITELIST.md`. Which datasets may be used and how.

## Five findings to keep in mind
1. A privacy number depends on how the attacker was trained. In every case we could re-measure (wearable IMU, chest X-ray) a published figure held only against a frozen or weak attacker.
2. Cheap floors (a label-supervised linear code, downsampling, noise/dither) matched or beat almost every learned mask or defender at equal utility. The one learned positive is a supervised adapter on JUMP with enough labels, and it works for sources it saw, not unseen ones.
3. Learned source-removal heads (adversarial or CORAL penalties) did not engage in any dataset where we checked, so "a learned mask hides the batch" was never demonstrated.
4. Chest X-ray: against a much stronger pair attacker, raw, 4x-downsampled and defended images are all linked at about 0.98 (shuffled-label control 0.55). Only a 14-number label code stays low (0.71).
5. Power, not effort, limited many conclusions; several comparisons could resolve only differences of 0.05 to 0.14.

## Where the code is
`research/fields/src`, `research/adversarial/src` (attackers, floors, aggregators), `research/fields/scripts` (queue worker, tick, collect), `research/*/jobs/specs` (every job as a YAML). The privacy-obfuscation codebase that preceded this work is `research/privacy-obfuscation`.
