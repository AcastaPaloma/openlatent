# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Chosen by Claude, not put to the user: Python standard-library server plus one HTML file, so `python -m privharness ui` runs on a laptop with nothing to install. The heavy dependencies (torch, GPUs) stay on the VMs.

## Users

- **Dataset owners** with no ML background. They hold labelled video or images and need to know whether a disguised version can be shared without exposing what must stay private.
- **The researcher running the harness**, who starts searches on rented GPU VMs and needs the full evidence behind every score.

## Product Purpose

`privharness` takes a labelled dataset with a task to keep and secrets to hide, tries many ways of disguising the footage, attacks each with attackers trained on the disguised data, and recommends the best measured balance of privacy and usefulness. The UI lets someone bring a dataset, start that process on a VM, follow it, and understand and trust the outcome.

Success: a dataset owner can reach a verdict and explain it to someone else, and a researcher can trace that verdict down to each attacker's score.

## Positioning

Privacy is measured by attacking, not assumed: scores come from fresh attackers trained on the disguised output, the worst attacker counts, and the pick is re-attacked with new seeds, longer training and a different attacker family before its score is called certified.

## Operating Context

- GPU work runs on SSH-reachable VMs with key-based login. Searches last hours to days, so the VM must keep working when the laptop closes.
- People check in occasionally rather than watching continuously.
- The UI is local to one person's machine and is never exposed on a network.

## Capabilities and Constraints

- Pipeline stages: ingest, profile, plan, screen, evaluate, robustness, report. Each resumes from its own output files.
- "Certified" means the worst result over the re-runs against the attackers tried. It is not a mathematical guarantee and the UI must not imply one.
- Known gaps the UI must not hide: no reconstruction attacker, and no sealed test set for selection yet.
- Dataset files stay on the VM. Licences may forbid redistribution, so the UI pulls only scores and logs.
- Never ask for passwords. SSH access relies on the user's existing keys and SSH config.

## Evidence on Hand

Real reports for five datasets in `results/harness/`, plus VP-UCF101 stopped after screening. No customer or testimonial evidence exists, and none may be invented.

## Product Principles

- Plain words first, evidence one step away: every verdict can be opened down to the attacker, interval and check behind it.
- Bad news stays as prominent as good news. A weak result reads as weak, and a failed check is never buried.
- Say what a score does not prove next to the score.
- A person should never need to know a regex, a tmux command or a metric name to finish the job.
- Long-running work belongs to the VM. The UI can close at any time and lose nothing.
