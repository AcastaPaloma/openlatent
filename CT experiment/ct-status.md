# Chest-CT obfuscation: where things stand (2026-11-03)

One question: can we change a CT scan's data so that a stranger cannot tell it belongs to the same patient as another scan, while doctors' disease readouts stay useful?

## What was built
- Frozen CT-FM encoder -> a small trained "adapter" -> a 256-number description of each scan.
- A hidden opponent ("adversary") is trained against the adapter so identity cannot be read. Evaluation uses fresh attackers (5 kinds, retrained, one given 4x more data), a scanner-metadata-only floor, and a constant-latent sanity check.
- Everything runs from the repo's `CT experiment/vm_run/` code on 8 A100s; the journal (`journal.md`) records every hypothesis, rejection and fix.

## What is established (with evidence)
1. Reproduced the original dev numbers within 0.015 (raw 0.207 vs 0.200, A3 0.078 vs 0.077, etc.). A scoring bug (NaN counted as a perfect match) was found by the constant-latent check; no real arm was affected.
2. On 538 never-used patients (sealed-2, frozen plan): raw features link 42.7%; A3 (1000 steps) 10.2% at 94% utility; PCA/random projection 27-33%; NoPeek/LEACE/noise reach 7.5-8.2% only by losing 27-45% of the utility.
3. Held-out confirmation on 169 original validation patients (sealed-3): A21 (A14's adversary + a stronger disease head) lowers linkage by 0.031 vs A3 (paired 95% interval -0.055 to -0.008) at the same utility; raw 0.474, A3 0.190, A21 0.159.
4. Not specific to one feature format: on the global 512-d view A3 removes 65% at 95% utility, A21 73% at 90%.
5. Adapters beat plain noise only at high utility (kept >= ~0.85): at 90% kept A21 0.077 vs noise 0.147. Below ~80% kept noise is as good.
6. The remaining leak is mostly body anatomy, not the scanner: with only same-scanner candidates A3 still links 9x above chance (against these hard negatives it removes 52% of the raw linkage, not 71%).

## What is NOT shown
- Anonymity. Every method stays 1.8x to 4x above the scanner-only floor (best: A14 1.8x on sealed-2; A3 3.0x; A21 3.7x on sealed-3).
- Pixel-space reconstruction attack: not run (argued unnecessary but owed).
- Other hospitals or encoders: one hospital (CT-RATE), one frozen encoder family (two views of CT-FM). Second encoder (TAP-CT) is waiting on the user's decision because it needs third-party code execution.
- Age stays decodable (R^2 0.62 -> 0.41-0.48). A21's nearest-neighbour "similar-case" utility is weak (0.64 of raw).
- Selection: ~150 configurations were tried on development data; held-out sets confirm A3/A21, but any tuned number should be read with that in mind. Labels are classifier-derived from reports.

## Rejected ideas (all logged)
Identity-direction erasure (L*), contrastive adversary as a free win (A12/A13), 5000-step training, input dropout/noise regularisation, a retrieval-aware term (A24/A25), a bigger latent (A22), loss reweighting (A23).

## Suggested next steps
1. Decide on TAP-CT (allow/skip). 2. Mutual-information (CLUB) penalty arm. 3. Explicit scanner-adversary + stronger head at kept >= 0.95. 4. Pixel-space reconstruction attack if a reviewer insists. 5. Second hospital data if obtainable.

## Weights
Adapter checkpoints (A3 / A14 / A21, seeds 0-2, safetensors) and a loader are in the **private** Hugging Face repo `Abby012/openlatent-ct-adapters` (CC BY-NC-SA 4.0, derived from CT-RATE). Latent files and CT-RATE data are not included. `vm_run/train_arms.py` with `SAVEW=1` exports them; `vm_run/load_adapter.py` + `verify_export.py` show loading and the round-trip check.
