# Focus A (PathoROB, source-invariant embeddings): working log

Owner: Builder-A. Code: `research/fields/src/a_pathorob/`. Job specs: `research/fields/jobs/specs/a_*.yaml`.
Everything below is a log of what was run; numbers come from `aggregate.py` on the JSONL/prediction files the jobs wrote
(HF `KuanKuanKuan/openlatent-fields`, `runs/a_*`). No in-training metric is reported as a result.

## Pre-registration (written before any adapter result was seen)

Protocol, per (encoder, dataset), seeds 0,1,2:

* Independent unit = slide (Camelyon), case (TCGA `slide_id[:12]`, Tolkach). Splits are unit-disjoint.
  `C` sealed test (25% of units per centre x class stratum, fixed across seeds), the rest split per seed into
  `A` adapter-train (45%), `B` attacker/label-probe train (35%), `V` validation (20%).
  The benchmark's unseen (OOD) centres are never used to fit an adapter; OOD1/OOD2 are unit halves for an
  unseen-centre attacker (train on OOD1, test OOD2), and the whole OOD set measures label accuracy on unseen centres.
* Secret = medical centre (closed set of the ID centres). Utility = tissue/diagnosis class. Metric = balanced accuracy.
* Attackers are retrained from scratch on the released representation of B: linear (L-BFGS softmax, L2 chosen on an
  inner unit-disjoint split), 2-layer MLP, cosine kNN, token-transformer (FT-Transformer style). Worst case = max over the four.
* Normalisation: chance (1/#centres) and the label-only baseline (predict centre from the TRUE label with B frequencies);
  `leak = (worst - floor) / (worst_raw - floor)`.
* Controls that must hold before any number is believed: constant embedding -> chance; Gaussian noise embedding -> chance;
  raw embedding with centre labels shuffled across units in B -> chance; raw embedding worst-case well above chance; the
  A-trained linear attacker is reported next to the B-trained one (adapter-overfit diagnostic).
* Selection: per method family, the configuration with the lowest worst-case centre accuracy on V among those whose V label
  accuracy is within delta (1, 2, 5 points) of the raw embedding's. Reported number = that configuration on C.
* Kill criterion A (FOCUS): the best learned adapter (vib, adv_retrain, dann, vq, supcon_xc; selected on V) must beat LEACE
  (ridge chosen on V) on worst-case centre accuracy on C at the same delta, with a paired unit-bootstrap 95% CI below 0,
  3 seeds. Otherwise stop A.
* Single-seed results are labelled "not yet replicated".

## Log
