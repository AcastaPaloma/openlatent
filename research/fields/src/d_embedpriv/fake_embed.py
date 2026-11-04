"""Mechanics-only test fixture: writes SYNTHETIC unit vectors (bag-of-words random projection + noise) so the pipeline can be smoke-tested on CPU
before the real extraction job has run.  Never used for any reported number."""
import sys
from pathlib import Path

import numpy as np

from common import World

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
W = World()
rng = np.random.default_rng(0)
R = rng.standard_normal((len(W.vocab), 768)).astype(np.float32) / np.sqrt(60)


def emb(B):
    X = B @ R + 0.5 * rng.standard_normal((B.shape[0], 768)).astype(np.float32)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


np.save(out / "nf_docs.npy", emb(W.nf_bags))
np.save(out / "sf_docs.npy", emb(W.sf_bags))
np.save(out / "nf_queries.npy", emb(W.bags(W.q_texts)))
for m in ("rare", "all"):
    np.save(out / f"nf_docs_mask_{m}.npy", emb(W.nf_bags))
    np.save(out / f"sf_docs_mask_{m}.npy", emb(W.sf_bags))
