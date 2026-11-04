"""R0 code-validity gate: S2 baseline mAP with the benchmark's own metric code vs the published baseline row (controls 0.90, nonrep 0.41).
Prints the three candidate statistics per column; the gate (JK0a) is evaluated by the pre-registered rule in results_notes/F.md (2.5, I6)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ap as AP  # noqa: E402
import common as C  # noqa: E402

PUB = dict(negcon=0.90, nonrep=0.41)
tag = sys.argv[1] if len(sys.argv) > 1 else "s2"
d = C.cache_dir(tag)
X = np.load(d / "X.npy")
m = pd.read_parquet(d / "meta.parquet")
m = m.rename(columns={"PertType": "Metadata_PertType"})
res = AP.benchmark_stats(m, X, ["TARGET2"])
stats = ["mean_average_precision", "fraction_below_p", "fraction_below_corrected_p"]
rows = {s: {k: res[k][s] for k in PUB} for s in stats}
ok = {s: all(abs(rows[s][k] - PUB[k]) <= 0.03 for k in PUB) for s in stats}
out = dict(tag=tag, n_wells=int(X.shape[0]), n_features=int(X.shape[1]), published=PUB, ours=res, stat_rows=rows, match_within_0p03=ok, R0_pass=bool(any(ok.values())))
(d / f"r0_{tag}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
