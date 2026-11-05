"""H3-016 second release seed: regenerate the dithered floors with release seed 1 through the FROZEN suite (suite.build_release is wrapped, no evaluator file is edited).
python ctrl_seed1.py --out DIR [--threads 4]"""
import os, sys, functools, argparse
sys.path.insert(0, os.path.dirname(__file__))
import suite as SU
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
import torch; torch.set_num_threads(a.threads)
SU.build_release = functools.partial(SU.build_release, seed=1)
for arm in ["act60_r1", "lda3_r1", "act8_r1", "lda3_r0.5"]:
    if os.path.exists(os.path.join(a.out, f"preds__{arm}_ns1__tab.npz")): continue
    SU.run(arm, ["O"], "tab", ["id", "act"], a.out, None, a.threads, True, "_ns1", "astar")
