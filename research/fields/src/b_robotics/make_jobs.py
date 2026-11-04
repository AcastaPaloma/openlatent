"""Write the main-matrix job specs (strong level): python make_jobs.py  (-> research/fields/jobs/specs/b_m_*.yaml)
Non-SAM arms and SAM arm are separate jobs so the SAM-free arms start while masks are computed."""
import os
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jobs", "specs")
PRE = "cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python"
NOSAM = "raw,rate4,gtmask,adv,advmask"          # lean arms for seeds 1-2 (queue is shared; rateb collapsed to chance at beta=3 and is dropped)
SEED0 = "raw,rate4,gtmask,adv,advmask,adv30,lmask"  # seed 0 also carries the sensitivity arms
PRIO = {"indep": 11, "corr_det": 11, "corr_noisy": 15, "corr_glued": 15}


def w(name, prio, cmd):
    open(os.path.join(SPEC, name + ".yaml"), "w").write(f"name: {name}\npriority: {prio}\ncmd: {cmd}\nhf_path: runs/{name}\n")


for kind, pr in PRIO.items():
    for seed in (0, 1, 2):
        w(f"b_m_{kind}_s{seed}", pr, f"{PRE} lam.py --kind {kind} --arms {SEED0 if seed == 0 else NOSAM} --epochs 30 --seed {seed} --out $OUT")
        w(f"b_msam_{kind}_s{seed}", pr + 1, f"until [ -f ~/work/data/b_mw/masks/sam_summary_strong_{kind}.json ]; do sleep 20; done; {PRE} lam.py --kind {kind} --arms sam --epochs 30 --seed {seed} --out $OUT")
    w(f"b_sam_{kind}", pr - 1, f"cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python sam_masks.py --kind {kind} --splits train,val,test --out ~/work/data/b_mw/masks && cp ~/work/data/b_mw/masks/sam_summary_strong_{kind}.json $OUT/")
w("b_sam_none", 14, "cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python sam_masks.py --kind none --splits test --out ~/work/data/b_mw/masks && cp ~/work/data/b_mw/masks/sam_summary_strong_none.json $OUT/")
for seed in (0, 1, 2):
    w(f"b_m_clean_s{seed}", 11, f"{PRE} lam.py --kind none --arms clean --epochs 30 --seed {seed} --out $OUT")
