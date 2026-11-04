"""Write the main-matrix job specs (strong level): python make_jobs.py  (-> research/fields/jobs/specs/b_m_*.yaml)
Non-SAM arms and SAM arm are separate jobs so the SAM-free arms start while masks are computed."""
import os
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jobs", "specs")
PRE = "cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python"
NOSAM = "raw,rate4,gtmask,adv"          # lean arms for seeds 1-2 (queue is shared; rateb collapsed to chance and lmask/advmask collapsed to the degenerate 'mask out everything that moves' solution on corr_det s0, so they are dropped)
SEED0 = "raw,rate4,gtmask,adv,adv30"  # seed 0 also carries the sensitivity arms
PRIO = {"indep": 11, "corr_det": 11, "corr_noisy": 15, "corr_glued": 15}


def w(name, prio, cmd):
    open(os.path.join(SPEC, name + ".yaml"), "w").write(f"name: {name}\npriority: {prio}\ncmd: {cmd}\nhf_path: runs/{name}\n")


for kind, pr in PRIO.items():
    for seed in (0, 1, 2):
        w(f"b_m_{kind}_s{seed}", pr, f"{PRE} lam.py --kind {kind} --arms {SEED0 if seed == 0 else NOSAM} --epochs 30 --seed {seed} --out $OUT")
        w(f"b_msam_{kind}_s{seed}", pr + 1, f"until [ -f ~/work/data/b_mw/masks/sam_summary_strong_{kind}.json ]; do sleep 20; done; {PRE} lam.py --kind {kind} --arms sam --epochs 30 --seed {seed} --out $OUT")
    if kind in ("indep", "corr_det"):
        w(f"b_sam_{kind}", pr - 1, f"cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python sam_masks.py --kind {kind} --splits train,val,test --out ~/work/data/b_mw/masks && cp ~/work/data/b_mw/masks/sam_summary_strong_{kind}.json $OUT/")
    else:   # masks + the three sam-arm seeds in one job
        w(f"b_sam_{kind}", pr - 1, f"cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python sam_masks.py --kind {kind} --splits train,val,test --out ~/work/data/b_mw/masks && cp ~/work/data/b_mw/masks/sam_summary_strong_{kind}.json $OUT/ && for s in 0 1 2; do B_LEVEL=strong ~/venvs/b/bin/python lam.py --kind {kind} --arms sam --epochs 30 --seed $s --out $OUT; done")
# (b_sam_none dropped: only needed for the SAM arm's OOD-on-clean column)
# w("b_sam_none", 14, "cd research/fields/src/b_robotics && B_LEVEL=strong ~/venvs/b/bin/python sam_masks.py --kind none --splits test --out ~/work/data/b_mw/masks && cp ~/work/data/b_mw/masks/sam_summary_strong_none.json $OUT/")
for seed in (0, 1, 2):
    w(f"b_m_clean_s{seed}", 11, f"{PRE} lam.py --kind none --arms clean --epochs 30 --seed {seed} --out $OUT")

# weakly supervised arms (2000 action labels also train an action head on z): label-matched control `sup`, `supadv`, and the learned-mask candidate `smask`
for kind in ("corr_det", "indep"):
    for seed in (0, 1, 2):
        w(f"b_w_{kind}_s{seed}", 8 if kind == "corr_det" else 10, f"{PRE} lam.py --kind {kind} --arms sup,supadv,smask --epochs 30 --seed {seed} --out $OUT")
