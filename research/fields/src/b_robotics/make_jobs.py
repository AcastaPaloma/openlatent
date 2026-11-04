"""Write the main-matrix job specs: python make_jobs.py  (-> research/fields/jobs/specs/b_lam_<kind>_s<seed>.yaml)"""
import os
SPEC = os.path.join(os.path.dirname(__file__), "..", "..", "jobs", "specs")
ARMS = "raw,rate4,rateb,sam,gtmask,adv,adv30"
for kind in ["indep", "corr_det", "corr_noisy", "corr_glued"]:
    for seed in (0, 1, 2):
        name = f"b_lam_{kind}_s{seed}"
        open(os.path.join(SPEC, name + ".yaml"), "w").write(
            f"name: {name}\npriority: 14\ncmd: cd research/fields/src/b_robotics && ~/venvs/b/bin/python lam.py --kind {kind} --arms {ARMS} --epochs 30 --seed {seed} --out $OUT\nhf_path: runs/{name}\n")
for seed in (0, 1, 2):
    name = f"b_lam_clean_s{seed}"
    open(os.path.join(SPEC, name + ".yaml"), "w").write(
        f"name: {name}\npriority: 14\ncmd: cd research/fields/src/b_robotics && ~/venvs/b/bin/python lam.py --kind none --arms clean --epochs 30 --seed {seed} --out $OUT\nhf_path: runs/{name}\n")
