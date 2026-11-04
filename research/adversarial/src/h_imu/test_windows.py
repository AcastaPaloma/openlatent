"""Unit test: standardisation stats and window counts of common.build_windows vs PrivDiffuser's own DataSampler (cwd must be the PD clone for its relative paths)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

trials = C.load_trials()
mean, std = C.pd_train_stats(trials)
w = C.build_windows(trials, mean, std, stride=10)
print("trials", len(trials), "windows stride10", len(w["X"]))
out = dict(my_mean=mean.tolist(), my_std=std.tolist(), my_n_windows_stride10=int(len(w["X"])))
for f in C.FOLDS:
    tr, te = C.fold_masks(w, f)
    out[f"fold{f}"] = dict(train=int(tr.sum()), test=int(te.sum()), test_trials=int(len(np.unique(C.trial_key({k: v[te] for k, v in w.items()})))))
os.chdir(C.PD_REPO); sys.path.insert(0, C.PD_REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from dataset_loader import DataSampler
DS = DataSampler("./datasets/motionsense/")
out.update(pd_mean=np.asarray(DS.train_mean).tolist(), pd_std=np.asarray(DS.train_std).tolist(),
           pd_train_windows=int(DS.train_data.shape[0]), pd_test_windows=int(DS.test_data.shape[0]))
out["mean_close"] = bool(np.allclose(mean, DS.train_mean, atol=1e-9)); out["std_close"] = bool(np.allclose(std, DS.train_std, atol=1e-9))
# alignment-independent check: per-(sub,act) number of PD test windows vs mine
pd_cnt = np.bincount(DS.id_test.astype(int) * 4 + DS.act_test.astype(int), minlength=96)
tr, te = C.fold_masks(w, "O")
my_cnt = np.bincount(w["sub"][te] * 4 + w["act"][te], minlength=96)
out["max_abs_count_diff_per_sub_act"] = int(np.abs(pd_cnt - my_cnt).max())
out["rel_count_diff"] = float(np.abs(pd_cnt - my_cnt).sum() / pd_cnt.sum())
print(json.dumps(out, indent=1))
os.makedirs(os.path.expanduser("~/work/h_imu/out"), exist_ok=True)
json.dump(out, open(os.path.expanduser("~/work/h_imu/out/test_windows.json"), "w"), indent=1)
