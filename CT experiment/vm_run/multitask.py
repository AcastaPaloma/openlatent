# multitask.py N TAG ARM... : extra utility task models on SAVED latents (no retraining). Exam rows = ~fit mask stored in Z file.
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, r2_score
from joblib import Parallel, delayed
N = sys.argv[1]; arms = sys.argv[2:]; D = "/home/ubuntu/ct"
lab = pd.concat([pd.read_csv(f"{D}/train_predicted_labels.csv"), pd.read_csv(f"{D}/valid_predicted_labels.csv")]).set_index("VolumeName")
md = pd.concat([pd.read_csv(f"{D}/train_metadata.csv"), pd.read_csv(f"{D}/validation_metadata.csv")]).set_index("VolumeName")
def one(arm, seed):
    z = np.load(f"{D}/runs/Z_{arm}_s{seed}_n{N}.npz", allow_pickle=True); Z = z["Z"].astype(np.float32); names = z["names"]; fit = z["fit"]
    Zs = (Z - Z[fit].mean(0)) / (Z[fit].std(0) + 1e-6); Y = lab.reindex(names).values.astype(np.float32); ev = ~fit
    # HistGB and kNN on 256-d latents (fresh, nonlinear task models); mean AUC over the 18 labels
    cols = [j for j in range(Y.shape[1]) if len(set(Y[fit][:, j])) > 1 and len(set(Y[ev][:, j])) > 1]
    from sklearn.decomposition import PCA
    pca = PCA(64, random_state=0).fit(Zs[fit]).transform(Zs)   # same 64-d view for every arm (raw 4096-d included)
    g = np.mean([roc_auc_score(Y[ev][:, j], HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0).fit(pca[fit], Y[fit][:, j]).predict_proba(pca[ev])[:, 1]) for j in cols])
    k = np.mean([roc_auc_score(Y[ev][:, j], KNeighborsClassifier(25).fit(Zs[fit], Y[fit][:, j]).predict_proba(Zs[ev])[:, 1]) for j in cols])
    age = pd.to_numeric(md.reindex(names).PatientAge.astype(str).str.extract(r"(\d+)")[0], errors="coerce").values; ok = ~np.isnan(age)
    r2 = r2_score(age[ev & ok], RidgeCV(alphas=[1, 10, 100, 1000]).fit(Zs[fit & ok], age[fit & ok]).predict(Zs[ev & ok]))
    return arm, seed, g, k, r2
jobs = [(a, s) for a in arms for s in (0, 1, 2)]
res = Parallel(n_jobs=24)(delayed(one)(a, s) for a, s in jobs)
print("arm        HistGB-AUC  kNN-AUC  age-R2(leak)")
raw = None
for a in arms:
    v = np.array([[r[2], r[3], r[4]] for r in res if r[0] == a]); m = v.mean(0)
    if a == "A0": raw = m
    print("%-10s %.3f       %.3f    %.3f   %s" % (a, m[0], m[1], m[2], "" if raw is None else "kept(HGB) %.2f kept(kNN) %.2f" % ((m[0] - .5) / (raw[0] - .5), (m[1] - .5) / (raw[1] - .5))))
