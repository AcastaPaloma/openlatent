
# run_s2.py -- raw baseline on cached CT-FM features. Deterministic; reads/writes only Drive.
import os, json, hashlib, ast, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
D = "/content/drive/MyDrive/openlatent_ct"; os.makedirs(f"{D}/runs", exist_ok=True)
SALT = "ctrate-openlatent-v1"
def h(p): return int(hashlib.sha256(f"{SALT}:{p}".encode()).hexdigest()[:8], 16) / 16**8

man = pd.read_csv(f"{D}/manifest_train.csv"); md = pd.read_csv(f"{D}/train_metadata.csv").set_index("VolumeName")
lab = pd.read_csv(f"{D}/train_predicted_labels.csv").set_index("VolumeName")
have = {f[:-4] for f in os.listdir(f"{D}/feats")}
man = man[man.name.map(lambda n: n[:-4] in have)].reset_index(drop=True)
npat = man.patient.nunique(); multi = man.groupby("patient").scan.nunique(); man = man[man.patient.isin(multi[multi >= 2].index)].reset_index(drop=True)
G, S = [], []
for n in man.name:
    z = np.load(f"{D}/feats/{n[:-4]}.npz"); G.append(z["g"]); S.append(z["s"].astype(np.float32))
G, S = np.stack(G), np.stack(S)
man["role"] = man.patient.map(lambda p: "fit" if h(p) < 0.5 else "eval")
fit, ev = (man.role == "fit").values, (man.role == "eval").values
print(f"{len(man)} scans, {man.patient.nunique()} patients with 2+ featurized scans | fit {fit.sum()} eval {ev.sum()}")

def cos_reid(F, m):
    """Gallery = earliest scan per eval patient, queries = later scans. Cosine kNN, ties at expected rank."""
    d = man[m].copy(); d["i"] = np.where(m)[0]; d = d.sort_values(["patient", "date", "scan"])
    gal = d.groupby("patient").head(1); q = d.drop(gal.index)
    Fg, Fq = F[gal.i.values], F[q.i.values]
    Fg = Fg / (np.linalg.norm(Fg, axis=1, keepdims=True) + 1e-9); Fq = Fq / (np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-9)
    sim = Fq @ Fg.T; gp = gal.patient.values; truth = np.array([np.where(gp == p)[0][0] for p in q.patient.values])
    ts = sim[np.arange(len(q)), truth][:, None]; rank = 1 + (sim > ts).sum(1) + 0.5 * ((sim == ts).sum(1) - 1)
    n = len(gal); return dict(top1=float((rank <= 1).mean()), top5=float((rank <= 5).mean()), mrr=float((1 / rank).mean()), chance_top1=1 / n, n_gallery=n, n_query=len(q))

def prep(X, k=256):
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6; Z = (X - mu) / sd
    p = PCA(min(k, fit.sum() - 1), whiten=True, random_state=0).fit(Z[fit]); return p.transform(Z)

out = dict(n_scans=len(man), n_patients=int(man.patient.nunique()), salt=SALT)
Xs = {"ctfm_global512": G, "ctfm_spatial4096": S}
for name, X in Xs.items():
    Zw = prep(X); r = {"reid_cosine_raw": cos_reid(X - X[fit].mean(0), ev), "reid_cosine_pcawhite": cos_reid(Zw, ev)}
    sx = md.loc[man.name, "PatientSex"].map({"M": 1, "F": 0}).values; ok = ~np.isnan(sx)
    c = LogisticRegression(C=0.1, max_iter=2000).fit(Zw[fit & ok], sx[fit & ok]); r["sex_auc"] = float(roc_auc_score(sx[ev & ok], c.decision_function(Zw[ev & ok])))
    age = md.loc[man.name, "PatientAge"].str.extract(r"(\d+)")[0].astype(float).values
    r["age_r2"] = float(r2_score(age[ev], Ridge(alpha=10).fit(Zw[fit], age[fit]).predict(Zw[ev])))
    Y = lab.reindex(man.name).values; aucs = {}
    for j, c_ in enumerate(lab.columns):
        y = Y[:, j]
        if np.isnan(y).any() or len(set(y[fit])) < 2 or len(set(y[ev])) < 2: continue
        aucs[c_] = float(roc_auc_score(y[ev], LogisticRegression(C=0.05, max_iter=2000).fit(Zw[fit], y[fit]).decision_function(Zw[ev])))
    r["abnormality_auc_mean"] = float(np.mean(list(aucs.values()))); r["abnormality_auc"] = aucs; out[name] = r

# metadata-only attacker: the shortcut floor an obfuscator must beat (no pixels, no dates)
cat = ["Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "SeriesDescription", "PatientPosition", "ExposureModulationType"]
num = ["ReconstructionDiameter", "TableHeight", "XRayTubeCurrent", "Exposure", "CTDIvol", "Rows", "Columns", "NumberofSlices", "ZSpacing", "RescaleIntercept"]
M = md.loc[man.name]; A = pd.get_dummies(M[cat].astype(str)).astype(float)
Nn = M[num].apply(pd.to_numeric, errors="coerce"); Nn = (Nn - Nn.mean()) / (Nn.std() + 1e-6); Nn = Nn.fillna(0)
xy = M.XYSpacing.map(lambda s: ast.literal_eval(s)[0] if isinstance(s, str) else np.nan); xy = ((xy - xy.mean()) / (xy.std() + 1e-6)).fillna(0)
Mf = np.hstack([A.values, Nn.values, xy.values[:, None]]); out["metadata_only_reid"] = cos_reid(Mf - Mf[fit].mean(0), ev)
print(json.dumps({k: (v if k.startswith("n_") or k == "salt" else {a: (b if not isinstance(b, dict) or a.startswith("reid") else "...") for a, b in v.items()} if isinstance(v, dict) and "sex_auc" in v else v) for k, v in out.items()}, indent=1)[:3500])
json.dump(out, open(f"{D}/runs/baseline_{len(man)}.json", "w"), indent=1)
