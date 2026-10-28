
# train_arms.py ARM SEED [STEPS]  -- pre-registered arms A0..A7 (A6 pending). Reads/writes Drive only.
import os, sys, json, time, hashlib, ast, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
D = "/content/drive/MyDrive/openlatent_ct"; ARM, SEED = sys.argv[1], int(sys.argv[2]); STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.RandomState(SEED)
h = lambda p: int(hashlib.sha256(f"ctrate-openlatent-v1:{p}".encode()).hexdigest()[:8], 16) / 16**8
man = pd.read_csv(f"{D}/manifest_train.csv"); have = {f[:-4] for f in os.listdir(f"{D}/feats")}
man = man[man.name.map(lambda n: n[:-4] in have)].reset_index(drop=True)
fz = f"{D}/frozen_scans.txt"
if os.path.exists(fz): man = man[man.name.isin(set(open(fz).read().split()))].reset_index(drop=True)
cnt = man.groupby("patient").scan.nunique(); man = man[man.patient.isin(cnt[cnt >= 2].index)].reset_index(drop=True)
mv = pd.read_csv(f"{D}/manifest_valid.csv"); mv = mv[mv.name.map(lambda n: n[:-4] in have)]; cv = mv.groupby("patient").scan.nunique(); mv = mv[mv.patient.isin(cv[cv >= 2].index)]
NTR = len(man); man = pd.concat([man, mv], ignore_index=True)
md = pd.concat([pd.read_csv(f"{D}/train_metadata.csv"), pd.read_csv(f"{D}/validation_metadata.csv")]).set_index("VolumeName").loc[man.name]
lab = pd.concat([pd.read_csv(f"{D}/train_predicted_labels.csv"), pd.read_csv(f"{D}/valid_predicted_labels.csv")]).set_index("VolumeName"); Y = lab.reindex(man.name).values.astype(np.float32)
S = np.stack([np.load(f"{D}/feats/{n[:-4]}.npz")["s"].astype(np.float32) for n in man.name])
fit = man.patient.map(lambda p: h(p) < 0.5).values & (np.arange(len(man)) < NTR); ev = np.arange(len(man)) >= NTR
mu, sd = S[fit].mean(0), S[fit].std(0) + 1e-6; X = (S - mu) / sd
pid = pd.factorize(man.patient)[0]; sex = md.PatientSex.map({"M": 1, "F": 0}).values
sc = (md.ManufacturerModelName.astype(str) + "|" + md.ConvolutionKernel.astype(str)).values
vc = pd.Series(sc[fit]).value_counts(); sc = np.where(pd.Series(sc).map(vc).fillna(0).values >= 5, sc, "other"); scid = pd.factorize(sc)[0]
print(f"{ARM} seed {SEED}: {len(man)} scans | fit {fit.sum()} eval {ev.sum()} | scanner classes {scid.max()+1}", flush=True)

# ---------- pair sampler (fit scans only) ----------
fi = np.where(fit)[0]; by_p = {}
for i in fi: by_p.setdefault(pid[i], []).append(i)
multi = [p for p, v in by_p.items() if len(v) >= 2]; by_sc = {}
for i in fi: by_sc.setdefault(scid[i], []).append(i)
def pairs(n):
    a, b, y = [], [], []
    for _ in range(n // 2):
        p = multi[rng.randint(len(multi))]; i, j = rng.choice(by_p[p], 2, replace=False); a.append(i); b.append(j); y.append(1)
    for _ in range(n // 2):
        i = fi[rng.randint(len(fi))]; pool = by_sc[scid[i]] if rng.rand() < 0.5 else fi
        j = pool[rng.randint(len(pool))]
        while pid[j] == pid[i]: j = fi[rng.randint(len(fi))]
        a.append(i); b.append(j); y.append(0)
    return np.array(a), np.array(b), np.array(y, dtype=np.float32)

# ---------- models ----------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(c, x, l): c.l = l; return x.view_as(x)
    @staticmethod
    def backward(c, g): return -c.l * g, None
def mlp(i, h_, o): return nn.Sequential(nn.Linear(i, h_), nn.GELU(), nn.Linear(h_, o))
def pair_head(d, w): return mlp(2 * d, w, 1)
pf = lambda z1, z2: torch.cat([(z1 - z2).abs(), z1 * z2], 1)
def dcor(x, z):
    def dc(a): d = torch.cdist(a, a); return d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()
    A, B = dc(x), dc(z); return (A * B).mean() / ((A * A).mean().sqrt() * (B * B).mean().sqrt() + 1e-9)
def leace_fit(Xf, C):
    """Closed-form linear erasure of concept matrix C from Xf (n x d). Returns (mean, P) with x' = P (x - mean) + mean."""
    mx = Xf.mean(0); Xc = Xf - mx; Cc = C - C.mean(0)
    cov = Xc.T @ Xc / len(Xc) + 1e-3 * np.eye(Xc.shape[1]); w, V = np.linalg.eigh(cov); W = V @ np.diag(w ** -.5) @ V.T; Wi = V @ np.diag(w ** .5) @ V.T
    M = W @ (Xc.T @ Cc / len(Xc)); U, s_, _ = np.linalg.svd(M, full_matrices=False); U = U[:, s_ > 1e-6]
    return mx, np.eye(Xc.shape[1]) - Wi @ U @ U.T @ W

inp = X; DIN = X.shape[1]
if ARM == "A7":
    pca = PCA(256, random_state=0).fit(X[fit]); Xp = pca.transform(X); ok = ~np.isnan(sex)
    Cm = np.hstack([sex[:, None].astype(float), np.eye(scid.max() + 1)[scid]]); mx, P = leace_fit(Xp[fit & ok], Cm[fit & ok])
    inp = ((Xp - mx) @ P.T + mx).astype(np.float32); DIN = 256
Xt = torch.tensor(inp, dtype=torch.float32, device=dev); Yt = torch.tensor(np.nan_to_num(Y), device=dev); Yok = ~np.isnan(Y)
LAM = {"A2": 1., "A3": 3., "A4": 1., "A8": 10., "A9": 3., "A10": 3., "A11": 3.}.get(ARM, 0.)
ZD = 64 if ARM == "A11" else 256; HW = (1024, 2048) if ARM == "A10" else (256, 512)
if ARM == "A10": STEPS *= 2
t0 = time.time()

def train():
    enc = nn.Sequential(mlp(DIN, 1024, ZD), nn.LayerNorm(ZD)).to(dev); task = nn.Linear(ZD, Y.shape[1]).to(dev)
    heads = lambda: [pair_head(ZD, HW[0]).to(dev), pair_head(ZD, HW[1]).to(dev)]
    sh_ = lambda: nn.Linear(ZD, scid.max() + 1).to(dev)
    adv, sadv = heads(), sh_(); mk = lambda: torch.optim.AdamW([p for m in adv + [sadv] for p in m.parameters()], lr=1e-3)
    oa = mk(); oe = torch.optim.AdamW(list(enc.parameters()) + list(task.parameters()), lr=1e-3, weight_decay=1e-2)
    for step in range(STEPS):
        if LAM > 0 and step % 100 == 0 and step > 0: adv, sadv = heads(), sh_(); oa = mk()
        a, b, y = pairs(256); z1, z2 = enc(Xt[a]), enc(Xt[b]); yt = torch.tensor(y, device=dev)
        m = torch.tensor(Yok[a], device=dev).float(); loss = (F.binary_cross_entropy_with_logits(task(z1), Yt[a], reduction="none") * m).sum() / m.sum().clamp(min=1)
        if LAM > 0:
            g1, g2 = GRL.apply(z1, LAM), GRL.apply(z2, LAM)
            loss = loss + sum(F.binary_cross_entropy_with_logits(hd(pf(g1, g2)).squeeze(1), yt) for hd in adv) / len(adv)
            if ARM in ("A4", "A9"): loss = loss + F.cross_entropy(sadv(GRL.apply(z1, 1.)), torch.tensor(scid[a], device=dev))
        if ARM == "A5": loss = loss + 1.0 * dcor(Xt[a], z1)
        oe.zero_grad(); oa.zero_grad(); loss.backward(); oe.step(); oa.step()
        if step % 300 == 0: print(f"  step {step} loss {loss.item():.3f}", flush=True)
    return enc.eval()
if ARM == "A0": Z = X.copy()
elif ARM == "B1": Z = PCA(256, random_state=0).fit(X[fit]).transform(X).astype(np.float32)
elif ARM == "B2": Z = X @ (np.random.RandomState(0).randn(X.shape[1], 256) / 16).astype(np.float32)
elif ARM == "B3": Z = X + float(sys.argv[4]) * np.random.RandomState(SEED).randn(*X.shape).astype(np.float32)
else:
    enc = train()
    with torch.no_grad(): Z = enc(Xt).cpu().numpy()
zf = lambda Zm: (Zm - Zm[fit].mean(0)) / (Zm[fit].std(0) + 1e-6)

# ---------- evaluation: fresh attackers, all trained on fit latents only ----------
d = man.assign(i=np.arange(len(man)))[ev].sort_values(["patient", "date", "scan"]); gal = d.groupby("patient").head(1); q = d.drop(gal.index)
gi, qi = gal.i.values, q.i.values; truth = np.array([np.where(gal.patient.values == p)[0][0] for p in q.patient.values]); NG = len(gi)
def rank_from_sim(sim):
    ts = sim[np.arange(len(qi)), truth][:, None]; return 1 + (sim > ts).sum(1) + 0.5 * ((sim == ts).sum(1) - 1)
def cos(A, B): A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9); B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9); return A @ B.T
def summ(r): return dict(top1=float((r <= 1).mean()), top5=float((r <= 5).mean()), mrr=float((1 / r).mean()), hits=(r <= 1).astype(int).tolist())
Zs = zf(Z); res = {}
res["knn_cosine"] = summ(rank_from_sim(cos(Zs[qi], Zs[gi])))
k = min(128, int(fit.sum()) - 1, Zs.shape[1]); pw = PCA(k, whiten=True, random_state=0).fit(Zs[fit]); Zw = pw.transform(Zs); res["knn_pca_white"] = summ(rank_from_sim(cos(Zw[qi], Zw[gi])))
def pair_scores(model):
    s = np.zeros((len(qi), NG), np.float32); Zt = torch.tensor(Zs, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for r_ in range(len(qi)): s[r_] = model(pf(Zt[qi[r_]].expand(NG, -1), Zt[gi])).squeeze(1).cpu().numpy()
    return s
def fit_pair(model, steps=1200):
    o = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2); Zt = torch.tensor(Zs, dtype=torch.float32, device=dev)
    for _ in range(steps):
        a, b, y = pairs(256); l = F.binary_cross_entropy_with_logits(model(pf(Zt[a], Zt[b])).squeeze(1), torch.tensor(y, device=dev)); o.zero_grad(); l.backward(); o.step()
    return model.eval()
dz = Zs.shape[1]
res["pair_logistic"] = summ(rank_from_sim(pair_scores(fit_pair(nn.Linear(2 * dz, 1).to(dev)))))
res["pair_mlp"] = summ(rank_from_sim(pair_scores(fit_pair(mlp(2 * dz, 512, 1).to(dev)))))
emb = mlp(dz, 512, 128).to(dev); o = torch.optim.AdamW(emb.parameters(), lr=1e-3, weight_decay=1e-2); Zt = torch.tensor(Zs, dtype=torch.float32, device=dev)
for _ in range(1200):
    a, b, _y = pairs(256); n_ = len(a) // 2; ea, eb = F.normalize(emb(Zt[a[:n_]]), dim=1), F.normalize(emb(Zt[b[:n_]]), dim=1)
    lg = ea @ eb.T / 0.1; t_ = torch.arange(n_, device=dev); l = (F.cross_entropy(lg, t_) + F.cross_entropy(lg.T, t_)) / 2; o.zero_grad(); l.backward(); o.step()
with torch.no_grad(): E = F.normalize(emb(Zt), dim=1).cpu().numpy()
res["siamese_contrastive"] = summ(rank_from_sim(E[qi] @ E[gi].T))
best = max(res, key=lambda n: res[n]["top1"]); hits = np.array(res[best]["hits"]); qp = q.patient.values; up = np.unique(qp); br = np.random.RandomState(0)
boot = [np.mean(np.concatenate([hits[qp == p] for p in br.choice(up, len(up))])) for _ in range(300)]
# inversion attack: decoder g: z -> standardized raw features (trained on fit), then the raw-trained PCA-256 cosine attacker on g(z)
gdec = mlp(dz, 1024, X.shape[1]).to(dev); og = torch.optim.AdamW(gdec.parameters(), lr=1e-3, weight_decay=1e-2); Xtr = torch.tensor(X, dtype=torch.float32, device=dev); fidx = torch.tensor(np.where(fit)[0], device=dev)
for _ in range(800):
    b_ = fidx[torch.randint(len(fidx), (256,), device=dev)]; l = F.mse_loss(gdec(Zt[b_]), Xtr[b_]); og.zero_grad(); l.backward(); og.step()
with torch.no_grad(): Xh = gdec(Zt).cpu().numpy()
praw = PCA(256, random_state=0).fit(X[fit]); Ph = praw.transform(Xh); inv = summ(rank_from_sim(cos(Ph[qi], Ph[gi])))
np.savez_compressed(f"{D}/sealed/Z_{ARM}{('_sigma' + sys.argv[4]) if ARM == 'B3' else ''}_s{SEED}_n{len(man)}.npz", Z=Z.astype(np.float16), names=man.name.values, fit=fit)
# metadata floor (same features as run_s2.py)
cat = ["Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "SeriesDescription", "PatientPosition", "ExposureModulationType"]
num = ["ReconstructionDiameter", "TableHeight", "XRayTubeCurrent", "Exposure", "CTDIvol", "Rows", "Columns", "NumberofSlices", "ZSpacing", "RescaleIntercept"]
A_ = pd.get_dummies(md[cat].astype(str)).astype(float).values; N_ = md[num].apply(pd.to_numeric, errors="coerce"); N_ = ((N_ - N_.mean()) / (N_.std() + 1e-6)).fillna(0).values
xy = md.XYSpacing.map(lambda s_: ast.literal_eval(s_)[0] if isinstance(s_, str) else np.nan); xy = ((xy - xy.mean()) / (xy.std() + 1e-6)).fillna(0).values[:, None]
Mf = np.hstack([A_, N_, xy]); Mf = Mf - Mf[fit].mean(0); floor = summ(rank_from_sim(cos(Mf[qi], Mf[gi])))["top1"]
# utility + sex
gk = GroupKFold(3); best_c, best_a = None, -1
for C in (0.001, 0.003, 0.01, 0.03):
    a_ = []
    for tr, va in gk.split(Zs[fit], groups=pid[fit]):
        for j in range(Y.shape[1]):
            yj = Y[fit][:, j]
            if len(set(yj[tr])) < 2 or len(set(yj[va])) < 2: continue
            a_.append(roc_auc_score(yj[va], LogisticRegression(C=C, max_iter=1000).fit(Zs[fit][tr], yj[tr]).decision_function(Zs[fit][va])))
    if np.mean(a_) > best_a: best_c, best_a = C, np.mean(a_)
au = {c_: roc_auc_score(Y[ev][:, j], LogisticRegression(C=best_c, max_iter=2000).fit(Zs[fit], Y[fit][:, j]).decision_function(Zs[ev])) for j, c_ in enumerate(lab.columns) if len(set(Y[fit][:, j])) > 1 and len(set(Y[ev][:, j])) > 1}
ok = ~np.isnan(sex); sx = roc_auc_score(sex[ev & ok], LogisticRegression(C=0.1, max_iter=2000).fit(Zs[fit & ok], sex[fit & ok]).decision_function(Zs[ev & ok]))
out = dict(arm=ARM + (("_sigma" + sys.argv[4]) if ARM == "B3" else ""), seed=SEED, steps=STEPS, n_scans=len(man), n_gallery=NG, n_query=len(qi), chance_top1=1 / NG, metadata_floor_top1=floor,
           attackers={n: {k_: v for k_, v in r.items() if k_ != "hits"} for n, r in res.items()}, best_attacker=best, best_top1=res[best]["top1"], inversion_top1=inv["top1"], inversion_top5=inv["top5"],
           best_top1_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], utility_auc_mean=float(np.mean(list(au.values()))), utility_C=best_c, sex_auc=float(sx),
           seconds=time.time() - t0, config_hash=hashlib.sha256(f"{ARM}{STEPS}".encode()).hexdigest()[:8])
os.makedirs(f"{D}/sealed", exist_ok=True); json.dump(out, open(f"{D}/sealed/arm_{out["arm"]}_s{SEED}_n{len(man)}.json", "w"), indent=1); open(f"{D}/sealed/ledger.jsonl", "a").write(json.dumps(out) + "\n")
print(json.dumps({k_: v for k_, v in out.items() if k_ != "attackers"}, indent=1)); print({n: round(r["top1"], 3) for n, r in res.items()})
