s = open("train_arms.py").read()
# 1) new input stage: identity-LDA erasure (L<k>A3 / L<k>) and constant-latent control (K0)
old = 'Xt = torch.tensor(inp, dtype=torch.float32, device=dev)'
new = '''if ARM.startswith("L"):
    KID = int(ARM[1:].split("A")[0]); Xp = PCA(256, random_state=0).fit(X[fit]).transform(X); Xf_ = Xp[fit]; pf_ = pid[fit]
    mp = {p: Xf_[pf_ == p].mean(0) for p in np.unique(pf_)}; Mu = np.stack([mp[p] for p in pf_]); Rw = Xf_ - Mu
    Sw = Rw.T @ Rw / len(Rw) + 1e-3 * np.eye(256); Pm = np.stack(list(mp.values())); Pc = Pm - Pm.mean(0); Sb = Pc.T @ Pc / len(Pc)
    w_, V_ = np.linalg.eigh(Sw); Wm = V_ @ np.diag(w_ ** -.5) @ V_.T; Wi = V_ @ np.diag(w_ ** .5) @ V_.T
    e_, U_ = np.linalg.eigh(Wm @ Sb @ Wm); U_ = U_[:, np.argsort(-e_)[:KID]]; Pe = np.eye(256) - Wi @ U_ @ U_.T @ Wm
    mx_ = Xp[fit].mean(0); inp = ((Xp - mx_) @ Pe.T + mx_).astype(np.float32); DIN = 256
Xt = torch.tensor(inp, dtype=torch.float32, device=dev)'''
assert old in s; s = s.replace(old, new, 1)
old = 'LAM = {"A2": 1.,'
assert old in s
s = s.replace('ZD = 64 if ARM == "A11"', 'LAM = 3. if (ARM.startswith("L") and ARM.endswith("A3")) else LAM\nZD = 64 if ARM == "A11"', 1)
old = 'if ARM == "A0": Z = X.copy()'
new = 'if ARM == "A0": Z = X.copy()\nelif ARM == "K0": Z = np.zeros((len(X), 256), np.float32)\nelif ARM.startswith("L") and not ARM.endswith("A3"): Z = inp.copy()'
assert old in s; s = s.replace(old, new, 1)
# 2) richer utility: per-label AUC, patient-bootstrap CI, nonlinear (MLP) probe
old = "ok = ~np.isnan(sex); sx ="
new = '''cols = [c_ for j, c_ in enumerate(lab.columns) if c_ in au]; Sc = {}
for j, c_ in enumerate(lab.columns):
    if c_ in au: Sc[c_] = LogisticRegression(C=best_c, max_iter=2000).fit(Zs[fit], Y[fit][:, j]).decision_function(Zs[ev])
pe = pid[ev]; Ye = Y[ev]; rb = np.random.RandomState(1); ups = np.unique(pe); ub = []
for _ in range(100):
    ch = rb.choice(ups, len(ups)); ix = np.concatenate([np.where(pe == p)[0] for p in ch]); a_ = []
    for j, c_ in enumerate(lab.columns):
        if c_ in Sc and len(set(Ye[ix, j])) > 1: a_.append(roc_auc_score(Ye[ix, j], Sc[c_][ix]))
    ub.append(np.mean(a_))
ut_ci = [float(np.percentile(ub, 2.5)), float(np.percentile(ub, 97.5))]
mp_ = nn.Sequential(nn.Linear(Zs.shape[1], 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, Y.shape[1])).to(dev); om = torch.optim.AdamW(mp_.parameters(), lr=1e-3, weight_decay=1e-2)
Zq = torch.tensor(Zs, dtype=torch.float32, device=dev); Yq = torch.tensor(np.nan_to_num(Y), device=dev); fi_ = torch.tensor(np.where(fit)[0], device=dev); torch.manual_seed(SEED)
for _ in range(400):
    b_ = fi_[torch.randint(len(fi_), (256,), device=dev)]; l = F.binary_cross_entropy_with_logits(mp_(Zq[b_]), Yq[b_]); om.zero_grad(); l.backward(); om.step()
with torch.no_grad(): Pm_ = mp_.eval()(Zq).cpu().numpy()
mlp_auc = float(np.mean([roc_auc_score(Y[ev][:, j], Pm_[ev][:, j]) for j in range(Y.shape[1]) if len(set(Y[fit][:, j])) > 1 and len(set(Y[ev][:, j])) > 1]))
ok = ~np.isnan(sex); sx ='''
assert old in s; s = s.replace(old, new, 1)
old = "utility_C=best_c,"
new = "utility_C=best_c, utility_ci95=ut_ci, utility_mlp_auc=mlp_auc, utility_per_label={k_: float(v) for k_, v in au.items()},"
assert old in s; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("patched ok")
