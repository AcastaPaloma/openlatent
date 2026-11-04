"""Floors (release generators) and the attacker suite for R-IMU.  Pre-registration: results_notes/H.md sections 2.3-2.4.

CLI:  python suite.py --arm lda3 --folds O,A,B --family tab|nn --targets id,act --out DIR [--pd_file PATH] [--threads 8]
Output per (arm, family): DIR/preds__<arm>__<family>.npz with, for every fold f: `f{f}_te_idx` (indices into A*), and `f{f}_{attacker}__{target}` = float16 scores (S, n_te, C).
Families: tab = linear, knn, lgbm, rf, svm_rbf (CPU) + tabular `mlp` (torch, CPU); nn = cnn1d, mlp_flat, transformer (window releases only; GPU if available).
"""
import os, sys, json, time, argparse, re, warnings, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

warnings.filterwarnings("ignore")
N_ID, N_ACT, N_GEN = 24, 4, 2
MAIN_TAB = ["linear", "mlp", "knn", "lgbm", "rf"]
HELD_TAB = ["svm_rbf"]
MAIN_NN = ["cnn1d", "mlp_flat"]
HELD_NN = ["transformer"]


# ------------------------------------------------------------------ floors
def _std_fit(F):
    mu, sd = F.mean(0), F.std(0) + 1e-6
    return mu, sd


def build_release(arm, S, PHI, tr_mask, fold, pd_file=None, seed=0):
    """Return (R, kind) with R aligned to A* windows; fitted on A* windows selected by tr_mask (training trials of the fold, j mod 4 == 0)."""
    X = S["X"]; N = len(X); rng = np.random.default_rng(10_000 + seed + zlib.crc32(arm.encode()) % 100_000)
    act = S["act"]
    if arm == "raw" or arm == "shuf_raw":
        return X, "win"
    m = re.fullmatch(r"noise_s([0-9.]+)", arm)
    if m:
        return (X + rng.standard_normal(X.shape).astype(np.float32) * float(m.group(1))).astype(np.float32), "win"
    m = re.fullmatch(r"quant_b(\d)", arm)
    if m:
        L = 2 ** int(m.group(1)); q = np.clip(np.floor((np.clip(X, -3, 3) + 3) / 6 * L), 0, L - 1)
        return (-3 + (q + 0.5) * 6 / L).astype(np.float32), "win"
    m = re.fullmatch(r"ds(\d+)", arm)
    if m:
        k = int(m.group(1)); return np.repeat(X.reshape(N, 2, 128 // k, k).mean(3), k, axis=2).astype(np.float32), "win"
    if arm == "demean":
        return (X - X.mean(2, keepdims=True)).astype(np.float32), "win"
    if arm == "ctrl_const":
        return np.zeros((N, 1), np.float32), "code"
    if arm == "ctrl_noise16":
        return rng.standard_normal((N, 16)).astype(np.float32), "code"
    if arm == "oracle_act":
        return np.eye(N_ACT, dtype=np.float32)[act], "code"
    if arm == "oracle_id":
        return np.eye(N_ID, dtype=np.float32)[S["sub"]], "code"
    if (arm.startswith("pd_") or arm in ("msda", "rep")) and pd_file is not None:
        d = np.load(pd_file); R = np.zeros((N, 2, 128), np.float32); R[d["idx"]] = d["release"]
        assert len(d["idx"]) == N, "PD release must cover all of A*"
        return R, "win"
    if arm.startswith("pdz60"):
        d = np.load(pd_file); Z = np.zeros((N, 60), np.float32); Z[d["idx"]] = d["z"]
        if arm.endswith("_r1"):
            sd = Z[tr_mask].std(0); Z = Z + rng.standard_normal(Z.shape).astype(np.float32) * sd
        return Z, "code"
    m = re.fullmatch(r"lda(\d)(?:_r([0-9.]+))?", arm)
    if m:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        mu, sd = _std_fit(PHI[tr_mask]); Fz = (PHI - mu) / sd
        lda = LDA(solver="svd").fit(Fz[tr_mask], act[tr_mask])
        Z = lda.transform(Fz)[:, :int(m.group(1))].astype(np.float32)
        if m.group(2):
            Z = Z + rng.standard_normal(Z.shape).astype(np.float32) * float(m.group(2)) * Z[tr_mask].std(0)
        return Z, "code"
    m = re.fullmatch(r"pcaphi(\d+)", arm)
    if m:
        from sklearn.decomposition import PCA
        mu, sd = _std_fit(PHI[tr_mask]); Fz = (PHI - mu) / sd
        return PCA(int(m.group(1)), random_state=0).fit(Fz[tr_mask]).transform(Fz).astype(np.float32), "code"
    m = re.fullmatch(r"pcaraw(\d+)", arm)
    if m:
        from sklearn.decomposition import PCA
        flat = X.reshape(N, -1)
        return PCA(int(m.group(1)), random_state=0).fit(flat[tr_mask]).transform(flat).astype(np.float32), "code"
    m = re.fullmatch(r"rp(\d+)(?:_seed(\d))?", arm)
    if m:
        k = int(m.group(1)); g = np.random.default_rng(777 + int(m.group(2) or 0)); G = g.standard_normal((256, k)).astype(np.float32) / np.sqrt(k)
        return X.reshape(N, -1) @ G, "code"
    m = re.fullmatch(r"act(\d+)(?:_r([0-9.]+))?", arm)
    if m:
        return act_bottleneck(PHI, act, tr_mask, int(m.group(1)), float(m.group(2) or 0), seed), "code"
    if arm == "predlabel":  # H-A4 floor: release only the PREDICTED activity (hard one-hot) of an MLP trained on the fold's training windows
        return act_bottleneck(PHI, act, tr_mask, 60, 0.0, seed, onehot=True), "code"
    if arm == "predlabel_pd":  # H-A4 floor: one-hot of the argmax of PD's own shipped surrogate classifier head applied to its 60-d embedding
        import torch
        sd = torch.load(os.path.join(C.PD_REPO, "models", "privdiffuser_motion_1", "pub_classifier_motion_1.pt"), map_location="cpu")
        W, b = sd["fc6.weight"].numpy(), sd["fc6.bias"].numpy()
        d = np.load(pd_file); Z = np.zeros((N, 60), np.float32); Z[d["idx"]] = d["z"]
        return np.eye(N_ACT, dtype=np.float32)[(Z @ W.T + b).argmax(1)], "code"
    raise ValueError(f"unknown arm {arm}")


def act_bottleneck(PHI, act, tr_mask, k, r, seed, onehot=False):
    """Label-supervised wide code: MLP activity classifier on standardised phi with an L2-normalised k-d bottleneck; release = bottleneck (+ N(0,(r sd)^2))."""
    import torch
    from torch import nn
    torch.manual_seed(seed); torch.set_num_threads(8)
    mu, sd = _std_fit(PHI[tr_mask]); Fz = torch.from_numpy(((PHI - mu) / sd).astype(np.float32))
    net = nn.Sequential(nn.Linear(Fz.shape[1], 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, k))
    head = nn.Linear(k, N_ACT)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt = Fz[tr_mask]; yt = torch.from_numpy(act[tr_mask].astype(np.int64))
    for ep in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]; z = nn.functional.normalize(net(Xt[b]), dim=-1)
            loss = nn.functional.cross_entropy(head(z), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Zt = nn.functional.normalize(net(Fz), dim=-1)
        if onehot:
            return np.eye(N_ACT, dtype=np.float32)[head(Zt).argmax(1).numpy()]
        Z = Zt.numpy().astype(np.float32)
    if r > 0:
        rng = np.random.default_rng(55 + seed); Z = Z + rng.standard_normal(Z.shape).astype(np.float32) * r * Z[tr_mask].std(0)
    return Z


# ------------------------------------------------------------------ attackers
def _fit_predict_tab(name, Ftr, ytr, Fte, ncls, seed, threads):
    mu, sd = _std_fit(Ftr)
    if np.all(sd <= 1e-5 + 1e-6):  # constant release: no information, uniform scores
        return np.full((len(Fte), ncls), 1.0 / ncls, np.float32)
    A, B = ((Ftr - mu) / sd).astype(np.float32), ((Fte - mu) / sd).astype(np.float32)
    cls = None
    if name == "linear":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=500, n_jobs=threads).fit(A, ytr)
        P = m.predict_proba(B); cls = m.classes_
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        P = KNeighborsClassifier(15, metric="cosine", n_jobs=threads).fit(A, ytr).predict_proba(B)
    elif name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, colsample_bytree=0.8, n_jobs=threads, random_state=seed, verbose=-1).fit(A, ytr)
        P = m.predict_proba(B); cls = m.classes_
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(500, min_samples_leaf=2, n_jobs=threads, random_state=seed).fit(A, ytr); P = m.predict_proba(B); cls = m.classes_
    elif name == "svm_rbf":
        from sklearn.svm import SVC
        P = SVC(C=10, kernel="rbf", gamma="scale", decision_function_shape="ovr").fit(A, ytr).decision_function(B)
        if ncls == 2: P = np.stack([-P, P], 1)
    elif name == "mlp":
        P = _torch_fit_predict("mlp_tab", A, ytr, B, ncls, seed)
    else:
        raise ValueError(name)
    out = np.zeros((len(Fte), ncls), np.float32)
    if cls is not None:
        out[:, cls] = P
    else:
        out[:] = P
    return out


def _make_net(kind, in_shape, ncls):
    import torch
    from torch import nn
    if kind == "mlp_tab":
        d = int(np.prod(in_shape)); return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, ncls))
    if kind == "mlp_flat":  # PD-style MLP on the flat 256-d window
        return nn.Sequential(nn.Flatten(), nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, ncls))
    if kind == "cnn1d":
        def blk(i, o, pool):
            L = [nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU()]
            return L + ([nn.MaxPool1d(2)] if pool else [])
        return nn.Sequential(*blk(2, 32, True), *blk(32, 64, True), *blk(64, 128, True), *blk(128, 128, False), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, ncls))
    if kind == "transformer":
        class T(nn.Module):
            def __init__(s):
                super().__init__(); s.emb = nn.Linear(16, 64); s.pos = nn.Parameter(torch.zeros(1, 16, 64))
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2); s.fc = nn.Linear(64, ncls)
            def forward(s, x):  # (B,2,128) -> 16 patches of 8 samples x 2 channels
                B = x.shape[0]; p = x.reshape(B, 2, 16, 8).permute(0, 2, 1, 3).reshape(B, 16, 16)
                return s.fc(s.enc(s.emb(p) + s.pos).mean(1))
        return T()
    raise ValueError(kind)


EPOCHS = dict(mlp_tab=30, mlp_flat=30, cnn1d=20, transformer=20)


def _torch_fit_predict(kind, Xtr, ytr, Xte, ncls, seed):
    import torch
    from torch import nn
    dev = torch.device("cuda:0" if (torch.cuda.is_available() and kind != "mlp_tab") else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = _make_net(kind, Xtr.shape[1:], ncls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32)).to(dev); yt = torch.from_numpy(ytr.astype(np.int64)).to(dev)
    Xe = torch.from_numpy(np.ascontiguousarray(Xte, dtype=np.float32)).to(dev)
    for ep in range(EPOCHS[kind]):
        net.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            if len(b) < 2: continue
            loss = nn.functional.cross_entropy(net(Xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xe), 2048):
            out.append(torch.softmax(net(Xe[i:i + 2048]), -1).cpu().numpy())
    return np.concatenate(out)


def seeds_for(name):
    return [0, 1, 2] if name in ("mlp", "cnn1d", "mlp_flat", "transformer") else [0]


def labels_for(S, target):
    return {"id": S["sub"], "act": S["act"], "gen": C.gender_by_subject()[S["sub"]]}[target], {"id": N_ID, "act": N_ACT, "gen": N_GEN}[target]


def run(arm, folds, family, targets, out_dir, pd_file=None, threads=8, held=True, tag=""):
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    S = C.analysis_set(); N = len(S["X"]); PHI = C.phi(S["X"])
    pack = {}
    for f in folds:
        te_trials = np.array([(C.ACTS[a], int(t)) in C.FOLD_TEST[f] for a, t in zip(S["act"], S["trial"])])
        tr_mask = (~te_trials) & S["is_train_window"]
        te_mask = te_trials  # all A* windows of the test trials (fold O: stride 20; folds A/B: stride 40)
        R, kind = build_release(arm, S, PHI, tr_mask, f, pd_file)
        if family == "nn" and kind != "win":
            continue
        Fr = C.phi(R) if (kind == "win" and family == "tab") else R
        te_idx = np.where(te_mask)[0]; pack[f"f{f}_te_idx"] = te_idx.astype(np.int32)
        names = (MAIN_TAB + (HELD_TAB if held else [])) if family == "tab" else (MAIN_NN + (HELD_NN if held else []))
        for tg in targets:
            y, ncls = labels_for(S, tg)
            ytr = y[tr_mask].copy()
            if arm == "shuf_raw" and tg == "id":  # trial-level permutation of the subject labels of the training trials
                keys = C.trial_key(S)[tr_mask]; uk = np.unique(keys); rr = np.random.default_rng(99).permutation(N_ID)
                m = {k: rr[int(k // 1000)] for k in uk}; ytr = np.array([m[k] for k in keys])
            for nm in names:
                Ps = []
                for sd in seeds_for(nm):
                    if family == "tab":
                        if nm == "mlp":
                            Ps.append(_fit_predict_tab("mlp", Fr[tr_mask], ytr, Fr[te_mask], ncls, sd, threads))
                        else:
                            Ps.append(_fit_predict_tab(nm, Fr[tr_mask], ytr, Fr[te_mask], ncls, sd, threads))
                    else:
                        mu = Fr[tr_mask].mean((0, 2), keepdims=True); sdv = Fr[tr_mask].std((0, 2), keepdims=True) + 1e-6
                        Ps.append(_torch_fit_predict(nm, (Fr[tr_mask] - mu) / sdv, ytr, (Fr[te_mask] - mu) / sdv, ncls, sd))
                pack[f"f{f}_{nm}__{tg}"] = np.stack(Ps).astype(np.float16)
        print(f"[{time.time()-t0:.0f}s] arm {arm} fold {f} family {family} done", flush=True)
    np.savez_compressed(os.path.join(out_dir, f"preds__{arm}{tag}__{family}.npz"), **pack)
    json.dump(dict(arm=arm, tag=tag, family=family, folds=folds, targets=targets, pd_file=pd_file, seconds=round(time.time() - t0, 1), n_astar=int(N)), open(os.path.join(out_dir, f"preds__{arm}{tag}__{family}.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="one arm or a comma-separated list (run sequentially in this process)"); ap.add_argument("--folds", default="O,A,B"); ap.add_argument("--family", default="tab", choices=["tab", "nn"])
    ap.add_argument("--targets", default="id,act"); ap.add_argument("--out", required=True); ap.add_argument("--pd_file", default=None)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--no_held", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch; torch.set_num_threads(a.threads)
    for arm in a.arm.split(","):
        if os.path.exists(os.path.join(a.out, f"preds__{arm}{a.tag}__{a.family}.npz")):
            print("skip existing", arm, flush=True); continue
        run(arm, a.folds.split(","), a.family, a.targets.split(","), a.out, a.pd_file, a.threads, held=not a.no_held, tag=a.tag)
