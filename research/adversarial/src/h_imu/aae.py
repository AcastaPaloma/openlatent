"""Anonymizing AutoEncoder (`msda`, IoTDI'19, arXiv 1810.11546) and replacement AE (`rep`, W-P2DS'18): shipped in github.com/mmalekzadeh/motion-sense/msda (MIT).
Models are Keras 2.2.2 JSON + h5; loaded with the legacy `tf_keras` 2.19 package (pip --target ~/work/h_imu/pylib, TF_USE_LEGACY_KERAS=1).
Release = output 0 of the multi-output model (the decoder output, shape (N,2,128,1)); inference mode, deterministic.

  python aae.py rel --out DIR              # releases for the analysis set A*  -> msda_s0.npz, rep_s0.npz   (CPU)
  python aae.py r0  --out DIR [--runs 3]   # R0-AAE gate: PD-loader windows, raw-trained frozen Estimator CNNs (torch re-implementation) scored on raw / msda / rep
"""
import os, sys, json, time, argparse, io, contextlib
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.expanduser("~/work/h_imu/pylib"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C

MSDA_DIR = os.path.expanduser(os.environ.get("MSDA_DIR", "~/work/h_imu/msda"))


def load_anon(name):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tf_keras as keras
    m = keras.models.model_from_json(open(os.path.join(MSDA_DIR, f"{name}_anon_model.json")).read())
    m.load_weights(os.path.join(MSDA_DIR, f"{name}_anon_model_weights.h5"))
    return m


def anon(model, X):
    """X (N,2,128) standardised float32 -> release (N,2,128)."""
    out = model.predict(X.reshape(-1, 2, 128, 1).astype(np.float32), batch_size=1024, verbose=0)[0]
    return out.reshape(-1, 2, 128).astype(np.float32)


def run_rel(a):
    os.makedirs(a.out, exist_ok=True)
    S = C.analysis_set(); N = len(S["X"]); t0 = time.time()
    res = {}
    for nm in ["msda", "rep"]:
        m = load_anon(nm); R = anon(m, S["X"]); assert np.isfinite(R).all()
        np.savez_compressed(os.path.join(a.out, f"{nm}_s0.npz"), release=R, idx=np.arange(N))
        res[nm] = dict(n=N, mse_vs_input=float(np.mean((R - S["X"]) ** 2)), release_mean=R.mean((0, 2)).tolist(), release_std=R.std((0, 2)).tolist())
    res["input_mean_std"] = [S["X"].mean((0, 2)).tolist(), S["X"].std((0, 2)).tolist()]
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "aae_release.json"), "w"), indent=1); print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- R0-AAE
def _estimator(ncls):
    import torch
    from torch import nn

    def blk(i, o):
        return [nn.Conv2d(i, o, (2, 5), padding="same"), nn.ReLU(), nn.BatchNorm2d(o), nn.MaxPool2d((1, 2)), nn.Dropout(0.25)]
    return nn.Sequential(*blk(1, 64), *blk(64, 64), *blk(64, 32), *blk(32, 32), nn.Flatten(),
                         nn.Linear(32 * 2 * 8, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
                         nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.Linear(64, ncls))


def _train_estimator(Xtr, ytr, ncls, seed, dev):
    """Notebook protocol: shuffle(random_state=0), first 20% = validation, Adam 1e-3, batch 128, <= 50 epochs, early stop patience 5 on validation F1, class weights, L2 1e-3 on kernels."""
    import torch
    from torch import nn
    from sklearn.metrics import f1_score
    from sklearn.utils import shuffle
    Xtr, ytr = shuffle(Xtr, ytr, random_state=0)
    nv = len(Xtr) // 5
    Xv, yv, Xt, yt = Xtr[:nv], ytr[:nv], Xtr[nv:], ytr[nv:]
    torch.manual_seed(seed); np.random.seed(seed)
    net = _estimator(ncls).to(dev)
    cnt = np.bincount(yt, minlength=ncls).astype(np.float64); cw = torch.tensor(cnt.max() / np.maximum(cnt, 1), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt_t = torch.from_numpy(Xt).float().to(dev); yt_t = torch.from_numpy(yt).long().to(dev); Xv_t = torch.from_numpy(Xv).float().to(dev)
    wts = [m.weight for m in net if isinstance(m, (nn.Conv2d, nn.Linear))][:-1]
    best, best_state, bad = -1, None, 0
    for ep in range(50):
        net.train(); perm = torch.randperm(len(Xt_t), device=dev)
        for i in range(0, len(Xt_t), 128):
            b = perm[i:i + 128]
            if len(b) < 2: continue
            loss = nn.functional.cross_entropy(net(Xt_t[b]), yt_t[b], weight=cw) + 1e-3 * sum((w ** 2).sum() for w in wts)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xv_t[i:i + 2048]).argmax(1) for i in range(0, len(Xv_t), 2048)]).cpu().numpy()
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best: best, bad = f1, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    net.load_state_dict(best_state); net.eval()
    return net, dict(epochs=ep + 1, best_val_f1=float(best))


def _score(net, X, y, dev):
    import torch
    Xt = torch.from_numpy(X).float().to(dev)
    with torch.no_grad():
        p = torch.cat([net(Xt[i:i + 2048]).argmax(1) for i in range(0, len(Xt), 2048)]).cpu().numpy()
    return float((p == y).mean())


def run_r0(a):
    import torch
    from scipy.signal import resample
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    cwd = os.getcwd(); os.chdir(C.PD_REPO); sys.path.insert(0, C.PD_REPO)
    from dataset_loader import DataSampler  # PD's loader: the AAE notebooks use the identical windows (60,059 train / 13,344 test)
    with contextlib.redirect_stdout(io.StringIO()):
        DS = DataSampler("./datasets/motionsense/")
    os.chdir(cwd)
    Xtr = DS.train_data.astype(np.float32).reshape(-1, 1, 2, 128); Xte = DS.test_data.astype(np.float32).reshape(-1, 1, 2, 128)
    idtr, idte = DS.id_train.astype(int), DS.id_test.astype(int); actr, acte = DS.act_train.astype(int), DS.act_test.astype(int)
    rel = {"raw": Xte}
    for nm in ["msda", "rep"]:
        rel[nm] = anon(load_anon(nm), Xte.reshape(-1, 2, 128)).reshape(-1, 1, 2, 128)
    for hz in [10, 5]:  # resampling baselines of the notebook (scipy.signal.resample down then up)
        ns = (128 * hz) // 50; X2 = Xte.copy()
        d = np.stack([resample(x, ns, axis=-1) for x in Xte[:, 0]], 0)  # (N,2,ns)
        X2[:, 0] = resample(d, 128, axis=-1); rel[f"resample_{hz}Hz"] = X2.astype(np.float32)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = dict(n_train=int(len(Xtr)), n_test=int(len(Xte)), runs=[])
    for r in range(a.runs):
        row = dict(run=r)
        for tgt, ytr_, yte_, ncls in [("id", idtr, idte, 24), ("act", actr, acte, 4)]:
            net, info = _train_estimator(Xtr, ytr_, ncls, r, dev)
            row[f"{tgt}_train"] = info
            for k, X in rel.items():
                row[f"{tgt}_acc__{k}"] = _score(net, X, yte_, dev)
        out["runs"].append(row); print(json.dumps(row), flush=True)
    keys = [k for k in out["runs"][0] if "_acc__" in k]
    out["mean"] = {k: float(np.mean([r[k] for r in out["runs"]])) for k in keys}
    out["sd"] = {k: float(np.std([r[k] for r in out["runs"]])) for k in keys}
    pub = dict(raw_id=(95.5, 3.0), msda_id=(6.98, 3.0), rep_id=(15.92, 4.0))
    m = out["mean"]
    out["published"] = dict(raw_id_pct="94.9-96.2 (5 runs, notebook)", msda_id_pct=6.98, rep_id_pct=15.92, msda_act_claim=">92%", resample10_id_pct=31.08, resample5_id_pct=13.53)
    out["R0_AAE_pass"] = bool(abs(100 * m["id_acc__raw"] - pub["raw_id"][0]) <= pub["raw_id"][1] and abs(100 * m["id_acc__msda"] - pub["msda_id"][0]) <= pub["msda_id"][1]
                              and abs(100 * m["id_acc__rep"] - pub["rep_id"][0]) <= pub["rep_id"][1])
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "r0_aae.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["mean", "sd", "R0_AAE_pass", "published"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["rel", "r0"]); ap.add_argument("--out", required=True); ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    run_rel(a) if a.mode == "rel" else run_r0(a)
