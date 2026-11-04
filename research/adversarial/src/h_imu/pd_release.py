"""PrivDiffuser sampler (shipped checkpoint, MIT, commit 4a16222) and the R0 gate.  GPU job.

Architecture classes Surrogate_Utility_Classifier / Aux_Priv_Classifier are copied from PrivDiffuser.ipynb (motion branch) so that the shipped
state dicts load.  Sampling mirrors notebook cells 22-27: surrogate embedding -> conditional embedding -> DDIM (50 steps, eta 0, linear) with
classifier-free guidance w1 and negative privacy-classifier guidance w2 using the TRUE gender label of the window's subject.
NOTE (faithfulness): the notebook never calls diffusion.model.eval() when it loads the shipped checkpoint (the training loop does not run because
last_epoch == 80), so the UNet is in train mode and ResBlock dropout (p = 0.1) is active during sampling.  `--unet_eval 1` switches dropout off (sensitivity).

Usage:
  python pd_release.py r0   --out DIR [--limit N]                # PD's own protocol, PD loader, shipped Keras attackers
  python pd_release.py rel  --out DIR --name pd_pub --w1 7.8 --w2 0.05 --seed 0 [--limit N] [--unet_eval 0|1] [--set astar]
"""
import os, sys, json, time, argparse, io, contextlib
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

import torch
from torch import nn


class Surrogate_Utility_Classifier(nn.Module):  # copied from PrivDiffuser.ipynb cell 9 (motion branch)
    def __init__(self, num_classes, z_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1)
        self.fc2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.fc3 = nn.Linear(66560, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, z_dim)
        self.fc6 = nn.Linear(z_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.flatten(self.relu(self.fc2(h1)))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        h5 = self.relu(self.fc5(h4))
        z = torch.nn.functional.normalize(h5, dim=-1)
        return self.fc6(z), z


class Aux_Priv_Classifier(nn.Module):  # copied from PrivDiffuser.ipynb cell 19 (motion branch); cdim = 60
    def __init__(self, num_classes, z_dim, cdim=60):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1)
        self.fc2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.fc3 = nn.Linear(66560 + cdim, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, z_dim)
        self.fc6 = nn.Linear(z_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, emb):
        h1 = self.relu(self.fc1(x))
        h2 = self.flatten(self.relu(self.fc2(h1)))
        h2 = torch.cat((h2, emb), dim=1)
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        h5 = self.relu(self.fc5(h4))
        z = torch.nn.functional.normalize(h5, dim=-1)
        return self.fc6(z), z


class PD:
    def __init__(self, device, unet_eval=False):
        sys.path.insert(0, C.PD_REPO)
        from unet import Unet
        import diffusion as D
        from embedding import ConditionalEmbedding
        from utils import get_named_beta_schedule
        D.tqdm = lambda x, **k: x
        torch.cuda.empty_cache = lambda: None  # PD calls it every DDIM step
        self.D = D
        self.device = device
        mdir = os.path.join(C.PD_REPO, "models", "privdiffuser_motion_1")
        ep = torch.load(os.path.join(mdir, "last_epoch.pt"))["last_epoch"]
        assert ep == 80, ep
        net = Unet(in_ch=1, mod_ch=256, out_ch=1, ch_mul=[1, 2], num_res_blocks=2, cdim=60, use_conv=True, droprate=0.1, dtype=torch.float32)
        betas = get_named_beta_schedule(num_diffusion_timesteps=1000)
        self.diffusion = D.GaussianDiffusion(dtype=torch.float32, model=net, betas=betas, w=1.8, v=0.1, device=device)
        self.cemb = ConditionalEmbedding(10, 60, 60).to(device)
        ck = torch.load(os.path.join(mdir, "ckpt_80_checkpoint.pt"), map_location="cpu")
        net.load_state_dict(ck["net"]); self.cemb.load_state_dict(ck["cemblayer"])
        self.sur = Surrogate_Utility_Classifier(4, 60); self.sur.load_state_dict(torch.load(os.path.join(mdir, "pub_classifier_motion_1.pt"), map_location="cpu")); self.sur.to(device).eval()
        self.priv = Aux_Priv_Classifier(2, 60); self.priv.load_state_dict(torch.load(os.path.join(mdir, "priv_classifier_motion_gender_1.pt"), map_location="cpu")); self.priv.to(device)
        if unet_eval:
            net.eval(); self.cemb.eval()  # sensitivity: dropout off
        # else: default nn.Module.training == True, as in the notebook
        self.train_mode = not unet_eval

    def embed(self, X):
        """X (N,2,128) float32 -> z (N,60): the surrogate activity embedding (PD's conditioning vector)."""
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                x = torch.from_numpy(X[i:i + 1024]).unsqueeze(1).to(self.device)
                out.append(self.sur(x)[1].cpu().numpy())
        return np.concatenate(out)

    def sample(self, X, gender, w1, w2, seed, batch=128):
        """X (N,2,128); gender (N,) in {0,1}. Fixed order; last partial batch padded by repeating the last window (then dropped)."""
        torch.manual_seed(seed); np.random.seed(seed)
        N = len(X); out = np.zeros((N, 2, 128), dtype=np.float32); Z = np.zeros((N, 60), dtype=np.float32)
        for i in range(0, N, batch):
            xb = X[i:i + batch]; gb = gender[i:i + batch]; n = len(xb)
            if n < batch:
                xb = np.concatenate([xb, np.repeat(xb[-1:], batch - n, axis=0)]); gb = np.concatenate([gb, np.repeat(gb[-1:], batch - n)])
            x = torch.from_numpy(xb).unsqueeze(1).to(self.device); g = torch.from_numpy(gb.astype(np.int64)).to(self.device)
            with torch.no_grad():
                _o, emb = self.sur(x); emb = emb.detach(); cemb = self.cemb(emb)
            with contextlib.redirect_stdout(io.StringIO()):
                gen = self.diffusion.ddim_sample((batch, 1, 2, 128), 50, 0.0, "linear", priv_classifier=self.priv, priv_y=g, emb=emb, w1=w1, w2=w2, cemb=cemb)
            out[i:i + n] = gen[:n, 0].detach().cpu().numpy(); Z[i:i + n] = emb[:n].cpu().numpy()
        return out, Z


def run_r0(a):
    """PD's own protocol: PD's DataSampler (stride 10, 13,344 test windows) + shipped Keras models."""
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cwd = os.getcwd(); os.chdir(C.PD_REPO); sys.path.insert(0, C.PD_REPO)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from tensorflow.keras.models import load_model
    from dataset_loader import DataSampler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    with contextlib.redirect_stdout(io.StringIO()):
        DS = DataSampler("./datasets/motionsense/")
    Xt = DS.test_data.astype(np.float32).reshape(-1, 2, 128)  # (N,256) -> (N,2,128): row-major, ch0 then ch1, as in the loader
    act = DS.act_test.astype(int); sub = DS.id_test.astype(int)
    gender = C.gender_by_subject()[sub]
    if a.limit:
        sel = np.random.default_rng(0).choice(len(Xt), a.limit, replace=False); sel.sort()
        Xt, act, sub, gender = Xt[sel], act[sel], sub[sel], gender[sel]
    eval_act = load_model("eval_models/MotionSense/activity_model_mlp.hdf5", compile=False)
    eval_gen = load_model("eval_models/MotionSense/gender_model_mlp.hdf5", compile=False)

    def score(X):
        flat = X.reshape(len(X), 256)
        pa = np.argmax(eval_act.predict(flat, verbose=0), axis=1)
        pg = (eval_gen.predict(flat, verbose=0)[:, 0] > 0.5).astype(int)
        return dict(har_acc=float(accuracy_score(act, pa)), har_f1=float(f1_score(act, pa, average=None).mean()),
                    gender_acc=float(accuracy_score(gender, pg)), gender_bacc=float(balanced_accuracy_score(gender, pg)),
                    gender_f1=float(f1_score(gender, pg, average=None).mean()), frac_pred_male=float(pg.mean()))

    res = dict(n_windows=int(len(Xt)), majority_gender_baseline=float(max(gender.mean(), 1 - gender.mean())), male_frac_windows=float(gender.mean()),
               pd_commit="4a16222", w1=a.w1, w2=a.w2, seed=a.seed, unet_train_mode_dropout_on=not a.unet_eval)
    res["raw"] = score(Xt)
    os.chdir(cwd)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pd_ = PD(dev, unet_eval=bool(a.unet_eval))
    t1 = time.time()
    rel, Z = pd_.sample(Xt, gender, a.w1, a.w2, a.seed, batch=128)
    res["sample_seconds"] = round(time.time() - t1, 1)
    res["pd"] = score(rel)
    res["mse_raw_vs_release"] = float(np.mean((Xt - rel) ** 2))
    res["total_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "r0.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "r0_release.npz"), release=rel.astype(np.float16), act=act.astype(np.int8), sub=sub.astype(np.int8))  # no raw copy of the public data
    print(json.dumps(res, indent=1))


def run_rel(a):
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = C.analysis_set()
    N = len(S["X"])
    idx = np.arange(N)
    if a.limit:
        idx = np.sort(np.random.default_rng(0).choice(N, a.limit, replace=False))
    X = S["X"][idx]; sub = S["sub"][idx]
    gender = C.gender_by_subject()[sub]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pd_ = PD(dev, unet_eval=bool(a.unet_eval))
    t1 = time.time()
    rel, Z = pd_.sample(X, gender, a.w1, a.w2, a.seed, batch=a.batch)
    sec = time.time() - t1
    assert np.isfinite(rel).all()
    np.savez_compressed(os.path.join(a.out, f"{a.name}_s{a.seed}.npz"), release=rel, z=Z, idx=idx)
    json.dump(dict(name=a.name, w1=a.w1, w2=a.w2, seed=a.seed, n=int(len(idx)), sample_seconds=round(sec, 1), windows_per_second=round(len(idx) / sec, 2),
                   unet_train_mode_dropout_on=not a.unet_eval, batch=a.batch, total_seconds=round(time.time() - t0, 1)), open(os.path.join(a.out, f"{a.name}_s{a.seed}.json"), "w"), indent=1)
    print("done", a.name, a.seed, len(idx), f"{len(idx)/sec:.1f} win/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["r0", "rel"]); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="pd_pub"); ap.add_argument("--w1", type=float, default=7.8); ap.add_argument("--w2", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--unet_eval", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    run_r0(a) if a.mode == "r0" else run_rel(a)
