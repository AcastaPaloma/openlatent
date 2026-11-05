"""Re-implementation of the Anonymizing AutoEncoder (arXiv 1810.11546) in PyTorch, trained INSIDE the programme (H.md 6.2).
Architecture = the shipped msda_anon_model.json (Enc / Dec / EncReg / DecReg / ActReg); loss L = b_i (Li(EncReg(Y)) + Li(DecReg(X'))) + b_a CE(ActReg(X'), T) + b_d MSE(X, X'),
Li = -(sum_k U_k log(1 - Uhat_k) + log(1 - max Uhat)); alternating schedule of the paper's Fig. 6 (pretrain X->X; rounds of {regularisers one epoch, AE one epoch with regularisers frozen}).
Departures from the paper (it gives no beta, epochs, optimiser, lr, batch): Adam lr 1e-3, batch 128, 3 pretraining epochs, R = 8 rounds, no re-initialisation after pretraining, last-round AE used.
  python aae_train.py r0b --out DIR                       # gate R0-B on the dev split, tries C1 -> C2 -> C3 and stops at the first pass
  python aae_train.py fit --set sealed --config C1 --out DIR [--folds S0,S1,S2]   # releases for all windows of the set, one model per fold
"""
import os, sys, json, time, argparse, io, contextlib
import numpy as np
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import datasets as D

CONFIGS = {"C1": (1.0, 1.0, 1.0), "C2": (4.0, 1.0, 1.0), "C3": (4.0, 1.0, 0.25)}  # (beta_i, beta_a, beta_d)
EPS = 1e-6


def sbn(c):
    return [nn.BatchNorm2d(c, momentum=0.01, eps=1e-3)]  # Keras BN momentum 0.99 -> torch momentum 0.01, epsilon 1e-3


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        L = []
        L += [nn.Conv2d(1, 100, (1, 3), padding=(0, 1)), nn.SELU()] + sbn(100)
        L += [nn.Conv2d(100, 100, (1, 3), padding=(0, 1)), nn.SELU()] + sbn(100) + [nn.MaxPool2d((1, 2))]
        L += [nn.Conv2d(100, 100, (1, 3), padding=(0, 1)), nn.SELU()] + sbn(100) + [nn.MaxPool2d((1, 2))]
        L += [nn.Conv2d(100, 100, (1, 3), padding=(0, 1)), nn.SELU()] + sbn(100)
        L += [nn.Conv2d(100, 1, (1, 3), padding=(0, 1))] + sbn(1)
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        L = [nn.ConvTranspose2d(1, 100, (1, 3), padding=(0, 1)), nn.SELU()] + sbn(100)
        L += [nn.ConvTranspose2d(100, 100, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)), nn.SELU()] + sbn(100)
        L += [nn.ConvTranspose2d(100, 100, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)), nn.SELU()] + sbn(100)
        L += [nn.ConvTranspose2d(100, 100, (1, 3), padding=(0, 1)), nn.SELU()]
        L += [nn.Conv2d(100, 1, (1, 3), padding=(0, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, y): return self.net(y)


def reg_net(width, ncls, n_pool, d1, d2, c5=False):
    """EncReg / DecReg / ActReg as in the shipped JSON (3-5 conv (2,5) x32 with dropout .25, max-pools (1,2), dense d1 / d2 with dropout .5)."""
    def conv(i): return [nn.Conv2d(i, 32, (2, 5), padding="same"), nn.ReLU(), nn.Dropout(0.25)]
    if n_pool == 2:  # ENC_ID: conv, conv, conv, pool, conv, conv, pool
        L = conv(1) + conv(32) + [nn.Conv2d(32, 32, (2, 5), padding="same"), nn.ReLU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.25)] + conv(32) \
            + [nn.Conv2d(32, 32, (2, 5), padding="same"), nn.ReLU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.25)]
        feat = 32 * 2 * (width // 4)
    else:  # DEC_ID / ACT_ID: conv, conv, pool, conv, pool, conv, pool, conv, pool
        L = [nn.Conv2d(1, 32, (2, 5), padding="same"), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(32, 32, (2, 5), padding="same"), nn.ReLU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.25)]
        for _ in range(3):
            L += [nn.Conv2d(32, 32, (2, 5), padding="same"), nn.ReLU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.25)]
        feat = 32 * 2 * (width // 16)
    L += [nn.Flatten(), nn.Linear(feat, d1), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d1, d2), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d2, ncls)]
    return nn.Sequential(*L)


def li_loss(logits, y):
    p = torch.softmax(logits, -1)
    return -((y * torch.log(1 - p + EPS)).sum(1) + torch.log(1 - p.max(1).values + EPS)).mean()


def train_aae(Xtr, id_tr, act_tr, n_id, n_act, cfg, seed, dev, log, rounds=8, pre_epochs=3, bs=128):
    bi, ba, bd = CONFIGS[cfg]
    torch.manual_seed(seed); np.random.seed(seed)
    enc, dec = Enc().to(dev), Dec().to(dev)
    for m in list(enc.modules()) + list(dec.modules()):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    enc_reg = reg_net(32, n_id, 2, 128, 32).to(dev); dec_reg = reg_net(128, n_id, 4, 256, 64).to(dev); act_reg = reg_net(128, n_act, 4, 256, 64).to(dev)
    X = torch.from_numpy(Xtr).float().unsqueeze(1).to(dev); idl = torch.from_numpy(id_tr).long().to(dev); actl = torch.from_numpy(act_tr).long().to(dev)
    nv = len(X) // 5; perm0 = torch.randperm(len(X), device=dev); vi, ti = perm0[:nv], perm0[nv:]  # 20% validation windows for monitoring only
    idoh = nn.functional.one_hot(idl, n_id).float()
    opt_ae = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    opt_r = torch.optim.Adam(list(enc_reg.parameters()) + list(dec_reg.parameters()) + list(act_reg.parameters()), lr=1e-3)
    ce = nn.functional.cross_entropy
    # --- pretraining X -> X
    for ep in range(pre_epochs):
        enc.train(); dec.train(); p = ti[torch.randperm(len(ti), device=dev)]
        for i in range(0, len(p), bs):
            b = p[i:i + bs]
            if len(b) < 4: continue
            loss = nn.functional.mse_loss(dec(enc(X[b])), X[b]); opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    hist = []
    for r in range(rounds):
        # --- regularisers: one epoch (all three on the current Y, X')
        enc.eval(); dec.eval(); [m.train() for m in (enc_reg, dec_reg, act_reg)]
        p = ti[torch.randperm(len(ti), device=dev)]
        for i in range(0, len(p), bs):
            b = p[i:i + bs]
            if len(b) < 4: continue
            with torch.no_grad():
                y = enc(X[b]); xp = dec(y)
            loss = ce(enc_reg(y), idl[b]) + ce(dec_reg(xp), idl[b]) + ce(act_reg(xp), actl[b]); opt_r.zero_grad(); loss.backward(); opt_r.step()
        # --- AE: one epoch with regularisers frozen
        enc.train(); dec.train(); [m.eval() for m in (enc_reg, dec_reg, act_reg)]
        for q in list(enc_reg.parameters()) + list(dec_reg.parameters()) + list(act_reg.parameters()): q.requires_grad_(False)
        p = ti[torch.randperm(len(ti), device=dev)]
        for i in range(0, len(p), bs):
            b = p[i:i + bs]
            if len(b) < 4: continue
            y = enc(X[b]); xp = dec(y)
            loss = bi * (li_loss(enc_reg(y), idoh[b]) + li_loss(dec_reg(xp), idoh[b])) + ba * ce(act_reg(xp), actl[b]) + bd * nn.functional.mse_loss(xp, X[b])
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        for q in list(enc_reg.parameters()) + list(dec_reg.parameters()) + list(act_reg.parameters()): q.requires_grad_(True)
        enc.eval(); dec.eval()
        with torch.no_grad():
            y = enc(X[vi]); xp = dec(y)
            row = dict(round=r + 1, val_mse=float(nn.functional.mse_loss(xp, X[vi])), val_encreg_id=float((enc_reg(y).argmax(1) == idl[vi]).float().mean()),
                       val_decreg_id=float((dec_reg(xp).argmax(1) == idl[vi]).float().mean()), val_actreg_act=float((act_reg(xp).argmax(1) == actl[vi]).float().mean()))
        hist.append(row); log(json.dumps(row))
    enc.eval(); dec.eval()
    return enc, dec, hist


def release(enc, dec, X, dev, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(dec(enc(torch.from_numpy(X[i:i + bs]).float().unsqueeze(1).to(dev))).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run_r0b(a):
    """gate R0-B: dev split (PD split), paper protocol with frozen raw-trained ID CNN and a retrained activity CNN."""
    import aae as A
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trials = C.load_trials(); mean, std = C.pd_train_stats(trials)
    allw = C.build_windows(trials, mean, std, stride=10); j = allw["start"] // 10
    te_o = np.array([(C.ACTS[x], int(t)) in C.FOLD_TEST["O"] for x, t in zip(allw["act"], allw["trial"])])
    tr = (~te_o) & (j % 2 == 0)
    # PD-loader windows for the paper protocol
    cwd = os.getcwd(); os.chdir(C.PD_REPO); sys.path.insert(0, C.PD_REPO)
    from dataset_loader import DataSampler
    with contextlib.redirect_stdout(io.StringIO()):
        DS = DataSampler("./datasets/motionsense/")
    os.chdir(cwd)
    Xtr_pd = DS.train_data.astype(np.float32).reshape(-1, 2, 128); Xte_pd = DS.test_data.astype(np.float32).reshape(-1, 2, 128)
    idtr, idte = DS.id_train.astype(int), DS.id_test.astype(int); actr, acte = DS.act_train.astype(int), DS.act_test.astype(int)
    # frozen raw-trained ID CNN (3 runs), reused for every config
    id_nets = []
    for r in range(3):
        net, info = A._train_estimator(Xtr_pd.reshape(-1, 1, 2, 128), idtr, 24, r, dev); id_nets.append(net)
    res = dict(configs={}, chosen=None)
    logf = open(os.path.join(a.out, f"r0b_train{a.tag}.log"), "a")
    def log(s): logf.write(s + "\n"); logf.flush()
    for cfg in a.configs.split(","):
        log(f"== {cfg}")
        enc, dec, hist = train_aae(allw["X"][tr], allw["sub"][tr], allw["act"][tr], 24, 4, cfg, 0, dev, log)
        Rtr, Rte = release(enc, dec, Xtr_pd, dev), release(enc, dec, Xte_pd, dev)
        frozen = [A._score(n, Rte.reshape(-1, 1, 2, 128), idte, dev) for n in id_nets]
        act_net, ainfo = A._train_estimator(Rtr.reshape(-1, 1, 2, 128), actr, 4, 0, dev)
        act_acc = A._score(act_net, Rte.reshape(-1, 1, 2, 128), acte, dev)
        ok = bool(np.mean(frozen) * 100 <= 10.0 and act_acc * 100 >= 90.0)
        res["configs"][cfg] = dict(beta=CONFIGS[cfg], frozen_id_acc_runs=frozen, frozen_id_acc_mean=float(np.mean(frozen)), retrained_act_acc=float(act_acc), mse_release_vs_input=float(np.mean((Rte - Xte_pd) ** 2)), passes=ok, history_last=hist[-1], seconds=round(time.time() - t0, 1))
        log(json.dumps(res["configs"][cfg]))
        json.dump(res, open(os.path.join(a.out, f"r0b{a.tag}.json"), "w"), indent=1)
        if ok:
            res["chosen"] = cfg; break
    res["R0B_pass"] = res["chosen"] is not None
    if res["chosen"] is None: res["chosen"] = a.configs.split(",")[0]
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, f"r0b{a.tag}.json"), "w"), indent=1); print(json.dumps(res, indent=1))


def run_fit(a):
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    S = D.get_set(a.set); X = S["X"]
    pack = {}; hists = {}
    logf = open(os.path.join(a.out, f"aae_fit_{a.set}{a.tag}.log"), "a")
    def log(s): logf.write(s + "\n"); logf.flush()
    for f in a.folds.split(","):
        te = D.test_mask(S, f); tr = ~te
        log(f"== fold {f} train windows {int(tr.sum())}")
        enc, dec, hist = train_aae(X[tr], S["sub"][tr], S["act"][tr], S["n_id"], S["n_act"], a.config, a.seed, dev, log)
        R = release(enc, dec, S["X"], dev)
        assert np.isfinite(R).all()
        pack[f"release_{f}"] = R; hists[f] = hist
        if a.save_weights:
            torch.save(dict(enc=enc.state_dict(), dec=dec.state_dict(), config=a.config, seed=a.seed, fold=f), os.path.join(a.out, f"weights_{a.set}{a.tag}_{f}.pt"))
        log(f"fold {f} done {time.time()-t0:.0f}s")
    np.savez_compressed(os.path.join(a.out, f"aae_re_{a.set}{a.tag}.npz"), idx=np.arange(len(X)), **pack)
    json.dump(dict(set=a.set, config=a.config, beta=CONFIGS[a.config], seed=a.seed, history=hists, seconds=round(time.time() - t0, 1)), open(os.path.join(a.out, f"aae_fit_{a.set}{a.tag}.json"), "w"), indent=1)
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["r0b", "fit"]); ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="sealed"); ap.add_argument("--config", default="C1"); ap.add_argument("--folds", default="S0,S1,S2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="C1,C2,C3"); ap.add_argument("--tag", default=""); ap.add_argument("--save_weights", action="store_true")
    a = ap.parse_args()
    run_r0b(a) if a.mode == "r0b" else run_fit(a)
