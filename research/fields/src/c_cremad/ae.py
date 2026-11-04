"""Task-agnostic adversarial autoencoder on pooled frozen features (Focus C, pre-registered in results_notes/C.md section 2c).

No emotion labels anywhere.  x = standardised per-clip mean+std of the frozen layer mix [N,1536].  Encoder -> (mu, logvar) of width d_z,
decoder reconstructs x; beta * KL is the rate term; lam * confusion makes two identity adversaries (linear + MLP, re-initialised and re-trained
from scratch in every round on the current z of the encoder-train actors) uncertain.  Same structure as Builder A's `train_ae`
(research/fields/src/a_pathorob/adapters.py).  The released representation is mu (no sampling noise).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import DEV


def train_ae(P, tr_idx, actor_idx, dz, lam, beta, seed, rounds=12, steps_e=100, steps_a=150, hid=512, bs=256, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    P = P.float().to(DEV)
    mu_x, sd_x = P[tr_idx].mean(0), P[tr_idx].std(0) + 1e-5
    X = (P - mu_x) / sd_x
    Xtr = X[tr_idx]
    d = X.shape[1]
    actors = np.unique(actor_idx[tr_idx])
    amap = {a: i for i, a in enumerate(actors)}
    y = torch.as_tensor(np.array([amap[a] for a in actor_idx[tr_idx]]), device=DEV)
    K = len(actors)
    enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 2 * dz)).to(DEV)
    dec = nn.Sequential(nn.Linear(dz, hid), nn.GELU(), nn.Linear(hid, d)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-2)

    def chan(x):
        mu, lv = enc(x).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
        return z, kl, mu

    adv_logs = []
    for r in range(rounds):
        advs = None
        if lam > 0:
            enc.eval()
            with torch.no_grad():
                Z = chan(Xtr)[0]
            zm, zs = Z.mean(0), Z.std(0) + 1e-5
            Zn = (Z - zm) / zs
            advs = nn.ModuleList([nn.Linear(dz, K), nn.Sequential(nn.Linear(dz, 256), nn.GELU(), nn.Linear(256, K))]).to(DEV)
            oa = torch.optim.Adam(advs.parameters(), lr=3e-3)
            for s in range(steps_a):
                idx = torch.randint(0, len(Zn), (512,), device=DEV)
                loss = sum(F.cross_entropy(a_(Zn[idx]), y[idx]) for a_ in advs)
                oa.zero_grad(); loss.backward(); oa.step()
            with torch.no_grad():
                adv_logs.append(round(float(np.mean([(a_(Zn).argmax(1) == y).float().mean().item() for a_ in advs])), 3))
        enc.train(); dec.train()
        for s in range(steps_e):
            idx = torch.randint(0, len(Xtr), (bs,), device=DEV)
            z, kl, _ = chan(Xtr[idx])
            loss = F.mse_loss(dec(z), Xtr[idx]) + beta * kl
            if lam > 0:
                zn = (z - zm) / zs
                conf = 0.0
                for a_ in advs:
                    p = F.softmax(a_(zn), -1)
                    conf = conf + (p * torch.log(p.clamp(min=1e-8))).sum(-1).mean() + math.log(K)
                loss = loss + lam * conf / len(advs)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    with torch.no_grad():
        mu = enc(X).chunk(2, -1)[0]
        mse_tr = float(F.mse_loss(dec(mu[tr_idx]), Xtr))
        mse_all = float(F.mse_loss(dec(mu), X))
    return mu.cpu(), dict(recon_mse_train=mse_tr, recon_mse_all=mse_all, adv_acc_per_round=adv_logs)
