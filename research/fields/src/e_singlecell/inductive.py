"""E2: inductive learned encoders (expression -> 16-d code) for HLCA.  Everything is fitted on the fold's TRAINING cells only; the training batch id is a
training-time covariate (cVAE decoder) or adversary label only; at inference the encoder sees one cell's standardised HVG expression and nothing else.
Families (E.md section 6.4): vae (no-adversary rate control), cvae (decoder conditioned on the training batch), vaeadv / cvaeadv (gradient-reversal batch adversary,
re-initialised every 250 steps), vaeplc (placebo adversary on permuted batch labels).
"""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BS, LR, WD, ANNEAL_EPOCHS, ADV_REINIT = 30, 1024, 1e-3, 1e-4, 5, 250

CONFIGS = ([f"vae_b{b}" for b in (2, 10, 50)] + [f"cvae_b{b}" for b in (2, 10, 50)] +
           [f"vaeadv_b10_l{l}" for l in (1, 4, 16)] + [f"cvaeadv_b10_l{l}" for l in (1, 4, 16)] + ["vaeplc_b10_l4"])
DIAG = ["vae_b2", "vae_b10", "vae_b50", "vaeadv_b10_l1", "vaeadv_b10_l4", "vaeadv_b10_l16"]


def parse(name):
    m = re.match(r"^(vae|cvae|vaeadv|cvaeadv|vaeplc)_b(\d+)(?:_l(\d+))?$", name)
    fam, beta, lam = m.group(1), float(m.group(2)), float(m.group(3) or 0)
    return dict(fam=fam, beta=beta, lam=lam, cond=fam.startswith("cvae"), adv=fam in ("vaeadv", "cvaeadv", "vaeplc"), placebo=fam == "vaeplc")


def hvg_standardised(X, T, n_hvg=2000):
    """same HVG list / standardisation as arms.encoder0 (HVG on training cells, train mean/sd, clip +-10); returns float32 (n, G)"""
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X[T])
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_hvg)
    hv = np.where(a.var["highly_variable"].to_numpy())[0]
    Xh = X[:, hv].toarray().astype(np.float32)
    mu, sd = Xh[T].mean(0), Xh[T].std(0)
    sd[sd < 1e-6] = 1.0
    return np.clip((Xh - mu) / sd, -10, 10).astype(np.float32)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, G, d, B, cond, h1=512, h2=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(G, h1), nn.GELU(), nn.Linear(h1, h2), nn.GELU())
        self.mu, self.lv = nn.Linear(h2, d), nn.Linear(h2, d)
        self.cond, self.B = cond, B
        self.dec = nn.Sequential(nn.Linear(d + (B if cond else 0), h2), nn.GELU(), nn.Linear(h2, h1), nn.GELU(), nn.Linear(h1, G))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h).clamp(-10, 6)

    def decode(self, z, b):
        if self.cond:
            z = torch.cat([z, F.one_hot(b, self.B).float()], 1)
        return self.dec(z)


def new_adv(d, B):
    return nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, B)).to(DEV)


def train_encoder(Xtr, btr, B, cfg, seed, d=16, beta_scale=1.0):
    """Xtr: float32 numpy (n, G) training cells; btr: int training-batch codes in 0..B-1.  Returns (net, train log dict)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    X = torch.as_tensor(Xtr, device=DEV)
    b = torch.as_tensor(btr, dtype=torch.long, device=DEV)
    n, G = X.shape
    net = Net(G, d, B, cfg["cond"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    cnt = torch.bincount(b, minlength=B).float()
    cw = torch.where(cnt > 0, cnt.sum() / (B * cnt.clamp(min=1)), torch.zeros_like(cnt))
    y_adv = b[torch.randperm(n, device=DEV, generator=gen)] if cfg["placebo"] else b
    adv = opt_adv = None
    spe = int(np.ceil(n / BS))
    total, anneal = EPOCHS * spe, ANNEAL_EPOCHS * spe
    beta = cfg["beta"] * beta_scale
    step, log = 0, dict(kl=[], rec=[], adv_ce=[])
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEV, generator=gen)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            if len(idx) < 8:
                continue
            ramp = min(1.0, (step + 1) / anneal)
            x = X[idx]
            mu, lv = net.encode(x)
            z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, device=DEV, generator=gen)
            xh = net.decode(z, b[idx])
            rec = 0.5 * ((x - xh) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = rec + beta * ramp * kl
            if cfg["adv"]:
                if adv is None or step % ADV_REINIT == 0:
                    adv = new_adv(d, B)
                    opt_adv = torch.optim.AdamW(adv.parameters(), lr=LR, weight_decay=WD)
                ce = F.cross_entropy(adv(GRL.apply(z, cfg["lam"] * ramp)), y_adv[idx], weight=cw)
                loss = loss + ce
                opt_adv.zero_grad()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg["adv"]:
                opt_adv.step()
            if step % 50 == 0:
                log["kl"].append(float(kl))
                log["rec"].append(float(rec))
                if cfg["adv"]:
                    log["adv_ce"].append(float(ce))
            step += 1
    net.eval()
    return net, log


@torch.no_grad()
def release(net, Xall, seed, sample=True, bs=8192):
    """released code: z = mu + sigma * eps (fixed seeded eps) for ALL cells; Xall float32 numpy"""
    gen = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    out = []
    for i in range(0, len(Xall), bs):
        mu, lv = net.encode(torch.as_tensor(Xall[i:i + bs], device=DEV))
        z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, device=DEV, generator=gen) if sample else mu
        out.append(z.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def train_stats(net, Xtr, btr, cfg, max_n=30000, seed=0):
    """training-cell diagnostics: mean KL (nats), reconstruction R^2 from the posterior mean"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(max_n, len(Xtr)), replace=False)
    x = torch.as_tensor(Xtr[idx], device=DEV)
    b = torch.as_tensor(btr[idx], dtype=torch.long, device=DEV)
    mu, lv = net.encode(x)
    kl = float((-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean())
    xh = net.decode(mu, b)
    r2 = float(1 - ((x - xh) ** 2).sum() / (x ** 2).sum())
    return kl, r2
