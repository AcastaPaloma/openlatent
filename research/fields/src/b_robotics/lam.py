"""Latent action model (LAPO-style IDM+FDM on 64x64 frames) with probes for the B' kill test.

Arms (one trained model each):
  clean   LAM on distractor-free frames (oracle upper bound for utility, leak = 0 by construction)
  raw     LAM on distractor frames
  rate4   raw + hard capacity limit (z dim 4)                      [rate-bottleneck control, no adversary]
  rateb   raw + strong KL (beta large)                             [rate-bottleneck control, no adversary]
  sam     frames and loss restricted to the zero-shot (oracle-prompted) SAM arm mask
  gtmask  frames and loss restricted to the exact arm silhouette   [best any mask can do]
  adv     raw + adversary on distractor state (confusion loss, lambda 10; fresh attackers retrained at eval); adv30: lambda 30
Utility = linear probe z -> true macro action (NMSE, 1.0 = chance).  Leakage = fresh probes z -> distractor state
(normalised to chance, 0 = chance), worst case over linear and MLP attackers and over targets.
"""
import os, sys, json, time, argparse, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import data as Dt, distractors as D

dev = "cuda" if torch.cuda.is_available() else "cpu"
S = 64
TRAIN_KINDS = ["none", "indep", "corr_det", "corr_noisy", "corr_glued"]
ARMS = {
    "clean": dict(train_kind="none", mask=None, zdim=16, beta=0.1, adv=0.0),
    "raw": dict(mask=None, zdim=16, beta=0.1, adv=0.0),
    "rate4": dict(mask=None, zdim=4, beta=0.1, adv=0.0),
    "rateb": dict(mask=None, zdim=16, beta=3.0, adv=0.0),
    "sam": dict(mask="sam", zdim=16, beta=0.1, adv=0.0),
    "gtmask": dict(mask="gt", zdim=16, beta=0.1, adv=0.0),
    "adv": dict(mask=None, zdim=16, beta=0.1, adv=10.0),
    "adv30": dict(mask=None, zdim=16, beta=0.1, adv=30.0),
}


# ----------------------------------------------------------------------------------------------- data
class GData:
    """One split under one distractor kind, resident on the GPU."""
    def __init__(self, split, kind, mask_root=None, limit=None):
        ep = Dt.load_split(split)
        st = Dt.dist_states(split, kind)
        self.kind, self.split = kind, split
        L = slice(0, limit)
        self.frames = torch.from_numpy(ep["frames"][L]).to(dev)
        self.acts = torch.from_numpy(ep["acts"][L]).to(dev)
        self.pos = torch.from_numpy(st["pos"][L]).to(dev)
        self.cls = torch.from_numpy(st["cls"][L]).to(dev)
        self.vres = torch.from_numpy(st["vres"][L]).to(dev)
        self.vtot = torch.from_numpy(st["vtot"][L]).to(dev)
        self.arm = torch.from_numpy(Dt.unpack(ep["arm"][L])).to(dev)
        self.eepix = torch.from_numpy(ep["eepix"][L]).to(dev)
        self.sam = None
        if mask_root is not None:
            f = os.path.join(mask_root, f"sam_{D.MASK_PREFIX}{kind}_{split}.npz")
            if os.path.exists(f):
                self.sam = torch.from_numpy(Dt.unpack(np.load(f)["mask"])[L]).to(dev)
        self.N, self.T1 = self.frames.shape[:2]
        self.T = self.T1 - 1

    def pairs(self):
        e = torch.arange(self.N, device=dev)[:, None].expand(self.N, self.T).reshape(-1)
        t = torch.arange(self.T, device=dev)[None].expand(self.N, self.T).reshape(-1)
        return e, t

    def batch(self, e, t, mask):
        """x (B,2,3,64,64) in [0,1] (masked frames if mask), m (B,2,1,64,64) or None."""
        B = len(e)
        tt = torch.stack([t, t + 1], 1)  # (B,2)
        ee = e[:, None].expand(B, 2)
        clean = self.frames[ee, tt].reshape(B * 2, 128, 128, 3).permute(0, 3, 1, 2).float() / 255
        pos = self.pos[ee, tt].reshape(B * 2, -1, 2)
        cls = self.cls[e][:, None].expand(B, 2, -1).reshape(B * 2, -1)
        img, _ = D.composite(clean, pos, cls)
        x = F.avg_pool2d(img, 2)
        m = None
        if mask == "gt":
            m = self.arm[ee, tt].reshape(B * 2, 1, 128, 128).float()
        elif mask == "sam":
            assert self.sam is not None, f"no SAM masks for {self.kind}/{self.split}"
            m = self.sam[ee, tt].reshape(B * 2, 1, 128, 128).float()
        if m is not None:
            m = (F.avg_pool2d(m, 2) > 0.5).float()
            x = x * m + 0.5 * (1 - m)
            m = m.reshape(B, 2, 1, S, S)
        return x.reshape(B, 2, 3, S, S), m

    def targets(self, e, t):
        rel = (self.pos[e, t + 1] - self.eepix[e, t + 1][:, None, :]).reshape(len(e), -1) / 128.0   # sprite position relative to the end effector
        if self.kind == "none":
            rel = torch.zeros_like(rel)
        return dict(a=self.acts[e, t], n1=self.vres[e, t].reshape(len(e), -1), n2=rel,
                    n3=self.cls[e], n4=self.vtot[e, t].reshape(len(e), -1))


# ----------------------------------------------------------------------------------------------- model
def cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU())


def ub(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU())


class IDM(nn.Module):
    def __init__(s, zdim):
        super().__init__()
        s.net = nn.Sequential(cb(6, 32), cb(32, 64), cb(64, 128), cb(128, 256))
        s.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 16, 512), nn.GELU(), nn.Linear(512, 2 * zdim))

    def forward(s, x):  # x (B,2,3,S,S)
        h = s.fc(s.net(x.flatten(1, 2)))
        mu, lv = h.chunk(2, -1)
        return mu, lv.clamp(-8, 3)


class FDM(nn.Module):
    def __init__(s, zdim):
        super().__init__()
        s.e1, s.e2, s.e3, s.e4 = cb(3, 32), cb(32, 64), cb(64, 128), cb(128, 256)
        s.zf = nn.Sequential(nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 2 * (256 + 128 + 64 + 32)))
        s.d4, s.d3, s.d2, s.d1 = ub(256, 128), ub(256, 64), ub(128, 32), ub(64, 32)
        s.out = nn.Conv2d(32 + 3, 3, 3, 1, 1)

    def film(s, h, g, b):
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(s, x, z):
        gb = s.zf(z)
        sz = [256, 128, 64, 32]
        parts = list(gb.split([2 * c for c in sz], -1))
        (g4, b4), (g3, b3), (g2, b2), (g1, b1) = [p.chunk(2, -1) for p in parts]
        h1 = s.e1(x); h2 = s.e2(h1); h3 = s.e3(h2); h4 = s.e4(h3)       # 32x32,16x16,8x8,4x4
        h4 = s.film(h4, g4, b4)
        u = s.film(s.d4(h4), g3, b3)                                  # 8x8, 128ch
        u = s.film(s.d3(torch.cat([u, h3], 1)), g2, b2)               # 16x16, 64ch
        u = s.film(s.d2(torch.cat([u, h2], 1)), g1, b1)               # 32x32, 32ch
        u = s.d1(torch.cat([u, h1], 1))                               # 64x64, 32ch
        return x + s.out(torch.cat([u, x], 1))


class Adv(nn.Module):
    def __init__(s, zdim, n_reg, n_cls=None, k=6):
        super().__init__()
        n_cls = n_cls or D.K_SPR
        s.n_reg, s.n_cls, s.k = n_reg, n_cls, k
        s.net = nn.Sequential(nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, n_reg + n_cls * k))

    def forward(s, z):
        o = s.net(z)
        return o[:, :s.n_reg], o[:, s.n_reg:].reshape(len(z), s.n_cls, s.k)


class LAM(nn.Module):
    def __init__(s, zdim):
        super().__init__()
        s.idm, s.fdm = IDM(zdim), FDM(zdim)


# ----------------------------------------------------------------------------------------------- training
def train_arm(cfg, tr, seed, epochs, bs=128, lr=3e-4, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LAM(cfg["zdim"]).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    e_all, t_all = tr.pairs()
    n = len(e_all)
    iters = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    use_adv = cfg["adv"] > 0
    if use_adv:
        has_n1 = bool(tr.vres.abs().sum() > 0)
        n_reg = (2 * D.K_SPR if has_n1 else 0) + 2 * D.K_SPR
        adv = Adv(cfg["zdim"], n_reg).to(dev)
        opt_a = torch.optim.AdamW(adv.parameters(), lr=1e-3, weight_decay=1e-4)
        # standardisers for regression targets
        tg = tr.targets(e_all, t_all)
        mu1, sd1 = tg["n1"].mean(0), tg["n1"].std(0).clamp_min(1e-6)
        mu2, sd2 = tg["n2"].mean(0), tg["n2"].std(0).clamp_min(1e-6)

        def reg_t(tg):
            parts = ([(tg["n1"] - mu1) / sd1] if has_n1 else []) + [(tg["n2"] - mu2) / sd2]
            return torch.cat(parts, 1)
    t0 = time.time()
    for it in range(iters):
        idx = torch.randint(0, n, (bs,), device=dev)
        e, t = e_all[idx], t_all[idx]
        x, mk = tr.batch(e, t, cfg["mask"])
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            mu, lv = m.idm(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            pred = m.fdm(x[:, 0], z)
        err = ((pred.float() - x[:, 1]) ** 2)
        recon = err.flatten(1).sum(1).mean()
        kl = (0.5 * (mu.float() ** 2 + lv.float().exp() - 1 - lv.float())).sum(1).mean()
        loss = recon + cfg["beta"] * kl
        if use_adv:
            lam = cfg["adv"] * min(1.0, it / (0.2 * iters))
            tg = tr.targets(e, t)
            ro, co = adv(mu.float())
            # confusion loss: push attacker outputs to the prior (0 for standardised regression, uniform for classes)
            conf = (ro ** 2).mean() + (-(F.log_softmax(co, -1).mean(-1)) - math.log(co.shape[-1])).mean()
            loss = loss + lam * conf
        opt.zero_grad(set_to_none=True)
        if use_adv:
            opt_a.zero_grad(set_to_none=True)
        loss.backward()
        if use_adv:
            opt_a.zero_grad(set_to_none=True)  # drop the confusion gradients on the attacker's own parameters
            ro, co = adv(mu.detach().float())
            la = F.mse_loss(ro, reg_t(tg)) + F.cross_entropy(co.reshape(-1, co.shape[-1]), tg["n3"].reshape(-1))
            la.backward()
            opt_a.step()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if it % 500 == 0 or it == iters - 1:
            log(f"  it {it}/{iters} recon {recon.item():.2f} kl {kl.item():.2f}" + (f" adv {la.item():.3f}" if use_adv else "") + f" {time.time()-t0:.0f}s")
    return m


@torch.no_grad()
def embed(m, ds, mask, bs=512):
    e, t = ds.pairs()
    out = []
    m.eval()
    for i in range(0, len(e), bs):
        x, _ = ds.batch(e[i:i + bs], t[i:i + bs], mask)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            out.append(m.idm(x)[0].float())
    return torch.cat(out), e, t



# ----------------------------------------------------------------------------------------------- probes
def std_fit(Ztr, *others):
    mu, sd = Ztr.mean(0), Ztr.std(0).clamp_min(1e-6)
    return [(Ztr - mu) / sd] + [(o - mu) / sd for o in others]


def ridge_fit(Z, Y, lam):
    Zb = torch.cat([Z, torch.ones(len(Z), 1, device=Z.device)], 1)
    A = Zb.T @ Zb + lam * torch.eye(Zb.shape[1], device=Z.device)
    A[-1, -1] -= lam
    return torch.linalg.solve(A, Zb.T @ Y)


def ridge_pred(W, Z):
    return torch.cat([Z, torch.ones(len(Z), 1, device=Z.device)], 1) @ W


def linear_probe(Ztr, Ytr, Zva, Yva, Zte_list, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    Ztr_s, Zva_s, *Zte_s = std_fit(Ztr, Zva, *Zte_list)
    ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-8)
    Yn = (Ytr - ymu) / ysd
    best = None
    for lam in lams:
        W = ridge_fit(Ztr_s, Yn, lam)
        v = ((ridge_pred(W, Zva_s) - (Yva - ymu) / ysd) ** 2).mean().item()
        if best is None or v < best[0]:
            best = (v, lam, W)
    W = best[2]
    return [ridge_pred(W, z) * ysd + ymu for z in Zte_s], best[1]


def mlp_probe(Ztr, Ytr, Zva, Yva, Zte_list, classes=None, epochs=40, seed=0):
    """MLP attacker.  classes=None: regression on standardised Y; else Y (n,m) int labels, m heads of `classes`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    Ztr_s, Zva_s, *Zte_s = std_fit(Ztr, Zva, *Zte_list)
    if classes is None:
        ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-8)
        nout = Ytr.shape[1]
        Yt, Yv = (Ytr - ymu) / ysd, (Yva - ymu) / ysd
    else:
        nout = Ytr.shape[1] * classes
        Yt, Yv = Ytr, Yva
    net = nn.Sequential(nn.Linear(Ztr.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, nout)).to(Ztr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)

    def lossf(o, y):
        if classes is None:
            return F.mse_loss(o, y)
        return F.cross_entropy(o.reshape(-1, classes), y.reshape(-1))

    best, best_sd, n = 1e9, None, len(Ztr_s)
    for ep in range(epochs):
        net.train()
        p = torch.randperm(n, device=Ztr.device)
        for i in range(0, n, 256):
            j = p[i:i + 256]
            opt.zero_grad(); lossf(net(Ztr_s[j]), Yt[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Zva_s), Yv).item()
        if v < best:
            best, best_sd = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_sd); net.eval()
    with torch.no_grad():
        outs = [net(z) for z in Zte_s]
    if classes is None:
        return [o * ysd + ymu for o in outs]
    return [o.reshape(len(o), -1, classes).argmax(-1) for o in outs]


def nmse_parts(pred, y, ymu_train, ep_idx, n_ep):
    """per-episode (sum sq err, sum sq dev from train mean) so the bootstrap over episodes is exact."""
    se = ((pred - y) ** 2).mean(1)               # mean over dims -> (n,)
    sv = ((y - ymu_train) ** 2).mean(1)
    num = torch.zeros(n_ep, device=pred.device).index_add_(0, ep_idx, se)
    den = torch.zeros(n_ep, device=pred.device).index_add_(0, ep_idx, sv)
    return num.cpu().numpy(), den.cpu().numpy()


def boot_ci(num, den, B=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num)
    r = [num[i].sum() / max(den[i].sum(), 1e-12) for i in (rng.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(m, cfg, kind_id, sets, seed, mask_root, log=print):
    """sets: dict split -> GData for the ID kind (train/val/test).  Returns metrics dict + per-episode parts."""
    mk = cfg["mask"]
    Ztr, etr, ttr = embed(m, sets["train"], mk)
    Zva, eva, tva = embed(m, sets["val"], mk)
    Zte, ete, tte = embed(m, sets["test"], mk)
    Ttr, Tva, Tte = sets["train"].targets(etr, ttr), sets["val"].targets(eva, tva), sets["test"].targets(ete, tte)
    n_te = sets["test"].N
    res, parts = {}, {}

    # ---- utility: linear probe on 2000 labels (primary), all labels, MLP
    g = torch.Generator(device="cpu").manual_seed(1234 + seed)
    sub = torch.randperm(len(Ztr), generator=g)[:2000].to(dev)
    for name, ZZ, YY in (("lin2k", Ztr[sub], Ttr["a"][sub]), ("linfull", Ztr, Ttr["a"])):
        (pte,), lam = linear_probe(ZZ, YY, Zva, Tva["a"], [Zte])
        mu = YY.mean(0)
        for dims, dn in ((slice(0, 3), "xyz"), (slice(0, 4), "xyzg")):
            num, den = nmse_parts(pte[:, dims], Tte["a"][:, dims], mu[dims], ete, n_te)
            res[f"util_{name}_{dn}"] = float(num.sum() / den.sum()); parts[f"util_{name}_{dn}"] = (num, den)
            res[f"util_{name}_{dn}_ci"] = boot_ci(num, den)
        res[f"util_{name}_lambda"] = lam
    (pm,) = mlp_probe(Ztr, Ttr["a"], Zva, Tva["a"], [Zte], seed=seed)
    num, den = nmse_parts(pm[:, :3], Tte["a"][:, :3], Ttr["a"].mean(0)[:3], ete, n_te)
    res["util_mlp_xyz"] = float(num.sum() / den.sum()); parts["util_mlp_xyz"] = (num, den)
    # control: shuffled labels -> chance
    sh = torch.randperm(len(Ztr), generator=g).to(dev)
    (pc,), _ = linear_probe(Ztr[sub], Ttr["a"][sh][sub], Zva, Tva["a"], [Zte])
    num, den = nmse_parts(pc[:, :3], Tte["a"][:, :3], Ttr["a"].mean(0)[:3], ete, n_te)
    res["ctrl_shuffled_util_xyz"] = float(num.sum() / den.sum())

    # ---- OOD utility: the ID-trained 2k probe applied to every other distractor kind
    sub_ids = sub
    ymu_, ysd_ = Ttr["a"][sub_ids].mean(0), Ttr["a"][sub_ids].std(0).clamp_min(1e-8)
    zmu, zsd = Ztr[sub_ids].mean(0), Ztr[sub_ids].std(0).clamp_min(1e-6)
    W = ridge_fit((Ztr[sub_ids] - zmu) / zsd, (Ttr["a"][sub_ids] - ymu_) / ysd_, res["util_lin2k_lambda"])
    res["util_by_test_kind"] = {}
    for k2 in D.KINDS:
        ds2 = GData("test", k2, mask_root, sets["test"].N) if k2 != kind_id else sets["test"]
        if mk == "sam" and ds2.sam is None:
            res["util_by_test_kind"][k2] = None
            continue
        Z2, e2, t2 = embed(m, ds2, mk)
        p2 = ridge_pred(W, (Z2 - zmu) / zsd) * ysd_ + ymu_
        a2 = ds2.acts[e2, t2]
        num, den = nmse_parts(p2[:, :3], a2[:, :3], ymu_[:3], e2, n_te)
        res["util_by_test_kind"][k2] = float(num.sum() / den.sum())
        if k2 != kind_id:
            del ds2

    # ---- leakage: attackers retrained on the frozen latents (train split), scored on test
    def reg_leak(tn, label):
        Y_tr, Y_va, Y_te = Ttr[tn], Tva[tn], Tte[tn]
        if Y_tr.std(0).max() < 1e-6:
            return
        (pl,), _ = linear_probe(Ztr, Y_tr, Zva, Y_va, [Zte])
        (pmm,) = mlp_probe(Ztr, Y_tr, Zva, Y_va, [Zte], seed=seed)
        for aname, pp in (("lin", pl), ("mlp", pmm)):
            num, den = nmse_parts(pp, Y_te, Y_tr.mean(0), ete, n_te)
            res[f"leak_{label}_{aname}"] = float(max(0.0, 1 - num.sum() / den.sum()))
            parts[f"leak_{label}_{aname}"] = (num, den)
            res[f"leak_{label}_{aname}_ci"] = [max(0.0, 1 - c) for c in boot_ci(num, den)][::-1]

    reg_leak("n1", "resid_vel")      # independent part of distractor motion
    reg_leak("n2", "position")       # sprite position relative to the end effector at t+1
    reg_leak("n4", "total_vel")      # diagnostic: total sprite velocity (contains the action response for corr kinds)
    # appearance (colour class of each sprite), classification
    if kind_id != "none":
        for aname in ("lin", "mlp"):
            if aname == "mlp":
                (pp,) = mlp_probe(Ztr, Ttr["n3"], Zva, Tva["n3"], [Zte], classes=6, seed=seed)
            else:  # softmax regression = MLP-free linear head
                (pp,) = _softmax_probe(Ztr, Ttr["n3"], Zva, Tva["n3"], [Zte], seed)
            acc = (pp == Tte["n3"]).float().mean().item()
            res[f"leak_appearance_{aname}"] = float(max(0.0, (acc - 1 / 6) / (5 / 6)))
        # control: shuffled secret labels -> attacker at chance
        sh2 = torch.randperm(len(Ztr), generator=g).to(dev)
        (pc,) = mlp_probe(Ztr, Ttr["n3"][sh2], Zva, Tva["n3"], [Zte], classes=6, seed=seed)
        res["ctrl_shuffled_appearance_mlp_acc"] = float((pc == Tte["n3"]).float().mean().item())
    main = [v for k, v in res.items() if k.startswith("leak_") and not k.endswith("_ci") and "total_vel" not in k]
    res["leak_worst"] = float(max(main)) if main else 0.0
    return res, parts


def _softmax_probe(Ztr, Ytr, Zva, Yva, Zte_list, seed, epochs=40):
    torch.manual_seed(seed)
    Ztr_s, Zva_s, *Zte_s = std_fit(Ztr, Zva, *Zte_list)
    net = nn.Linear(Ztr.shape[1], Ytr.shape[1] * 6).to(Ztr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-3)
    best, sd = 1e9, None
    for ep in range(epochs):
        p = torch.randperm(len(Ztr_s), device=Ztr.device)
        for i in range(0, len(p), 256):
            j = p[i:i + 256]
            opt.zero_grad(); F.cross_entropy(net(Ztr_s[j]).reshape(-1, 6), Ytr[j].reshape(-1)).backward(); opt.step()
        with torch.no_grad():
            v = F.cross_entropy(net(Zva_s).reshape(-1, 6), Yva.reshape(-1)).item()
        if v < best:
            best, sd = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(sd)
    with torch.no_grad():
        return [net(z).reshape(len(z), -1, 6).argmax(-1) for z in Zte_s]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=TRAIN_KINDS)
    ap.add_argument("--arms", default="clean,raw,rate4,rateb,sam,gtmask,adv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", type=int, default=None, help="limit episodes per split (smoke tests)")
    ap.add_argument("--out", default=os.environ.get("OUT", "./out"))
    ap.add_argument("--mask_root", default=os.path.join(Dt.ROOT, "masks"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = {}
    for arm in a.arms.split(","):
        cfg = dict(ARMS[arm])
        kind = cfg.pop("train_kind", a.kind)
        t0 = time.time()
        lim = a.smoke
        sets = {s: GData(s, kind, a.mask_root, lim) for s in ("train", "val", "test")}
        print(f"== {arm} kind={kind} cfg={cfg} seed={a.seed}", flush=True)
        m = train_arm(cfg, sets["train"], a.seed, a.epochs, log=lambda s: print(s, flush=True))
        res, parts = evaluate(m, cfg, kind, sets, a.seed, a.mask_root, log=print)
        res.update(arm=arm, train_kind=kind, seed=a.seed, epochs=a.epochs, cfg=cfg, seconds=round(time.time() - t0, 1))
        allres[arm] = res
        json.dump(res, open(os.path.join(a.out, f"res_{a.kind}_{arm}_s{a.seed}.json"), "w"), indent=1)
        np.savez(os.path.join(a.out, f"parts_{a.kind}_{arm}_s{a.seed}.npz"), **{f"{k}__{i}": v[i] for k, v in parts.items() for i in (0, 1)})
        torch.save(m.state_dict(), os.path.join(a.out, f"lam_{a.kind}_{arm}_s{a.seed}.pt"))
        brief = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k.startswith(("util_lin2k_xyz", "leak_worst", "ctrl"))}
        print("RESULT", arm, brief, "by_kind", {k: (round(v, 3) if v is not None else None) for k, v in res["util_by_test_kind"].items()}, flush=True)
        del sets, m
        torch.cuda.empty_cache()
    json.dump(allres, open(os.path.join(a.out, f"all_{a.kind}_s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
