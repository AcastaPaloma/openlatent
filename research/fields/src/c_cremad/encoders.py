"""Learned encoders E(frozen SSL features) -> z with rate / adversarial / VQ bottlenecks (Focus C).

Input: Ragged frame features [Nfr, 4 layers, 768] (fp16).  The encoder layer-normalises each layer, mixes them with a
learned softmax weighting, runs a small conv+transformer, pools to one vector per clip (`pool='utt'`) or to segments of
P frames (`pool=P`), and passes through a bottleneck:
    vib : z = mu (+ sigma*eps while training when beta>0); KL(q||N(0,I)) is the rate penalty (nats per clip/segment)
    vq  : product quantiser, G groups x K codes of dim e (rate = G*log2 K bits per clip/segment), EMA codebooks
    none: deterministic linear map to d_z
Heads: emotion MLP on mean(z).  Optional identity adversaries (ensemble of K MLPs that are re-initialised from scratch
every `adv_reinit` epochs and warmed up before the encoder sees them) trained on the encoder-train actors only; the
encoder is pushed to make their posterior uniform (confusion loss).
Nothing in training is reported as a result: after training the evaluation retrains fresh attackers on the released z.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import MLP, Ragged, Rep, DEV, uar, cosine_sched


class VQ(nn.Module):
    def __init__(self, groups, codes, dim, decay=0.99, commit=0.25):
        super().__init__()
        self.G, self.K, self.e, self.decay, self.commit = groups, codes, dim, decay, commit
        cb = torch.randn(groups, codes, dim) * 0.5
        self.register_buffer("cb", cb)
        self.register_buffer("ema_n", torch.ones(groups, codes))
        self.register_buffer("ema_s", cb.clone())
        self.register_buffer("usage", torch.zeros(groups, codes))

    def forward(self, h):  # h [..., G*e]
        shp = h.shape[:-1]
        hg = h.reshape(-1, self.G, self.e)
        d = ((hg[:, :, None, :] - self.cb[None]) ** 2).sum(-1)  # [n,G,K]
        idx = d.argmin(-1)  # [n,G]
        q = torch.stack([self.cb[g][idx[:, g]] for g in range(self.G)], 1)  # [n,G,e]
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(idx, self.K).float()  # [n,G,K]
                n = oh.sum(0)
                s = torch.einsum("ngk,nge->gke", oh, hg.detach())
                self.ema_n.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_s.mul_(self.decay).add_(s, alpha=1 - self.decay)
                tot = self.ema_n.sum(1, keepdim=True)
                nn_ = (self.ema_n + 1e-5) / (tot + self.K * 1e-5) * tot
                self.cb.copy_(self.ema_s / nn_[..., None])
                self.usage.add_(n)
        loss = self.commit * F.mse_loss(hg, q.detach())
        z = hg + (q - hg).detach()
        return z.reshape(*shp, self.G * self.e), loss, idx.reshape(*shp, self.G)

    def restart_dead(self, hbatch):
        """Re-seed codes that were (almost) never used since the last call with random encoder outputs."""
        hg = hbatch.detach().reshape(-1, self.G, self.e)
        dead = 0
        for g in range(self.G):
            m = self.usage[g] < 1
            k = int(m.sum())
            if k:
                pick = hg[torch.randint(0, hg.shape[0], (k,), device=hg.device), g]
                self.cb[g][m] = pick; self.ema_s[g][m] = pick; self.ema_n[g][m] = 1.0
                dead += k
        self.usage.zero_()
        return dead

    def perplexity(self):
        p = self.usage / self.usage.sum(1, keepdim=True).clamp(min=1)
        return float(torch.exp(-(p * torch.log(p + 1e-9)).sum(1)).mean())


class Encoder(nn.Module):
    def __init__(self, n_layers=4, din=768, dh=256, dz=64, pool="utt", bott="vib", beta=0.0, vq_groups=1, vq_codes=64,
                 vq_dim=32, n_emo=6, dropout=0.1):
        super().__init__()
        self.pool, self.bott, self.beta = pool, bott, beta
        self.lw = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Linear(din, dh)
        self.dw = nn.Conv1d(dh, dh, 15, padding=7, groups=dh)
        enc = nn.TransformerEncoderLayer(dh, 4, 2 * dh, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, 2)
        self.norm = nn.LayerNorm(dh)
        self.att = nn.Linear(dh, 1)
        if bott == "vq":
            self.pre = nn.Linear(dh, vq_groups * vq_dim)
            self.vq = VQ(vq_groups, vq_codes, vq_dim)
            self.dz = vq_groups * vq_dim
            self.bits = vq_groups * math.log2(vq_codes)
        else:
            self.mu = nn.Linear(dh, dz)
            self.lv = nn.Linear(dh, dz) if bott == "vib" else None
            self.dz = dz
            self.bits = None
        self.emo = MLP(self.dz, n_emo, hidden=128, p=0.1, depth=1)

    def frames(self, x, mask):
        x = F.layer_norm(x, (x.shape[-1],))
        w = torch.softmax(self.lw, 0)
        x = (x * w[None, None, :, None]).sum(2)
        h = self.proj(x)
        h = h + self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.tr(h, src_key_padding_mask=~mask)
        return self.norm(h)

    def pool_h(self, h, mask):
        if self.pool == "utt":
            a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(1)
            return (a[..., None] * h).sum(1), None
        P = int(self.pool)
        B, T, D = h.shape
        S = math.ceil(T / P)
        pad = S * P - T
        hp = F.pad(h, (0, 0, 0, pad)); mp = F.pad(mask, (0, pad))
        mf = mp.float().view(B, S, P, 1)
        seg = (hp.view(B, S, P, D) * mf).sum(2) / mf.sum(2).clamp(min=1)
        return seg, mp.view(B, S, P).any(-1)

    def forward(self, x, mask):
        h = self.frames(x, mask)
        hp, smask = self.pool_h(h, mask)
        out = dict(smask=smask)
        if self.bott == "vq":
            pre = self.pre(hp)
            z, vql, idx = self.vq(pre)
            out.update(z=z, mu=z, kl=torch.zeros(z.shape[:-1], device=x.device), aux=vql, idx=idx, pre_h=pre.detach())
        else:
            mu = self.mu(hp)
            if self.bott == "vib":
                lv = self.lv(hp).clamp(-8, 4)
                kl = 0.5 * (mu ** 2 + lv.exp() - lv - 1).sum(-1)  # [B] or [B,S]
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if (self.training and self.beta > 0) else mu
            else:
                kl, z = torch.zeros(mu.shape[:-1], device=mu.device), mu
            out.update(z=z, mu=mu, kl=kl, aux=torch.zeros((), device=x.device))
        zz = out["z"]
        if smask is None:
            out["zvec"] = zz
            out["zstat"] = zz
        else:
            mf = smask.float()[..., None]
            n = mf.sum(1).clamp(min=1)
            m = (zz * mf).sum(1) / n
            sd = ((((zz - m[:, None]) ** 2) * mf).sum(1) / n).clamp(min=1e-8).sqrt()
            out["zvec"] = m
            out["zstat"] = torch.cat([m, sd], -1)
        out["emo_logits"] = self.emo(out["zvec"])
        return out


class AdvEnsemble(nn.Module):
    def __init__(self, din, ncls, k=3, hidden=256):
        super().__init__()
        self.din, self.ncls, self.k, self.hidden = din, ncls, k, hidden
        self.nets = nn.ModuleList([MLP(din, ncls, hidden, p=0.1, depth=2) for _ in range(k)])
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=1e-3)

    def reinit(self):
        self.nets = nn.ModuleList([MLP(self.din, self.ncls, self.hidden, p=0.1, depth=2) for _ in range(self.k)]).to(DEV)
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=1e-3)

    def step(self, zstat, y):
        self.nets.train()
        loss = sum(F.cross_entropy(n(zstat), y) for n in self.nets) / self.k
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        return float(loss)

    def confusion(self, zstat):
        """CE of adversary posterior against the uniform distribution (minimised when the adversary is maximally unsure)."""
        self.nets.eval()
        return sum(-F.log_softmax(n(zstat), -1).mean(-1).mean() for n in self.nets) / self.k

    @torch.no_grad()
    def acc(self, zstat, y):
        self.nets.eval()
        return float(np.mean([(n(zstat).argmax(1) == y).float().mean().item() for n in self.nets]))


def train_encoder(cfg, rag4, emo, actor_idx, train_clips, val_clips, seed, log=print):
    """cfg keys: dz, pool, bott, beta, lam (adversary weight), vq_*, epochs, bs, lr, adv_reinit, adv_warm.
    rag4: Ragged over [Nfr,4,768]; emo: np int labels [N]; actor_idx: np int actor labels [N] (0..90)."""
    torch.manual_seed(seed); np.random.seed(seed)
    c = dict(dz=64, pool="utt", bott="vib", beta=0.0, lam=0.0, vq_groups=1, vq_codes=64, vq_dim=32, epochs=40, bs=64,
             lr=1e-3, adv_reinit=10, adv_warm=100, adv_k=3, ramp=5, dropout=0.1)
    c.update(cfg)
    release = c_release = cfg.get("release", "z")
    E = Encoder(n_layers=rag4.flat.shape[1], din=rag4.flat.shape[2], dz=c["dz"], pool=c["pool"], bott=c["bott"], beta=c["beta"], vq_groups=c["vq_groups"], vq_codes=c["vq_codes"],
                vq_dim=c["vq_dim"], dropout=c["dropout"]).to(DEV)
    E.release = release
    emo_t = torch.as_tensor(emo, device=DEV)
    train_clips = np.asarray(train_clips)
    # contiguous adversary labels over the encoder-train actors only
    tr_actors = np.unique(actor_idx[train_clips])
    amap = {a: i for i, a in enumerate(tr_actors)}
    adv_t = torch.as_tensor(np.array([amap.get(a, -1) for a in actor_idx]), device=DEV)
    adv = None
    if c["lam"] > 0:
        adv = AdvEnsemble(E.dz * (1 if c["pool"] == "utt" else 2), len(tr_actors), c["adv_k"]).to(DEV)
    opt = torch.optim.AdamW(E.parameters(), lr=c["lr"], weight_decay=1e-2)
    steps = c["epochs"] * math.ceil(len(train_clips) / c["bs"])
    sched = cosine_sched(opt, steps)
    hist = []
    t0 = time.time()

    def adv_warm(n):
        E.eval()
        for _ in range(n):
            b = np.random.choice(train_clips, c["bs"], replace=False)
            x, m = rag4.batch(b)
            with torch.no_grad():
                o = E(x.float(), m)
            adv.step(o["zstat"], adv_t[torch.as_tensor(b, device=DEV)])
        E.train()

    for ep in range(c["epochs"]):
        if adv is not None and ep % c["adv_reinit"] == 0:
            adv.reinit(); adv_warm(c["adv_warm"])
        perm = np.random.permutation(train_clips)
        lam = c["lam"] * min(1.0, (ep + 1) / max(1, c["ramp"]))
        E.train()
        tot = dict(ce=0.0, kl=0.0, conf=0.0, adv=0.0, n=0)
        for s in range(0, len(perm), c["bs"]):
            b = perm[s:s + c["bs"]]
            if len(b) < 4:
                continue
            bt = torch.as_tensor(b, device=DEV)
            x, m = rag4.batch(b)
            o = E(x.float(), m)
            ce = F.cross_entropy(o["emo_logits"], emo_t[bt])
            kl = o["kl"].mean() if o["smask"] is None else (o["kl"] * o["smask"]).sum() / o["smask"].sum()
            loss = ce + c["beta"] * kl + o["aux"]
            if adv is not None:
                conf = adv.confusion(o["zstat"])
                loss = loss + lam * conf
                tot["conf"] += float(conf)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(E.parameters(), 5.0)
            opt.step(); sched.step()
            if adv is not None:
                tot["adv"] += adv.step(o["zstat"].detach(), adv_t[bt])
            tot["ce"] += float(ce); tot["kl"] += float(kl); tot["n"] += 1
        if c["bott"] == "vq":
            E.vq.restart_dead(o["pre_h"])
        if ep % 5 == 4 or ep == c["epochs"] - 1:
            v = eval_head(E, rag4, emo, val_clips)
            n = max(tot["n"], 1)
            h = dict(ep=ep + 1, ce=tot["ce"] / n, kl=tot["kl"] / n, conf=tot["conf"] / n, adv_ce=tot["adv"] / n,
                     val_uar_head=v, sec=round(time.time() - t0, 1))
            hist.append(h); log("   enc", h)
    return E, hist


@torch.no_grad()
def eval_head(E, rag4, emo, clips, bs=128):
    E.eval()
    P = []
    for s in range(0, len(clips), bs):
        x, m = rag4.batch(clips[s:s + bs])
        P.append(E(x.float(), m)["emo_logits"].argmax(1).cpu().numpy())
    return uar(emo[clips], np.concatenate(P), 6)


@torch.no_grad()
def encode_all(E, rag4, bs=128):
    """Released representation of every clip: z = mu (vib/none) or the quantised vector (vq).  Returns Rep + rate stats."""
    E.eval()
    n = rag4.n
    pooled, seqs, lens, kls = [], [], [], []
    hist = torch.zeros(E.vq.G, E.vq.K) if E.bott == "vq" else None
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        x, m = rag4.batch(idx)
        o = E(x.float(), m)
        if hist is not None:
            ix = o["idx"].reshape(-1, E.vq.G).cpu()
            for g in range(E.vq.G):
                hist[g] += torch.bincount(ix[:, g], minlength=E.vq.K).float()
        rel = getattr(E, "release", "z")
        if rel == "posterior":
            pooled.append(torch.softmax(o["emo_logits"], -1).cpu())
        elif rel == "label":
            pooled.append(F.one_hot(o["emo_logits"].argmax(-1), 6).float().cpu())
        else:
            pooled.append(o["zstat"].cpu())
        kls.append(o["kl"].mean(1).cpu() if o["smask"] is not None else o["kl"].cpu())
        if o["smask"] is not None and getattr(E, "release", "z") == "z":
            for i in range(len(idx)):
                seqs.append(o["z"][i][o["smask"][i]].cpu())
    pooled = torch.cat(pooled, 0)
    seq = None
    if seqs and getattr(E, "release", "z") == "z":
        lens = np.array([len(q) for q in seqs])
        seq = Ragged(torch.cat(seqs, 0), np.concatenate([[0], np.cumsum(lens)]))
    rate = dict(kl_nats=float(torch.cat(kls).mean()), bits_vq=E.bits if getattr(E, "release", "z") == "z" else (float(np.log2(6)) if E.release == "label" else None))
    if E.bott == "vq":
        p = hist / hist.sum(1, keepdim=True).clamp(min=1)  # code usage over all 7,442 clips (the training-time usage buffer is reset every epoch)
        rate["vq_perplexity"] = float(torch.exp(-(p * torch.log(p + 1e-9)).sum(1)).mean())
        rate["vq_codes_used"] = float((hist > 0).sum(1).float().mean())
    return Rep(pooled, seq, "learned"), rate
