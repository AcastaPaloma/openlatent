"""F2: ONE supervised nonlinear adapter family: MLP trained with a supervised-contrastive (compound) loss on the Fit set only.

  D -> 1024 -> 512 -> d  (z-scored input, LayerNorm, GELU, dropout 0.1), released vector = L2-normalised output.
  Batches: 192 production compounds x 2 wells (distinct wells of the compound) + 32 Target2 compounds x 4 wells; SupCon, temperature 0.1,
  AdamW lr 1e-3 cosine to 0, wd 1e-4, 8,000 steps, no early stopping, no checkpoint selection.  No adversary, no V/H label.
Pre-registration: results_notes/F.md section 6.3-6.4.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.1
N_PROD, K_PROD, N_T2, K_T2 = 192, 2, 32, 4
STEPS = 8000


class Net(nn.Module):
    def __init__(self, D, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.f = nn.Sequential(nn.Linear(D, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, d))

    def forward(self, x):
        return F.normalize(self.f((x - self.mu) / self.sd), dim=1)


def supcon(z, y):
    n = z.shape[0]
    sim = z @ z.T / TAU
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1)
    ok = npos > 0
    loss = -(logp * pos).sum(1)[ok] / npos[ok]
    return loss.mean()


def train_supcon(Xfit, lab, n_prod, d, seed, steps=STEPS):
    """Xfit (nf, D) float32 tensor on DEV.  lab: int array (nf,): production compound classes are 0..n_prod-1 (each with >= 2 wells),
    Target2 compound classes are n_prod.., -1 = not used (DMSO, V/H, unlabeled).  Returns (model, info)."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    lab = np.asarray(lab)
    ok = np.where(lab >= 0)[0]
    order = ok[np.argsort(lab[ok], kind="stable")]
    cl, start, cnt = np.unique(lab[order], return_index=True, return_counts=True)
    mu, sd = Xfit.mean(0, keepdim=True), Xfit.std(0, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    net = Net(Xfit.shape[1], d, mu, sd).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    order_t = torch.as_tensor(order, device=DEV)
    start_t, cnt_t, cl_t = (torch.as_tensor(a, device=DEV) for a in (start, cnt, cl))
    prod_c = torch.where(cl_t < n_prod)[0]
    t2_c = torch.where(cl_t >= n_prod)[0]
    losses = []
    net.train()
    for step in range(steps):
        pc = prod_c[torch.randint(len(prod_c), (N_PROD,), device=DEV, generator=g)]
        tc = t2_c[torch.randint(len(t2_c), (N_T2,), device=DEV, generator=g)]
        n = cnt_t[pc]                                            # >= 2 wells
        a = torch.minimum((torch.rand(N_PROD, device=DEV, generator=g) * n).long(), n - 1)
        off = 1 + torch.minimum((torch.rand(N_PROD, device=DEV, generator=g) * (n - 1)).long(), n - 2)
        b = (a + off) % n                                         # a distinct second well
        idx_p = torch.stack([start_t[pc] + a, start_t[pc] + b], 1).reshape(-1)
        lab_p = cl_t[pc].repeat_interleave(2)
        n2 = cnt_t[tc]
        r = torch.minimum((torch.rand(N_T2, K_T2, device=DEV, generator=g) * n2[:, None]).long(), (n2 - 1)[:, None])
        idx_t = (start_t[tc][:, None] + r).reshape(-1)
        lab_t = cl_t[tc].repeat_interleave(K_T2)
        wells = order_t[torch.cat([idx_p, idx_t])]
        y = torch.cat([lab_p, lab_t])
        loss = supcon(net(Xfit[wells]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.detach())
    losses = torch.stack(losses).cpu().numpy()
    net.eval()
    info = dict(loss_first20=float(losses[:20].mean()), loss_last200=float(losses[-200:].mean()), steps=steps, n_prod_classes=int(len(prod_c)), n_t2_classes=int(len(t2_c)),
                loss_curve=[float(losses[i:i + 400].mean()) for i in range(0, steps, 400)], nan=bool(np.isnan(losses).any()))
    return net, info


@torch.no_grad()
def embed(net, X, bs=65536):
    return torch.cat([net(X[i:i + bs]) for i in range(0, X.shape[0], bs)])


def effective_rank(E):
    E = E.double()
    E = E - E.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(E.T @ E / (E.shape[0] - 1)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())
