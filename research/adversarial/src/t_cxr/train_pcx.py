"""In-programme retrain of a PriCheXy-style defender (T.md A3 / section 7.2).  EVERY component is trained on S-role patients only (the generator's own patients);
nothing from A / V / T is read.  Re-implemented from the paper text and from reading the released training loop (`utils.train`); no GPL-3 code is copied.

Stages (one job each, weights stay in --wdir on the VM, only small JSON logs go to $OUT):
  base : (1) verifier  Siamese ResNet-50, ImageNet init, BCE on S pairs (random negatives, <= 2 positives per S multi-visit patient), early stopping on PriCheXy's validation pairs
         (2) classifier  DenseNet-121 (14 sigmoid outputs), ImageNet init, BCE on S images
         (3) pretrain  U-Net generator by reconstruction (mu = 0.01 pipeline), images of the training pairs
  adv  : adversarial training for one mu: generator loss = BCE-with-logits(aux classifier on the deformed image1, 14 findings) + (-log(1 - sigmoid(verifier(deformed1, real2)))),
         then the verifier is updated on (deformed1.detach(), real2) pairs and the classifier on deformed1.detach() (as in the released loop); checkpoint = lowest VALIDATION total loss.
Flow construction is generator.warp (F = id - mu * G(x), Gaussian filter 9 / 2 with zero padding, grid_sample border / align_corners).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attackers as A, generator as G, meta, pairs, stats as S, utility as U
from .run_suite import planted_bank

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dense121(pretrained=True):
    import torchvision.models as tvm
    m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
    return m


def clf_logits(m, x):
    f = F.relu(m.features(x), inplace=False)
    return m.classifier[0](F.adaptive_avg_pool2d(f, 1).flatten(1))


def norm3(x, size=None):
    x = x.expand(-1, 3, -1, -1)
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_data(a):
    """metadata (+ roles), the bank, and the paper's pair lists restricted to ok images; synthetic mode substitutes N0 pairs of S patients."""
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d)
    if a.synthetic:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(sorted(seen), size=a.synthetic, replace=False))
        d = d[d.pid.isin(keep)].reset_index(drop=True); d["row"] = np.arange(len(d)); bank = planted_bank(d, a.res, 1.0)
        seen = keep
    else:
        bank = np.load(a.bank, mmap_mode="r")
    d = meta.assign_roles(d, seen, "S-only", 0)
    d.loc[d.role != "S", "ok"] = False                       # nothing outside S is ever read
    return d, bank


def pair_lists(a, d):
    if a.synthetic:
        tr, _ = pairs.build(d, "S", "N0", 1, "pcx-tr", col="role")
        pid = tr.pid_a.unique(); half = set(pid[: len(pid) // 5])
        va = tr[tr.pid_a.isin(half) | tr.pid_b.isin(half)]; tr = tr[~(tr.pid_a.isin(half) | tr.pid_b.isin(half))]
        return tr.reset_index(drop=True), va.reset_index(drop=True)
    out = []
    for f in (meta.PRI):
        p = pairs.from_pair_file(f"{a.pri}/{f}", d)
        bad = ~d["ok"].to_numpy()
        out.append(p[~(bad[p.a.to_numpy()] | bad[p.b.to_numpy()])].reset_index(drop=True))
    return out[0], out[1]


def to_gpu(bank, rows, dev):
    return torch.from_numpy(np.ascontiguousarray(bank[rows])).to(dev).half().div_(255.0) if bank.dtype == np.uint8 else torch.as_tensor(bank[rows]).to(dev).half()


def findings(d, a):
    import pandas as pd
    raw = pd.read_csv(f"{a.data}/Data_Entry_2017_v2020.csv"); raw.columns = [c.strip() for c in raw.columns]
    lab = dict(zip(raw["Image Index"], raw["Finding Labels"].astype(str)))
    fl = d["img"].map(lab).str.split("|")
    return np.array([[f in row for f in meta.FINDINGS] for row in fl], dtype=np.float32)


def stage_base(a, d, bank, dev, log):
    wd = Path(a.wdir); wd.mkdir(parents=True, exist_ok=True)
    tr, va = pair_lists(a, d)
    # ---- (1) verifier on S pairs
    pS, rS = pairs.build(d, "S", "N0", 2, "pcx-ver", col="role")
    need = np.unique(np.r_[pS.a, pS.b, va.a, va.b]); pos = {r: i for i, r in enumerate(need)}
    X = to_gpu(bank, need, dev)
    mk = lambda p: dict(xa=X, xb=X, ia=np.array([pos[r] for r in p.a]), ib=np.array([pos[r] for r in p.b]), y=p.label.to_numpy().astype(np.float32))
    t0 = time.time()
    out = A.train_siamese(mk(pS), mk(va), mk(va), arch="resnet50", pretrained=not a.no_pretrain, seed=0, max_epochs=a.ver_epochs, patience=3, bs=32, device=dev, amp=dev == "cuda", log=log, return_model=True)
    torch.save(out["model"].state_dict(), wd / "verifier.pth")
    auc_va = S.auc(va.label.to_numpy(), out["val_score"])
    log(f"verifier: {out['epochs']} epochs, S-validation AUC {auc_va:.3f}, {time.time() - t0:.0f}s")
    del X, out
    # ---- (2) classifier on S images
    rows = d.index[d.ok].to_numpy()
    Y = findings(d, a)[rows]
    Xs = to_gpu(bank, d["row"].to_numpy()[rows], dev)
    clf = dense121(not a.no_pretrain)
    t0 = time.time()
    clf = U.finetune(clf, Xs, Y, epochs=a.clf_epochs, lr=1e-4, bs=32, seed=0, device=dev, size=a.clf_size, log=log)
    torch.save(clf.state_dict(), wd / "classifier.pth")
    vr = np.unique(np.r_[va.a, va.b]); Xv = to_gpu(bank, d["row"].to_numpy()[vr], dev)
    P = U.predict(clf, Xv, dev, size=a.clf_size); uv, _ = U.mean_auc(findings(d, a)[vr].astype(np.uint8), P)
    log(f"classifier: S-validation mean AUC {uv:.3f}, {time.time() - t0:.0f}s")
    del Xs, Xv
    # ---- (3) generator reconstruction pre-training (mu = 0.01), images of the training pairs
    imgs = np.unique(tr.a.to_numpy()); Xg = to_gpu(bank, d["row"].to_numpy()[imgs], dev)
    gen = G.UNet(1, 2, 32).to(dev)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
    gl = torch.Generator(); gl.manual_seed(0); hist = []
    t0 = time.time()
    for ep in range(a.pre_epochs):
        gen.train(); tot = 0.0
        for idx in torch.randperm(len(Xg), generator=gl).split(a.bs):
            x = Xg[idx].float().unsqueeze(1)
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                raw = gen(x)
            y = G.warp(x, raw.float(), 0.01)
            loss = F.mse_loss(y, x)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        hist.append(tot / len(Xg)); log(f"  pretrain ep {ep} mse {hist[-1]:.3e}")
    torch.save(gen.state_dict(), wd / "gen_pretrained.pth")
    return dict(verifier_epochs_auc_val=auc_va, classifier_val_auc=uv, pretrain_mse=hist, n_verifier_pairs=len(pS), n_clf_images=len(rows), pair_report=rS)


def stage_adv(a, d, bank, dev, log):
    wd = Path(a.wdir); mu = a.mu
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    c1_name = f"generator_pcxr_mu{mu:g}_{a.tag}_C1.pth" if a.tag else f"generator_pcxr_mu{mu:g}.pth"
    c2_name = f"generator_pcxr_mu{mu:g}_{a.tag}_C2.pth" if a.tag else f"generator_pcxr_mu{mu:g}_C2.pth"
    tr, va = pair_lists(a, d)
    rows_all = np.unique(np.r_[tr.a, tr.b, va.a, va.b]); pos = {r: i for i, r in enumerate(rows_all)}
    X = to_gpu(bank, d["row"].to_numpy()[rows_all], dev)
    Yall = torch.as_tensor(findings(d, a)[rows_all])
    pk = lambda p: (torch.as_tensor([pos[r] for r in p.a]), torch.as_tensor([pos[r] for r in p.b]), torch.as_tensor(p.label.to_numpy(), dtype=torch.float32))
    (tia, tib, ty), (via, vib, vy) = pk(tr), pk(va)
    ver = A.Siamese("resnet50", False).to(dev); ver.load_state_dict(torch.load(wd / "verifier.pth", map_location=dev)); ver.eval()
    clf = dense121(False).to(dev); clf.load_state_dict(torch.load(wd / "classifier.pth", map_location=dev)); clf.eval()
    gen = G.UNet(1, 2, 32).to(dev); gen.load_state_dict(torch.load(wd / "gen_pretrained.pth", map_location=dev))
    og = torch.optim.Adam(gen.parameters(), lr=1e-4); ov = torch.optim.Adam(ver.parameters(), lr=1e-4)
    oc = torch.optim.SGD(clf.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    ac = lambda x: dev == "cuda"
    gl = torch.Generator(); gl.manual_seed(a.seed)
    best, hist, t_start = 1e9, [], time.time()
    for ep in range(a.adv_epochs):
        t0 = time.time(); agg = np.zeros(3); n = 0
        for idx in torch.randperm(len(tia), generator=gl).split(a.bs):
            x1, x2 = X[tia[idx]].float().unsqueeze(1), X[tib[idx]].float().unsqueeze(1)
            lab, yv = Yall[tia[idx]].to(dev), ty[idx].to(dev)
            gen.train()
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                raw = gen(x1)
            fake = G.warp(x1, raw.float(), mu)
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                acl = F.binary_cross_entropy_with_logits(clf_logits(clf, norm3(fake, a.clf_size)).float(), lab)
                p = torch.sigmoid(ver(norm3(fake), norm3(x2)).float())
            lik = (-torch.log(1 - p + 1e-6)).mean()
            tot = acl + lik
            og.zero_grad(set_to_none=True); tot.backward(); og.step()
            fd = fake.detach()
            ver.train()
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                lv = F.binary_cross_entropy_with_logits(ver(norm3(fd), norm3(x2)).float(), yv)
            ov.zero_grad(set_to_none=True); lv.backward(); ov.step(); ver.eval()
            clf.train()
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                pc = clf(norm3(fd, a.clf_size))
            lc = F.binary_cross_entropy(pc.float().clamp(1e-6, 1 - 1e-6), lab)           # BCE outside autocast (unsafe to autocast)
            oc.zero_grad(set_to_none=True); lc.backward(); oc.step(); clf.eval()
            agg += np.array([float(acl), float(lik), float(tot)]) * len(idx); n += len(idx)
            if a.max_iters and n >= a.max_iters * a.bs:
                break
        # ---- validation (S validation pairs): total loss decides the checkpoint, as in the paper
        gen.eval(); vs = np.zeros(3); vn = 0; scores, labs = [], []
        with torch.no_grad():
            for idx in torch.arange(len(via)).split(a.bs):
                x1, x2 = X[via[idx]].float().unsqueeze(1), X[vib[idx]].float().unsqueeze(1); lab = Yall[via[idx]].to(dev)
                with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    fake = G.warp(x1, gen(x1).float(), mu)
                    acl = F.binary_cross_entropy_with_logits(clf_logits(clf, norm3(fake, a.clf_size)).float(), lab)
                    pr = torch.sigmoid(ver(norm3(fake), norm3(x2)).float())
                lik = (-torch.log(1 - pr + 1e-6)).mean()
                vs += np.array([float(acl), float(lik), float(acl + lik)]) * len(idx); vn += len(idx); scores.append(pr.cpu()); labs.append(vy[idx])
        vs /= vn
        auc_adv = S.auc(torch.cat(labs).numpy(), torch.cat(scores).numpy())
        hist.append(dict(epoch=ep, train=(agg / max(n, 1)).tolist(), val_ac=float(vs[0]), val_lik=float(vs[1]), val_total=float(vs[2]), val_adversary_auc=float(auc_adv), seconds=round(time.time() - t0, 1)))
        log(f"  adv mu={mu} ep {ep}: train total {agg[2] / max(n, 1):.3f} | val ac {vs[0]:.3f} lik {vs[1]:.3f} total {vs[2]:.3f} | co-trained adversary AUC {auc_adv:.3f} | {hist[-1]['seconds']}s")
        if vs[2] < best:
            best = vs[2]; torch.save(gen.state_dict(), wd / c1_name); hist[-1]["saved"] = True
        if (time.time() - t_start) / 3600 > a.max_hours:
            log(f"  STOP: --max-hours {a.max_hours} reached after epoch {ep}"); hist[-1]["truncated_by_max_hours"] = True
            break
    torch.save(gen.state_dict(), wd / c2_name)                                         # C2 = the LAST epoch's generator (T.md 8.4)
    return dict(mu=mu, seed=a.seed, tag=a.tag, history=hist, best_val_total=float(best), c1=c1_name, c2=c2_name, epochs_completed=len(hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "adv"]); ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", default=None); ap.add_argument("--exclude", default=None)
    ap.add_argument("--wdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ver-epochs", type=int, default=20); ap.add_argument("--clf-epochs", type=int, default=5); ap.add_argument("--pre-epochs", type=int, default=20)
    ap.add_argument("--adv-epochs", type=int, default=40); ap.add_argument("--max-hours", type=float, default=1.8); ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--clf-size", type=int, default=224); ap.add_argument("--no-pretrain", action="store_true"); ap.add_argument("--tag", default="", help="checkpoint tag, e.g. s1 (T3)")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--res", type=int, default=64)
    a = ap.parse_args()
    if not a.synthetic and not a.exclude:
        raise SystemExit("--exclude exclude_dups.json is required for real runs (T.md A2)")
    dev = a.device
    d, bank = load_data(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); lines.append(s)
    t0 = time.time()
    res = stage_base(a, d, bank, dev, log) if a.stage == "base" else stage_adv(a, d, bank, dev, log)
    res.update(stage=a.stage, seconds=round(time.time() - t0, 1), synthetic=bool(a.synthetic), n_S_images_ok=int(d["ok"].sum()), epochs=dict(ver=a.ver_epochs, clf=a.clf_epochs, pre=a.pre_epochs, adv=a.adv_epochs))
    (out / f"train_{a.stage}{'_mu' + format(a.mu, 'g') + ('_' + a.tag if a.tag else '') if a.stage == 'adv' else ''}.json").write_text(json.dumps(res, indent=1, default=float))
    print("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
