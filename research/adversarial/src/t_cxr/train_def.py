"""T7 (T.md section 15): a LEARNED image defender trained adversarially against an F1+-like pair attacker, with a diagnostic (14-finding) utility term.
Trained on S-role patients only (the defender's own patients); 10% of the S multi-visit patients are HELD OUT (never seen by the generator, the in-loop adversary or the diagnostic head)
and give the in-loop gate numbers.  Nothing from A / V / T is read.

  G  : ResUNet, released image = clamp(x + U(x), 0, 1), zero-initialised (starts at the identity), full 256 px
  D  : in-loop adversary = Siamese ResNet-50, 512-d xl head, pair feature [|ea-eb|, ea*eb, ea+eb] (the F1+ design), ImageNet init, trained with the F1+ augmentation on (G(x1), x2) pairs
  C  : diagnostic head DenseNet-121 (14 sigmoid outputs, starts from the S-trained classifier.pth), updated on G(x1).detach()
  Loss for G:  BCE(C(G(x1)), findings) + lambda * BCE_with_logits(D(G(x1), x2), 0.5)      (domain confusion: D's pair logit pushed to 0 on EVERY pair)
  Updates alternate each step: G, then D (BCE on the true pair labels, G(x1) detached), then C.  After `warm` D-only steps.
The checkpoint is the LAST step (no selection).  Gate numbers are measured every `eval_every` seconds on the held-out S patients: D's AUC on held-out N2 pairs, the AUC gap between held-in and held-out pairs
(memorisation), C's macro AUC on the defended held-out images versus the raw classifier on the raw images (collapse), and the AUC of D on pairs of pure-noise images (null-input floor)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import attackers as A, generator as G, meta, pairs, stats as S, utility as U
from .train_pcx import clf_logits, dense121, findings, load_data, norm3, to_gpu


def split_hold(d, frac: float, salt: str):
    ok = d[d["ok"]]
    cnt = ok.groupby("pid").size()
    multi = cnt[cnt >= 2].index
    return {p for p in multi if meta.unit(p, salt) < frac}


@torch.no_grad()
def evaluate(gen, ver, clf, clf0, X, Yall, pa, pb, py, ia_img, dev, bs=64, size=224):
    """D's AUC on the pairs (G(x1) vs raw x2) and the diagnostic macro AUC of C on G(x1) for the anchor images `ia_img`."""
    gen.eval(); ver.eval(); clf.eval()
    sc = []
    for idx in torch.arange(len(pa)).split(bs):
        x1, x2 = X[pa[idx]].float().unsqueeze(1), X[pb[idx]].float().unsqueeze(1)
        with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
            sc.append(ver(norm3(gen(x1)), norm3(x2)).float().cpu())
    auc = S.auc(py.numpy(), torch.cat(sc).numpy())
    out = {}
    for nm, net, defend in (("C_defended", clf, True), ("C0_raw", clf0, False)):
        P = []
        for idx in torch.arange(len(ia_img)).split(bs):
            x = X[ia_img[idx]].float().unsqueeze(1)
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
                P.append(torch.sigmoid(clf_logits(net, norm3(gen(x) if defend else x, size)).float()).cpu())
        out[nm] = U.mean_auc(Yall[ia_img].numpy().astype(np.uint8), torch.cat(P).numpy())[0]
    return dict(auc=float(auc), **{k: float(v) for k, v in out.items()})


@torch.no_grad()
def noise_floor(gen, ver, dev, n=2000, res=256, bs=64):
    """null-input floor: pairs of independent pure-noise images with random 50/50 labels carry no identity; D's AUC on them must be 0.5."""
    g = torch.Generator(device="cpu"); g.manual_seed(12345)
    y = (torch.rand(n, generator=g) < 0.5).float()
    gen.eval(); ver.eval(); sc = []
    for s in range(0, n, bs):
        k = min(bs, n - s)
        x1 = (0.5 + 0.2 * torch.randn(k, 1, res, res, generator=g)).clamp(0, 1).to(dev); x2 = (0.5 + 0.2 * torch.randn(k, 1, res, res, generator=g)).clamp(0, 1).to(dev)
        with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda"):
            sc.append(ver(norm3(gen(x1)), norm3(x2)).float().cpu())
    return float(S.auc(y.numpy(), torch.cat(sc).numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--lam", type=float, required=True); ap.add_argument("--budget-s", type=int, default=3000); ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", default=None); ap.add_argument("--exclude", default=None)
    ap.add_argument("--wdir", required=True, help="dir with classifier.pth (S-trained DenseNet); the defender weights are saved here as generator_defl_<tag>.pth (VM only)"); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--warm", type=int, default=150)
    ap.add_argument("--hold-frac", type=float, default=0.10); ap.add_argument("--clf-size", type=int, default=224); ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--res", type=int, default=256); ap.add_argument("--arch", default="resnet50"); ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    if not a.synthetic and not a.exclude:
        raise SystemExit("--exclude exclude_dups.json is required for real runs (T.md A2)")
    dev = a.device
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d, bank = load_data(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); wd = Path(a.wdir); wd.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); lines.append(s)

    hold = split_hold(d, a.hold_frac, "t7:hold")
    dtr, dho = d.copy(), d.copy()
    dtr.loc[dtr.pid.isin(hold), "ok"] = False; dho.loc[~dho.pid.isin(hold), "ok"] = False
    pTr, rTr = pairs.build(dtr, "S", "N2", 2, "t7tr", col="role"); pHo, rHo = pairs.build(dho, "S", "N2", 1, "t7ho", col="role")
    sub = pTr.sample(n=min(len(pHo), len(pTr)), random_state=0)                                     # held-in pairs, same count as the held-out ones (memorisation gap)
    rows_all = np.unique(np.r_[pTr.a, pTr.b, pHo.a, pHo.b]); pos = {r: i for i, r in enumerate(rows_all)}
    X = to_gpu(bank, d["row"].to_numpy()[rows_all], dev)
    Yall = torch.as_tensor(findings(d, a)[rows_all])
    pk = lambda p: (torch.as_tensor([pos[r] for r in p.a]), torch.as_tensor([pos[r] for r in p.b]), torch.as_tensor(p.label.to_numpy(), dtype=torch.float32))
    (tia, tib, ty), (hia, hib, hy), (sia, sib, sy) = pk(pTr), pk(pHo), pk(sub)
    log(f"S patients held out: {len(hold)}; train pairs {len(pTr)}, held-out pairs {len(pHo)}; images on device {len(rows_all)}")
    ver = A.Siamese(a.arch, not a.no_pretrain, "xl").to(dev).to(memory_format=torch.channels_last)
    clf = dense121(False).to(dev); clf0 = dense121(False).to(dev)
    sd = torch.load(wd / "classifier.pth", map_location=dev) if (wd / "classifier.pth").exists() else None
    if sd is not None:
        clf.load_state_dict(sd); clf0.load_state_dict(sd)
    clf0.eval()
    gen = G.ResUNet(32).to(dev)
    og = torch.optim.Adam(gen.parameters(), lr=1e-4); ov = torch.optim.Adam(ver.parameters(), lr=1e-4); oc = torch.optim.SGD(clf.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    held_img = torch.unique(torch.cat([hia, hib]))                                                  # anchor images for the diagnostic gate: held-out patients' images
    ac = dict(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev == "cuda")
    gl = torch.Generator(); gl.manual_seed(a.seed)
    hist, step, t_start, next_eval = [], 0, time.time(), a.eval_every

    def do_eval(tag):
        e_ho = evaluate(gen, ver, clf, clf0, X, Yall, hia, hib, hy, held_img, dev, size=a.clf_size)
        e_tr = evaluate(gen, ver, clf, clf0, X, Yall, sia, sib, sy, held_img[:64], dev, size=a.clf_size)
        r = dict(tag=tag, step=step, seconds=round(time.time() - t_start), heldout_auc=e_ho["auc"], heldin_auc=e_tr["auc"], memorisation_gap=e_ho["auc"] - e_tr["auc"], C_defended=e_ho["C_defended"], C0_raw=e_ho["C0_raw"], C_drop=e_ho["C0_raw"] - e_ho["C_defended"])
        hist.append(r); log("  EVAL", json.dumps(r))
        return r

    ver.train()
    done = False
    while not done:
        for idx in torch.randperm(len(tia), generator=gl).split(a.bs):
            x1, x2 = X[tia[idx]].float().unsqueeze(1), X[tib[idx]].float().unsqueeze(1)
            lab, yv = Yall[tia[idx]].to(dev), ty[idx].to(dev)
            if step >= a.warm:                                                                       # ---- G step (D and C frozen in eval mode)
                gen.train(); ver.eval(); clf.eval()
                with torch.autocast(**ac):
                    fake = gen(x1)
                    dl = ver(norm3(fake), norm3(x2)).float(); cl = clf_logits(clf, norm3(fake, a.clf_size)).float()
                l_conf = F.binary_cross_entropy_with_logits(dl, torch.full_like(dl, 0.5)); l_diag = F.binary_cross_entropy_with_logits(cl, lab)
                tot = l_diag + a.lam * l_conf
                og.zero_grad(set_to_none=True); tot.backward(); og.step()
                fd = fake.detach()
            else:
                with torch.no_grad():
                    gen.eval(); fd = gen(x1)
                l_conf = l_diag = torch.zeros(())
            ver.train()                                                                              # ---- D step on (G(x1).detach(), x2) with the true pair labels, F1+ augmentation
            with torch.autocast(**ac):
                lg = ver(norm3(A.augment(fd.squeeze(1).float()).unsqueeze(1)), norm3(A.augment(x2.squeeze(1)).unsqueeze(1))).float()
            lv = F.binary_cross_entropy_with_logits(lg, yv)
            ov.zero_grad(set_to_none=True); lv.backward(); ov.step()
            if step >= a.warm:                                                                       # ---- C step
                clf.train()
                with torch.autocast(**ac):
                    pc = clf_logits(clf, norm3(fd, a.clf_size)).float()
                lc = F.binary_cross_entropy_with_logits(pc, lab)
                oc.zero_grad(set_to_none=True); lc.backward(); oc.step(); clf.eval()
            step += 1
            if step % 100 == 0:
                log(f"  step {step} t={time.time() - t_start:.0f}s  diag {float(l_diag):.3f} conf {float(l_conf):.3f} D {float(lv):.3f}")
            if time.time() - t_start >= next_eval:
                do_eval(f"t{next_eval}"); next_eval += a.eval_every; ver.train()
            if time.time() - t_start >= a.budget_s or (a.max_steps and step >= a.max_steps):
                done = True; break
    final = do_eval("final")
    nf = noise_floor(gen, ver, dev, n=2000 if not a.synthetic else 200, res=X.shape[-1])
    torch.save(gen.state_dict(), wd / f"generator_defl_{a.tag}.pth")
    res = dict(tag=a.tag, lam=a.lam, steps=step, seconds=round(time.time() - t_start), budget_s=a.budget_s, final=final, noise_null_auc=nf, history=hist, synthetic=bool(a.synthetic),
               n_heldout_patients=len(hold), n_train_pairs=len(pTr), n_heldout_pairs=len(pHo), pair_report_train=rTr, pair_report_heldout=rHo)
    (out / f"train_def_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    (out / f"train_def_{a.tag}.log").write_text("\n".join(lines))
    print("done", res["seconds"], "s; noise null AUC", round(nf, 3))


if __name__ == "__main__":
    main()
