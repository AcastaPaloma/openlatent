"""One release x one rotation: pairs -> defended images -> attackers -> hashed-id predictions + summary (JSON only in $OUT).

  python -m t_cxr.run_suite --release rw2_mu0.01 --rot 0 --attackers f1,f3,f4,f5 --conds match,mismatch,anon2 --seeds 0,1,2 --out $OUT
Post-gate it reads the real bank (`--bank cxr256.npy`) and the custodian's salt (`--salt-file`).  `--synthetic N` builds a PLANTED-LEAK bank (res 64) on a
random patient subset with the real CSV metadata and a provisional salt, to test the whole pipeline on CPU; synthetic output is labelled and never a result.
Outputs: `$OUT/<release>_rot<r>/{summary.json, preds_<attacker>.json}` (pair_id, sha256-hashed patient id, y, scores for V and T).  No attacker weights,
defended images or embeddings are written to $OUT (whitelist condition e); numpy caches go to `--work`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import attackers as A, generator as G, heads as H, ledger, meta, pairs, releases as R, stats as S

from .meta import PRI, hid


def planted_bank(d: pd.DataFrame, res: int, alpha: float, seed: int = 0) -> np.ndarray:
    """synthetic images: patient-specific smooth texture (identity, strength alpha) + age/sex-dependent tint + per-image noise; shape (len(d),res,res) uint8."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    pat = {p: gaussian_filter(rng.normal(0, 1, (res, res)), 3) for p in d.pid.unique()}
    out = np.empty((len(d), res, res), np.uint8)
    for i, (p, age, male) in enumerate(zip(d.pid.to_numpy(), d.age.to_numpy(), d.male.to_numpy())):
        x = alpha * 3 * pat[p] + gaussian_filter(rng.normal(0, 1, (res, res)), 2) + 0.01 * (age - 50) + 0.3 * male
        out[i] = np.clip(128 + 40 * x, 0, 255).astype(np.uint8)
    return out


def split_pairs(pr: pd.DataFrame, pos_of: dict):
    return dict(ia=np.array([pos_of[r] for r in pr.a]), ib=np.array([pos_of[r] for r in pr.b]), y=pr.label.to_numpy().astype(np.float32))


def cond_sets(cond, Xd, Xr, parts):
    """xa/xb tensors per phase for a condition (see attackers.py header)."""
    out = {}
    for ph in ("tr", "va", "te"):
        xa = Xd
        xb = Xr if (cond == "match" or (cond == "mismatch" and ph == "te")) else Xd
        out[ph] = dict(xa=xa, xb=xb, **parts[ph])
    return out


def auc_ci(y, s, cluster, B, seed):
    pb = S.PatientBootstrap(cluster, B=B, seed=seed)
    return S.summarize(S.auc(y, s), S.auc_boot(y, s, pb.codes(cluster), pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--attackers", default="f1,f3,f4,f5"); ap.add_argument("--conds", default="match,mismatch,anon2"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data", default=None); ap.add_argument("--pri", default=None); ap.add_argument("--bank", default=None); ap.add_argument("--salt-file", default=None)
    ap.add_argument("--gen-dir", default=None); ap.add_argument("--flow-stats", default=None); ap.add_argument("--work", default="/tmp/t_work"); ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet50"); ap.add_argument("--max-epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--no-pretrain", action="store_true"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--synthetic", type=int, default=0, help="N patients; builds a planted-leak bank at --res")
    ap.add_argument("--res", type=int, default=64); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--feat", default="dinov2", choices=["dinov2", "random"]); ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--exclude", default=None, help="exclude_dups.json (T.md A2 b); required for real runs")
    ap.add_argument("--screen", action="store_true", help="T2 screen: validation only. T pairs / T images are never built, no ledger entry; output dir suffix _screen")
    ap.add_argument("--pcxr-dir", default=None, help="dir with generator_pcxr_mu*.pth (train_pcx.py)")
    ap.add_argument("--ladder-s", type=int, default=0, help="T4a: add this many S-role multi-visit patients (2 triples each) to the attacker-training pairs; -1 = all of them")
    ap.add_argument("--r5", action="store_true", help="T5 (T.md section 12): attacker F1+ (xl head, augmentation, cosine lr, validation-AUC selection, time budget)")
    ap.add_argument("--design", default="N2", choices=["N2", "N3", "N3s"], help="T4b: N3 = negatives matched on the 14-finding vector (T.md section 11); N3s = N3 restricted to N2 tiers 0-1 (A12)")
    ap.add_argument("--shuffle-val", action="store_true", help="with --shuffle-labels: ALSO permute the validation labels (removes the epoch-selection leak of the control; T.md D3); suffix _shuf2")
    ap.add_argument("--shuffle-labels", action="store_true", help="control: permute the TRAINING pair labels (attackers must score at chance); outputs are suffixed _shuf")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out) / f"{a.release}_rot{a.rot}{('_shuf2' if a.shuffle_val else '_shuf') if a.shuffle_labels else ''}{'_screen' if a.screen else ''}{('_ladder' + ('all' if a.ladder_s < 0 else str(a.ladder_s))) if a.ladder_s else ''}{'_' + a.design.lower() if a.design != 'N2' else ''}{'_r5' if a.r5 else ''}"; out.mkdir(parents=True, exist_ok=True)
    if not a.synthetic and not a.exclude:
        raise SystemExit("--exclude exclude_dups.json is required for real runs (T.md A2)")
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in PRI], d)
    if a.synthetic:
        salt = "PROVISIONAL-selftest"
        rng = np.random.default_rng(0)
        keep = set(rng.choice(d.pid.unique(), size=a.synthetic, replace=False))
        d = d[d.pid.isin(keep)].reset_index(drop=True); d["row"] = np.arange(len(d))
        bank = planted_bank(d, a.res, a.alpha)
    else:
        salt = Path(a.salt_file).read_text().strip()                  # refuses to run without the custodian's salt
        bank = np.load(a.bank, mmap_mode="r")
    d = meta.assign_roles(d, seen, salt, a.rot)
    labcode = None
    if a.design in ("N3", "N3s"):
        Lm = meta.findings_matrix(d, f"{a.data}/Data_Entry_2017_v2020.csv").astype(np.int64)
        labcode = (Lm * (1 << np.arange(14))).sum(1)
    pA, rA = pairs.build(d, "A", a.design, 2, "tr0", col="role", labcode=labcode); pV, rV = pairs.build(d, "V", a.design, 1, "v0", col="role", labcode=labcode)
    if a.ladder_s:                                                   # T4a: extra attacker-training pairs from the S role (valid for floors / raw; defender-seen for pcxr)
        posS = pairs.positives(d, "S", 2, "lad", col="role")
        order = posS.pid.drop_duplicates().map(lambda p: meta.unit(p, f"{salt}:ladder")).sort_values().index
        keep_pid = set(posS.pid.drop_duplicates().loc[order][: (None if a.ladder_s < 0 else a.ladder_s)])
        posS = posS[posS.pid.isin(keep_pid)].reset_index(drop=True)
        pS, rS = pairs.build(d, "S", "N2", 2, "lad", positives_df=posS, col="role")
        pS["triple"] = pS["triple"] + 10**6
        pA = pd.concat([pA, pS], ignore_index=True); rA = dict(rA, ladder_S=rS)
    if a.screen:
        pT, rT = pV.copy(), rV                                      # SCREEN: the 'test' slot is the validation pairs; no T image or T pair is ever built
    else:
        pT, rT = pairs.build(d, "T", a.design, 1, "te0", col="role", labcode=labcode)
    need = np.unique(np.r_[pA.a, pA.b, pV.a, pV.b, pT.a, pT.b]); pos_of = {r: i for i, r in enumerate(need)}
    parts = dict(tr=split_pairs(pA, pos_of), va=split_pairs(pV, pos_of), te=split_pairs(pT, pos_of))
    sfx = ""
    if a.shuffle_labels:
        parts["tr"]["y"] = np.random.default_rng(777).permutation(parts["tr"]["y"]); sfx = "_shuf"
        if a.shuffle_val:
            parts["va"]["y"] = np.random.default_rng(778).permutation(parts["va"]["y"]); sfx = "_shuf2"
    cfg = dict(attackers=a.attackers, conds=a.conds, seeds=a.seeds, arch=a.arch, max_epochs=a.max_epochs, synthetic=a.synthetic, salt_id=hid("salt-id", salt), shuffle_val=a.shuffle_val, ladder_s=a.ladder_s, design=a.design, r5=a.r5, **(dict(shuffle_labels=True) if a.shuffle_labels else {}))   # salt id: a new partition is new sealed material
    ledger_rec = None if (a.no_ledger or a.synthetic or a.screen) else ledger.sealed_query(a.release, cfg, "T", a.rot)
    dev = a.device
    flow = torch.load(a.flow_stats) if a.flow_stats else None
    pca = R.fit_pca(bank, d.loc[(d.role == "A") & d.ok, "row"].to_numpy()[:10000], size=min(64, a.res)) if a.release.startswith("pca_") else None
    rel = R.Releases(dev, gen_dir=a.gen_dir, flow_stats=flow, pca=pca, pcxr_dir=a.pcxr_dir)
    ids = d["img"].to_numpy()[need].tolist()
    Xr = rel.materialize("raw", bank, need, ids)
    Xd = Xr if a.release == "raw" else rel.materialize(a.release, bank, need, ids)
    summ = dict(release=a.release, rot=a.rot, screen=bool(a.screen), synthetic=bool(a.synthetic), salt_provisional=bool(a.synthetic), evaluator_sha=ledger.evaluator_sha()[:16], ledger=ledger_rec,
                n_triples=dict(A=int(len(pA) // 2), V=int(len(pV) // 2), T=int(len(pT) // 2)), pair_reports=dict(A=rA, V=rV, T=rT), role_counts=meta.role_counts(d), attackers={})
    cl_te = pT.anchor_pid.map(lambda p: hid(p, salt)).to_numpy(); cl_va = pV.anchor_pid.map(lambda p: hid(p, salt)).to_numpy()

    def record(name, sv, st, extra=None):
        yv, yt = parts["va"]["y"], parts["te"]["y"]
        r = dict(val_auc=S.auc(yv, sv), test=auc_ci(yt, st, cl_te, a.boot, 0), **(extra or {}))
        summ["attackers"][name] = r
        (out / f"preds_{name}.json").write_text(json.dumps(dict(pair_id=list(range(len(yt))), cluster=cl_te.tolist(), y=yt.tolist(), score_test=np.round(st, 5).tolist(),
                                                              score_val=np.round(sv, 5).tolist(), cluster_val=cl_va.tolist(), y_val=yv.tolist())))
        print(f"  {name}: val {r['val_auc']:.3f}  test {r['test']['point']:.3f} [{r['test']['lo']:.3f},{r['test']['hi']:.3f}]", flush=True)

    todo = a.attackers.split(",")
    if "f1" in todo:
        for cond in a.conds.split(","):
            sc = []
            for sd in [int(s) for s in a.seeds.split(",")]:
                o = A.train_siamese(**{k: v for k, v in zip(("tr", "va", "te"), (lambda cs: (cs["tr"], cs["va"], cs["te"]))(cond_sets(cond, Xd, Xr, parts)))}, arch=a.arch, pretrained=not a.no_pretrain,
                                    seed=sd, max_epochs=a.max_epochs, bs=a.bs, device=dev, amp=dev == "cuda", log=(print if a.r5 else (lambda *x: None)),
                                    **(dict(head="xl", select="auc", aug=True, cosine=True, patience=10, time_budget_s=2100) if a.r5 else {}))
                sc.append((o["val_score"], o["test_score"], o["epochs"]))
                record(f"f1_{cond}_s{sd}", o["val_score"], o["test_score"], dict(epochs=o["epochs"], truncated=bool(o.get("truncated")), best_val_auc=float(o.get("best_val_auc", -1)), best_val_loss=float(o["best_val_loss"]), curve=[[round(float(v), 5) for v in c] for c in o["curve"]]))
            record(f"f1_{cond}_ens", np.mean([z[0] for z in sc], 0), np.mean([z[1] for z in sc], 0))
    if "f4" in todo:
        for cond in ("match", "anon2"):
            xb = Xr if cond == "match" else Xd
            sv = A.structural_scores(Xd, xb, parts["va"]["ia"], parts["va"]["ib"], size=min(128, Xd.shape[-1]), device=dev if dev != "cuda" else "cuda")
            st = A.structural_scores(Xd, xb, parts["te"]["ia"], parts["te"]["ib"], size=min(128, Xd.shape[-1]), device=dev if dev != "cuda" else "cuda")
            record(f"f4_{cond}", sv, st)
    if "f3" in todo or "f5" in todo:
        if a.feat == "random":
            g = torch.Generator().manual_seed(0); W = torch.randn(Xd.shape[-1] ** 2, 128, generator=g)
            feat = lambda X: (X.float().flatten(1).cpu() @ W).numpy()
        else:
            from .features import dino_features as feat
        Fd = feat(Xd)
        cache = Path(a.work) / f"rawfeat_{a.feat}_rot{a.rot}_{'screen' if a.screen else 'full'}_{len(need)}_{int(need.sum()) % 10**9}.npy"
        if cache.exists():
            Fr = np.load(cache)
        else:
            Fr = feat(Xr)
            cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, Fr)
        tr_, va_, te_ = ({**parts[k]} for k in ("tr", "va", "te"))
        FF = np.concatenate([Fd, Fr])
        for k, p in (("tr", tr_), ("va", va_), ("te", te_)):
            p["ib"] = p["ib"] + len(Fd)
        if "f3" in todo:
            roleA = np.isin(d["role"].to_numpy()[need], ["A", "S"] if a.ladder_s else ["A"])
            maskA = np.r_[roleA, roleA]
            o = H.fit_f3(FF, tr_, FF, va_, FF, te_, stats_from=FF[maskA])
            for nm, (sv, st) in o.items():
                record(nm, sv, st)
        if "f5" in todo:
            dtr = d.set_index("row").loc[need]
            # element A is estimated from the defended image; element B from the real image (F5a) -- both columns of FF are indexed by position
            dem = dict(pid=np.r_[dtr.pid.to_numpy(), dtr.pid.to_numpy()], age=np.r_[dtr.age.to_numpy(), dtr.age.to_numpy()], male=np.r_[dtr.male.to_numpy(), dtr.male.to_numpy()], pa=np.r_[dtr.pa.to_numpy(), dtr.pa.to_numpy()])
            roleA = np.isin(d["role"].to_numpy()[need], ["A", "S"] if a.ladder_s else ["A"])
            sv, st = H.fit_f5(FF, dem, np.r_[roleA, roleA], tr_, va_, te_)
            record("f5a", sv, st)
            sv, st = H.fit_f5(FF, dem, np.r_[roleA, roleA], tr_, va_, te_, true_rows=np.r_[np.zeros(len(Fd), bool), np.ones(len(Fr), bool)])
            record("f5b", sv, st)
    if "f2" in todo:                                                     # ConvNeXt-T Siamese, condition match, 1 seed (T.md 3.5)
        cs = cond_sets("match", Xd, Xr, parts)
        o = A.train_siamese(cs["tr"], cs["va"], cs["te"], arch="convnext_tiny", pretrained=not a.no_pretrain, seed=0, max_epochs=a.max_epochs, bs=a.bs, device=dev, amp=dev == "cuda", log=lambda *x: None)
        record("f2_match_s0", o["val_score"], o["test_score"], dict(epochs=o["epochs"])); record("f2_match_ens", o["val_score"], o["test_score"])
    if "f6" in todo:                                                     # HELD-OUT view: hand statistics + LightGBM (never used for any selection)
        from . import handstats as HS
        Dd, Dr = HS.descriptors(Xd.float().cpu()), HS.descriptors(Xr.float().cpu())
        sv, st = HS.fit_f6(Dd, Dr, parts["tr"], parts["va"], parts["te"])
        record("f6", sv, st)
    summ["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("done", out, summ["seconds"], "s")


if __name__ == "__main__":
    main()
