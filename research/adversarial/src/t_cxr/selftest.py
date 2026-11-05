"""Synthetic-data self-tests for the R-CXR evaluator and attackers (CPU, < 2 min).  Usage:
   python -m t_cxr.selftest --data DIR   (DIR holds Data_Entry_2017_v2020.csv, test_list.txt, train_val_list.txt)
Nothing here reads an image or produces a model number about ChestX-ray14; the only real-data part is the metadata pair-design check."""
import argparse
import sys
import time

import numpy as np
import torch

from . import attackers as A, generator as G, heads as H, meta, pairs, releases as R, stats as S

OK = []


def check(name, cond, info=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)


def binormal_scores(y, auc_target, rng):
    from scipy.stats import norm
    d = np.sqrt(2) * norm.ppf(auc_target)
    return rng.normal(0, 1, len(y)) + d * y


def t_stats():
    rng = np.random.default_rng(0)
    # 1. AUC and bootstrap recover a planted AUC; coverage of the 95% interval over simulated datasets (cluster structure: 2 units per patient)
    P, cover, ses = 600, 0, []
    from scipy.stats import norm
    dd = np.sqrt(2) * norm.ppf(0.70)
    true_auc = float(norm.cdf(dd / np.sqrt(2 * 1.25)))      # population AUC once the shared patient effect (var .25) is added to both classes
    for sim in range(60):
        pat = np.repeat(np.arange(P), 2)
        y = np.tile([1, 0], P)
        s = binormal_scores(y, 0.70, rng) + np.repeat(rng.normal(0, 0.5, P), 2)       # patient-level shared effect
        pb = S.PatientBootstrap(pat, B=200, seed=sim)
        b = S.auc_boot(y, s, pb.codes(pat), pb)
        r = S.summarize(S.auc(y, s), b)
        ses.append(r["se"])
        cover += r["lo"] <= true_auc <= r["hi"]
    check("bootstrap interval covers the planted AUC (>= 85% of 60 sims, 200 resamples)", cover / 60 >= 0.85, f"coverage {cover / 60:.2f}, mean SE {np.mean(ses):.4f}")
    check("bootstrap SE close to Hanley-McNeil for iid units (within 35%)", abs(np.mean(ses) / S.hm_se(true_auc, P, P) - 1) < 0.35, f"boot {np.mean(ses):.4f} vs HM {S.hm_se(true_auc, P, P):.4f}")
    # 2. paired difference: power at the MDE is about 80%
    P = 1400
    pat = np.repeat(np.arange(P), 2); y = np.tile([1, 0], P)
    pb = S.PatientBootstrap(pat, B=300, seed=1)
    codes = pb.codes(pat)
    # two attackers with error correlation 0.6
    def pair(delta_auc, rho=0.6):
        shared = rng.normal(0, 1, len(y)); e1 = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, len(y)); e2 = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, len(y))
        from scipy.stats import norm
        d1, d2 = np.sqrt(2) * norm.ppf(0.70 + delta_auc), np.sqrt(2) * norm.ppf(0.70)
        return e1 + d1 * y, e2 + d2 * y
    s1, s2 = pair(0.0)
    ba, bb = S.auc_boot(y, s1, codes, pb), S.auc_boot(y, s2, codes, pb)
    r = S.paired(ba, bb, S.auc(y, s1), S.auc(y, s2))
    check("paired SE at n=1400 patients, rho=.6 within 40% of the analytic value", abs(r["se"] / (S.hm_mde_paired(0.70, P, P, 0.6) / S.Z_MDE) - 1) < 0.4, f"SE {r['se']:.4f}, analytic {S.hm_mde_paired(0.70, P, P, 0.6) / S.Z_MDE:.4f}, MDE80 {r['mde80']:.4f}")
    # 3. envelope toy (FOCUS A2): floors on a frontier; a defence ABOVE the frontier beats it, one ON it does not
    B = 400
    rngb = np.random.default_rng(5)
    shared = rngb.standard_normal(B) * 0.004                  # resample-level noise shared by all releases = paired
    mkb = lambda L, U: (L + shared + 0.002 * rngb.standard_normal(B), U + shared + 0.002 * rngb.standard_normal(B))
    fl = [(0.50, 0.50), (0.60, 0.70), (0.70, 0.75), (0.80, 0.78), (0.95, 0.805)]
    FL = np.stack([mkb(l, u)[0] for l, u in fl]); FU = np.stack([mkb(l, u)[1] for l, u in fl])
    pl, pu = np.array([l for l, _ in fl]), np.array([u for _, u in fl])
    for nm, (L, U) in dict(good=(0.65, 0.78), on_frontier=(0.65, 0.725), leaks_less=(0.55, 0.70)).items():
        LD, UD = mkb(L, U)
        bt = S.envelope_boot(FL, FU, LD, UD)
        v = S.envelope_verdict(pl, pu, L, U, bt, margin_u=0.02, margin_l=0.03)
        globals().setdefault("_V", {})[nm] = v
    check("envelope: a defence above the frontier beats it on the utility axis", _V["good"]["utility_axis_beats"] and _V["good"]["beats_envelope"], f"dU {_V['good']['point_dU']:.3f} [{_V['good']['dU_lo']:.3f}, {_V['good']['dU_hi']:.3f}]")
    check("envelope: a defence ON the frontier (interpolated) does not beat it", not _V["on_frontier"]["beats_envelope"], f"dU {_V['on_frontier']['point_dU']:.3f}, dL {_V['on_frontier']['point_dL']:.3f}")
    check("envelope: a defence that leaks less at equal utility beats it on the leakage axis", _V["leaks_less"]["leak_axis_beats"], f"dL {_V['leaks_less']['point_dL']:.3f} lo {_V['leaks_less']['dL_lo']:.3f}")
    check("KT2 fires when no defence beats the envelope, not otherwise", S.kill_floor_envelope([_V["on_frontier"]]) and not S.kill_floor_envelope([_V["on_frontier"], _V["good"]]))
    # 4. kill predicates
    check("KT1 fires for a weak raw attacker and for a failed control, passes otherwise",
          S.kill_measurability(0.62, 0.52, [(0.5, 0.011), (0.51, 0.011)], 1400, 1400)["fires"] and S.kill_measurability(0.80, 0.52, [(0.5, 0.011), (0.60, 0.011)], 1400, 1400)["fires"]
          and not S.kill_measurability(0.80, 0.52, [(0.5, 0.011), (0.51, 0.011)], 1400, 1400)["fires"])
    check("KT3 fires when demographics explain >= 90% of the excess", S.kill_demographics(0.885, 0.90)["fires"] and not S.kill_demographics(0.70, 0.90)["fires"])
    # 5. power table sanity: MDE shrinks with n
    t = {r["units"]: r for r in S.power_table(aucs=(0.7,))}
    check("MDE80 shrinks with the number of independent patients", t["all multi-visit patients (cross-fit)"]["mde80_rho0.6"] < t["official test, pairs from 1,424 patients"]["mde80_rho0.6"] < t["PriCheXy 5,000-pair list: 492 patients behind the positives"]["mde80_rho0.6"])


def t_generator_releases():
    g = G.border_artifact()
    check("zero-flow pipeline is NOT the identity at the border (zero-padded smoothing)", g["at_px0"] > 10 and g["interior_max"] < 1e-3, str({k: round(v, 3) for k, v in g.items()}))
    net = G.UNet(1, 2, 8).eval()
    x = torch.rand(2, 1, 256, 256)
    with torch.no_grad():
        f = net(x)
    check("UNet output is bounded in [-1,1] with 2 channels", f.shape == (2, 2, 256, 256) and f.abs().max() <= 1)
    d = G.displacement_px(torch.ones(1, 2, 256, 256), 0.01)[0, 0, 128, 64:192]
    check("a constant raw flow of 1 at mu=.01 displaces by 1.275 px in the interior", abs(float(d.abs().mean()) - 1.275) < 0.01, f"{float(d.abs().mean()):.3f}")
    # spectrum recovery
    gen = torch.Generator(); gen.manual_seed(0)
    n = 256
    true_p = torch.exp(-torch.arange(n // 2 + 1).float() / 6.0)
    fields = torch.stack([G.spectrum_matched_field(true_p, torch.tensor(0.5), gen) for _ in range(40)])
    check("spectrum-matched field has the requested RMS", abs(float(fields.pow(2).mean((1, 2, 3)).sqrt().mean()) - 0.5) < 0.02)
    P = (torch.fft.fft2(fields).abs() ** 2).sum(1).mean(0)
    fy = torch.fft.fftfreq(n)[:, None].expand(n, n); fx = torch.fft.fftfreq(n)[None, :].expand(n, n)
    rad = ((fx ** 2 + fy ** 2).sqrt() * n).round().long().clamp(max=n // 2)
    got = torch.bincount(rad.flatten(), weights=P.flatten(), minlength=n // 2 + 1) / torch.bincount(rad.flatten(), minlength=n // 2 + 1).float()
    r = (got[1:30] / got[1:30].sum()); t = (true_p[1:30] / true_p[1:30].sum())
    check("measured radial spectrum of sampled fields matches the generating spectrum (corr > .95)", float(np.corrcoef(r.numpy(), t.numpy())[0, 1]) > 0.95)
    # releases
    rel = R.Releases("cpu", flow_stats={"mu0.01": dict(rms_px=torch.tensor([0.3, 0.5, 0.7]), radial_power=true_p ** 2)}, pca=None)
    rng = np.random.default_rng(0)
    base = (np.clip(np.cumsum(np.cumsum(rng.normal(0, 1, (4, 256, 256)), 1), 2) * 2 + 128, 0, 255)).astype(np.uint8)    # smooth-ish textures
    ids = [f"img{i}" for i in range(4)]
    xt = torch.from_numpy(base)
    pca = R.fit_pca(base, np.arange(4), size=32, kmax=4)
    rel.pca = pca
    res = {}
    for nm in ["raw", "const", "warp0", "rw1_mu0.01", "rw2_mu0.01", "rw3_mu0.01", "blur_s2", "down_f4", "noise_s0.1", "quant_l8", "quantd_l8", "jpeg_q10", "pca_k4", "dppix_b8"]:
        res[nm] = rel.apply(nm, xt, ids).float()
        again = rel.apply(nm, xt, ids).float()
        check(f"release {nm}: deterministic, shape/dtype ok", torch.equal(res[nm], again) and res[nm].shape == (4, 256, 256), "")
    x01 = xt.float() / 255
    check("raw is identity, const is zero", torch.allclose(res["raw"], x01, atol=1e-3) and res["const"].abs().max() == 0)
    check("warp0 equals raw in the interior", (res["warp0"] - x01)[:, 16:-16, 16:-16].abs().max() < 1e-3 and (res["warp0"] - x01)[:, :3].abs().max() > 1e-3)
    d1 = (res["rw2_mu0.01"] - res["warp0"])[:, 16:-16, 16:-16]
    check("rw2 at mu=.01 differs from the zero-flow pipeline by a bounded smooth warp (nonzero, small)", 0 < float(d1.abs().mean()) < 0.1)
    hf = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean())
    check("blur and downsample reduce high-frequency energy monotonically", hf(res["blur_s2"]) < hf(res["raw"]) and hf(rel.apply("blur_s4", xt, ids).float()) < hf(res["blur_s2"]) and hf(res["down_f4"]) < hf(res["raw"]))
    check("quant_l8 has <= 8 grey levels", len(torch.unique(res["quant_l8"])) <= 8)
    check("noise: different image ids give different noise", float((rel.apply("noise_s0.1", xt, ids).float()[0] - x01[0]).std()) > 0.05 and not torch.equal(rel.apply("noise_s0.1", xt, ids[::-1]).float(), res["noise_s0.1"]))
    errs = [float((rel.apply(f"pca_k{k}", xt, ids).float() - x01).pow(2).mean()) for k in (1, 2, 4)]
    check("PCA reconstruction error decreases with k", errs[0] > errs[1] > errs[2], str([round(e, 5) for e in errs]))


def _pr(x):
    return A._prep(x)


def t_attackers_synthetic():
    rng = np.random.default_rng(0)
    n, N = 64, 120
    base = rng.normal(0, 1, (N, n, n)).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    base = np.stack([gaussian_filter(b, 3) for b in base]); base = (base - base.mean((1, 2), keepdims=True)) / base.std((1, 2), keepdims=True)
    # patient k has images 2k and 2k+1: the same texture shifted by up to 3 px, plus noise
    def make(noise):
        imgs = np.zeros((N, n, n), np.float32)
        for k in range(N // 2):
            imgs[2 * k] = base[k] + rng.normal(0, noise, (n, n))
            sh = rng.integers(-3, 4, 2)
            imgs[2 * k + 1] = np.roll(base[k], tuple(sh), (0, 1)) + rng.normal(0, noise, (n, n))
        return torch.from_numpy(((imgs - imgs.min()) / (imgs.max() - imgs.min())).astype(np.float32))
    ia = np.arange(0, N, 2); ib = ia + 1
    ineg = (ib + 2) % N
    ia2, ib2 = np.r_[ia, ia], np.r_[ib, ineg]
    y = np.r_[np.ones(len(ia)), np.zeros(len(ia))]
    aucs = []
    for noise in (0.1, 1.0, 4.0):
        x = make(noise)
        sc = A.structural_scores(x, x, ia2, ib2, size=64, highpass=True)
        aucs.append(S.auc(y, sc))
    check("structural NCC attacker: AUC ~1 at low noise, falls with noise, ~chance for unrelated pairs",
          aucs[0] > 0.95 and aucs[0] >= aucs[1] >= aucs[2] - 0.02, str([round(a, 3) for a in aucs]))
    x0 = make(0.1)
    sc_rand = A.structural_scores(x0, x0[torch.randperm(N, generator=torch.Generator().manual_seed(1))], ia2, ib2, size=64)
    check("structural attacker on shuffled partners is near chance", abs(S.auc(y, sc_rand) - 0.5) < 0.12, f"{S.auc(y, sc_rand):.3f}")
    # Siamese smoke test: runs end-to-end, no pretrained download
    xs = make(0.5).half()
    tr = dict(xa=xs, xb=xs, ia=ia2[:60], ib=ib2[:60], y=y[:60]); va = dict(xa=xs, xb=xs, ia=ia2[60:90], ib=ib2[60:90], y=y[60:90]); te = va
    t0 = time.time()
    out = A.train_siamese(tr, va, te, arch="resnet18", pretrained=False, max_epochs=2, bs=16, device="cpu", amp=False, log=lambda *a: None)
    check("Siamese (resnet18, 64px, CPU) trains 2 epochs and returns finite scores", np.isfinite(out["test_score"]).all() and len(out["test_score"]) == 30, f"{time.time() - t0:.0f}s")
    # T5 / R5: the F1+ attacker (xl head, augmentation, cosine lr, validation-AUC selection, time budget)
    xa_ = xs[:8].float()
    xaug = A.augment(xa_)
    check("R5 augmentation: same shape, values in [0,1], differs from the input, identity-free of NaN", xaug.shape == xa_.shape and bool(((xaug >= 0) & (xaug <= 1)).all()) and not torch.allclose(xaug, xa_) and bool(torch.isfinite(xaug).all()))
    net = A.Siamese("resnet18", False, head="xl")
    lg = net(_pr(xs[:4]), _pr(xs[4:8]))
    check("R5 head 'xl': 512-d embedding, pair feature 1536 -> 256 -> 1, one logit per pair; default head unchanged (128 -> 1)", lg.shape == (4,) and net.pair[0].in_features == 1536 and A.Siamese("resnet18", False).fc_end.in_features == 128)
    o5 = A.train_siamese(tr, va, te, arch="resnet18", pretrained=False, max_epochs=3, bs=16, device="cpu", amp=False, log=lambda *a: None, head="xl", select="auc", aug=True, cosine=True, patience=10, time_budget_s=1e-3)
    check("R5 trainer: AUC selection records val AUC per epoch, the time budget truncates and is flagged, scores finite", o5["truncated"] and o5["epochs"] == 1 and len(o5["curve"][0]) == 4 and np.isfinite(o5["test_score"]).all(), str(o5["curve"][:1]))
    o5b = A.train_siamese(tr, va, te, arch="resnet18", pretrained=False, max_epochs=3, bs=16, device="cpu", amp=False, log=lambda *a: None, head="xl", select="auc", aug=True, cosine=True, patience=10)
    check("R5 trainer without a budget runs the full cap (3 epochs), not truncated, best_val_auc is the max of the curve", (not o5b["truncated"]) and o5b["epochs"] == 3 and abs(o5b["best_val_auc"] - max(c[3] for c in o5b["curve"])) < 1e-9)


def t_heads_planted():
    rng = np.random.default_rng(0)
    P, d = 400, 64
    ident = rng.normal(0, 1, (P, d))
    res = []
    for alpha in (0.0, 0.5, 1.5):
        Fe = np.concatenate([ident * alpha + rng.normal(0, 1, (P, d)), ident * alpha + rng.normal(0, 1, (P, d))])    # rows k and P+k same patient
        ia = np.r_[np.arange(P), np.arange(P)]; ib = np.r_[P + np.arange(P), P + (np.arange(P) + 7) % P]; y = np.r_[np.ones(P), np.zeros(P)]
        half = P // 2 * 1
        tr = dict(ia=ia[:half].tolist() + ia[P:P + half].tolist(), ib=ib[:half].tolist() + ib[P:P + half].tolist(), y=np.r_[y[:half], y[P:P + half]])
        tr = {k: np.asarray(v) for k, v in tr.items()}
        te_idx = np.r_[np.arange(half, P), np.arange(P + half, 2 * P)]
        te = dict(ia=ia[te_idx], ib=ib[te_idx], y=y[te_idx])
        o = H.fit_f3(Fe, tr, Fe, te, Fe, te)
        res.append((alpha, S.auc(te["y"], o["f3a_logreg"][1]), S.auc(te["y"], o["f3c_canberra"][1]), S.auc(te["y"], o["f3c_cosine"][1])))
    check("planted leak: F3 AUC ~0.5 at alpha=0 and increases monotonically with alpha (logreg, cosine)", abs(res[0][1] - 0.5) < 0.1 and res[0][1] < res[1][1] < res[2][1] and res[0][3] < res[1][3] < res[2][3],
          str([(a, round(b, 3), round(c, 3), round(e, 3)) for a, b, c, e in res]))


def t_roles(data, pri):
    P = data
    d = meta.load(f"{P}/Data_Entry_2017_v2020.csv", f"{P}/test_list.txt", f"{P}/train_val_list.txt")
    seen = meta.seen_patients([f"{pri}/image_pairs_training_10000.txt", f"{pri}/image_pairs_validation_2000.txt"], d)
    dr = meta.assign_roles(d, seen, "PROVISIONAL-selftest", 0)
    rc = meta.role_counts(dr)
    sets = {r: set(dr.loc[dr.role == r, "pid"]) for r in "SAVT"}
    check("roles are patient-disjoint and cover all 30,805 patients", all(not (sets[a] & sets[b]) for a in "SAVT" for b in "SAVT" if a < b) and sum(len(v) for v in sets.values()) == 30805, str({k: v["ge2"] for k, v in rc.items()}))
    check("generator-seen patients: 10,795 (8,714 multi-visit), none in T/V/A", rc["S"]["patients"] == 10795 and rc["S"]["ge2"] == 8714)
    pte, rte = pairs.build(dr, "T", "N2", 1, "te0", col="role"); ptr, rtr = pairs.build(dr, "A", "N2", 2, "tr0", col="role"); pv, rv = pairs.build(dr, "V", "N2", 1, "v0", col="role")
    check("N2 on T: <= 1% positives dropped; gate: metadata-only GBM (all fields) <= 0.56 on T and on V", rte["dropped"] / rte["positives_requested"] < 0.01 and pairs.metadata_floor(ptr, pte, dr, "all") <= 0.56 and pairs.metadata_floor(ptr, pv, dr, "all") <= 0.56,
          f"T {pairs.metadata_floor(ptr, pte, dr, 'all'):.3f} V {pairs.metadata_floor(ptr, pv, dr, 'all'):.3f} device-only T {pairs.metadata_floor(ptr, pte, dr, 'device'):.3f}")


def _fake_tree(root, rels, P=500, seed=0, with_f5=True):
    """results tree with planted (L, U) per release: binormal pair scores for every attacker, 3 images per patient with 14 planted-label scores."""
    import json
    from pathlib import Path
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    pat = np.array([f"p{i:04d}" for i in range(P)])
    y = np.tile([1, 0], P); cl = np.repeat(pat, 2)
    yv = np.tile([1, 0], P // 3); clv = np.repeat(pat[: P // 3], 2)
    Yimg = (rng.random((P * 3, 14)) < 0.25).astype(int); pimg = np.repeat(pat, 3)
    Yv = (rng.random((300, 14)) < 0.25).astype(int)
    root = Path(root)
    for name, (L, Uu, Ff5) in rels.items():
        d = root / f"{name}_rot0"; d.mkdir(parents=True, exist_ok=True)
        att = {"f1_match_ens": L, "f1_mismatch_ens": L - 0.03, "f1_anon2_ens": L - 0.02, "f3a_logreg": L - 0.04, "f4_match": max(0.5, L - 0.1)}
        if with_f5:
            att["f5a"] = Ff5
        for nm, a in att.items():
            sh = np.sqrt(2) * norm.ppf(a)
            (d / f"preds_{nm}.json").write_text(json.dumps(dict(pair_id=list(range(len(y))), cluster=cl.tolist(), y=y.tolist(), score_test=(rng.normal(0, 1, len(y)) + sh * y).tolist(),
                                                                score_val=(rng.normal(0, 1, len(yv)) + sh * yv).tolist(), cluster_val=clv.tolist(), y_val=yv.tolist())))
        (d / "summary.json").write_text("{}")
        sh = np.sqrt(2) * norm.ppf(Uu)
        mk = lambda Y: (rng.normal(0, 1, Y.shape) + sh * Y).round(4).tolist()
        PT = mk(Yimg)
        (root / f"utility_{name}_rot0.json").write_text(json.dumps(dict(T=dict(patient=pimg.tolist(), Y=Yimg.tolist(), P=PT, P_fixed=PT), V=dict(Y=Yv.tolist(), P=mk(Yv)))))
    # const control + shuffled-label control
    return root


def t_aggregate():
    import tempfile
    from . import aggregate as AG
    rels = {"raw": (0.90, 0.80, 0.52), "const": (0.50, 0.50, 0.50), "rim_w24": (0.55, 0.52, 0.50), "blur_s2": (0.75, 0.74, 0.50), "down_f4": (0.65, 0.70, 0.50), "rw2_mu0.01": (0.85, 0.79, 0.50),
            "pcxr_mu0.01": (0.60, 0.78, 0.50), "pcxr_mu0.005": (0.70, 0.70, 0.50)}
    with tempfile.TemporaryDirectory() as td:
        root = _fake_tree(td, rels)
        import json
        from pathlib import Path
        g0 = Path(td) / "g0.json"; g0.write_text(json.dumps(dict(rot0=dict(N2=dict(all=dict(auc=0.53))), passes_all_rotations=True)))
        # shuffled-label control for raw at chance
        sh = Path(td) / "raw_rot0_shuf"; sh.mkdir()
        rngc = np.random.default_rng(1); P = 500; y = np.tile([1, 0], P)
        for nm in ("f1_match_ens", "f3a_logreg", "f5a"):
            (sh / f"preds_{nm}.json").write_text(json.dumps(dict(pair_id=[], cluster=np.repeat([f"p{i:04d}" for i in range(P)], 2).tolist(), y=y.tolist(), score_test=rngc.normal(0, 1, len(y)).tolist(),
                                                                 score_val=rngc.normal(0, 1, 2 * (P // 3)).tolist(), cluster_val=np.repeat([f"p{i:04d}" for i in range(P // 3)], 2).tolist(), y_val=np.tile([1, 0], P // 3).tolist())))
        res = AG.run(td, str(g0), [0], B=300, prereg_commit="test", out=td + "/out")
        k2 = res["KT"]["KT2"]["verdicts"]
        check("aggregate: KT1 passes on a planted measurable raw release with controls at chance", not res["KT"]["KT1"]["fires"], f"raw L {res['releases']['raw']['L']:.3f}, controls dev max {max(abs(c['auc'] - 0.5) for c in res['KT']['KT1']['controls']):.3f}")
        check("aggregate: a defence planted ABOVE the frontier beats the envelope; one BELOW the frontier does not", k2["pcxr_mu0.01"]["primary"]["beats_envelope"] and not k2["pcxr_mu0.005"]["primary"]["beats_envelope"],
              f"dU {k2['pcxr_mu0.01']['primary']['point_dU']:.3f} vs {k2['pcxr_mu0.005']['primary']['point_dU']:.3f}")
        check("aggregate: verdict_original is GO and result/ floors_table files exist", res["verdict_original"].startswith("GO") and (Path(td) / "out" / "result.json").exists() and (Path(td) / "out" / "floors_table.md").exists(), res["verdict_original"])
        check("aggregate: result.json carries evaluator_sha, prereg_commit, verdict_original and verdict_tightened", all(k in res for k in ("evaluator_sha", "prereg_commit", "verdict_original", "verdict_tightened")))
    with tempfile.TemporaryDirectory() as td:
        rels2 = {k: v for k, v in rels.items() if k != "pcxr_mu0.01"}
        _fake_tree(td, rels2)
        res = AG.run(td, None, [0], B=300, out=td + "/out")
        check("aggregate: with only a defence ON/below the frontier the verdict is NO-GO (KT2)", res["verdict_original"].startswith("NO-GO") or res["KT"]["KT2"]["fires_primary"], res["verdict_original"])
    with tempfile.TemporaryDirectory() as td:
        rels3 = dict(rels); rels3["raw"] = (0.90, 0.80, 0.89)
        _fake_tree(td, rels3)
        res = AG.run(td, None, [0], B=300, out=td + "/out")
        check("aggregate: KT3 fires when the demographic-feature attacker explains >= 90% of the raw excess", res["KT"]["KT3"]["fires"], f"share {res['KT']['KT3']['share']:.2f}")


def t_f5_matched():
    rng = np.random.default_rng(0)
    P, d = 900, 32
    age = rng.integers(20, 80, P).astype(float)
    male = rng.integers(0, 2, P)
    w = rng.normal(0, 1, (2, d))
    feat = lambda a, m: np.outer(a - 50, w[0]) / 20 + np.outer(m, w[1]) + rng.normal(0, 1, (len(a), d))
    ageB = age + rng.integers(0, 2, P)
    Fa, Fb = feat(age, male), feat(ageB, male)                                   # two images per patient (age differs by 0-1 year)
    F = np.concatenate([Fa, Fb])                                                  # rows 0..P-1 = element A table, P..2P-1 = element B table
    dem = dict(pid=np.r_[np.arange(P), np.arange(P)], age=np.r_[age, ageB], male=np.r_[male, male], pa=np.zeros(2 * P))
    # matched negatives: partner of the same sex and age as the true partner but another patient
    key = {(a, m): np.where((ageB == a) & (male == m))[0] for a, m in set(zip(ageB, male))}
    neg = np.array([rng.choice(key[(ageB[i], male[i])][key[(ageB[i], male[i])] != i]) if len(key[(ageB[i], male[i])]) > 1 else i for i in range(P)])
    rnd = rng.permutation(P)
    mask = np.r_[np.ones(P // 2, bool), np.zeros(P - P // 2, bool)]
    mask = np.r_[mask, mask]
    def pairset(sel, negs):
        ia = np.r_[sel, sel]; ib = P + np.r_[sel, negs[sel]]
        return dict(ia=ia, ib=ib, y=np.r_[np.ones(len(sel)), np.zeros(len(sel))])
    tr, va, te = np.arange(0, P // 2), np.arange(P // 2, P // 2 + 150), np.arange(P // 2 + 150, P)
    out = {}
    for nm, negs in (("matched", neg), ("random", rnd)):
        sv, st = H.fit_f5(F, dem, mask, pairset(tr, negs), pairset(va, negs), pairset(te, negs))
        out[nm] = S.auc(pairset(te, negs)["y"], st)
    check("F5 demographic attacker: ~chance on demographically matched negatives, clearly above chance on random negatives",
          abs(out["matched"] - 0.5) < 0.09 and out["random"] > 0.7, str({k: round(v, 3) for k, v in out.items()}))


def t_handstats_utility():
    import torch.nn as nn
    from . import handstats as HS, utility as U
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    n, N = 64, 200
    base = np.stack([gaussian_filter(rng.normal(0, 1, (n, n)), 4) for _ in range(N // 2)])
    imgs = np.zeros((N, n, n), np.float32)
    for k in range(N // 2):
        imgs[2 * k] = base[k] + 0.3 * rng.normal(0, 1, (n, n)); imgs[2 * k + 1] = np.roll(base[k], (1, 2), (0, 1)) + 0.3 * rng.normal(0, 1, (n, n))
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    X = torch.from_numpy(imgs)
    D = HS.descriptors(X, size=32)
    ia = np.arange(0, N, 2); ib = ia + 1
    neg = (ib + 6) % N
    mk = lambda sl: dict(ia=np.r_[ia[sl], ia[sl]], ib=np.r_[ib[sl], neg[sl]], y=np.r_[np.ones(len(ia[sl])), np.zeros(len(ia[sl]))])
    sv, st = HS.fit_f6(D, D, mk(slice(0, 60)), mk(slice(60, 80)), mk(slice(80, 100)))
    t = mk(slice(80, 100))
    check("F6 hand-statistics attacker links shifted copies of the same texture (AUC > 0.85)", S.auc(t["y"], st) > 0.85, f"{S.auc(t['y'], st):.3f}")
    # utility smoke: tiny classifier, mean_auc on planted labels
    Y = (rng.random((N, 14)) < 0.3).astype(np.uint8)
    X2 = torch.from_numpy((imgs + 0.5 * Y[:, :1, None] ).astype(np.float32)).clamp(0, 1)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 14), nn.Sigmoid())
    wrapped = nn.Sequential(*list(model.children()))
    m = U.finetune(wrapped, X2, Y, epochs=2, bs=32, device="cpu", size=32, log=lambda *a: None)
    P = U.predict(m, X2, device="cpu", size=32)
    a, per = U.mean_auc(Y, P)
    check("utility pipeline runs (fine-tune, predict, macro AUC finite)", np.isfinite(a) and per.shape == (14,), f"mean AUC {a:.3f}")


def t_ladder_r5():
    """fabricated prediction files: the aggregator reproduces L4, applies the pre-registered rule, and the ordering test on planted scores."""
    import json, tempfile
    from . import ladder_r5 as L5
    rng = np.random.default_rng(3)
    n = 1200
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    cl = [f"c{i % 300}" for i in range(n)]
    root = tempfile.mkdtemp()

    def put(rel, rot, suffix, name, sc):
        d = __import__("pathlib").Path(root) / f"{rel}_rot{rot}{suffix}"; d.mkdir(exist_ok=True)
        (d / f"preds_{name}.json").write_text(json.dumps(dict(pair_id=list(range(n)), cluster=cl, y=y.tolist(), score_test=sc.tolist(), score_val=sc.tolist(), cluster_val=cl, y_val=y.tolist())))
        sm = d / "summary.json"
        j = json.loads(sm.read_text()) if sm.exists() else dict(attackers={})
        j["attackers"][name] = dict(val_auc=0.7, epochs=12, truncated=False, best_val_auc=0.8); sm.write_text(json.dumps(j))
    gain = lambda g: y * g + rng.normal(0, 1, n)
    for rot in (0, 1):
        for rel, g1, g4, g5 in (("down_f4", 1.0, 1.3, 1.31), ("pcxr_mu0.01_s1_C2", 1.0, 1.28, 1.29)):
            put(rel, rot, "", "f1_match_ens", gain(g1)); put(rel, rot, "", "f3b_mlp", gain(g1)); put(rel, rot, "", "f4_match", gain(0.2)); put(rel, rot, "", "f5a", gain(0.1))
            put(rel, rot, "_ladderall", "f1_match_ens", gain(g4)); put(rel, rot, "_ladderall", "f3b_mlp", gain(g1))
            put(rel, rot, "_ladderall_r5", "f1_match_ens", gain(g5))
        put("labels14", rot, "", "f3b_mlp", gain(0.6)); put("labels14", rot, "_ladderall", "f3b_mlp", gain(0.65)); put("labels14", rot, "_ladderall_r5", "f3x_mlp", gain(0.66))
    import subprocess
    out = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, "-m", "t_cxr.ladder_r5", "--root", root, "--rots", "0,1", "--out", out, "--B", "300"], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    ok = r.returncode == 0 and (__import__("pathlib").Path(out) / "ladder_r5.json").exists()
    check("ladder_r5 aggregator runs on fabricated predictions", ok, (r.stderr or r.stdout)[-300:])
    if ok:
        j = json.loads((__import__("pathlib").Path(out) / "ladder_r5.json").read_text())
        dn = j["releases"]["down_f4"]
        check("ladder_r5: L5 >= L4 in every resample (D5 lower bound >= 0) and the planted tiny D5 gets SIGNED-WITH-CAVEAT", dn["D5"]["ci"][0] >= -1e-12 and dn["D5"]["diff"] < 0.02 and j["leg2_R5"]["verdict_original"] == "SIGNED-WITH-CAVEAT", str(dn["D5"]))
        check("ladder_r5: the rule function maps (0.01,0.03)->SIGNED, (0.025,0.06)->UNSIGNABLE, (0.015,0.05)->UNRESOLVED", L5.rule(0.01, 0.03) == "SIGNED-WITH-CAVEAT" and L5.rule(0.025, 0.06).startswith("PERMANENTLY") and L5.rule(0.015, 0.05).startswith("UNRESOLVED"))
        check("ladder_r5: ordering test reports floor_ahead_by with the right sign and no spurious reversal", "R5" in j["ordering"] and not j["ordering"]["R5"]["defender_beats_floor"], str(j["ordering"]["R5"]))


def t_dups(data, pri, exclude):
    import json
    d = meta.load(f"{data}/Data_Entry_2017_v2020.csv", f"{data}/test_list.txt", f"{data}/train_val_list.txt", exclude_json=exclude)
    ex = json.loads(open(exclude).read())
    seen = meta.seen_patients([f"{pri}/{f}" for f in meta.PRI], d)
    check("duplicate rule: D2 images flagged not-ok, D1 groups labelled, the bank row of every image unchanged", int((~d.ok).sum()) == len(ex["D2_exclude"]) and int((d.dupgroup >= 0).sum()) == sum(len(g) for g in ex["D1_groups"]) and (d.row.to_numpy() == np.arange(len(d))).all(),
          f"{int((~d.ok).sum())} excluded, {int((d.dupgroup >= 0).sum())} in D1 groups")
    bad_img = set(d.loc[~d.ok, "row"]); d1 = d.set_index("row")["dupgroup"]
    viol = 0; dropped = 0
    for rot in (0, 1, 2):
        dr = meta.assign_roles(d, seen, "PROVISIONAL-selftest", rot)
        for role, per in (("T", 1), ("V", 1), ("A", 2)):
            pr, rep = pairs.build(dr, role, "N2", per, role + "0", col="role")
            dropped += rep["patients_without_valid_positive"]
            viol += int(pr.a.isin(bad_img).sum() + pr.b.isin(bad_img).sum())
            pos = pr[pr.label == 1]
            viol += int(((d1.loc[pos.a.to_numpy()].to_numpy() >= 0) & (d1.loc[pos.a.to_numpy()].to_numpy() == d1.loc[pos.b.to_numpy()].to_numpy())).sum())
    check("duplicate rule: no excluded image and no D1 duplicate positive pair in any pair of any role/rotation", viol == 0, f"violations {viol}, patients with no valid positive due to duplicates {dropped}")


def t_pairs(data):
    P = data
    d = meta.load(f"{P}/Data_Entry_2017_v2020.csv", f"{P}/test_list.txt", f"{P}/train_val_list.txt")
    c = meta.counts(d)
    check("published counts: 112,120 images, 30,805 patients, 13,302 multi-visit, 1,424 in the official test split",
          c["all"]["images"] == 112120 and c["all"]["patients"] == 30805 and c["all"]["ge2"] == 13302 and c["te"]["ge2"] == 1424, str(c))
    pte, rte = pairs.build(d, "te", "N2", 1, "te0"); ptr, rtr = pairs.build(d, "tr", "N2", 2, "tr0")
    check("N2 test pairs: patient-disjoint from train, <= 1% positives dropped", rte["dropped"] / rte["positives_requested"] < 0.01 and not (set(pte.pid_a) | set(pte.pid_b)) & (set(ptr.pid_a) | set(ptr.pid_b)), str(rte))
    check("every positive has the same patient on both sides and every negative differs", (pte[pte.label == 1].pid_a == pte[pte.label == 1].pid_b).all() and (pte[pte.label == 0].pid_a != pte[pte.label == 0].pid_b).all())
    f_all = pairs.metadata_floor(ptr, pte, d, "all")
    f_demo = pairs.metadata_floor(ptr, pte, d, "demo")
    check("N2 metadata-only attacker (age, sex, view, W, H, spacing, GBM) <= 0.56 on test pairs", f_all <= 0.56, f"all {f_all:.3f}, demo {f_demo:.3f}")
    p0, _ = pairs.build(d, "te", "N0", 1, "te0"); q0, _ = pairs.build(d, "tr", "N0", 2, "tr0")
    f0 = pairs.metadata_floor(q0, p0, d, "all")
    check("N0 (random negatives) metadata-only attacker >= 0.95 (reproduces the stress-test finding 0.983-0.988)", f0 >= 0.95, f"{f0:.3f}")
    est = pairs.metadata_floor(q0, p0, d, "demo_raw", noise=dict(age_sd=5.0, sex_flip=0.03))
    check("N0 with image-estimated age/sex (simulated sd 5y, 3% flips) stays high (0.85-0.92) -> a demographics number", 0.80 <= est <= 0.95, f"{est:.3f}")
    est2 = pairs.metadata_floor(ptr, pte, d, "demo_raw", noise=dict(age_sd=5.0, sex_flip=0.03))
    check("same simulated estimator on N2 pairs is at chance (|AUC-.5| < .05)", abs(est2 - 0.5) < 0.05, f"{est2:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--skip", default=""); ap.add_argument("--pri", default=None, help="dir with PriCheXy image_pairs_*.txt"); ap.add_argument("--exclude", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    for nm, fn in [("stats", t_stats), ("generator+releases", t_generator_releases), ("attackers", t_attackers_synthetic), ("heads", t_heads_planted), ("f5", t_f5_matched), ("aggregate", t_aggregate), ("ladder_r5", t_ladder_r5), ("handstats+utility", t_handstats_utility), ("pairs", lambda: t_pairs(a.data))] + ([("roles", lambda: t_roles(a.data, a.pri))] if a.pri else []) + ([("dups", lambda: t_dups(a.data, a.pri, a.exclude))] if a.exclude and a.pri else []):
        if nm in a.skip:
            continue
        print(f"== {nm}", flush=True)
        t0 = time.time(); fn(); print(f"   ({time.time() - t0:.0f}s)", flush=True)
    print(f"{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)
