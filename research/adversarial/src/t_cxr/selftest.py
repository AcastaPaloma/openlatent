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
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--skip", default=""); ap.add_argument("--pri", default=None, help="dir with PriCheXy image_pairs_*.txt")
    a = ap.parse_args()
    torch.set_num_threads(4)
    for nm, fn in [("stats", t_stats), ("generator+releases", t_generator_releases), ("attackers", t_attackers_synthetic), ("heads", t_heads_planted), ("pairs", lambda: t_pairs(a.data))] + ([("roles", lambda: t_roles(a.data, a.pri))] if a.pri else []):
        if nm in a.skip:
            continue
        print(f"== {nm}", flush=True)
        t0 = time.time(); fn(); print(f"   ({time.time() - t0:.0f}s)", flush=True)
    print(f"{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)
