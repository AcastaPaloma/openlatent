"""One queue job = one method config x seeds x 4 actor-disjoint folds.

  python run_exp.py --name c_rate_d16 --method learned --cfg '{"dz":16,"bott":"vib"}' --seeds 0
  methods: raw | mean | smooth | pca | leace_tr | leace_oracle | learned | ctrl_constant | ctrl_length | layers
  --shuffle id|emo  trains attackers/probes on permuted labels (must give chance)
Writes $OUT/result_s<seed>.json  (per-actor counts for bootstrap over actors) and prints a compact summary.
"""
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import data, baselines, evaluate as ev
from lib import Ragged, Rep, constant_rep, length_rep, DEV
from encoders import train_encoder, encode_all, eval_head

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--backbone", default="wavlm-base-plus")
    ap.add_argument("--shuffle", default="none")
    ap.add_argument("--no_secrets", action="store_true")
    ap.add_argument("--feat", default=None, help="hand-crafted feature set name under FEAT_ROOT/handcrafted (method feat)")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    out = Path(os.environ.get("OUT", f"/tmp/{a.name}")); out.mkdir(parents=True, exist_ok=True)
    feats = data.FeatStore(a.backbone)
    ctx = ev.Ctx(feats)
    frames = feats.frames(mmap=True)
    need4 = a.method == "learned"
    t_load = time.time()
    x0 = None
    if a.method in ("raw", "mean", "smooth", "pca", "leace_tr", "leace_oracle"):
        x0 = baselines.build_x0(frames, feats.offsets)
    rag4 = None
    if need4:
        rag4 = Ragged(torch.from_numpy(np.ascontiguousarray(frames)), feats.offsets)
    stats = feats.stats() if a.method == "layers" else None
    feat_rep = None
    if a.method == "feat":
        hc = data.FEAT_ROOT / "handcrafted"
        F_ = np.load(hc / f"{a.feat}.npy").astype(np.float32)
        ctx.keep = np.load(hc / f"{a.feat}.keep.npy")
        if cfg.get("cols"):
            F_ = F_[:, cfg["cols"]]
        if cfg.get("actor_norm"):  # ORACLE per-actor standardisation (uses the actor's own statistics, incl. test actors)
            for ac in np.unique(ctx.actor):
                mm = (ctx.actor == ac) & ctx.keep
                F_[mm] = (F_[mm] - F_[mm].mean(0)) / (F_[mm].std(0) + 1e-6)
        feat_rep = Rep(torch.from_numpy(F_), None, a.feat)
        print(f"[{a.name}] feat {a.feat} {F_.shape} kept {ctx.keep.mean():.3f}", flush=True)
    print(f"[{a.name}] loaded in {time.time()-t_load:.0f}s, method={a.method} cfg={cfg}", flush=True)
    sp = ctx.sp
    for seed in a.seeds:
        t0 = time.time()
        acc, extra = {}, dict(folds=[])
        for fold in a.folds:
            S = ctx.fold_sets(fold)
            tf = time.time()
            torch.manual_seed(seed * 100 + fold); np.random.seed(seed * 100 + fold)
            head_pred, info = None, {}
            if a.method == "raw":
                rep = baselines.rep_from_ragged(x0, "raw")
            elif a.method == "mean":
                rep = baselines.mean_rep(x0)
            elif a.method == "smooth":
                rep = baselines.rep_from_ragged(baselines.smooth(x0, cfg["P"]), "smooth")
            elif a.method == "pca":
                A, b = baselines.fit_pca(x0, S["tr"], cfg["k"])
                rep = baselines.rep_from_ragged(baselines.affine(x0, A, b), "pca")
            elif a.method in ("leace_tr", "leace_oracle"):
                if a.method == "leace_tr":
                    clips, lab = S["tr"], ctx.actor_idx[S["tr"]]
                else:
                    clips = np.where(ctx.masks["atk_train"])[0]; lab = ctx.actor_idx[clips]
                A, b, r = baselines.fit_leace(x0, clips, lab, 91)
                info["leace_rank"] = r
                rep = baselines.rep_from_ragged(baselines.affine(x0, A, b), a.method)
            elif a.method == "feat":
                rep = feat_rep
            elif a.method == "ctrl_constant":
                rep = constant_rep(feats.ids.shape[0])
            elif a.method == "ctrl_length":
                rep = length_rep(ctx.lengths)
            elif a.method == "layers":
                rep = None
            elif a.method == "learned":
                E, hist = train_encoder(cfg, rag4, ctx.emo, ctx.actor_idx, S["tr"], S["val"], seed * 100 + fold,
                                        log=lambda *x: print(f"[{a.name} s{seed} f{fold}]", *x, flush=True))
                rep, rate = encode_all(E, rag4)
                head_pred = []
                with torch.no_grad():
                    for s in range(0, len(S["te"]), 128):
                        x, m = rag4.batch(S["te"][s:s + 128])
                        head_pred.append(E(x.float(), m)["emo_logits"].argmax(1).cpu().numpy())
                head_pred = np.concatenate(head_pred)
                info.update(rate=rate, hist=hist, layer_w=torch.softmax(E.lw, 0).tolist(),
                            val_uar_head=hist[-1]["val_uar_head"])
            if a.method == "layers":
                acc.setdefault("layers", {})
                for l in range(stats.shape[1]):
                    r_ = Rep(torch.from_numpy(stats[:, l].reshape(stats.shape[0], -1)), None, f"layer{l}")
                    o = ev.evaluate_rep(ctx, r_, fold, seed, secrets=False)
                    ev.merge_fold(acc["layers"].setdefault(l, {}), o)
                print(f"[{a.name} s{seed} f{fold}] layers done {time.time()-tf:.0f}s", flush=True)
                continue
            indep = a.method in ("raw", "mean", "smooth", "feat", "ctrl_constant", "ctrl_length", "leace_oracle")
            id_mode = "fold" if not indep else ("all" if fold == a.folds[0] else "skip")
            o = ev.evaluate_rep(ctx, rep, fold, seed, shuffle=a.shuffle, secrets=not a.no_secrets, extra_emo=head_pred, id_mode=id_mode)
            ev.merge_fold(acc, o)
            extra["folds"].append(dict(fold=fold, seconds=round(time.time() - tf, 1), **info))
            # compact log line (progress only; headline numbers come from aggregate.py with intervals)
            emo = {k: float(np.mean([np.trace(np.array(C)) / max(np.array(C).sum(), 1) for C in v.values()])) for k, v in o["emo"].items()}
            idt = {k: float(np.mean([c / max(n, 1) for c, n in t["T1"]["unseen"].values()])) for k, t in o["id"].items() if "T1" in t}
            print(f"[{a.name} s{seed} f{fold}] {time.time()-tf:.0f}s  emo_acc(actor-mean) {emo}  id_T1_unseen(actor-mean) {idt}", flush=True)
        res = dict(name=a.name, method=a.method, cfg=cfg, backbone=a.backbone, seed=seed, shuffle=a.shuffle,
                   commit=git_rev(), seconds=round(time.time() - t0, 1), splits=dict(S_TE=sp["S_TE"], S_TR=sp["S_TR"], E_TR=sp["E_TR"], E_TE=sp["E_TE"]),
                   **acc, **extra)
        json.dump(res, open(out / f"result_s{seed}.json", "w"))
        print(f"[{a.name}] seed {seed} done in {res['seconds']}s -> {out}/result_s{seed}.json", flush=True)


if __name__ == "__main__":
    main()
