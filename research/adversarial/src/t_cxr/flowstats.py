"""Displacement statistics of a learned generator on A-role images (T.md 3.4 `rw3`; also descriptive: how far does a PriCheXy-style warp really move pixels?).
Saves a torch dict {"mu<mu>": {"rms_px": tensor, "radial_power": tensor}} (consumed by run_suite --flow-stats) in --wout (VM only) and a JSON summary in $OUT.
   python -m t_cxr.flowstats --generators pcxr_mu0.01,pcxr_mu0.005,pcx_mu0.01 ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import generator as G, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", required=True); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--gen-dir", required=True); ap.add_argument("--pcxr-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--wout", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d)
    d = meta.assign_roles(d, seen, Path(a.salt_file).read_text().strip(), 0)
    rows = d.loc[(d.role == "A") & d.ok, "row"].to_numpy()[: a.n]
    bank = np.load(a.bank, mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(bank[rows])).float().div(255).unsqueeze(1)
    stats, summ = {}, {}
    for g in a.generators.split(","):
        kind, mu = g.split("_mu"); mu = float(mu)
        path = f"{a.pcxr_dir}/generator_pcxr_mu{mu:g}.pth" if kind == "pcxr" else f"{a.gen_dir}/generator_lowest_total_loss_mu_{mu:g}.pth"
        gen = G.load_generator(path, a.device)
        st = G.flow_statistics(gen, x, mu)
        stats[g] = dict(rms_px=st["rms_px"], radial_power=st["radial_power"])
        stats[f"mu{mu:g}"] = stats.get(f"mu{mu:g}", stats[g]) if kind != "pcxr" else stats[g]           # rw3 uses the retrained generator when present
        q = lambda t: [float(np.percentile(t.numpy(), p)) for p in (5, 50, 95)]
        summ[g] = dict(mu=mu, rms_px_p5_p50_p95=q(st["rms_px"]), sup_px_p5_p50_p95=q(st["sup_px"]), bound_px=float(mu * G.PX_PER_UNIT), n_images=int(len(rows)))
        print(g, summ[g], flush=True)
    Path(a.wout).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in stats.items() if k.startswith("mu")}, a.wout)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "flowstats.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
