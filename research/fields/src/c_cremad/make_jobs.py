"""Generate queue job specs (research/fields/jobs/specs/c_*.yaml) for the Focus C sweeps.
  python make_jobs.py ref | sweep0 | seeds12
Each learned job = one config x one seed x 4 actor-disjoint folds (one GPU, ~10 min).
"""
import json, sys
from pathlib import Path

SPECS = Path(__file__).resolve().parents[2] / "jobs" / "specs"
RUN = "python3 research/fields/src/c_cremad/run_exp.py"


def spec(name, method, cfg=None, seeds=(0,), prio=20, extra=""):
    cmd = f"{RUN} --name {name} --method {method} --cfg '{json.dumps(cfg or {})}' --seeds {' '.join(map(str, seeds))} {extra}".strip()
    # json.dumps gives a valid double-quoted YAML scalar (a plain scalar containing ': ' from the JSON cfg is a YAML error
    # that crashes every worker's claim() -- happened once, 08:55Z)
    (SPECS / f"{name}.yaml").write_text(f"name: {name}\npriority: {prio}\ncmd: {json.dumps(cmd)}\nhf_path: runs/{name}\n")
    return name


def ref():
    out = [spec("c_ref_raw", "raw", seeds=(0, 1, 2), prio=11),
           spec("c_ctrl_constant", "ctrl_constant", seeds=(0, 1, 2), prio=12, extra="--no_secrets"),
           spec("c_ctrl_length", "ctrl_length", seeds=(0,), prio=12, extra="--no_secrets"),
           spec("c_ctrl_shuf_id", "raw", seeds=(0,), prio=12, extra="--shuffle id --no_secrets"),
           spec("c_ctrl_shuf_emo", "raw", seeds=(0,), prio=12, extra="--shuffle emo --no_secrets"),
           spec("c_layers", "layers", seeds=(0,), prio=12),
           spec("c_fix_mean", "mean", seeds=(0, 1, 2), prio=13),
           spec("c_fix_leace_tr", "leace_tr", seeds=(0, 1, 2), prio=13),
           spec("c_fix_leace_oracle", "leace_oracle", seeds=(0, 1, 2), prio=13)]
    for P in (5, 25, 100):
        out.append(spec(f"c_fix_smooth{P}", "smooth", {"P": P}, seeds=(0, 1, 2), prio=14))
    for k in (4, 16, 64):
        out.append(spec(f"c_fix_pca{k}", "pca", {"k": k}, seeds=(0, 1, 2), prio=14))
    return out


def trim():
    """Speech-only variants (VAD-trimmed frames): identity net of the non-speech channel cue (H-C-006)."""
    return [spec("c_ref_raw_trim", "raw", seeds=(0, 1, 2), prio=12, extra="--trim"),
            spec("c_fix_mean_trim", "mean", seeds=(0,), prio=13, extra="--trim")]


def learned_configs():
    cf = {}
    for d in (2, 4, 8, 16, 32, 64, 128, 256):
        cf[f"c_rate_d{d}"] = dict(dz=d, bott="vib", beta=0.0, lam=0.0)
    for b in (3e-3, 3e-2, 3e-1):
        cf[f"c_vib_d64_b{b:g}"] = dict(dz=64, bott="vib", beta=b, lam=0.0)
    for d in (16, 64):
        for lam in (0.3, 1.0, 3.0, 10.0):
            cf[f"c_adv_d{d}_l{lam:g}"] = dict(dz=d, bott="vib", beta=0.0, lam=lam)
    for g, k in ((1, 8), (1, 64), (2, 64), (4, 64), (4, 256)):
        cf[f"c_vq_g{g}k{k}"] = dict(bott="vq", vq_groups=g, vq_codes=k, vq_dim=16, lam=0.0)
    cf["c_vqadv_g4k64_l1"] = dict(bott="vq", vq_groups=4, vq_codes=64, vq_dim=16, lam=1.0)
    cf["c_rate_d1"] = dict(dz=1, bott="vib", beta=0.0, lam=0.0)
    for d in (16, 64):  # harder-working adversary: 5x longer warm-up after every re-initialisation, reinit every 5 epochs
        cf[f"c_advstrong_d{d}_l3"] = dict(dz=d, bott="vib", beta=0.0, lam=3.0, adv_warm=500, adv_reinit=5)
    cf["c_post6"] = dict(dz=64, bott="vib", beta=0.0, lam=0.0, release="posterior")  # minimal-rate reference: 6 class posteriors
    cf["c_label6"] = dict(dz=64, bott="vib", beta=0.0, lam=0.0, release="label")  # hard emotion label, log2(6)=2.6 bits
    for d, lam in ((8, 0.0), (32, 0.0), (32, 1.0)):
        cf[f"c_seg25_d{d}_l{lam:g}"] = dict(dz=d, bott="vib", beta=0.0, lam=lam, pool=25)
    return cf


def sweep(seeds, prio):
    out = []
    for name, cfg in learned_configs().items():
        for s in seeds:
            out.append(spec(f"{name}_s{s}", "learned", cfg, seeds=(s,), prio=prio + s))
    return out


def sweeptrim(seeds=(0,), prio=22):
    """Key learned configs again on speech-trimmed input (net of the non-speech channel cue)."""
    keep = ["c_rate_d4", "c_rate_d16", "c_rate_d64", "c_adv_d16_l1", "c_adv_d16_l3", "c_adv_d64_l3", "c_vq_g4k64"]
    cf = learned_configs()
    return [spec(f"{n.replace('c_', 'c_trim_', 1)}_s{sd}", "learned", cf[n], seeds=(sd,), prio=prio + sd, extra="--trim")
            for n in keep for sd in seeds]


if __name__ == "__main__":
    what = sys.argv[1]
    names = sweeptrim() if what == "sweeptrim" else trim() if what == "trim" else ref() if what == "ref" else sweep((0,), 14) if what == "sweep0" else sweep((1, 2), 30)
    print(len(names), "specs", names[:3], "...")
