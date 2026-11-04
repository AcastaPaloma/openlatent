"""Shared loader: clean episodes (envgen.py) + deterministic distractor states (distractors.py)."""
import os, json
import numpy as np
import distractors as D

ROOT = os.path.expanduser(os.environ.get("B_DATA", "~/work/data/b_mw"))
SPLIT_SEED = {"train": 11, "val": 22, "test": 33}
_cache = {}


def load_split(split, root=ROOT):
    k = ("eps", split, root)
    if k not in _cache:
        z = np.load(os.path.join(root, f"{split}.npz"))
        _cache[k] = {n: z[n] for n in z.files}
    return _cache[k]


def unpack(packed):
    """(..., H, W/8) uint8 -> (..., H, W) bool"""
    return np.unpackbits(packed, axis=-1).astype(bool)


def global_params(root=ROOT):
    k = ("gp", root)
    if k not in _cache:
        tr = load_split("train", root)
        M = D.fit_ee_response(tr["acts"], tr["eepix"])
        maps = D.make_maps(M)
        # std of the total sprite speed in corr_noisy, used to match indep speed
        st = D.sample("corr_noisy", tr["acts"], tr["eepix"][:, 0], maps, seed=0)
        _cache[k] = dict(M_ee=M, maps=maps, indep_std=float(st["vtot"].std()))
    return _cache[k]


def dist_states(split, kind, root=ROOT):
    k = ("dist", split, kind, root)
    if k not in _cache:
        ep = load_split(split, root)
        gp = global_params(root)
        seed = SPLIT_SEED[split] * 100 + D.KINDS.index(kind)
        _cache[k] = D.sample(kind, ep["acts"], ep["eepix"][:, 0], gp["maps"], seed, indep_std=gp["indep_std"])
    return _cache[k]
