"""PathoROB data assembly and the unit-disjoint A/B/V/C(+OOD) split protocol.

Independent unit = slide (camelyon), case (tcga: slide_id[:12]; tolkach: case).  All splits are by unit.
  C     sealed test units (fixed, independent of the run seed)           25% of units per (centre, class) stratum
  A     adapter-train units                                              45% of the rest
  B     attacker / label-probe train units (disjoint from A)             35% of the rest
  V     validation units (selection of hyper-parameters / methods)       20% of the rest
  OOD   the benchmark's unseen centres: OOD1 = attacker-train half, OOD2 = test half (fixed split by unit); the whole OOD set
        is also used to measure label accuracy on unseen centres.  Never used to fit an adapter.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import common as C

ID_CENTRES = {
    "camelyon": ["RUMC", "UMCU"],
    "tcga": ["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"],
    "tolkach_esca": ["VALSET2_WNS", "VALSET4_CHA_FULL"],
}
OOD_CENTRES = {
    "camelyon": ["CWZ", "RST", "LPON"],
    "tcga": ["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium", "Johns Hopkins"],
    "tolkach_esca": ["VALSET1_UKK", "VALSET3_TCGA"],
}
META_CSV = {"camelyon": "camelyon", "tcga": "tcga_4x4", "tolkach_esca": "tolkach_esca"}


@dataclass
class PData:
    name: str
    X: np.ndarray          # (n, d) float32 raw frozen embeddings
    y: np.ndarray          # biological class idx
    c: np.ndarray          # centre idx (ID centres first, then OOD)
    u: np.ndarray          # unit idx
    ood: np.ndarray        # bool, patch belongs to an unseen (OOD) centre
    classes: list
    centres: list
    n_id_centres: int
    unit_names: list


def load_pdata(model, dataset):
    X = C.load_embeddings(model, dataset)
    idx = C.load_index(dataset)
    meta = pd.read_csv(C.PATHOROB_REPO / "data" / "metadata" / f"{META_CSV[dataset]}.csv", dtype=str)
    key = idx.slide_id.astype(str) + "|" + idx.patch_id.astype(str)
    assert key.is_unique, "duplicate (slide, patch) keys in parquet index"
    pos = pd.Series(np.arange(len(idx)), index=key.values)
    mkey = meta.slide_id.astype(str) + "|" + meta.patch_id.astype(str)
    rows = pos.reindex(mkey.values)
    assert rows.notna().all(), "metadata patches missing from parquet"
    rows = rows.values.astype(int)
    meta = meta.reset_index(drop=True)
    assert (idx.medical_center.values[rows] == meta.medical_center.values).all()
    assert (idx.biological_class.values[rows] == meta.biological_class.values).all()
    centres = ID_CENTRES[dataset] + OOD_CENTRES[dataset]
    classes = sorted(meta.biological_class.unique())
    unit_key = meta.slide_id.str[:12] if dataset == "tcga" else meta.slide_id
    unit_names = sorted(unit_key.unique())
    return PData(
        name=dataset, X=X[rows], y=meta.biological_class.map({k: i for i, k in enumerate(classes)}).values.astype(int),
        c=meta.medical_center.map({k: i for i, k in enumerate(centres)}).values.astype(int),
        u=unit_key.map({k: i for i, k in enumerate(unit_names)}).values.astype(int),
        ood=(meta["subset"] == "OOD").values, classes=classes, centres=centres,
        n_id_centres=len(ID_CENTRES[dataset]), unit_names=unit_names)


def _unit_table(P, mask):
    """one row per unit: centre, a stratum label (centre, class if the unit is single-class else 'multi')."""
    df = pd.DataFrame(dict(u=P.u[mask], c=P.c[mask], y=P.y[mask]))
    g = df.groupby("u").agg(c=("c", "nunique"), ny=("y", "nunique"), y=("y", "first"), n=("y", "size")).reset_index()
    assert (g.c == 1).all(), "unit spans several centres"
    cen = df.groupby("u").c.first()
    g["c"] = g.u.map(cen)
    if (g.ny == 1).all():
        g["stratum"] = [f"{c}|{y}" for c, y in zip(g.c, g.y)]
    else:                                  # units span classes (tolkach cases): stratify by centre only
        g["stratum"] = [str(c) for c in g.c]
    return g


def _take(rng, units, n):
    units = list(units)
    rng.shuffle(units)
    return units[:n], units[n:]


def confound_A(P, idxA, rho, seed):
    """Make the adapter-train set shortcut-prone: class j has a 'home' centre (j mod #centres); a share `rho` of the class-j patches
    in A comes from the home centre, the rest from the other centres (patch-level subsampling of the units already in A)."""
    rng = np.random.RandomState(20_000 + seed)
    Kc = P.n_id_centres
    keep = []
    for j in range(len(P.classes)):
        ij = idxA[P.y[idxA] == j]
        h, o = ij[P.c[ij] == j % Kc], ij[P.c[ij] != j % Kc]
        n_o = int(round(len(h) * (1 - rho) / rho))
        keep.append(h)
        if len(o):
            keep.append(rng.choice(o, size=min(n_o, len(o)), replace=False))
    return np.sort(np.concatenate(keep))


def seen_held_classes(P):
    K = len(P.classes)
    seen = list(range((K + 1) // 2))
    return seen, [k for k in range(K) if k not in seen]


def make_splits(P, seed, test_frac=0.25, fracs=(0.45, 0.35, 0.20), confound=None, holdout=False):
    """returns dict of patch-index arrays: A, B, V, C (ID centres), OOD1, OOD2, OOD (all unseen-centre patches).
    confound=rho: A is subsampled to centre-class correlation rho.  holdout: A keeps only the 'seen' half of the classes."""
    idm = ~P.ood
    g = _unit_table(P, idm)
    unit_split = {}
    rng_c = np.random.RandomState(1234)      # sealed test split: independent of the run seed
    rng_s = np.random.RandomState(10_000 + seed)
    for _, grp in g.groupby("stratum"):
        us = sorted(grp.u.tolist())
        nt = max(1, int(np.ceil(test_frac * len(us))))
        test, rest = _take(rng_c, us, nt)
        rest = sorted(rest)
        na = max(1, int(round(fracs[0] * len(rest))))
        nb = max(1, int(round(fracs[1] * len(rest))))
        a, rest = _take(rng_s, rest, na)
        b, v = _take(rng_s, rest, nb)
        if len(v) == 0:                       # tiny stratum: borrow from A
            v, a = a[-1:], a[:-1]
        for name, us_ in (("C", test), ("A", a), ("B", b), ("V", v)):
            for u in us_:
                unit_split[u] = name
    sp = {k: np.where(idm & np.array([unit_split.get(u, "") == k for u in P.u]))[0] for k in "ABVC"}
    if holdout:
        seen, _ = seen_held_classes(P)
        sp["A"] = sp["A"][np.isin(P.y[sp["A"]], seen)]
    if confound is not None:
        sp["A"] = confound_A(P, sp["A"], confound, seed)
    # OOD: fixed halves by unit, stratified by centre
    oodm = P.ood
    go = _unit_table(P, oodm)
    rng_o = np.random.RandomState(4321)
    o1 = []
    for _, grp in go.groupby("c"):
        us = sorted(grp.u.tolist())
        h, _ = _take(rng_o, us, int(np.ceil(len(us) / 2)))
        o1 += h
    o1 = set(o1)
    sp["OOD1"] = np.where(oodm & np.array([u in o1 for u in P.u]))[0]
    sp["OOD2"] = np.where(oodm & np.array([u not in o1 for u in P.u]))[0]
    sp["OOD"] = np.where(oodm)[0]
    # hygiene: unit-disjoint
    sets = {k: set(P.u[sp[k]].tolist()) for k in ["A", "B", "V", "C", "OOD1", "OOD2"]}
    ks = list(sets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            assert not (sets[ks[i]] & sets[ks[j]]), f"units shared between {ks[i]} and {ks[j]}"
    return sp
