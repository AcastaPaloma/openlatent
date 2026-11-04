"""CREMA-D metadata, fixed splits and feature-cache loading (Focus C).

Everything that defines the benchmark lives here so the split is frozen in one place:
  * 91 actors, 12 sentences, 6 emotions (ANG DIS FEA HAP NEU SAD), 7,442 clips.
  * `make_splits()` is deterministic (numpy RandomState seeds fixed below) and is written to splits.json.
  * Actor folds (4, stratified by sex then race): fold f is the sealed TEST actors of "fold f";
    12 further non-test actors are VALIDATION actors, the rest are encoder TRAIN actors.
  * Sentence groups for the identity attacker: S_TR (8 sentences, of which 2 are attacker-validation S_VA), S_TE (4).
  * Emotion groups for the identity attacker: E_TR (ANG HAP NEU SAD), E_TE (DIS FEA).
    Attacker trains on S_TR x E_TR only.  Test sets: T1 = S_TE x E_TR (unseen sentences, seen emotions; PRIMARY),
    T2 = S_TR x E_TE (unseen emotions), T3 = S_TE x E_TE (unseen sentences AND emotions; hardest).
"""
import csv, json, os
from pathlib import Path
import numpy as np

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
SENTENCES = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]
E_TR = ["ANG", "HAP", "NEU", "SAD"]
E_TE = ["DIS", "FEA"]
DATA_ROOT = Path(os.environ.get("CREMAD_ROOT", Path.home() / "work/data/cremad"))
FEAT_ROOT = Path(os.environ.get("CREMAD_FEATS", Path.home() / "work/feats/cremad"))
SPLIT_FILE = Path(__file__).with_name("splits.json")


def load_meta(root=DATA_ROOT):
    """Return dict of numpy arrays aligned to sorted clip ids."""
    names = sorted(r["Filename"] for r in csv.DictReader(open(Path(root) / "SentenceFilenames.csv")))
    parts = [n.split("_") for n in names]
    actor = np.array([int(p[0]) for p in parts])
    sent = np.array([p[1] for p in parts])
    emo = np.array([p[2] for p in parts])
    level = np.array([p[3] for p in parts])
    dem = {int(r["ActorID"]): r for r in csv.DictReader(open(Path(root) / "VideoDemographics.csv"))}
    return dict(ids=np.array(names), actor=actor, sent=sent, emo=emo, level=level, dem=dem)


def make_splits(meta, seed=0):
    rs = np.random.RandomState(seed)
    actors = sorted(set(meta["actor"].tolist()))
    dem = meta["dem"]
    # stratify by (sex, race): shuffle within stratum, deal round-robin into 4 folds
    strata = {}
    for a in actors:
        strata.setdefault((dem[a]["Sex"], dem[a]["Race"]), []).append(a)
    folds = [[] for _ in range(4)]
    k = 0
    for key in sorted(strata):
        lst = list(strata[key]); rs.shuffle(lst)
        for a in lst:
            folds[k % 4].append(a); k += 1
    folds = [sorted(f) for f in folds]
    out_folds = []
    for f in range(4):
        test = folds[f]
        rest = sorted(set(actors) - set(test))
        rs2 = np.random.RandomState(1000 + f)
        val = sorted(rs2.choice(rest, 12, replace=False).tolist())
        train = sorted(set(rest) - set(val))
        out_folds.append(dict(fold=f, train=train, val=val, test=test))
    rs3 = np.random.RandomState(7)
    sperm = list(rs3.permutation(SENTENCES))
    s_te = sorted(sperm[:4]); s_va = sorted(sperm[4:6]); s_tr = sorted(sperm[4:])
    return dict(seed=seed, folds=out_folds, S_TE=s_te, S_TR=s_tr, S_VA=s_va, E_TR=E_TR, E_TE=E_TE)


def get_splits(meta=None):
    if SPLIT_FILE.exists():
        return json.load(open(SPLIT_FILE))
    sp = make_splits(meta or load_meta())
    json.dump(sp, open(SPLIT_FILE, "w"), indent=1)
    return sp


def attack_masks(meta, sp):
    """Boolean masks over clips for the identity attacker (independent of fold)."""
    s_tr, s_te, s_va = map(set, (sp["S_TR"], sp["S_TE"], sp["S_VA"]))
    e_tr, e_te = set(sp["E_TR"]), set(sp["E_TE"])
    S, E = meta["sent"], meta["emo"]
    in_ = lambda arr, s: np.array([x in s for x in arr])
    m = dict(
        atk_train=in_(S, s_tr) & in_(E, e_tr),
        atk_fit=in_(S, s_tr - s_va) & in_(E, e_tr),   # used while selecting hyper-parameters
        atk_val=in_(S, s_va) & in_(E, e_tr),
        T1=in_(S, s_te) & in_(E, e_tr),
        T2=in_(S, s_tr) & in_(E, e_te),
        T3=in_(S, s_te) & in_(E, e_te),
    )
    return m


def actor_attrs(meta):
    """Per-clip secondary secrets: sex (0/1), race (0 Caucasian,1 African American,2 Asian,-1 unknown), age."""
    dem = meta["dem"]
    race_map = {"Caucasian": 0, "African American": 1, "Asian": 2}
    sex = np.array([1 if dem[a]["Sex"] == "Female" else 0 for a in meta["actor"]])
    race = np.array([race_map.get(dem[a]["Race"], -1) for a in meta["actor"]])
    age = np.array([float(dem[a]["Age"]) for a in meta["actor"]])
    return sex, race, age


class FeatStore:
    """Frame-level features cached by extract_feats.py (flat fp16 [Nfr, L, D] + offsets) and pooled stats."""

    def __init__(self, backbone="wavlm-base-plus", root=FEAT_ROOT):
        self.dir = Path(root) / backbone
        self.info = json.load(open(self.dir / "info.json"))
        self.ids = np.array(open(self.dir / "clip_ids.txt").read().split())
        self.offsets = np.load(self.dir / "offsets.npy")
        self.layers = self.info["layers"]

    def frames(self, mmap=True):
        return np.load(self.dir / "frames.npy", mmap_mode="r" if mmap else None)

    def stats(self):
        return np.load(self.dir / "stats.npy")  # [N, n_all_layers, 2, D] float32 (mean,std)

    def lengths(self):
        return np.diff(self.offsets)
