"""Per-fold evaluation of a released representation: emotion utility, identity leakage, secondary secrets.

Utility  : emotion probes (linear / MLP / kNN / transformer) retrained on the released rep.  Fit on encoder-train actors,
           hyper-parameters chosen on validation actors, refit on train+val actors, scored on the sealed TEST actors only.
Privacy  : closed-set 91-way identity attackers retrained on the released rep.  Trained on S_TR x E_TR clips of ALL 91 actors
           (the attacker's own enrolment data), scored on T1 (unseen sentences), T2 (unseen emotions), T3 (both).
           Counts are split into actors the encoder never saw (test actors, group 'unseen') and encoder-train actors ('seen').
Secrets  : sex, race (3-way), age, attackers trained on train+val actors, scored on test actors (actor-disjoint).
"""
import numpy as np
import data
from lib import attack, suite_for, actor_counts, actor_confusions, uar

AGE_MU, AGE_SD = 30.0, 12.0


class Ctx:
    def __init__(self, feats):
        self.meta = data.load_meta()
        assert (self.meta["ids"] == feats.ids).all(), "feature cache not aligned with metadata"
        self.sp = data.get_splits(self.meta)
        self.masks = data.attack_masks(self.meta, self.sp)
        self.actors_sorted = sorted(set(self.meta["actor"].tolist()))
        a2i = {a: i for i, a in enumerate(self.actors_sorted)}
        self.actor = self.meta["actor"]
        self.actor_idx = np.array([a2i[a] for a in self.actor])
        self.emo = np.array([data.EMOTIONS.index(e) for e in self.meta["emo"]])
        self.sex, self.race, self.age = data.actor_attrs(self.meta)
        self.lengths = feats.lengths()
        self.feats = feats
        self.keep = np.ones(len(self.actor), dtype=bool)  # clips that have a representation (hand-crafted sets may drop some)

    def fold_sets(self, fold):
        f = self.sp["folds"][fold]
        m = lambda lst: np.isin(self.actor, lst) & self.keep
        return dict(tr=np.where(m(f["train"]))[0], val=np.where(m(f["val"]))[0], te=np.where(m(f["test"]))[0],
                    te_actors=f["test"], tr_actors=f["train"])


def _label_only_identity(ctx, tr_idx, test_idx):
    """Best identity guess from the emotion label alone (trained on the attacker's training clips)."""
    cnt = np.zeros((6, 91))
    np.add.at(cnt, (ctx.emo[tr_idx], ctx.actor_idx[tr_idx]), 1)
    guess = cnt.argmax(1)
    return guess[ctx.emo[test_idx]] == ctx.actor_idx[test_idx]


def evaluate_rep(ctx, rep, fold, seed, shuffle="none", secrets=True, extra_emo=None, log=print):
    S = ctx.fold_sets(fold)
    rs = np.random.RandomState(1234 + seed)
    out = dict(emo={}, id={}, secrets={}, hp={})
    # ------------------------------------------------------------------ emotion utility on sealed test actors
    y = ctx.emo.copy()
    train_all = np.concatenate([S["tr"], S["val"]])
    if shuffle == "emo":
        y[train_all] = rs.permutation(y[train_all])
    res = attack(rep, y, S["tr"], S["val"], train_all, dict(test=S["te"]), suite_for(rep), "cls", 6, seed)
    for k, r in res.items():
        p = r["preds"]["test"]
        out["emo"][k] = actor_confusions(ctx.actor[S["te"]], ctx.emo[S["te"]], p, 6)
        out["hp"][f"emo_{k}"] = dict(r["hp"], val=r["val"])
    if extra_emo is not None:  # in-pipeline head of a learned encoder (not a retrained probe)
        out["emo"]["head"] = actor_confusions(ctx.actor[S["te"]], ctx.emo[S["te"]], extra_emo, 6)
    # ------------------------------------------------------------------ identity attackers (91-way closed set)
    M = ctx.masks
    idx = {k: np.where(v & ctx.keep)[0] for k, v in M.items()}
    y = ctx.actor_idx.copy()
    if shuffle == "id":
        y[idx["atk_train"]] = rs.permutation(y[idx["atk_train"]])
    evs = {t: idx[t] for t in ("T1", "T2", "T3")}
    res = attack(rep, y, idx["atk_fit"], idx["atk_val"], idx["atk_train"], evs, suite_for(rep), "cls", 91, seed)
    te_set = set(S["te_actors"]); tr_set = set(S["tr_actors"])
    for k, r in res.items():
        out["id"][k] = {}
        for t, ii in evs.items():
            correct = r["preds"][t] == ctx.actor_idx[ii]
            act = ctx.actor[ii]
            un = np.isin(act, list(te_set)); se = np.isin(act, list(tr_set))
            out["id"][k][t] = dict(unseen=actor_counts(act[un], correct[un]), seen=actor_counts(act[se], correct[se]))
        out["hp"][f"id_{k}"] = dict(r["hp"], val=r["val"])
    lo = {}
    for t, ii in evs.items():
        correct = _label_only_identity(ctx, idx["atk_train"], ii)
        act = ctx.actor[ii]
        un = np.isin(act, list(te_set)); se = np.isin(act, list(tr_set))
        lo[t] = dict(unseen=actor_counts(act[un], correct[un]), seen=actor_counts(act[se], correct[se]))
    out["id"]["label_only"] = lo
    # ------------------------------------------------------------------ secondary secrets (actor-disjoint)
    if secrets and shuffle == "none":
        for name, lab, ncls in (("sex", ctx.sex, 2), ("race", ctx.race, 3)):
            ok = (lab >= 0) & ctx.keep
            f_, v_, a_, t_ = (np.intersect1d(S[k], np.where(ok)[0]) for k in ("tr", "val", "tr", "te"))
            a_ = np.concatenate([f_, v_])
            res = attack(rep, lab, f_, v_, a_, dict(test=t_), ["linear", "mlp"], "cls", ncls, seed)
            out["secrets"][name] = {k: actor_confusions(ctx.actor[t_], lab[t_], r["preds"]["test"], ncls) for k, r in res.items()}
        yreg = (ctx.age - AGE_MU) / AGE_SD
        res = attack(rep, yreg, S["tr"], S["val"], train_all, dict(test=S["te"]), ["ridge", "mlp"], "reg", 1, seed)
        ages = {}
        for k, r in res.items():
            err = np.abs(r["preds"]["test"] * AGE_SD + AGE_MU - ctx.age[S["te"]])
            ages[k] = {int(a): [float(err[ctx.actor[S["te"]] == a].sum()), int((ctx.actor[S["te"]] == a).sum())] for a in S["te_actors"]}
        base = ctx.age[train_all].mean()
        err = np.abs(base - ctx.age[S["te"]])
        ages["const_mean"] = {int(a): [float(err[ctx.actor[S["te"]] == a].sum()), int((ctx.actor[S["te"]] == a).sum())] for a in S["te_actors"]}
        out["secrets"]["age"] = ages
    return out


def merge_fold(acc, fold_out):
    """Accumulate per-fold outputs into one dict (actors are disjoint across folds for 'unseen'/emotion; 'seen' sums)."""
    for k, v in fold_out["emo"].items():
        acc.setdefault("emo", {}).setdefault(k, {}).update(v)
    for k, d in fold_out["id"].items():
        for t, g in d.items():
            for grp, cnts in g.items():
                tgt = acc.setdefault("id", {}).setdefault(k, {}).setdefault(t, {}).setdefault(grp, {})
                for a, (c, n) in cnts.items():
                    o = tgt.setdefault(a, [0, 0]); o[0] += c; o[1] += n
    for name, d in fold_out["secrets"].items():
        for k, v in d.items():
            acc.setdefault("secrets", {}).setdefault(name, {}).setdefault(k, {}).update(v)
    acc.setdefault("hp", []).append(fold_out["hp"])
    return acc
