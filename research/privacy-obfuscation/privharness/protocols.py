"""Attack protocols: which items an attacker trains on and which it is tested on.

    within          the dataset's own split. The headline protocol: an attacker that has seen obfuscated
                    footage of these people/attributes in training is tested on new footage of them.
    cross_utility   attackers train on items of two thirds of the primary utility classes and are tested on the
                    other third (all splits). A secret that is only "how this person does this task" stops
                    counting; what remains transfers to activities the attacker never saw.
    cross_<group>   attackers train on items of some values of a group key (scene, session, camera) and are tested
                    on unseen values: does the secret survive a new recording condition?
    hardest         cross_utility on the dataset's test split only, with the test split's group values unseen in
                    training when the dataset's split already holds out a group.

Multiclass secrets (identity) are only scored on test items whose class occurs in the attacker's training items.
A protocol is infeasible (skipped, with the reason recorded) when fewer than `min_test` items or fewer than two
secret classes remain.
"""

import numpy as np


def utility_partition(spec, seed=0):
    """Deterministic 2/3 - 1/3 partition of the primary utility classes, balanced by item count."""
    prim = spec.primary
    if prim.kind != "multiclass" or prim.n < 3:
        return None
    counts = np.bincount([it["labels"][prim.name] for it in spec.items if prim.name in it["labels"]], minlength=prim.n)
    order = np.random.default_rng(seed).permutation(prim.n)
    held, total, target = [], 0, counts.sum() / 3
    for c in order:  # greedy: add classes to the held-out third until it holds about a third of the items
        if total < target and len(held) < prim.n - 2:
            held.append(int(c))
            total += counts[c]
    return sorted(held)


def build(spec, secret, which, min_test=20, seed=0):
    """-> dict(name, train, test, note) for the attacker of `secret`, or dict(name, skip=reason)."""
    lab = [it for it in spec.items if secret.name in it["labels"]]
    tr, te, note = None, None, ""
    if which == "within":
        tr, te = [it for it in lab if it["split"] == "train"], [it for it in lab if it["split"] == "test"]
        note = "dataset split"
    elif which in ("cross_utility", "hardest"):
        held = utility_partition(spec, seed)
        if held is None:
            return dict(name=which, skip="primary utility target is not multiclass with >= 3 classes")
        prim = spec.primary.name
        unseen = lambda it: prim in it["labels"] and it["labels"][prim] in held  # noqa: E731
        seen = lambda it: prim in it["labels"] and it["labels"][prim] not in held  # noqa: E731
        if which == "cross_utility":
            tr, te = [it for it in lab if seen(it)], [it for it in lab if unseen(it)]
        else:
            tr = [it for it in lab if seen(it) and it["split"] == "train"]
            te = [it for it in lab if unseen(it) and it["split"] == "test"]
        note = f"utility classes {[spec.primary.classes[c] for c in held]} held out"
    elif which.startswith("cross_"):
        key = which[len("cross_"):]
        if key not in spec.groups:
            return dict(name=which, skip=f"no group '{key}'")
        values = sorted({str(it["groups"][key]) for it in lab})
        if len(values) < 2:
            return dict(name=which, skip=f"group '{key}' has one value")
        rng = np.random.default_rng(seed)
        held = set(rng.choice(values, max(1, len(values) // 3), replace=False).tolist())
        tr = [it for it in lab if str(it["groups"][key]) not in held]
        te = [it for it in lab if str(it["groups"][key]) in held]
        note = f"{key} values {sorted(held)} held out"
    else:
        raise ValueError(which)
    if secret.kind == "multiclass":
        seen_cls = {it["labels"][secret.name] for it in tr}
        dropped = sum(it["labels"][secret.name] not in seen_cls for it in te)
        te = [it for it in te if it["labels"][secret.name] in seen_cls]
        if dropped:
            note += f"; {dropped} test items of classes absent from training dropped"
        n_cls = len({it["labels"][secret.name] for it in te})
        if n_cls < 2:
            return dict(name=which, skip=f"{n_cls} secret classes left in test")
    if len(te) < min_test or len(tr) < min_test:
        return dict(name=which, skip=f"too few items ({len(tr)} train / {len(te)} test)")
    src_tr, src_te = {it["groups"]["source"] for it in tr}, {it["groups"]["source"] for it in te}
    if src_tr & src_te:  # cross protocols draw from both splits; never test on footage cut from a training recording
        te = [it for it in te if it["groups"]["source"] not in src_tr]
        note += f"; test items sharing a recording with training removed ({len(src_tr & src_te)} recordings)"
    return dict(name=which, train=tr, test=te, note=note)


def candidates(spec):
    """Protocols worth running for this dataset, headline first."""
    out = ["within"]
    if spec.primary.kind == "multiclass" and spec.primary.n >= 3:
        out += ["cross_utility", "hardest"]
    out += [f"cross_{g}" for g in spec.groups if g not in ("source", "subject", "identity")]
    return out
