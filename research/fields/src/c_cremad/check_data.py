"""Dataset + split sanity checks (CPU only): counts, label priors, duplicate audio, near-duplicate clips across splits.
  python check_data.py --backbone wavlm-base-plus --out check.json
"""
import argparse, collections, json
import numpy as np
import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="wavlm-base-plus")
    ap.add_argument("--out", default="check.json")
    a = ap.parse_args()
    meta = data.load_meta(); sp = data.get_splits(meta)
    fs = data.FeatStore(a.backbone)
    out = {}
    out["n_clips"] = len(meta["ids"]); out["n_actors"] = len(set(meta["actor"].tolist()))
    out["emotions"] = dict(collections.Counter(meta["emo"].tolist()))
    out["sentences"] = len(set(meta["sent"].tolist()))
    out["emotion_prior_max"] = max(out["emotions"].values()) / out["n_clips"]
    per_actor = collections.Counter(meta["actor"].tolist())
    out["clips_per_actor"] = [min(per_actor.values()), max(per_actor.values())]
    out["most_frequent_actor_fraction"] = max(per_actor.values()) / out["n_clips"]
    out["ids_aligned"] = bool((fs.ids == meta["ids"]).all())
    sha = open(fs.dir / "wav_sha1.txt").read().split()
    dup = [v for v in collections.Counter(sha).values() if v > 1]
    out["duplicate_wav_files"] = int(sum(dup))
    dur = fs.lengths() * 0.02
    out["duration_s"] = dict(mean=float(dur.mean()), min=float(dur.min()), max=float(dur.max()))
    # split integrity
    folds = sp["folds"]
    te = [set(f["test"]) for f in folds]
    out["test_actors_disjoint_across_folds"] = all(not (te[i] & te[j]) for i in range(4) for j in range(i + 1, 4))
    out["test_union"] = len(set().union(*te))
    out["fold_train_val_test_disjoint"] = all(not (set(f["train"]) & set(f["val"])) and not (set(f["train"]) & set(f["test"])) and not (set(f["val"]) & set(f["test"])) for f in folds)
    m = data.attack_masks(meta, sp)
    out["attack_sets"] = {k: int(v.sum()) for k, v in m.items()}
    out["attack_train_vs_tests_overlap"] = int(sum((m["atk_train"] & m[t]).sum() for t in ("T1", "T2", "T3")))
    # near-duplicates: cosine similarity of layer-6 mean embeddings, test clips vs attacker-train clips of OTHER sentences
    S = fs.stats()[:, 6, 0, :]
    S = S - S.mean(0)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    tr = np.where(m["atk_train"])[0]
    for t in ("T1", "T3"):
        ti = np.where(m[t])[0]
        sim = S[ti] @ S[tr].T
        out[f"max_cos_{t}_to_atk_train"] = dict(mean_of_max=float(sim.max(1).mean()), global_max=float(sim.max()),
                                                 n_over_0999=int((sim.max(1) > 0.999).sum()))
    # same-actor vs different-actor similarity (identity signal in a trivial embedding)
    ti = np.where(m["T1"])[0]
    sim = S[ti] @ S[tr].T
    same = meta["actor"][ti][:, None] == meta["actor"][tr][None]
    out["mean_cos_same_actor"] = float(sim[same].mean()); out["mean_cos_diff_actor"] = float(sim[~same].mean())
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
