"""Synthetic stand-in for the feature cache (dev/smoke only): actor + emotion + sentence structure, small dims.
  CREMAD_FEATS=<dir> python synth_feats.py   -> <dir>/synthetic/{frames,offsets,stats,clip_ids,info}
"""
import json, numpy as np
import data

def main(dim=64, nl=4):
    meta = data.load_meta()
    rs = np.random.RandomState(0)
    actors = sorted(set(meta["actor"].tolist()))
    A = {a: rs.randn(dim) for a in actors}
    E = {e: rs.randn(dim) for e in data.EMOTIONS}
    S = {s: rs.randn(dim) for s in data.SENTENCES}
    frames, offs, stats = [], [0], []
    for i in range(len(meta["ids"])):
        T = rs.randint(20, 40)
        base = 0.8 * A[meta["actor"][i]] + 0.7 * E[meta["emo"][i]] + 0.5 * S[meta["sent"][i]]
        x = base[None, None, :] + rs.randn(T, nl, dim) * 1.0 + np.sin(np.arange(T))[:, None, None] * 0.3
        frames.append(x.astype(np.float16)); offs.append(offs[-1] + T)
        st = np.stack([x.mean(0), x.std(0)], 1)  # [nl,2,dim]
        stats.append(np.concatenate([st] * 3 + [st[:1]], 0)[:13])
    d = data.FEAT_ROOT / "synthetic"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.npy", np.concatenate(frames, 0)); np.save(d / "offsets.npy", np.array(offs))
    np.save(d / "stats.npy", np.stack(stats, 0).astype(np.float32))
    (d / "clip_ids.txt").write_text("\n".join(meta["ids"]) + "\n")
    (d / "info.json").write_text(json.dumps(dict(layers=[3, 6, 9, 12], n_clips=len(meta["ids"]), dim=dim)))

if __name__ == "__main__":
    main()
