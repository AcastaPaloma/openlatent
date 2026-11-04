import os, sys, glob, numpy as np, csv
from sklearn.ensemble import RandomForestClassifier
D = sys.argv[1]
TRIALS = {'dws': [1, 2, 11], 'ups': [3, 4, 12], 'wlk': [7, 8, 15], 'jog': [9, 16], 'std': [6, 14], 'sit': [5, 13]}
LAST = {'dws': 11, 'ups': 12, 'wlk': 15, 'jog': 16, 'std': 14, 'sit': 13}
cols = None
data = {}
for act, trs in TRIALS.items():
    for t in trs:
        for s in range(1, 25):
            fn = f'{D}/{act}_{t}/sub_{s}.csv'
            rows = list(csv.reader(open(fn)))
            hdr, arr = rows[0], np.array(rows[1:], dtype=float)
            if cols is None: cols = hdr[1:]
            data[(act, t, s)] = arr[:, 1:]
print(cols)
idx = {c: i for i, c in enumerate(cols)}
def sel(names): return [idx[n] for n in names]
ROT = ['rotationRate.x', 'rotationRate.y', 'rotationRate.z']
ACC = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']
ALL12 = cols
def windows(x, W=128, S=64):
    return [x[i:i + W] for i in range(0, len(x) - W + 1, S)]
def feats(w, demean=True, mag=False, keepmean=False):
    if mag:
        w = np.linalg.norm(w, axis=1, keepdims=True)
    mu = w.mean(0)
    wc = w - mu if demean else w
    sd = wc.std(0)
    mad = np.abs(np.diff(wc, axis=0)).mean(0)
    pc = np.percentile(wc, [10, 25, 50, 75, 90], axis=0).ravel()
    F = np.abs(np.fft.rfft(wc - wc.mean(0), axis=0))[1:17]
    f = [sd, mad, pc, F.ravel()]
    if keepmean: f.append(mu)
    return np.concatenate(f)
def build(chs, acts, **kw):
    ci = sel(chs)
    Xtr, ytr, Xte, yte, tid = [], [], [], [], []
    for (act, t, s), arr in data.items():
        if act not in acts: continue
        ws = windows(arr[:, ci])
        for w in ws:
            f = feats(w, **kw)
            if t == LAST[act]:
                Xte.append(f); yte.append(s - 1); tid.append((act, t, s))
            else:
                Xtr.append(f); ytr.append(s - 1)
    return np.array(Xtr), np.array(ytr), np.array(Xte), np.array(yte), tid
def run(name, chs, acts, seed=0, **kw):
    Xtr, ytr, Xte, yte, tid = build(chs, acts, **kw)
    rf = RandomForestClassifier(300, n_jobs=8, random_state=seed).fit(Xtr, ytr)
    P = rf.predict_proba(Xte)
    # trial score = mean log prob
    trials = {}
    for p, y, t in zip(P, yte, tid):
        trials.setdefault(t, []).append(p)
    ok = []
    for t, ps in trials.items():
        m = np.log(np.array(ps) + 1e-3).mean(0)
        ok.append(int(np.argmax(m) == t[2] - 1))
    win = (P.argmax(1) == yte).mean()
    print(f'{name:55s} trials={len(ok):3d} trial-acc={np.mean(ok):.3f} win-acc={win:.3f}  ntrain={len(Xtr)}', flush=True)
ALLACT = list(TRIALS)
run('A all12 incl means, all acts', ALL12, ALLACT, keepmean=True, demean=True)
run('B rot+acc demeaned, all acts', ROT + ACC, ALLACT)
run('C acc magnitude only, all acts', ACC, ALLACT, mag=True)
run('D static only (sit,std): all12 incl means', ALL12, ['sit', 'std'], keepmean=True)
run('E static only: attitude+gravity means only', ['attitude.roll','attitude.pitch','attitude.yaw','gravity.x','gravity.y','gravity.z'], ['sit','std'], keepmean=True)
run('F static only: rot+acc demeaned', ROT + ACC, ['sit', 'std'])
run('G dynamic only: rot+acc demeaned', ROT + ACC, ['dws','ups','wlk','jog'])
