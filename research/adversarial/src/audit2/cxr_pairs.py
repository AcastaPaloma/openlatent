import csv, sys, numpy as np, collections
d = sys.argv[1]
rows = list(csv.reader(open(d + '/Data_Entry_2017_v2020.csv')))[1:]
meta = {r[0]: dict(pid=r[3], fu=int(r[2]), age=int(r[4]), sex=r[5], view=r[6], w=r[7], h=r[8], sp=r[9]) for r in rows}
pairs = [l.strip().split('\t') for l in open(d + '/pairs5000.txt') if l.strip()]
y = np.array([float(p[2]) for p in pairs])
print('pairs', len(pairs), 'pos', int(y.sum()))
a = [meta[p[0]] for p in pairs]; b = [meta[p[1]] for p in pairs]
pids = set(m['pid'] for m in a) | set(m['pid'] for m in b)
print('distinct patients', len(pids))
pos_pids = set(m['pid'] for m, yy in zip(a, y) if yy == 1)
print('patients behind positives', len(pos_pids))
agediff = np.array([abs(x['age'] - z['age']) for x, z in zip(a, b)])
samesex = np.array([x['sex'] == z['sex'] for x, z in zip(a, b)])
sameview = np.array([x['view'] == z['view'] for x, z in zip(a, b)])
samedim = np.array([(x['w'], x['h']) == (z['w'], z['h']) for x, z in zip(a, b)])
samesp = np.array([(x['sp']) == (z['sp']) for x, z in zip(a, b)])
def auc(score, y):
    order = np.argsort(score, kind='mergesort'); s = score[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[order[i:j+1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print('AUC -|age diff| only', round(auc(-agediff.astype(float), y), 4))
print('AUC samesex only', round(auc(samesex.astype(float), y), 4))
print('AUC -|agediff| + 100*samesex', round(auc(-agediff + 100 * samesex, y), 4))
print('AUC samedim', round(auc(samedim.astype(float), y), 4), 'samesp', round(auc(samesp.astype(float), y), 4), 'sameview', round(auc(sameview.astype(float), y), 4))
print('AUC -agediff+100*samesex+5*samedim+3*sameview', round(auc(-agediff + 100 * samesex + 5 * samedim + 3 * sameview, y), 4))
print('pos: same sex rate', samesex[y==1].mean(), 'neg', samesex[y==0].mean(), 'median age diff pos', np.median(agediff[y==1]), 'neg', np.median(agediff[y==0]))
# are all test-pair images in official test list?
tl = set(x.strip() for x in open(d + '/test_list.txt'))
allimgs = set(p[0] for p in pairs) | set(p[1] for p in pairs)
print('pair images', len(allimgs), 'in HF official test_list', len(allimgs & tl))
pc = set(x.strip() for x in open(d + '/pc_test_list.txt'))
print('PriCheXy test_list == HF test_list', pc == tl, len(pc), len(tl))
# matched negatives feasibility: same sex & |age diff|<=3, same dims
neg = [(x, z) for x, z, yy in zip(a, b, y) if yy == 0]
print('neg fraction matched (same sex & age<=3)', np.mean([x['sex']==z['sex'] and abs(x['age']-z['age'])<=3 for x, z in neg]))
pos_ = [(x, z) for x, z, yy in zip(a, b, y) if yy == 1]
print('pos fu gap median', np.median([abs(x['fu']-z['fu']) for x, z in pos_]), 'frac same view', np.mean([x['view']==z['view'] for x,z in pos_]), 'frac same dims', np.mean([(x['w'],x['h'])==(z['w'],z['h']) for x,z in pos_]))
print('pos with age diff 0', np.mean([x['age']==z['age'] for x,z in pos_]), 'pos with age diff<=1', np.mean([abs(x['age']-z['age'])<=1 for x,z in pos_]))
# fu gap distribution
g = np.array([abs(x['fu']-z['fu']) for x,z in pos_]); print('fu gap quantiles', np.percentile(g,[10,25,50,75,90]))
