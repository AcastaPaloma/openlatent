import csv, sys, collections
d = sys.argv[1]
rows = list(csv.reader(open(d + '/Data_Entry_2017_v2020.csv')))
hdr, rows = rows[0], rows[1:]
print(hdr)
n_img = len(rows)
by = collections.defaultdict(list)
for r in rows:
    by[r[3]].append(r)
print('images', n_img, 'patients', len(by))
multi = {p for p, v in by.items() if len(v) >= 2}
print('multi>=2', len(multi), 'ge3', sum(len(v)>=3 for v in by.values()), 'ge5', sum(len(v)>=5 for v in by.values()), 'ge10', sum(len(v)>=10 for v in by.values()))
test = set(x.strip() for x in open(d + '/test_list.txt') if x.strip())
tv = set(x.strip() for x in open(d + '/train_val_list.txt') if x.strip())
print('test imgs', len(test), 'trainval imgs', len(tv), 'overlap', len(test & tv))
pt_test = collections.defaultdict(int)
for r in rows:
    if r[0] in test: pt_test[r[3]] += 1
print('test patients', len(pt_test), 'test multi', sum(v>=2 for v in pt_test.values()))
pt_tv = {r[3] for r in rows if r[0] in tv}
print('patient overlap train_val/test', len(pt_tv & set(pt_test)))
# followup index gaps / same-day? follow-up # is index only
# patients with PA and AP both
mixed = sum(1 for p in multi if len({r[6] for r in by[p]})>1)
print('multi patients with mixed views', mixed, 'of', len(multi))
# age anomalies
ages = [int(r[4]) for r in rows]
print('age>100', sum(a>100 for a in ages), 'max', max(ages))
# within patient age non-monotone or sex change
sexchg = sum(1 for p, v in by.items() if len({r[5] for r in v})>1)
print('patients with >1 sex value', sexchg)
# device dims: share of multi patients whose all images share same width/height
same_dim = sum(1 for p in multi if len({(r[7], r[8]) for r in by[p]})==1)
print('multi patients with constant (w,h)', same_dim)
# number of distinct (w,h,spacing) combos
combos = collections.Counter((r[7], r[8], r[9]) for r in rows)
print('distinct (w,h,spacing) combos', len(combos), 'top5', combos.most_common(5))
# followup number distribution for test multi: consecutive within 1 day? unknown
# Patient ID range vs follow up
mx = max(int(r[2]) for r in rows); print('max followup', mx)
