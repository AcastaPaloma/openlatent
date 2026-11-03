import json,glob,numpy as np
def load(a,tag):
    L=[json.load(open(f))["utility_per_label"] for f in glob.glob(f"runs/arm_{a}_s*_n5094{tag}.json") if not f.endswith("_st5000.json")]
    return {k:np.mean([l[k] for l in L if k in l]) for k in L[0]}
raw=load("A0","_attx"); a3=load("A3","_attx"); a14=load("A14","_attx"); b3=load("B3_sigma1","_attx")
print("label".ljust(36),"raw   A3    A14   noise1")
for k in sorted(raw,key=lambda k:-raw[k]): print(k[:36].ljust(36),"%.3f %.3f %.3f %.3f"%(raw[k],a3[k],a14[k],b3[k]))
print("labels where A3 loses >0.03:", [k[:20] for k in raw if raw[k]-a3[k]>0.03])
print("mean change A3 %.3f A14 %.3f noise %.3f"%tuple(np.mean([x[k]-raw[k] for k in raw]) for x in (a3,a14,b3)))
