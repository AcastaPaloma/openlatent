import json,glob,numpy as np,collections
d=collections.defaultdict(list)
for f in glob.glob("runs/arm_*_seal2*.json"):
    o=json.load(open(f)); d[o["arm"]+"@"+("1000" if f.endswith("_st1000.json") else "1500")].append(o)
raw=d["A0@1500"]; ru=np.mean([o["utility_auc_mean"] for o in raw]); rt=np.mean([o["best_top1"] for o in raw]); rm=np.mean([o["utility_mlp_auc"] for o in raw])
o0=raw[0]; print("n_gallery",o0["n_gallery"],"n_query",o0["n_query"],"chance %.4f"%o0["chance_top1"],"floor %.3f"%o0["metadata_floor_top1"],"raw top1 %.3f util %.3f mlp %.3f"%(rt,ru,rm))
print("%-12s n  top1+-sd   CI(first seed)      leak_removed  util  kept   utilCI(seed0)   mlp   keptMLP  inv   sex"%"arm")
for k in sorted(d,key=lambda k:np.mean([o["best_top1"] for o in d[k]])):
    v=d[k]; t=np.array([o["best_top1"] for o in v]); u=np.mean([o["utility_auc_mean"] for o in v]); m=np.mean([o["utility_mlp_auc"] for o in v])
    ci=v[0]["best_top1_ci95"]; uc=v[0]["utility_ci95"]
    print("%-12s %d  %.3f+-%.3f  [%.3f,%.3f]  %5.2f   %.3f  %.2f  [%.3f,%.3f]  %.3f  %.2f  %.3f  %.3f"%(k,len(v),t.mean(),t.std(),ci[0],ci[1],(rt-t.mean())/(rt-v[0]["chance_top1"]),u,(u-.5)/(ru-.5),uc[0],uc[1],m,(m-.5)/(rm-.5),np.mean([o["inversion_top1"] for o in v]),np.mean([o["sex_auc"] for o in v])))
a3=d["A3@1000"]; print("A3 attackers:",{n:round(np.mean([o["attackers"][n]["top1"] for o in a3]),3) for n in a3[0]["attackers"]})
print("raw attackers:",{n:round(np.mean([o["attackers"][n]["top1"] for o in raw]),3) for n in raw[0]["attackers"]})
