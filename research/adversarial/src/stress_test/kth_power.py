import json, numpy as np, os, sys
from scipy.stats import beta
rng=np.random.default_rng(0)
ROOT=os.path.expanduser("~/work/stress_probe/h/harness")
def load(ds,name,fam="within"):
    d=os.path.join(ROOT,ds,"eval",name)
    z=np.load(os.path.join(d,"preds.npz"),allow_pickle=False)
    r=json.load(open(os.path.join(d,"result.json")))
    ids=r["secrets"]["identity"][fam]["test_ids"]
    y=np.array([int(i.split("_")[0].replace("person",""))-1 for i in ids])
    rec=np.array([i.rsplit("_",1)[0] for i in ids])
    out={}
    for k in z.files:
        p=k.split("|")
        if len(p)==5 and p[1]=="identity" and p[2]==fam and p[3]=="retrained":
            P=z[k]; out[p[4]]=(P.argmax(1)==y).astype(float)
    return out,y,rec
def cluster_boot(c,rec,B=2000):
    ur,inv=np.unique(rec,return_inverse=True)
    s=np.bincount(inv,weights=c); n=np.bincount(inv)
    m=len(ur); res=[]
    for _ in range(B):
        idx=rng.integers(0,m,m); res.append(s[idx].sum()/n[idx].sum())
    return np.array(res)
def paired_boot(c1,c2,rec,B=2000):
    ur,inv=np.unique(rec,return_inverse=True)
    d=c1-c2
    s=np.bincount(inv,weights=d); n=np.bincount(inv); m=len(ur); res=[]
    for _ in range(B):
        idx=rng.integers(0,m,m); res.append(s[idx].sum()/n[idx].sum())
    return np.array(res)
for ds in ["kth_ilic","kth_scenario"]:
    print("=====",ds)
    names=sorted(os.listdir(os.path.join(ROOT,ds,"eval")))
    rows={}
    for nm in names:
        try: c,y,rec=load(ds,nm)
        except Exception as e: print("skip",nm,e); continue
        rows[nm]=(c,y,rec)
    nm0=names[0]; _,y,rec=rows[nm0]
    n=len(y); nrec=len(set(rec)); nper=len(set(y))
    print("n_clips",n,"n_recordings",nrec,"clips/rec",n/nrec,"n_persons",nper,"recordings/person",nrec/nper)
    print("%-26s %6s %6s | %7s %7s | %9s %9s %5s"%("release","fr50","r3d","best","selSE","clipSE","clusSE","DEFF"))
    rows={k:v for k,v in rows.items() if len(v[0])>0}
    for nm,(c,y,rec) in rows.items():
        accs={a:v.mean() for a,v in c.items()}
        best=max(accs.values())
        # attacker with best -> cluster SE
        a=max(accs,key=accs.get); v=c[a]
        se_b=np.sqrt(best*(1-best)/n); bs=cluster_boot(v,rec,1000); se_c=bs.std()
        print("%-26s %6.3f %6.3f | %7.3f %7s | %9.4f %9.4f %5.1f"%(nm,accs.get("frame_r50",np.nan),accs.get("r3d",accs.get("dino",np.nan)),best,"-",se_b,se_c,(se_c/se_b)**2 if se_b>0 else np.nan))
    # paired comparisons of best-of-2
    def best_vec(nm):
        c=rows[nm][0]; a=max(c,key=lambda k:c[k].mean()); return c[a]
    pairs=[("pose_canon_down16","bdq_down64"),("pose_canon_down16","pose_canon_down32"),("pose_canon_down16","pose_canon_down8"),("pose_canon_down16","motion_down32"),("pose_canon_down16","pose_canon_down16_s1"),("pose_canon_down16_s1","pose_canon_down16_s2"),("pose_canon_down16","pose_canon_down16_e24"),("bdq_down64","bdq_down32")]
    print("paired difference of best-of-2 attackers, recording-cluster bootstrap")
    for a,b in pairs:
        if a in rows and b in rows:
            va,vb=best_vec(a),best_vec(b); d=va.mean()-vb.mean()
            bs=paired_boot(va,vb,rows[a][2]); se=bs.std()
            print("  %-22s - %-22s diff %+.3f  SE %.4f  MDE80(2.8SE) %.3f  CI[%+.3f,%+.3f]"%(a,b,d,se,2.8*se,np.percentile(bs,2.5),np.percentile(bs,97.5)))
    # winner curse: split recordings in halves; pick best of 2 on half A, evaluate on B; compare naive
    ur=np.unique(rows[nm0][2])
    gaps=[]
    for nm,(c,y,rec) in rows.items():
        if len(c)<2: continue
        ur=np.unique(rec); bias=[]
        for _ in range(300):
            half=set(rng.choice(ur,len(ur)//2,replace=False)); mA=np.array([r in half for r in rec])
            accA={a:v[mA].mean() for a,v in c.items()}; accB={a:v[~mA].mean() for a,v in c.items()}
            pick=max(accA,key=accA.get)
            bias.append(accA[pick]-accB[pick])  # optimism of in-sample max vs held-out
        gaps.append((nm,np.mean(bias)))
    print("mean in-sample optimism of best-of-2 (half-split), points:",np.round(np.mean([g for _,g in gaps])*100,2),"max",np.round(max(g for _,g in gaps)*100,2))
    # seed spread for pose_canon_down16
    sp=[rows[k][0]["r3d"].mean() for k in ["pose_canon_down16","pose_canon_down16_s1","pose_canon_down16_s2"] if k in rows]
    sp2=[max(rows[k][0].values(),key=lambda v:v.mean()).mean() for k in ["pose_canon_down16","pose_canon_down16_s1","pose_canon_down16_s2"] if k in rows]
    print("pose_canon_down16 best-of-2 over seeds",np.round(sp2,3),"sd",np.std(sp2,ddof=1) if len(sp2)>1 else None)
