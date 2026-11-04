import numpy as np, pandas as pd, time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(1)
d=pd.read_csv("/home/ubuntu/work/stress_probe/cxr/data/Data_Entry_2017_v2020.csv"); d.columns=[c.strip() for c in d.columns]
d=d.rename(columns={"OriginalImage[Width":"W","Height]":"H","OriginalImagePixelSpacing[x":"sx","y]":"sy","Patient Age":"age","Patient Gender":"sex","View Position":"view","Follow-up #":"fu","Patient ID":"pid","Image Index":"img"})
d=d[d.age<120].reset_index(drop=True)
te=set(open("/home/ubuntu/work/stress_probe/cxr/data/test_list.txt").read().split()); tv=set(open("/home/ubuntu/work/stress_probe/cxr/data/train_val_list.txt").read().split())
age=d.age.values.astype(float); sxn=(d.sex=="M").values.astype(int); W=d.W.values; H=d.H.values; spx=d.sx.values; vw=(d.view=="PA").values.astype(int); pid=d.pid.values
def sample(idx, n, matched):
    idx=np.asarray(idx); pp=pid[idx]
    order=np.argsort(pp,kind="stable"); sp=pp[order]; si=idx[order]
    starts=np.r_[0,np.where(sp[1:]!=sp[:-1])[0]+1]; ends=np.r_[starts[1:],len(sp)]
    multi=np.where(ends-starts>=2)[0]
    I=[];J=[]
    for g in rng.choice(multi,n):
        s,e=starts[g],ends[g]; a,b=rng.choice(np.arange(s,e),2,replace=False); I.append(si[a]); J.append(si[b])
    I=np.array(I);J=np.array(J)
    NI=[];NJ=[]
    for i in I:
        while True:
            c=idx[rng.integers(len(idx),size=300)]
            ok=(pid[c]!=pid[i])
            if matched: ok&=(sxn[c]==sxn[i])&(np.abs(age[c]-age[i])<=3)
            c=c[ok]
            if len(c): NI.append(i); NJ.append(c[0]); break
    if not matched:  # random negatives: random anchor too
        NI=idx[rng.integers(len(idx),size=n)]; NJ=[]
        for i in NI:
            while True:
                c=idx[rng.integers(len(idx))]
                if pid[c]!=pid[i]: NJ.append(c); break
    I2=np.r_[I,np.array(NI)]; J2=np.r_[J,np.array(NJ)]; y=np.r_[np.ones(n),np.zeros(n)]
    return I2,J2,y
def feat(I,J,asig,serr,dev,ae,se):
    cols=[np.abs(ae[I]-ae[J]),(se[I]==se[J]).astype(float)]
    if dev: cols+=[(W[I]==W[J]).astype(float),(H[I]==H[J]).astype(float),np.abs(spx[I]-spx[J]),(vw[I]==vw[J]).astype(float)]
    return np.c_[tuple(cols)]
tr_idx=np.where(d.img.isin(tv))[0]; te_idx=np.where(d.img.isin(te))[0]
print("train imgs",len(tr_idx),"test imgs",len(te_idx),flush=True)
for matched in [False,True]:
    t0=time.time(); Itr,Jtr,ytr=sample(tr_idx,4000,matched); Ite,Jte,yte=sample(te_idx,3000,matched); print("sampled",matched,round(time.time()-t0,1),flush=True)
    for (asig,serr,dev,label) in [(0,0,False,"true age+sex"),(5,0.03,False,"image-estimated age (sd5y~MAE4) + sex (97%)"),(8,0.05,False,"weaker estimator (sd8y, 95% sex)"),(0,0,True,"true age+sex+device size/spacing/view"),(0,0,"dev_only","device size/spacing/view only (no age, no sex)")]:
        ae=age+rng.normal(0,asig,len(age)); se=np.where(rng.random(len(sxn))<serr,1-sxn,sxn)
        if dev=="dev_only":
            f=lambda I,J: np.c_[(W[I]==W[J]).astype(float),(H[I]==H[J]).astype(float),np.abs(spx[I]-spx[J]),(vw[I]==vw[J]).astype(float)]
        else:
            f=lambda I,J: feat(I,J,asig,serr,dev,ae,se)
        m=GradientBoostingClassifier(n_estimators=120,max_depth=3).fit(f(Itr,Jtr),ytr)
        print("%-8s negatives | %-48s | test AUC %.3f"%("matched" if matched else "random",label,roc_auc_score(yte,m.predict_proba(f(Ite,Jte))[:,1])),flush=True)
