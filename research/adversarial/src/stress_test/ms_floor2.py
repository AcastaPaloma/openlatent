import numpy as np, glob, os, re, collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
rng=np.random.default_rng(0)
R=os.path.expanduser("~/work/stress_probe/imu/ms_x/A_DeviceMotion_data")
cols=None; trials=[]  # (subject, activity, trialid, array)
for d in sorted(glob.glob(R+"/*_*")):
    nm=os.path.basename(d); act,tid=nm.split("_"); tid=int(tid)
    for f in glob.glob(d+"/sub_*.csv"):
        s=int(os.path.basename(f).split("_")[1].split(".")[0])
        a=np.genfromtxt(f,delimiter=",",skip_header=1)[:,1:]
        trials.append((s,act,tid,a))
names=["roll","pitch","yaw","grav_x","grav_y","grav_z","rot_x","rot_y","rot_z","acc_x","acc_y","acc_z"]
print("trials",len(trials),"subjects",len({t[0] for t in trials}))
# hold out the LAST trial id per (subject, activity)
key=collections.defaultdict(list)
for t in trials: key[(t[0],t[1])].append(t[2])
test_ids={(s,a):max(v) for (s,a),v in key.items()}
W=128; ST=64
def feat(x,mode):
    # x: (128,12)
    if mode=="acc_rot_only": x=x[:,6:12]
    if mode=="acc_only": x=x[:,9:12]
    if mode=="no_attitude_gravity": x=x[:,6:12]
    if mode=="acc_magnitude": x=np.linalg.norm(x[:,9:12],axis=1,keepdims=True)
    parts=[]
    keep_mean = mode in ("all",)
    if keep_mean: parts.append(x.mean(0))
    xc=x-x.mean(0)
    parts+= [xc.std(0), np.abs(np.diff(xc,axis=0)).mean(0), np.percentile(xc,[10,50,90],axis=0).ravel()]
    F=np.abs(np.fft.rfft(xc,axis=0))[1:17]; parts.append(F.reshape(-1)[::1])
    return np.concatenate(parts)
def make(mode):
    Xtr=[];ytr=[];Xte=[];yte=[];gte=[]
    for (s,a,tid,arr) in trials:
        is_te = tid==test_ids[(s,a)]
        for st in range(0,len(arr)-W,ST):
            f=feat(arr[st:st+W],mode)
            if is_te: Xte.append(f); yte.append(s-1); gte.append((s,a,tid))
            else: Xtr.append(f); ytr.append(s-1)
    return np.array(Xtr),np.array(ytr),np.array(Xte),np.array(yte),gte
def trial_acc(P,yte,gte,classes):
    from collections import defaultdict
    g=defaultdict(list)
    for p,y,k in zip(P,yte,gte): g[k].append((p,y))
    ok=[]
    for k,v in g.items():
        pm=np.mean([x[0] for x in v],0); ok.append(float(classes[pm.argmax()]==v[0][1]))
    return np.mean(ok),len(ok)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
ACT=sorted({t[1] for t in trials}); amap={a:i for i,a in enumerate(ACT)}
def make2(mode):
    Xtr=[];ytr=[];atr=[];Xte=[];yte=[];ate=[];gte=[]
    for (s_,a,tid,arr) in trials:
        is_te = tid==test_ids[(s_,a)]
        for st in range(0,len(arr)-W,ST):
            f=feat(arr[st:st+W],mode)
            if is_te: Xte.append(f); yte.append(s_-1); ate.append(amap[a]); gte.append((s_,a,tid))
            else: Xtr.append(f); ytr.append(s_-1); atr.append(amap[a])
    return np.array(Xtr),np.array(ytr),np.array(atr),np.array(Xte),np.array(yte),np.array(ate),gte
Xtr,ytr,atr,Xte,yte,ate,gte=make2("no_attitude_gravity")
sc=StandardScaler().fit(Xtr); a,b=sc.transform(Xtr),sc.transform(Xte)
lda=LDA(solver="svd").fit(a,atr)
print("activity accuracy (window) raw-feature LDA:",(lda.predict(b)==ate).mean())
for k in [5,3,2,1]:
    Ztr=lda.transform(a)[:,:k]; Zte=lda.transform(b)[:,:k]
    res=[]
    for nm,m in [("LR",LogisticRegression(max_iter=500)),("RF",RandomForestClassifier(300,n_jobs=32,random_state=0)),("kNN",__import__("sklearn.neighbors",fromlist=["x"]).KNeighborsClassifier(15))]:
        m.fit(Ztr,ytr); P=m.predict_proba(Zte); w=(m.classes_[P.argmax(1)]==yte).mean(); t,n=trial_acc(P,yte,gte,m.classes_); res.append(f"{nm}: window {w:.3f} trial {t:.3f}")
    # activity from k-dim code
    ma=RandomForestClassifier(200,n_jobs=32,random_state=0).fit(Ztr,atr); aacc=(ma.predict(Zte)==ate).mean()
    print(f"activity-LDA code dim {k}: activity acc {aacc:.3f} | identity "+" | ".join(res))
# downsample / low-pass floor: average-pool x4 of raw channels then features
def feat_ds(x,fac): 
    x=x[:,6:12]; n=(len(x)//fac)*fac; x=x[:n].reshape(-1,fac,x.shape[1]).mean(1); xc=x-x.mean(0)
    return np.concatenate([xc.std(0),np.abs(np.diff(xc,axis=0)).mean(0),np.percentile(xc,[10,50,90],axis=0).ravel(),np.abs(np.fft.rfft(xc,axis=0))[1:9].reshape(-1)])
for fac in [4,8,16]:
    Xtr=[];ytr=[];Xte=[];yte=[];gte=[]
    for (s_,a,tid,arr) in trials:
        is_te = tid==test_ids[(s_,a)]
        for st in range(0,len(arr)-W,ST):
            f=feat_ds(arr[st:st+W],fac)
            if is_te: Xte.append(f); yte.append(s_-1); gte.append((s_,a,tid))
            else: Xtr.append(f); ytr.append(s_-1)
    Xtr,Xte=np.array(Xtr),np.array(Xte); sc=StandardScaler().fit(Xtr)
    m=RandomForestClassifier(300,n_jobs=32,random_state=0).fit(sc.transform(Xtr),ytr); P=m.predict_proba(sc.transform(Xte)); t,n=trial_acc(P,np.array(yte),gte,m.classes_)
    print(f"downsample x{fac} (50Hz->{50//fac}Hz) rot+acc: RF identity trial {t:.3f} window {(m.classes_[P.argmax(1)]==np.array(yte)).mean():.3f}")
