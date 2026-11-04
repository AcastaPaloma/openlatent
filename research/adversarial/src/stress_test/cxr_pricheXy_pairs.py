import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
d=pd.read_csv("cxr/data/Data_Entry_2017_v2020.csv"); d.columns=[c.strip() for c in d.columns]
d=d.rename(columns={"OriginalImage[Width":"W","Height]":"H","OriginalImagePixelSpacing[x":"sx","Patient Age":"age","Patient Gender":"sex","View Position":"view","Follow-up #":"fu","Patient ID":"pid","Image Index":"img"}).set_index("img")
def load(p):
    a=np.loadtxt(p,dtype=str); return a[:,0],a[:,1],a[:,2].astype(float)
def F(a,b,estimated=False):
    A=d.loc[a]; B=d.loc[b]
    age_a=A.age.values.astype(float); age_b=B.age.values.astype(float); sa=(A.sex=="M").values; sb=(B.sex=="M").values
    if estimated:
        age_a=age_a+rng.normal(0,5,len(a)); age_b=age_b+rng.normal(0,5,len(b))
        sa=np.where(rng.random(len(a))<.03,~sa,sa); sb=np.where(rng.random(len(b))<.03,~sb,sb)
    return np.c_[np.abs(age_a-age_b),(sa==sb).astype(float),(A.W.values==B.W.values).astype(float),(A.H.values==B.H.values).astype(float),np.abs(A.sx.values-B.sx.values),(A.view.values==B.view.values).astype(float)]
a,b,y=load("pairs_train.txt"); ta,tb,ty=load("pairs_test.txt")
print("train pairs",len(y),"pos frac",y.mean(),"test pairs",len(ty),"pos frac",ty.mean())
pa=d.loc[ta,"pid"].values; pb=d.loc[tb,"pid"].values
print("test: unique patients appearing", len(set(pa)|set(pb)), " same-pid among label1:",(pa==pb)[ty==1].mean()," among label0:",(pa==pb)[ty==0].mean())
print("test patients with >=2 images in list used by positives:", len(set(pa[ty==1])))
for name,cols,est in [("age+sex true",[0,1],False),("age+sex image-estimated (sd5y,97%)",[0,1],True),("device only (size/spacing/view)",[2,3,4,5],False),("all true metadata",[0,1,2,3,4,5],False),("device + estimated age/sex",[0,1,2,3,4,5],True)]:
    Xtr=F(a,b,est); Xte=F(ta,tb,est)
    m=GradientBoostingClassifier(n_estimators=120,max_depth=3).fit(Xtr[:,cols],y)
    print("%-38s AUC on PriCheXy test pairs: %.3f"%(name,roc_auc_score(ty,m.predict_proba(Xte[:,cols])[:,1])))
# also: view-match and sex-match rates by label
A=d.loc[ta]; B=d.loc[tb]
for lab in [1,0]:
    m=ty==lab; print("label",lab,"same sex %.3f"%((A.sex.values==B.sex.values)[m]).mean(),"same view %.3f"%((A.view.values==B.view.values)[m]).mean(),"same W,H %.3f"%(((A.W.values==B.W.values)&(A.H.values==B.H.values))[m]).mean(),"|dAge| median %.1f"%np.median(np.abs(A.age.values-B.age.values)[m]))
