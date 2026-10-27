
import os, hashlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D="/content/drive/MyDrive/openlatent_ct"
h=lambda p:int(hashlib.sha256(f"ctrate-openlatent-v1:{p}".encode()).hexdigest()[:8],16)/16**8
man=pd.read_csv(f"{D}/manifest_train.csv"); have={f[:-4] for f in os.listdir(f"{D}/feats")}
man=man[man.name.map(lambda n:n[:-4] in have)].reset_index(drop=True)
lab=pd.read_csv(f"{D}/train_predicted_labels.csv").set_index("VolumeName"); Y=lab.reindex(man.name).values
G=[];S=[]
for n in man.name:
    z=np.load(f"{D}/feats/{n[:-4]}.npz"); G.append(z["g"]); S.append(z["s"].astype(np.float32))
G,S=np.stack(G),np.stack(S); fit=man.patient.map(lambda p:h(p)<0.5).values; ev=~fit
print(len(man),"scans | fit",fit.sum(),"eval",ev.sum(),flush=True)
def run(name,X,C,k=None,white=False):
    mu,sd=X[fit].mean(0),X[fit].std(0)+1e-6; Z=(X-mu)/sd
    if k: Z=PCA(min(k,fit.sum()-1),whiten=white,random_state=0).fit(Z[fit]).transform(Z)
    a=[]
    for j in range(Y.shape[1]):
        y=Y[:,j]
        if len(set(y[fit]))<2 or len(set(y[ev]))<2: continue
        a.append(roc_auc_score(y[ev],LogisticRegression(C=C,max_iter=3000).fit(Z[fit],y[fit]).decision_function(Z[ev])))
    print(f"{name:34s} C={C:<7} mean AUC {np.mean(a):.3f}  best {np.max(a):.3f} worst {np.min(a):.3f}",flush=True)
for C in (0.0005,0.005,0.05): run("global512 standardized",G,C)
for C in (0.0001,0.001,0.01): run("spatial4096 standardized",S,C)
for C in (0.001,0.01): run("spatial4096 PCA128 unwhitened",S,C,k=128)
