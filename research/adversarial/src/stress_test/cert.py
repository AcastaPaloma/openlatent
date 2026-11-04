import numpy as np
from scipy.stats import beta, norm
def cp_ucb(k,n,a):  # one-sided upper bound at level 1-a
    return 1.0 if k>=n else beta.ppf(1-a,k+1,n-k)
print("Certificate upper bound (Clopper-Pearson, one-sided 95%, Bonferroni /K) when the sealed test shows the observed rate p_hat")
print("columns: n_units, K, p_hat -> UCB ; 'resolvable above chance' = UCB - chance")
for chance,label in [(0.04,"identity 25-way (chance 4%)"),(0.011,"identity 90-way (chance 1.1%)"),(0.5,"binary attribute (chance 50%)")]:
    print("\n",label)
    for n in [24,51,120,150,360,1424,4707]:
        row=[]
        for K in [1,10,30]:
            k=int(round(chance*n)); u=cp_ucb(k,n,0.05/K); row.append(f"K={K}: UCB {u:.3f} (+{(u-chance)*100:.1f}pt)")
        print(f"  n={n:5d}  "+" | ".join(row))
# MDE80 for paired difference of two proportions with discordance: SE ~ sqrt((p1(1-p1)+p2(1-p2)-2 rho sqrt(..))/n)
print("\nMDE80 (2.8*SE) for difference of two defenders' accuracy at p~0.12, unit-level n, rho=0.5 error correlation:")
for n in [24,51,120,150,360,1424,4707]:
    p=0.12; se=np.sqrt(2*p*(1-p)*(1-0.5)/n); print(f"  n={n:5d} SE={se*100:.2f}pt MDE80={2.8*se*100:.1f}pt")
# AUC SE Hanley-McNeil
def hm(A,n1,n2):
    Q1=A/(2-A); Q2=2*A*A/(1+A)
    return np.sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n2-1)*(Q2-A*A))/(n1*n2))
print("\nCXR verification AUC SE (Hanley-McNeil, units=patients with same-patient pair and diff-patient pair), A=0.8:")
for n in [1424,4707,7588,13302]:
    s=hm(0.8,n,n); print(f"  patients={n:6d} SE={s:.4f} unpaired MDE80={2.8*s*np.sqrt(2):.3f}  paired(rho=.6) MDE80={2.8*s*np.sqrt(2*(1-.6)):.3f}")
# Rank-1 
print("\nCXR Rank-1 (p=0.25) SE with n query patients:")
for n in [1424,4707]:
    se=np.sqrt(.25*.75/n); print(f"  n={n} SE={se*100:.2f}pt, paired MDE80 (rho .5) {2.8*np.sqrt(2*.25*.75*.5/n)*100:.1f}pt")
# subject-level attribute
print("\nsubject-level attribute (balanced acc ~0.5 null), SE=0.5/sqrt(n):")
for n in [9,24,30,51]:
    se=0.5/np.sqrt(n); print(f"  subjects={n}: SE={se:.3f}, one-arm MDE80 over chance = {2.8*se:.3f}; paired diff (rho .5) MDE80 = {2.8*0.5/np.sqrt(n)*np.sqrt(2*0.5):.3f}")
