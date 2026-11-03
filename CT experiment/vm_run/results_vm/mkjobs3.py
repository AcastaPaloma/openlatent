jobs=[]
for sd in (0,1,2):
    for a,st in (("A0",1500),("B1",1500),("A3",1000),("A14",1000),("A19",1000),("A20",1000)): jobs.append((a,sd,st,"x"))
open("jobs14.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
