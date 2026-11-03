import os
jobs=[]
for a in ["A1","A2","A5","A7"]:
    for st in (1000,1500):
        for sd in (0,1,2): jobs.append((a,sd,st,"x"))
for a in ["B1","B2"]:
    for sd in (0,1,2): jobs.append((a,sd,1500,"x"))
for sg in ("0.5","1","1.5"):
    for sd in (0,1): jobs.append(("B3",sd,1500,sg))
for sd in (0,1,2): jobs.append(("A8",sd,1000,"x"))
for sd in (0,1,2): jobs.append(("A15",sd,1000,"x"))
for sd in (0,1,2): jobs.append(("A0",sd,1500,"x"))
def out(j):
    a,sd,st,sg=j; name=a+("_sigma"+sg if a=="B3" else "")
    return f"runs/arm_{name}_s{sd}_n5094"+("_st%d"%st if st not in (1500,3000) else "")+".json"
todo=[j for j in jobs if not os.path.exists(out(j))]
open("jobs13.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in todo)+"\n"); print(len(jobs),"->",len(todo),"to run")
