jobs=[]
for a in ["A1","A2","A5","A7"]:
    for st in (1000,1500):
        for sd in (0,1,2): jobs.append((a,sd,st,""))
for a in ["B1","B2"]:
    for sd in (0,1,2): jobs.append((a,sd,1500,""))
for sg in ("0.5","1","1.5"):
    for sd in (0,1): jobs.append(("B3",sd,1500,sg))
for sd in (0,1,2): jobs.append(("A8",sd,1000,""))
for sd in (0,1): jobs.append(("A15",sd,1000,""))
for sd in (1,2): jobs.append(("A0",sd,1500,""))
jobs=[j for j in jobs if not (j[0]=="A1" and j[1]==0 and j[2]==1500)]
open("jobs13.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
