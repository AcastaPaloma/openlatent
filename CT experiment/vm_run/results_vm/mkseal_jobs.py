jobs=[]
spec=[("A0",1500,"x"),("B1",1500,"x"),("B2",1500,"x"),("B3",1500,"1"),("B3",1500,"1.5"),("A1",1500,"x"),("A2",1500,"x"),("A5",1500,"x"),("A7",1000,"x"),("A3",1000,"x"),("A14",1000,"x"),("A8",1000,"x"),("K0",1500,"x")]
for sd in (0,1,2):
    for a,st,sg in spec: jobs.append((a,sd,st,sg))
open("jobs_seal2.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
