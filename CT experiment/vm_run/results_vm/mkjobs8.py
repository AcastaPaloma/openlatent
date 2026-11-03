spec=[("A0",1500,"x"),("B1",1500,"x"),("B3",1500,"1"),("B3",1500,"1.5"),("A3",1000,"x"),("A14",1000,"x"),("A21",1000,"x"),("K0",1500,"x")]
jobs=[(a,sd,st,sg) for sd in (0,1,2) for a,st,sg in spec]
open("jobs18.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
