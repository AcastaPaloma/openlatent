jobs=[(a,sd,1000,"x") for sd in range(8) for a in ("A3","A14","A21")]
open("jobs15.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); open("jobs15b.txt","w").write(""); print(len(jobs))
