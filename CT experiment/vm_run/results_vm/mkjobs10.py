jobs=[(a,sd,1000,"x") for sd in (0,1,2) for a in ("A26","A27")]
open("jobs14.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
