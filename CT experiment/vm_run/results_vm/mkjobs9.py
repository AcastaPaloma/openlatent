jobs=[("B3",sd,1500,sg) for sd in (0,1,2) for sg in ("0.25","0.5","0.75")]
open("jobs18.txt","w").write("\n".join(" ".join(str(x) for x in j) for j in jobs)+"\n"); print(len(jobs))
