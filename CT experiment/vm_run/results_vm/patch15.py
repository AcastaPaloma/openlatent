s = open("train_arms.py").read()
old = "same_scanner_top1=ss_top1,"
assert old in s; s = s.replace(old, "best_hits=hits.tolist(), best_query_patients=[str(p) for p in qp], same_scanner_top1=ss_top1,", 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
