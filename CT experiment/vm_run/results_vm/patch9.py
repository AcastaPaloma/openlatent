s = open("train_arms.py").read()
old = '+ f"_s{SEED}_n{len(man)}" + ("_attx" if ATTX else "") + ".json"'
new = '+ f"_s{SEED}_n{len(man)}" + ("_attx" if ATTX else "") + (f"_st{STEPS}" if STEPS not in (1500, 3000) else "") + ".json"'
assert old in s; s = s.replace(old, new, 1); open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
