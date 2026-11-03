s = open("train_arms.py").read()
s = s.replace('LAM = {"A12": 3., "A13": 3.,', 'LAM = {"A12": 3., "A13": 3., "A14": 3., "A15": 3.,', 1)
s = s.replace('if ARM in ("A12", "A13"):', 'if ARM in ("A12", "A13", "A14", "A15"):', 1)
old = '(1. if ARM == "A12" else 3.)'
assert old in s; s = s.replace(old, '{"A12": 1., "A13": 3., "A14": .3, "A15": .1}[ARM]', 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
