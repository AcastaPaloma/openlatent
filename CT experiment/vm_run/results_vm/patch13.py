s = open("train_arms.py").read()
s = s.replace('LAM = {"A19": 3.,', 'LAM = {"A21": 3., "A22": 3., "A23": 3., "A19": 3.,', 1)
s = s.replace('if ARM in ("A12", "A13", "A14", "A15", "A19", "A20"):', 'if ARM in ("A12", "A13", "A14", "A15", "A19", "A20", "A21", "A22", "A23"):', 1)
s = s.replace('"A19": .2, "A20": .5}[ARM]', '"A19": .2, "A20": .5, "A21": .3, "A22": .3, "A23": .3}[ARM]', 1)
old = 'ZD = 64 if ARM == "A11" else 256'
assert old in s; s = s.replace(old, 'ZD = 64 if ARM == "A11" else (512 if ARM == "A22" else 256)', 1)
old = "task = nn.Linear(ZD, Y.shape[1]).to(dev)"
assert old in s; s = s.replace(old, 'task = (nn.Sequential(nn.Linear(ZD, 256), nn.GELU(), nn.Linear(256, Y.shape[1])) if ARM == "A21" else nn.Linear(ZD, Y.shape[1])).to(dev)', 1)
old = "        if LAM > 0:\n            g1, g2"
assert old in s; s = s.replace(old, "        if ARM == \"A23\": loss = loss * 3\n        if LAM > 0:\n            g1, g2", 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
