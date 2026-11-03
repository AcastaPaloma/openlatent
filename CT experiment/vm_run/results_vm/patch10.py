s = open("train_arms.py").read()
s = s.replace('LAM = {"A12": 3.,', 'LAM = {"A16": 3., "A17": 3., "A18": 3., "A12": 3.,', 1)
old = "def train():\n"
new = '''def aug(x):  # train-time input regularisers against memorising the fit patients
    if ARM == "A16": return F.dropout(x, 0.3)
    if ARM == "A17": return x + 0.5 * torch.randn_like(x)
    if ARM == "A18": return F.dropout(x, 0.6)
    return x
def train():
'''
assert old in s; s = s.replace(old, new, 1)
old = "z1, z2 = enc(Xt[a]), enc(Xt[b])"
assert old in s; s = s.replace(old, "z1, z2 = enc(aug(Xt[a])), enc(aug(Xt[b]))", 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
