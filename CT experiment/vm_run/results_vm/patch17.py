s = open("train_arms.py").read()
old = 'np.load(f"{D}/feats/{n[:-4]}.npz")["s"].astype(np.float32)'
assert old in s; s = s.replace(old, 'np.load(f"{D}/feats/{n[:-4]}.npz")[os.environ.get("FEATKEY", "s")].astype(np.float32)', 1)
old = "_s{SEED}_n{len(man)}{'_attx' if ATTX else ''}.npz"
assert old in s; s = s.replace(old, "_s{SEED}_n{len(man)}{'_attx' if ATTX else ''}{os.environ.get('TAG', '')}.npz", 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok", s.count("FEATKEY"))
