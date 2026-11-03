s = open("train_arms.py").read()
old = "def rank_from_sim(sim):\n"
new = "def rank_from_sim(sim):\n    sim = np.nan_to_num(np.asarray(sim, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)  # NaN would otherwise rank as a perfect hit\n"
assert old in s; s = s.replace(old, new, 1); open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
