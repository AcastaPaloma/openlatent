s = open("train_arms.py").read()
old = "    ts = sim[np.arange(len(qi)), truth][:, None]; return 1 + (sim > ts).sum(1) + 0.5 * ((sim == ts).sum(1) - 1)"
new = '''    ts = sim[np.arange(len(qi)), truth][:, None]
    tsc = scid[gi][truth]; mk_ = scid[gi][None, :] == tsc[:, None]; ms = np.where(mk_, sim, -np.inf)   # hard negatives: only gallery people scanned like the true match
    rm_ = 1 + (ms > ts).sum(1) + 0.5 * ((ms == ts).sum(1) - 1); MASKED.append((float((rm_ <= 1).mean()), float((1.0 / mk_.sum(1)).mean())))
    return 1 + (sim > ts).sum(1) + 0.5 * ((sim == ts).sum(1) - 1)'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace("def rank_from_sim(sim):", "MASKED = []\ndef rank_from_sim(sim):", 1)
old = "best = max(res, key=lambda n: res[n][\"top1\"]);"
new = "best = max(res, key=lambda n: res[n][\"top1\"]); ss_top1, ss_chance = MASKED[list(res).index(best)];"
assert old in s; s = s.replace(old, new, 1)
old = "utility_C=best_c,"
new = "same_scanner_top1=ss_top1, same_scanner_chance=ss_chance, utility_C=best_c,"
assert old in s; s = s.replace(old, new, 1)
s = s.replace('LAM = {"A16": 3.,', 'LAM = {"A19": 3., "A20": 3., "A16": 3.,', 1)
s = s.replace('if ARM in ("A12", "A13", "A14", "A15"):', 'if ARM in ("A12", "A13", "A14", "A15", "A19", "A20"):', 1)
s = s.replace('{"A12": 1., "A13": 3., "A14": .3, "A15": .1}[ARM]', '{"A12": 1., "A13": 3., "A14": .3, "A15": .1, "A19": .2, "A20": .5}[ARM]', 1)
old = '(f"_st{STEPS}" if STEPS not in (1500, 3000) else "") + ".json"'
new = '(f"_st{STEPS}" if STEPS not in (1500, 3000) else "") + os.environ.get("TAG", "") + ".json"'
assert old in s; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
