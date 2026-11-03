s = open("train_arms.py").read()
old = "fit = (man.patient.map(lambda p: h(p) < 0.5) | man.patient.isin(XP)).values; ev = ~fit"
new = '''isx = man.patient.isin(XP).values; fit0 = man.patient.map(lambda p: h(p) < 0.5).values & ~isx; ev0 = ~fit0 & ~isx
ATTX = bool(os.environ.get("ATTX"))   # ATTX: adapter trained on the original fit set only; attackers (and probes) later trained on fit + extra patients
fit = fit0 if ATTX else (fit0 | isx); ev = ev0 if ATTX else ~fit'''
assert old in s; s = s.replace(old, new, 1)
old = 'zf = lambda Zm: (Zm - Zm[fit].mean(0)) / (Zm[fit].std(0) + 1e-6)'
new = '''if ATTX:  # switch to the strong-attacker regime: attackers/probes see fit + extra patients; adapter is already frozen
    fit = fit0 | isx; ev = ev0
    fi = np.where(fit)[0]; by_p = {}
    for i in fi: by_p.setdefault(pid[i], []).append(i)
    multi = [p for p, v in by_p.items() if len(v) >= 2]; by_sc = {}
    for i in fi: by_sc.setdefault(scid[i], []).append(i)
    print("ATTX: attackers train on", fit.sum(), "scans; eval", ev.sum(), flush=True)
zf = lambda Zm: (Zm - Zm[fit].mean(0)) / (Zm[fit].std(0) + 1e-6)'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('json.dump(out, open(f"{D}/runs/arm_" + out["arm"] + f"_s{SEED}_n{len(man)}.json", "w"), indent=1)',
              'json.dump(out, open(f"{D}/runs/arm_" + out["arm"] + f"_s{SEED}_n{len(man)}" + ("_attx" if ATTX else "") + ".json", "w"), indent=1)', 1)
s = s.replace('f"{D}/runs/Z_{ARM}{(\'_sigma\' + sys.argv[4]) if ARM == \'B3\' else \'\'}_s{SEED}_n{len(man)}.npz"', 'f"{D}/runs/Z_{ARM}{(\'_sigma\' + sys.argv[4]) if ARM == \'B3\' else \'\'}_s{SEED}_n{len(man)}{\'_attx\' if ATTX else \'\'}.npz"', 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok", "_attx" in s)
