s = open("train_arms.py").read()
old = "md = pd.read_csv(f\"{D}/train_metadata.csv\").set_index(\"VolumeName\").loc[man.name]"
new = '''SEAL2 = bool(os.environ.get("SEAL2")); SP = set()
if SEAL2:  # fresh exam: patients never used anywhere; dev-eval patients are dropped, adapters/attackers train on fit + extra only
    man = man[(man.patient.map(lambda p: h(p) < 0.5) | man.patient.isin(XP)).values].reset_index(drop=True)
    sx2 = pd.read_csv(f"{D}/manifest_seal2.csv"); sx2 = sx2[sx2.name.map(lambda n: n[:-4] in have)]; c2 = sx2.groupby("patient").scan.nunique(); sx2 = sx2[sx2.patient.isin(c2[c2 >= 2].index)]
    SP = set(sx2.patient); assert not (SP & set(man.patient)); man = pd.concat([man, sx2[man.columns]], ignore_index=True); print("SEAL2:", len(SP), "exam patients,", len(sx2), "scans", flush=True)
md = pd.read_csv(f"{D}/train_metadata.csv").set_index("VolumeName").loc[man.name]'''
assert old in s; s = s.replace(old, new, 1)
old = "isx = man.patient.isin(XP).values; fit0 = man.patient.map(lambda p: h(p) < 0.5).values & ~isx; ev0 = ~fit0 & ~isx"
new = "isx = man.patient.isin(XP).values; isS = man.patient.isin(SP).values; fit0 = man.patient.map(lambda p: h(p) < 0.5).values & ~isx & ~isS; ev0 = isS if SEAL2 else (~fit0 & ~isx)"
assert old in s; s = s.replace(old, new, 1)
old = '("_attx" if ATTX else "") + (f"_st{STEPS}"'
new = '("_attx" if ATTX else "") + ("_seal2" if SEAL2 else "") + (f"_st{STEPS}"'
assert old in s; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
