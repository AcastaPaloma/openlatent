s = open("train_arms.py").read()
old = "md = pd.read_csv(f\"{D}/train_metadata.csv\").set_index(\"VolumeName\").loc[man.name]"
new = '''XP = set()
if os.environ.get("EXTRA"):  # extra fit-only patients (never evaluated): more labelled scans + more multi-visit pairs
    ex = pd.read_csv(f"{D}/manifest_extra.csv"); ex = ex[ex.name.map(lambda n: n[:-4] in have)]; ce = ex.groupby("patient").scan.nunique(); ex = ex[ex.patient.isin(ce[ce >= 2].index)]
    ex = ex.iloc[:int(os.environ["EXTRA"])] if int(os.environ["EXTRA"]) < len(ex) else ex
    XP = set(ex.patient); man = pd.concat([man, ex[man.columns]], ignore_index=True)
md = pd.read_csv(f"{D}/train_metadata.csv").set_index("VolumeName").loc[man.name]'''
assert old in s; s = s.replace(old, new, 1)
old = "fit = man.patient.map(lambda p: h(p) < 0.5).values; ev = ~fit"
new = "fit = (man.patient.map(lambda p: h(p) < 0.5) | man.patient.isin(XP)).values; ev = ~fit"
assert old in s; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
