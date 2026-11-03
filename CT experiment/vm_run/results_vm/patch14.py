s = open("train_arms.py").read()
old = 'sx2 = pd.read_csv(f"{D}/manifest_seal2.csv")'
assert old in s; s = s.replace(old, 'sx2 = pd.read_csv(f"{D}/" + os.environ.get("SEALMAN", "manifest_seal2.csv"))', 1)
old = 'md = pd.read_csv(f"{D}/train_metadata.csv").set_index("VolumeName").loc[man.name]'
assert s.count(old) == 1; s = s.replace(old, 'md = pd.concat([pd.read_csv(f"{D}/train_metadata.csv"), pd.read_csv(f"{D}/validation_metadata.csv")]).set_index("VolumeName").loc[man.name]', 1)
old = 'lab = pd.read_csv(f"{D}/train_predicted_labels.csv").set_index("VolumeName")'
assert old in s; s = s.replace(old, 'lab = pd.concat([pd.read_csv(f"{D}/train_predicted_labels.csv"), pd.read_csv(f"{D}/valid_predicted_labels.csv")]).set_index("VolumeName")', 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
