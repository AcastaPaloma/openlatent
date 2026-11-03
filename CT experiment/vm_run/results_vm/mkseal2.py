import re, random, pandas as pd, os
from huggingface_hub import HfApi
H = os.path.expanduser("~/ct")
fs = [f for f in HfApi().list_repo_files("ibrahimhamamci/CT-RATE", repo_type="dataset") if f.startswith("dataset/train/") and f.endswith("_1.nii.gz")]
rows = []
for f in fs:
    m = re.match(r"dataset/train/(train_\d+)/(train_\d+_([a-z]+))/(.+_1\.nii\.gz)$", f)
    if m: rows.append(dict(split="train", patient=m.group(1), scan=m.group(3), recon=1, path=f, name=m.group(4)))
al = pd.DataFrame(rows); old = set(pd.read_csv(f"{H}/manifest_train.csv").patient)
md = pd.read_csv(f"{H}/train_metadata.csv").set_index("VolumeName")
al = al[~al.patient.isin(old) & al.name.isin(md.index)].copy(); al["date"] = al.name.map(md.StudyDate)
cnt = al.groupby("patient").scan.nunique(); multi = sorted(cnt[cnt >= 2].index)
random.Random(1).shuffle(multi); used = set(multi[:1300]); rest = multi[1300:]
assert set(pd.read_csv(f"{H}/manifest_extra.csv").patient) == used
sx = al[al.patient.isin(set(rest))].sort_values(["patient", "date", "scan"]).groupby("patient").head(4)
sx.to_csv(f"{H}/manifest_seal2.csv", index=False); print("seal2 scans", len(sx), "patients", sx.patient.nunique())
