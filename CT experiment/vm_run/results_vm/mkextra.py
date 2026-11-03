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
cnt = al.groupby("patient").scan.nunique(); multi = sorted(cnt[cnt >= 2].index); print("candidate multi-scan patients", len(multi), "single", (cnt == 1).sum())
random.Random(1).shuffle(multi); pick = set(multi[:1300])
ex = al[al.patient.isin(pick)].sort_values(["patient", "date", "scan"]).groupby("patient").head(4)
ex.to_csv(f"{H}/manifest_extra.csv", index=False); print("extra scans", len(ex), "patients", ex.patient.nunique())
