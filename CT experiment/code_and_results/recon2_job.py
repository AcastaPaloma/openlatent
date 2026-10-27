
# recon2_job.py -- session 2: second reconstruction of each fit-patient scan -> CT-FM features on Drive (for arm A6). Resumable.
import os, sys, time, hashlib, subprocess, threading, numpy as np, pandas as pd, torch, torch.nn.functional as F
subprocess.run("pip install -q --no-deps lighter-zoo monai hf_xet nibabel", shell=True)
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from lighter_zoo import SegResEncoder
assert os.environ.get("HF_TOKEN"), "launch from the notebook cell, which passes HF_TOKEN in the environment"
D = "/content/drive/MyDrive/openlatent_ct"; OUT = f"{D}/feats2"; DL = "/content/dl2"; os.makedirs(OUT, exist_ok=True); os.makedirs(DL, exist_ok=True)
h = lambda p: int(hashlib.sha256(f"ctrate-openlatent-v1:{p}".encode()).hexdigest()[:8], 16) / 16**8
man = pd.read_csv(f"{D}/manifest_train.csv"); idx = pd.read_csv(f"{D}/ctrate_index.csv"); idx = idx[idx.split == "train"].copy()
md = pd.read_csv(f"{D}/train_metadata.csv").set_index("VolumeName")
idx["recon"] = idx.recon.astype(int); idx["name"] = idx.path.str.split("/").str[-1]
m1 = man[man.patient.map(lambda p: h(p) < 0.5)][["patient", "scan", "name"]].rename(columns={"name": "name1"})
cand = idx.merge(m1, on=["patient", "scan"]); cand = cand[cand.name != cand.name1].sort_values("recon").drop_duplicates(["patient", "scan"]).reset_index(drop=True)
cand.to_csv(f"{D}/manifest_recon2.csv", index=False); print(len(m1), "fit scans |", len(cand), "have a second reconstruction", flush=True)
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").cuda().eval(); lock = threading.Lock()
def work(r):
    o = f"{OUT}/{r['name1'][:-4]}.npz"          # keyed by the recon-1 name so A6 can pair views
    if os.path.exists(o): return 0
    for t in range(4):
        try: p = hf_hub_download("ibrahimhamamci/CT-RATE", r["path"], repo_type="dataset", local_dir=DL); break
        except Exception as e: time.sleep(5 * (t + 1)); p = None
    if p is None: print("FAIL", r["name"], flush=True); return 0
    m = md.loc[r["name"]]; a = np.asarray(nib.load(p).dataobj, dtype=np.float32)
    hu = np.clip(a * float(m.RescaleSlope) + float(m.RescaleIntercept), -1000, 1000); xy = eval(m.XYSpacing); sp = (float(xy[0]), float(xy[1]), float(m.ZSpacing))
    with lock:
        t = torch.from_numpy(hu).cuda()[None, None]; sz = [max(1, round(s * k / 2.0)) for s, k in zip(t.shape[2:], sp)]
        t = F.adaptive_avg_pool3d(t, sz) if all(z <= s for z, s in zip(sz, t.shape[2:])) else F.interpolate(t, size=sz, mode="trilinear")
        v = torch.full((192, 192, 192), -1000., device="cuda"); src = []; dst = []
        for s in t.shape[2:]: c = min(s, 192); src.append(slice((s - c) // 2, (s - c) // 2 + c)); dst.append(slice((192 - c) // 2, (192 - c) // 2 + c))
        v[tuple(dst)] = t[0, 0][tuple(src)]; v = v.half().float()[None, None]      # same fp16 rounding as the recon-1 cache
        x = ((v.clamp(-1000, 1000) + 1024) / 3072).clamp(0, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): f = model(x)[-1].float()
        np.savez(o, g=F.adaptive_avg_pool3d(f, 1).flatten().cpu().numpy(), s=F.adaptive_avg_pool3d(f, 2).flatten().cpu().numpy().astype(np.float16))
    os.remove(p); return 1
t0 = time.time(); n = 0
with ThreadPoolExecutor(6) as ex:
    for k in ex.map(work, [r for _, r in cand.iterrows()]):
        n += 1
        if n % 25 == 0: print(n, len(cand), f"{time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
