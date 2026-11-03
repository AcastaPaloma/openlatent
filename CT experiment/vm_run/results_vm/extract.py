# extract.py SPLIT SHARD NSHARDS -- CT-FM features (s=4096, g=512) for manifest_<SPLIT>.csv; one GPU per shard; resumable.
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn.functional as F, nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from lighter_zoo import SegResEncoder
split, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H = os.path.expanduser("~/ct"); OUT = f"{H}/feats"; DL = f"/tmp/dl_{split}_{shard}"
os.makedirs(OUT, exist_ok=True); os.makedirs(DL, exist_ok=True)
man = pd.read_csv(f"{H}/manifest_{split}.csv"); man = man.iloc[shard::nsh]
MDF = "train_metadata" if split in ("train", "extra", "seal2") else "validation_metadata"
md = pd.read_csv(f"{H}/hf/dataset/metadata/{MDF}.csv").set_index("VolumeName")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").cuda().eval()
def fetch(r):
    o = OUT + "/" + r["name"][:-4] + ".npz"
    if os.path.exists(o): return None
    for t in range(5):
        try: return hf_hub_download("ibrahimhamamci/CT-RATE", r["path"], repo_type="dataset", local_dir=DL)
        except Exception as e: time.sleep(5 * (t + 1))
    print("FAIL", r["name"], flush=True); return None
def feats(p, r):
    m = md.loc[r["name"]]; a = np.asarray(nib.load(p).dataobj, dtype=np.float32)
    hu = np.clip(a * float(m.RescaleSlope) + float(m.RescaleIntercept), -1000, 1000)
    xy = eval(m.XYSpacing); sp = (float(xy[0]), float(xy[1]), float(m.ZSpacing))
    t = torch.from_numpy(hu).cuda()[None, None]; sz = [max(1, round(s * k / 2.0)) for s, k in zip(t.shape[2:], sp)]
    t = F.adaptive_avg_pool3d(t, sz) if all(z <= s for z, s in zip(sz, t.shape[2:])) else F.interpolate(t, size=sz, mode="trilinear")
    v = torch.full((192, 192, 192), -1000., device="cuda"); src = []; dst = []
    for s in t.shape[2:]: c = min(s, 192); src.append(slice((s - c) // 2, (s - c) // 2 + c)); dst.append(slice((192 - c) // 2, (192 - c) // 2 + c))
    v[tuple(dst)] = t[0, 0][tuple(src)]; v = v.half().float()[None, None]
    x = ((v.clamp(-1000, 1000) + 1024) / 3072).clamp(0, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): f = model(x)[-1].float()
    return F.adaptive_avg_pool3d(f, 1).flatten().cpu().numpy(), F.adaptive_avg_pool3d(f, 2).flatten().cpu().numpy().astype(np.float16)
rows = [r for _, r in man.iterrows()]; t0 = time.time(); n = 0
with ThreadPoolExecutor(4) as ex:
    for r, p in zip(rows, ex.map(fetch, rows)):
        if p is not None:
            g, s = feats(p, r); np.savez(OUT + "/" + r["name"][:-4] + ".npz", g=g, s=s); os.remove(p)
        n += 1
        if n % 10 == 0: print(n, len(rows), f"{time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
