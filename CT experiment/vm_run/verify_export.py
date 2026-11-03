import sys, numpy as np, torch
sys.path.insert(0, "/home/ubuntu/ct"); import load_adapter as la
D = "/home/ubuntu/ct"
for arm in ("A3", "A14", "A21"):
    z = np.load(f"{D}/runs/Z_{arm}_s0_n4492_w.npz", allow_pickle=True); Z = z["Z"].astype(np.float32); names = z["names"]
    S = np.stack([np.load(f"{D}/feats/{n[:-4]}.npz")["s"].astype(np.float32) for n in names])
    Zr = la.encode(f"{D}/weights/{arm}_s0_1000steps.safetensors", S)
    err = np.abs(Zr - Z).max(); print(arm, "scans", len(names), "max|z_reloaded - z_run| = %.4f (fp16 storage ~1e-3)" % err, "OK" if err < 0.05 else "MISMATCH")
