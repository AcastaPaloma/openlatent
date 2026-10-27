
# run_batch.py PART NPARTS  -- runs the pre-registered grid, skipping finished (arm, seed, n) results. Needs frozen_scans.txt.
import os, sys, subprocess, json
D = "/content/drive/MyDrive/openlatent_ct"; part, nparts = int(sys.argv[1]), int(sys.argv[2])
n = len(open(f"{D}/frozen_scans.txt").read().split())
grid = [(a, s, None) for s in (0, 1, 2) for a in ("A0", "A1", "A2", "A3", "A4", "A5", "A7", "B1", "B2")] + [("B3", s, sg) for s in (0, 1, 2) for sg in ("1", "2", "4")]
mine = [g for i, g in enumerate(grid) if i % nparts == part]
for arm, seed, sg in mine:
    tag = f"{arm}{('_sigma' + sg) if sg else ''}"
    done = [f for f in os.listdir(f"{D}/runs") if f.startswith(f"arm_{tag}_s{seed}_n") and f.endswith(".json")]
    if done: print("skip", tag, seed); continue
    cmd = ["python", f"{D}/train_arms.py", arm, str(seed), "1500"] + ([sg] if sg else [])
    print("run", tag, seed, flush=True); r = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True)
    print((r.stdout[-400:] if r.returncode == 0 else "FAILED: " + r.stderr[-600:]), flush=True)
print("BATCH DONE")
