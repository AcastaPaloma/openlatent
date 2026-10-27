"""GPU job queue for one VM.

    ~/work/queue/pending/*.json   submitted configs (run in name order; prefix with a priority)
    ~/work/queue/running/         claimed by an atomic rename, so a job runs once
    ~/work/queue/done|failed/     by exit status; the job log is ~/work/logs/<name>.log

Before each launch the repo is fast-forwarded, so new code reaches the next job without
touching running ones. Jobs left in running/ by a crash or reboot are requeued at startup.

Usage: python3 worker.py [--slots 2]   (run under nohup/tmux)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

Q = os.path.expanduser("~/work/queue")
LOGS = os.path.expanduser("~/work/logs")
REPO = os.path.expanduser("~/openlatent")
CODE = os.path.join(REPO, "research", "privacy-obfuscation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    args = ap.parse_args()
    for d in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    for p in glob.glob(os.path.join(Q, "running", "*.json")):
        shutil.move(p, os.path.join(Q, "pending", os.path.basename(p)))
        print(f"requeued {os.path.basename(p)}", flush=True)

    procs = {}
    while True:
        for job, (proc, logfh) in list(procs.items()):
            if proc.poll() is not None:
                logfh.close()
                dest = "done" if proc.returncode == 0 else "failed"
                shutil.move(os.path.join(Q, "running", job), os.path.join(Q, dest, job))
                print(f"{time.strftime('%H:%M:%S')} {job} -> {dest} (exit {proc.returncode})", flush=True)
                del procs[job]
        while len(procs) < args.slots:
            pending = sorted(glob.glob(os.path.join(Q, "pending", "*.json")))
            if not pending:
                break
            job = os.path.basename(pending[0])
            try:
                os.rename(pending[0], os.path.join(Q, "running", job))
            except FileNotFoundError:
                continue
            subprocess.run(["git", "-C", REPO, "pull", "-q", "--ff-only"], capture_output=True)
            name = json.load(open(os.path.join(Q, "running", job)))["name"]
            logfh = open(os.path.join(LOGS, f"{name}.log"), "a")
            proc = subprocess.Popen(["python3", "-m", "ppv.run", os.path.join(Q, "running", job)],
                                    cwd=CODE, stdout=logfh, stderr=subprocess.STDOUT, start_new_session=True)
            procs[job] = (proc, logfh)
            print(f"{time.strftime('%H:%M:%S')} started {job} (pid {proc.pid})", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
