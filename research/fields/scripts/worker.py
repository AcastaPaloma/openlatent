#!/usr/bin/env python3
"""Per-GPU queue worker.  One process per GPU:  python worker.py --gpu 0

Queue layout on the VM (not in git):  ~/queue/{pending,running,done,failed}/<name>.yaml
A job is a YAML file (also kept in git under research/fields/jobs/specs/) with:
    name: str                 unique
    cmd: str                  shell command, run from the repo root, CUDA_VISIBLE_DEVICES set to this worker's GPU
    priority: int             lower runs first (default 50)
    out: str                  directory the job writes (default ~/work/runs/<name>); uploaded to HF on success
    hf_path: str              path inside the HF repo (default runs/<name>)
    hf_allow: list[str]       optional glob allow-list for the upload (default: everything); REQUIRED for t_ (chest X-ray) jobs: ["*.json","*.log","*.md"]
Claim = atomic os.rename pending -> running; a crashed worker leaves the file in running/ for the tick to requeue.
After each job: result.json (config, commit, exit code, timings) is written next to the logs, outputs go to HF.
"""
import argparse, json, os, shutil, subprocess, sys, time, socket
from pathlib import Path

import yaml

Q = Path.home() / "queue"
REPO = Path.home() / "openlatent"
WORK = Path.home() / "work"
HF_REPO = os.environ.get("FIELDS_HF_REPO", "KuanKuanKuan/openlatent-fields")


def claim(gpu, max_priority=None):
    keyed = []
    (Q / "bad").mkdir(exist_ok=True)
    for p in (Q / "pending").glob("*.yaml"):
        try:
            cfg = yaml.safe_load(p.read_text())
            assert isinstance(cfg, dict) and "name" in cfg and "cmd" in cfg
            if max_priority is not None and cfg.get("priority", 50) > max_priority:
                continue  # express worker: reserved-priority jobs only
            keyed.append(((cfg.get("priority", 50), p.stat().st_mtime, p.name), p))  # ties: oldest submission first, then name
        except Exception as e:  # one malformed spec must never kill a worker: quarantine it
            print(f"quarantine {p.name}: {str(e)[:120]}", flush=True)
            try:
                os.rename(p, Q / "bad" / p.name)
            except FileNotFoundError:
                pass
    files = [p for _, p in sorted(keyed)]
    running_names = {r.name.split("__", 1)[1] for r in (Q / "running").glob("gpu*__*.yaml")}
    for p in files:
        if p.name in running_names:  # a second copy of a spec already running shares its $OUT and log (06:02Z 2026-11-05): leave it pending
            continue
        dst = Q / "running" / f"gpu{gpu}__{p.name}"
        try:
            os.rename(p, dst)  # atomic; losing the race raises FileNotFoundError
            return dst
        except FileNotFoundError:
            continue
    return None


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def upload(out, hf_path, hf_allow=None):
    """hf_allow: optional list of glob patterns; when set ONLY matching files are uploaded
    (use ["*.json","*.log","*.md"] for jobs whose weights/images must not be published, e.g. chest X-ray linkage attackers)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_REPO, exist_ok=True, repo_type="model", private=False)
        kw = dict(allow_patterns=list(hf_allow)) if hf_allow else {}
        api.upload_folder(folder_path=str(out), path_in_repo=hf_path, repo_id=HF_REPO,
                          ignore_patterns=["*.tmp", "frames_cache/*"], **kw)
        return True
    except Exception as e:  # noqa
        print("HF upload failed:", e, flush=True)
        return False


def safe_pull():
    """git pull --ff-only; if untracked files would be overwritten, move them to ~/work/untracked_backup/<ts>/ and retry once.
    Returns (ok, note). A silent pull failure used to leave jobs on stale code (2026-11-04)."""
    cmd = ["git", "-C", str(REPO), "pull", "-q", "--ff-only", "origin", "fields-research"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return True, ""
    paths = [l.strip() for l in r.stderr.splitlines() if l.startswith("\t") and "would be overwritten" not in l]
    if "untracked working tree files would be overwritten" in r.stderr and paths:
        dest = WORK / "untracked_backup" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for rel in paths:
            src = REPO / rel
            if src.exists():
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest / rel))
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        return r2.returncode == 0, f"moved {len(paths)} untracked file(s) to {dest}; retry rc={r2.returncode}"
    return False, r.stderr.strip()[:300]


def run(job_path, gpu):
    cfg = yaml.safe_load(job_path.read_text())
    name = cfg["name"]
    out = Path(os.path.expanduser(cfg.get("out", f"~/work/runs/{name}")))
    out.mkdir(parents=True, exist_ok=True)
    logdir = WORK / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    pull_ok, pull_note = safe_pull()
    # thread cap enforced for every job (charter rule: <= 8 CPU threads); a job spec may override via its own cmd
    caps = {k: "8" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT=str(out), JOB_NAME=name, PYTHONUNBUFFERED="1")
    for k, v in caps.items():
        env.setdefault(k, v)
    commit0 = git("rev-parse", "--short", "HEAD")  # commit the job starts on (pull happened above)
    t0 = time.time()
    with open(logdir / f"{name}.log", "a") as lf:
        rc = subprocess.call(cfg["cmd"], shell=True, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    res = dict(name=name, config=cfg, commit=commit0, commit_after=git("rev-parse", "--short", "HEAD"), pull_ok=pull_ok, pull_note=pull_note, exit_code=rc, gpu=gpu,
               host=socket.gethostname(), seconds=round(time.time() - t0, 1), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    shutil.copy(logdir / f"{name}.log", out / "train.log")
    (out / "job_result.json").write_text(json.dumps(res, indent=1))
    ok = rc == 0 and upload(out, cfg.get("hf_path", f"runs/{name}"), cfg.get("hf_allow"))
    res["uploaded"] = bool(ok)
    (out / "job_result.json").write_text(json.dumps(res, indent=1))
    dest = Q / ("done" if rc == 0 else "failed")
    os.rename(job_path, dest / job_path.name.split("__", 1)[1])
    print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {name} rc={rc} {res['seconds']}s uploaded={ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max_priority", type=int, default=None, help="express worker: only claim jobs with priority <= this")
    a = ap.parse_args()
    for d in ("pending", "running", "done", "failed"):
        (Q / d).mkdir(parents=True, exist_ok=True)
    while True:
        j = claim(a.gpu, a.max_priority)
        if j is None:
            time.sleep(20)
            continue
        run(j, a.gpu)


if __name__ == "__main__":
    main()
