"""privharness: give it a labelled visual dataset, get back the obfuscator that best protects its secrets.

    python -m privharness ingest <description.json> <root>      raw media -> harness dataset (ingest.py)
    python -m privharness adapt <kth|ucf101_vp|hmdb51_vp|charades|utdmhad> <root> [--split ilic]
    python -m privharness profile <root>                        where the task and the secrets live (GPU)
    python -m privharness plan <root> [--budget 8]              candidates, attackers, protocols, metrics
    python -m privharness screen <root>                         cheap pass over the fixed candidates (GPU)
    python -m privharness evaluate <root> <candidate> [...]     full evaluation of one candidate (GPU)
    python -m privharness search <root> [--parallel 2]          profile -> plan -> screen -> evaluate the
                                                                shortlist -> robustness -> report (one GPU host)
    python -m privharness report <root>                         scores, checks, frontier, recommendation

Run from research/privacy-obfuscation (privharness reuses ppv's models, obfuscators and training loops).
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_to(root):
    path = os.path.join(root, "harness.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def run_evals(root, jobs, parallel, log):
    """Run `python -m privharness.evaluate` jobs, `parallel` at a time on this host's GPU; skip finished ones."""
    pending = list(jobs)
    procs = []
    while pending or procs:
        while pending and len(procs) < parallel:
            name, args = pending.pop(0)
            if os.path.exists(os.path.join(root, "eval", name, "result.json")):
                log(f"evaluate {name}: already done")
                continue
            os.makedirs(os.path.join(root, "logs"), exist_ok=True)
            fh = open(os.path.join(root, "logs", f"{name}.log"), "a")
            p = subprocess.Popen([sys.executable, "-m", "privharness.evaluate", root, *args], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((name, p, fh))
            log(f"evaluate {name}: started (pid {p.pid})")
        for t in list(procs):
            name, p, fh = t
            if p.poll() is not None:
                fh.close()
                procs.remove(t)
                log(f"evaluate {name}: {'done' if p.returncode == 0 else f'FAILED (exit {p.returncode}), see logs/{name}.log'}")
        time.sleep(10)


def search(root, parallel=2, budget=None, robustness=True):
    from . import planner, profile, report, screen
    from . import spec as S
    log = log_to(root)
    spec = S.load(root)
    if not os.path.exists(os.path.join(root, "profile.json")):
        profile.profile(root, log=log)
    prof = json.load(open(os.path.join(root, "profile.json")))
    p = planner.plan(spec, prof, budget=budget or 8)
    planner.save(root, p)
    for w in p["warnings"]:
        log(f"warning: {w}")
    screened = screen.screen(root, p, log=log)
    short = screen.shortlist(p, screened, budget)
    json.dump(dict(shortlist=short), open(os.path.join(root, "shortlist.json"), "w"), indent=1)
    log(f"shortlist: {short}")
    run_evals(root, [("raw", ["raw"])], 1, log)  # the reference every score is normalised by
    run_evals(root, [(n, [n]) for n in short if n != "raw"], parallel, log)
    rep = report.build(root)
    best = rep["recommendation"]["best"]
    if robustness and best:
        log(f"robustness on {best}: attacker seeds 1 and 2, 24-epoch attackers, DINOv2 fine-tune attacker")
        run_evals(root, [(f"{best}_s1", [best, "--seed", "1"]), (f"{best}_s2", [best, "--seed", "2"]),
                         (f"{best}_e24", [best, "--epochs", "24"]),
                         (f"{best}_dino", [best, "--attackers", "frame_dino_probe,frame_dino_ft", "--tag", "dino"])], parallel, log)
        rep = report.build(root)
    log(f"report: {os.path.join(root, 'report.md')}")
    return rep


def main():
    ap = argparse.ArgumentParser(prog="privharness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("description")
    a.add_argument("root")
    a.add_argument("--workers", type=int, default=16)
    a = sub.add_parser("adapt")
    a.add_argument("name")
    a.add_argument("root")
    a.add_argument("--split", default="ilic")
    for c in ("profile", "screen", "report"):
        sub.add_parser(c).add_argument("root")
    a = sub.add_parser("plan")
    a.add_argument("root")
    a.add_argument("--budget", type=int, default=8)
    a = sub.add_parser("evaluate")
    a.add_argument("root")
    a.add_argument("candidate")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--attackers", default=None)
    a.add_argument("--tag", default="")
    a = sub.add_parser("search")
    a.add_argument("root")
    a.add_argument("--parallel", type=int, default=2)
    a.add_argument("--budget", type=int, default=None)
    a.add_argument("--no-robustness", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ingest":
        from .ingest import ingest
        s, r = ingest(args.description, args.root, workers=args.workers)
        print(f"{s.name}: {len(s.items)} items; warnings: {r['warnings']}")
    elif args.cmd == "adapt":
        from .adapters.prepared import convert
        s, r = convert(args.name, args.root, args.split)
        print(f"{s.name}: {len(s.items)} items; warnings: {r['warnings']}")
    elif args.cmd == "profile":
        from .profile import profile
        profile(args.root, log=log_to(args.root))
    elif args.cmd == "plan":
        from . import planner
        from . import spec as S
        p = planner.plan(S.load(args.root), json.load(open(os.path.join(args.root, "profile.json"))), args.budget)
        planner.save(args.root, p)
        print(json.dumps({k: p[k] for k in ("screen", "learned", "metrics", "warnings")}, indent=1))
    elif args.cmd == "screen":
        from . import screen
        p = json.load(open(os.path.join(args.root, "plan.json")))
        print(screen.shortlist(p, screen.screen(args.root, p, log=log_to(args.root))))
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(args.root, args.candidate, args.seed, args.epochs, args.attackers.split(",") if args.attackers else None, args.tag)
    elif args.cmd == "search":
        search(args.root, args.parallel, args.budget, not args.no_robustness)
    elif args.cmd == "report":
        from .report import build
        print(json.dumps(build(args.root)["recommendation"], indent=1))


if __name__ == "__main__":
    main()
