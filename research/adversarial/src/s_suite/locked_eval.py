"""Locked evaluator: sealed labels stay with the harness, the attacker (hand-built code or an LLM agent) receives only a scalar.

Protocol
--------
* Item ids and their order are public; labels are not. The harness splits the units into DEV (queried through a Ladder) and SEALED (one
  final query bound to an artifact digest).
* `dev(probs)`: returns one float. Ladder rule (Blum and Hardt, 2015): the score shown is the best score revealed so far, and it moves only
  when the true score beats it by more than `eta`; the revealed value is rounded to a multiple of `eta`. A query budget caps the number of calls.
* `final(probs, digest)`: allowed exactly once; returns one float (top-1 accuracy rounded to `final_decimals`). The digest is logged so a
  later audit can tie the sealed score to the code and weights the agent committed to before seeing it.
* The agent never receives per-item correctness, labels, ids of sealed units' classes, confusion, or confidence intervals. The full certificate
  (`seal_report`) is for the harness owner only.

Isolation
---------
`LockedEvaluator` is an in-process object (honour system: Python cannot hide attributes). For real separation use `LockedClient`, which
spawns `python -m s_suite.locked_eval --serve labels.npz` as a separate process that alone reads the labels and answers JSON lines on
stdin/stdout; the client object holds no labels. OS-level separation (another user, container or host) is the deployer's job and is what the
agent-vs-hand-built ladder must use.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from typing import Callable, Optional

import numpy as np


def _enc(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype=np.float32)
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def _dec(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float32).reshape(d["shape"])


class BudgetExceeded(RuntimeError):
    pass


class AlreadyFinal(RuntimeError):
    pass


class LockedEvaluator:
    def __init__(self, y_dev, y_sealed, eta: float = 0.01, max_queries: int = 100, final_decimals: int = 3):
        self.__y_dev = np.asarray(y_dev).astype(int).copy()
        self.__y_sealed = np.asarray(y_sealed).astype(int).copy()
        self.n_dev = len(self.__y_dev)
        self.n_sealed = len(self.__y_sealed)
        self.eta = float(eta)
        self.max_queries = int(max_queries)
        self.final_decimals = int(final_decimals)
        self.__best = None
        self.queries = 0
        self.log = []
        self.__final_done = False
        self.__final_digest = None
        self.__final_score = None

    # ---- what the agent can call ----
    def dev(self, probs) -> float:
        if self.queries >= self.max_queries:
            raise BudgetExceeded(f"query budget {self.max_queries} exhausted")
        p = self.__check(probs, self.n_dev)
        self.queries += 1
        true = float((p.argmax(1) == self.__y_dev).mean())
        if self.__best is None:
            self.__best = np.floor(true / self.eta) * self.eta
        elif true > self.__best + self.eta:
            self.__best = np.floor(true / self.eta) * self.eta
        shown = float(round(self.__best, 10))
        self.log.append(dict(kind="dev", q=self.queries, digest=_digest(p), shown=shown))
        return shown

    def final(self, probs, artifact_digest: str) -> float:
        if self.__final_done:
            raise AlreadyFinal("the sealed split has been scored once already")
        p = self.__check(probs, self.n_sealed)
        self.__final_done = True
        self.__final_digest = str(artifact_digest)
        self.__final_score = float((p.argmax(1) == self.__y_sealed).mean())
        shown = round(self.__final_score, self.final_decimals)
        self.log.append(dict(kind="final", digest=_digest(p), artifact=self.__final_digest, shown=shown))
        return shown

    # ---- harness owner only ----
    def seal_report(self) -> dict:
        """Not exposed over the pipe. Raw sealed score and the audit trail."""
        return dict(final_score=self.__final_score, artifact_digest=self.__final_digest, n_queries=self.queries, log=list(self.log))

    @staticmethod
    def __check(probs, n):
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != n:
            raise ValueError(f"expected ({n}, C) scores, got {p.shape}")
        if not np.isfinite(p).all():
            raise ValueError("non-finite scores")
        return p


def _digest(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype=np.float32).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------------
# process isolation

def serve(labels_npz: str, eta: float = 0.01, max_queries: int = 100, stdin=None, stdout=None):
    z = np.load(labels_npz)
    ev = LockedEvaluator(z["y_dev"], z["y_sealed"], eta, max_queries)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            op = msg["op"]
            if op == "dev":
                r = dict(ok=True, score=ev.dev(_dec(msg["probs"])))
            elif op == "final":
                r = dict(ok=True, score=ev.final(_dec(msg["probs"]), msg.get("digest", "")))
            elif op == "status":
                r = dict(ok=True, queries=ev.queries, remaining=ev.max_queries - ev.queries, n_dev=ev.n_dev, n_sealed=ev.n_sealed, eta=ev.eta)
            elif op == "quit":
                stdout.write(json.dumps(dict(ok=True)) + "\n")
                stdout.flush()
                return
            else:
                r = dict(ok=False, error="unknown op")
        except Exception as e:  # noqa: BLE001 - report to the client, never crash the server with labels in memory
            r = dict(ok=False, error=f"{type(e).__name__}: {e}")
        stdout.write(json.dumps(r) + "\n")
        stdout.flush()


class LockedClient:
    """Talks to a separate process that holds the labels. Holds no label data itself."""

    def __init__(self, labels_npz: str, eta: float = 0.01, max_queries: int = 100):
        self._p = subprocess.Popen([sys.executable, "-m", "s_suite.locked_eval", "--serve", labels_npz, "--eta", str(eta), "--max-queries",
                                    str(max_queries)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                   cwd=str(_pkg_parent()))

    def _call(self, msg: dict) -> dict:
        self._p.stdin.write(json.dumps(msg) + "\n")
        self._p.stdin.flush()
        r = json.loads(self._p.stdout.readline())
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "server error"))
        return r

    def dev(self, probs) -> float:
        return self._call(dict(op="dev", probs=_enc(probs)))["score"]

    def final(self, probs, artifact_digest: str) -> float:
        return self._call(dict(op="final", probs=_enc(probs), digest=artifact_digest))["score"]

    def status(self) -> dict:
        return self._call(dict(op="status"))

    def close(self):
        try:
            self._call(dict(op="quit"))
        except Exception:  # noqa: BLE001
            pass
        self._p.wait(timeout=10)


def _pkg_parent():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


def write_labels(path: str, y_dev, y_sealed):
    np.savez(path, y_dev=np.asarray(y_dev).astype(int), y_sealed=np.asarray(y_sealed).astype(int))


# ---------------------------------------------------------------------------------------------------------------------------
# search loop through the locked interface, and its shuffled-label null

def run_search_loop(ev, design: Callable[[int], tuple], n_designs: int) -> dict:
    """Generic agent-style loop: for i in range(n_designs) the designer returns (dev_probs, sealed_probs, digest); the loop sees only
    the Ladder scalar for dev, keeps the design with the highest shown score (latest on ties), and spends the single final query on it."""
    best = (-1.0, None)
    shown_hist = []
    cache = {}
    for i in range(n_designs):
        dev_p, sealed_p, dig = design(i)
        s = ev.dev(dev_p)
        shown_hist.append(s)
        if s >= best[0]:
            best = (s, i)
        cache[i] = (sealed_p, dig)
    sealed_p, dig = cache[best[1]]
    final = ev.final(sealed_p, dig)
    return dict(best_index=best[1], best_dev_shown=best[0], final=final, shown=shown_hist)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--max-queries", type=int, default=100)
    a = ap.parse_args()
    serve(a.serve, a.eta, a.max_queries)
