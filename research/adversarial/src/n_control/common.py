"""Shared constants, key-derived RNG, canonical JSON and hashing for the negative-control builder (N).

Everything random is derived from a hex key by SHA-256 so a release is a pure function of (manifest entry, base release).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

N_ID = 24                      # pseudo-identities (= the number of MotionSense subjects)
T = 128                        # window length (samples, 50 Hz)
NBIN = T // 2 + 1              # rfft bins 0..64
FAMILIES = ("bias", "specin", "xphase", "ampfreq", "stealth", "pooled")
SEEN = ("bias", "specin")                  # mechanisms Builder S planted itself (bias, in-band spectral line)
UNSEEN = ("xphase", "ampfreq", "stealth")  # families S did not plant and whose features S does not declare
REGIME = ("pooled",)                       # S's in-band spectral mechanism, but a regime (weak per window, strong when a session is pooled) the certificate does not score
LEVELS = (1, 2, 3, 4, 5)
TARGET_WINDOW = (0.08, 0.15, 0.30, 0.60, 0.90)    # key-aware matched-filter window-level identification rate per level
TARGET_SESSION = (0.20, 0.40, 0.70, 0.90, 0.99)   # pooled: session-pooled matched-filter rate per level
N_Q = 12                       # ampfreq: channel-0 window-mean levels (duodeciles of the base release)
STEALTH_BINS = tuple(range(40, 64))   # 24 distinct bins; S's features read bins 1..16
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
EULER = 0.5772156649015329


def rs_from(key: str, tag: str) -> np.random.RandomState:
    h = hashlib.sha256(f"{key}|{tag}".encode()).digest()
    return np.random.RandomState(np.frombuffer(h[:16], dtype=np.uint32))


def canon(obj) -> bytes:
    """Canonical JSON bytes (sorted keys, no spaces, ascii) used for every committed hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()
