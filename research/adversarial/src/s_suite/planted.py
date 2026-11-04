"""Planted-leak generator: synthetic wearable-like windows with a known, dialled identity leak.

A unit is one recording session of one identity; its W windows share a session offset (so windows of one unit are NOT independent
and the unit is the resampling cluster). Each window is a (T, C) signal:

    x = (1 + gain_term) * (activity_wave + AR(1) noise) + session_offset + bias_term + spectral_term

with the identity leak entering through exactly one channel:
    bias      per-identity constant offset per sensor channel            (removed by de-meaning, survives in the mean feature)
    gain      per-identity amplitude gain per channel                    (zero mean: invisible to linear/kNN on raw windows)
    spectral  per-identity narrow spectral line with random phase        (cadence-like; phase-invariant, energy-band feature)
    null      dose is ignored: no identity signal at all (also what a shuffled-label release looks like)
The activity (public/utility label) changes amplitude and frequency and is independent of identity, so it is a nuisance for identity.

`oracle_accuracy` is the charter's oracle arm: the best in-family attacker trained with ~20x the data on the same identities, scored on
a large fresh sample; it is a lower bound on the Bayes accuracy and defines what "dose" means in accuracy units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

CHANNELS = ("null", "bias", "gain", "spectral")


@dataclass(frozen=True)
class PlantedSpec:
    channel: str = "gain"
    dose: float = 0.0
    n_id: int = 25
    T: int = 32
    C: int = 4
    W: int = 4            # windows per unit
    n_act: int = 6
    tau: float = 0.5      # session-offset std per channel, shared by the unit's windows
    ar: float = 0.5
    ident_seed: int = 0   # draws the identities' parameters; one value per replicate


def _world(spec: PlantedSpec):
    """Activity parameters (fixed world) and identity parameters (per replicate)."""
    rw = np.random.RandomState(12345)
    A = rw.uniform(0.5, 2.0, (spec.n_act, spec.C))
    fa = rw.randint(3, 10, spec.n_act)
    ri = np.random.RandomState(1000 + spec.ident_seed)
    b = ri.randn(spec.n_id, spec.C)
    g = ri.uniform(-1.0, 1.0, (spec.n_id, spec.C))
    fi = ri.permutation(np.arange(2, 2 + 14))[: spec.n_id] if spec.n_id <= 14 else ri.randint(2, 16, spec.n_id)
    h = ri.uniform(0.5, 1.5, (spec.n_id, spec.C))
    return A, fa, b, g, fi, h


def gen_split(spec: PlantedSpec, units_per_id: int, rs: np.random.RandomState):
    """-> X (n,T,C) float32, y (n,), unit (n,) int, act (n,). n = n_id * units_per_id * W."""
    A, fa, b, g, fi, h = _world(spec)
    U = spec.n_id * units_per_id
    uid = np.repeat(np.arange(spec.n_id), units_per_id)
    act_u = rs.randint(0, spec.n_act, U)
    off_u = rs.randn(U, spec.C) * spec.tau
    n = U * spec.W
    unit = np.repeat(np.arange(U), spec.W)
    ident = uid[unit]
    act = act_u[unit]
    off = off_u[unit]
    t = np.arange(spec.T)[None, :, None]
    phase = rs.uniform(0, 2 * np.pi, (n, spec.C))
    wave = A[act][:, None, :] * np.sin(2 * np.pi * fa[act][:, None, None] * t / spec.T + phase[:, None, :])
    z = rs.randn(n, spec.T + 16, spec.C)
    e = lfilter([np.sqrt(1 - spec.ar ** 2)], [1.0, -spec.ar], z, axis=1)[:, 16:, :]
    sig = wave + e
    d = spec.dose
    if spec.channel == "gain":
        gain = np.clip(1.0 + d * g[ident], 0.1, None)
        sig = sig * gain[:, None, :]
    sig = sig + off[:, None, :]
    if spec.channel == "bias":
        sig = sig + d * b[ident][:, None, :]
    elif spec.channel == "spectral":
        psi = rs.uniform(0, 2 * np.pi, (n, spec.C))
        sig = sig + d * h[ident][:, None, :] * np.sin(2 * np.pi * fi[ident][:, None, None] * t / spec.T + psi[:, None, :])
    return sig.astype(np.float32), ident.astype(np.int64), unit.astype(np.int64), act.astype(np.int64)


def features(X: np.ndarray) -> np.ndarray:
    """Hand features per window: per-channel mean, std (de-meaned), mean abs difference, 8 log band energies. -> (n, 11*C)."""
    n, T, C = X.shape
    mu = X.mean(1)
    xc = X - mu[:, None, :]
    sd = xc.std(1)
    mad = np.abs(np.diff(xc, axis=1)).mean(1)
    F = np.abs(np.fft.rfft(xc, axis=1)) ** 2          # (n, T/2+1, C)
    F = F[:, 1:17, :]
    bands = F.reshape(n, 8, 2, C).sum(2)               # 8 bands of 2 bins
    lb = np.log1p(bands)
    return np.concatenate([mu, sd, mad, lb.reshape(n, 8 * C)], axis=1).astype(np.float32)


def raw_view(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def generate_replicate(spec: PlantedSpec, seed: int, units_train: int = 12, units_val: int = 6, units_test: int = 57, units_pop: int = 100):
    """Independent train/val/test/pop splits for one replicate (units per identity). Population split gives the attackers' true accuracy."""
    rs = np.random.RandomState((7919 * (seed + 1) + 17) % (2 ** 32 - 1))
    out = {}
    for nm, u in (("train", units_train), ("val", units_val), ("test", units_test), ("pop", units_pop)):
        X, y, unit, act = gen_split(spec, u, rs)
        out[nm] = dict(X=X, y=y, unit=unit, act=act)
    return out
