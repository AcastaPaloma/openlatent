"""N2 leak families (fresh parameterisations and two families S evaluator v3 does not read by design), injectors and key-aware oracles.

Same windows, same pseudo-identity machinery, same oracle idea as families.py (N1); everything keyed by a fresh 128-bit key per release.

family   carries the identity in                                                                              class against evaluator v3
bias     (N1 mechanism, fresh keys)   channel-mean offset                                                       positive control (v1 and v3 read it)
specin   (N1 mechanism, fresh keys)   tone at an identity-specific (bin 3..14, channel)                          positive control (v1 and v3 read it)
hb2      a flat-amplitude tone at an identity-specific (bin 18..29, channel), random phase                       v3 target (hb view reads every bin); N1 stealth used bins 40..63 on both channels
phase2   phase lags of THREE bin pairs (bins 18..40, channel-1 amplitude half of channel 0), lags packed on the   v3 target (phase view reads every bin); N1 used two bins in 3..14 and equal amplitudes
         3-torus; common random phase per window and bin
af2      tone position T_q[i] (random permutation per level q, bins 20..31 x channel) with the level q = octile    v3 target (inter / lvlspec); N1 used duodeciles of channel 0 and bins 3..14, an affine-free
         of the CHANNEL-1 window mean                                                                           Latin table here is random per level
pac      PHASE-AMPLITUDE COUPLING in channel 0: a low-frequency carrier (bin 8..16) whose phase sets the           NEW, unseen: power spectra and cross-channel phases carry nothing; the identity is the
         amplitude of a high-frequency tone (bin 28..46); identity = preferred coupling phase (two pairs,          bispectral phase of (k_lo, k_hi, k_hi + k_lo); S.md 10.5 lists it as not covered
         torus-packed); every spectral power is identical across identities
order    WINDOW ORDER: the amplitude of a fixed tone (bin 18..40, channel 0, random phase per window) follows an  NEW, unseen: the per-window distribution is identical for all identities; only the sequence of
         identity-specific two-tone modulation (depths 0.6, 0.3) over the position of the window inside its session                     a session's windows carries the identity (S.md 10.5: structure across windows)

Oracles (key-aware, never trained): tone families use the complex-Gaussian tone likelihood with the base release's reference-spectrum noise model; phase2 the
vector (two-channel) version with a correlated-noise 2x2 covariance; af2 reads the tone-invariant level from the released window; pac uses the noise-normalised
bispectral triple product; order uses the periodogram of the noise-normalised tone-bin amplitude across the session's ordered windows. Omega is a lower bound on the Bayes rate.
For `order` a single window carries no information by construction: Omega_w is the chance level 1/24 and the ladder is defined on the session level.
"""
from __future__ import annotations

import numpy as np

from . import families as F1
from .common import N_ID, NBIN, T, rs_from
from .families import OFFS, _logi0, _tone_llr, noise_model, spectrum

TWO_PI = 2.0 * np.pi
FAMILIES2 = ("bias", "specin", "hb2", "phase2", "af2", "pac", "order")
CONTROLS2 = ("bias", "specin")
TARGETS2 = ("hb2", "phase2", "af2")         # re-parameterised families that evaluator v3 reads by design
NEW2 = ("pac", "order")                      # families v1 and v3 do not read by design
LEVELS2 = (1, 2, 3, 4)
TARGET_WINDOW2 = (0.15, 0.30, 0.60, 0.90)
TARGET_SESSION2 = (0.20, 0.50, 0.80, 0.95)  # `order` only
NQ1 = 8                                      # af2 levels (octiles of the channel-1 window mean)
NU_GRID = (0.05, 0.11, 0.17, 0.23, 0.29, 0.35)   # order: modulation frequencies in cycles per window (identity = ordered pair of distinct ones)


def session_scale(family):
    return family == "order"


def stats2(base):
    """Base stats plus the af2 level edges (octiles of the channel-1 window mean). Cached on the Base object."""
    if getattr(base, "_stats2", None) is None:
        st = dict(base.stats())
        m1 = base.X[:, 1, :].astype(np.float64).mean(-1)
        st["q1edges"] = np.quantile(m1, np.arange(1, NQ1) / NQ1)
        base._stats2 = st
    return base._stats2


def rank_in_session(sess):
    """Position of every window inside its session (file order; sessions are contiguous)."""
    first = np.zeros(sess.max() + 1, dtype=np.int64)
    first[sess[::-1]] = np.arange(len(sess))[::-1]
    return np.arange(len(sess)) - first[sess]


def _pack(rs, d, dim, n=N_ID, tries=300):
    best = None
    for _ in range(tries):
        P = rs.uniform(0, TWO_PI, (n, dim))
        D = np.abs(P[:, None, :] - P[None, :, :])
        D = np.minimum(D, TWO_PI - D)
        D = np.sqrt((D ** 2).sum(-1))
        D[np.arange(n), np.arange(n)] = np.inf
        if best is None or D.min() > best[0]:
            best = (D.min(), P)
    return best[1]


def make_params2(family, key):
    if family in ("bias", "specin"):
        return F1.make_params(family, key)
    rs = rs_from(key, "params2|" + family)
    if family == "hb2":
        combos = [(b, c) for b in range(18, 30) for c in (0, 1)]
        perm = rs.permutation(len(combos))
        return dict(bin=np.array([combos[i][0] for i in perm]), chan=np.array([combos[i][1] for i in perm]))
    if family == "phase2":
        while True:
            k = np.sort(rs.choice(np.arange(18, 41), 3, replace=False))
            if np.diff(k).min() >= 5:
                break
        return dict(bins=k, delta=_pack(rs, None, 3), ratio=0.5)
    if family == "af2":
        lab = rs.permutation(24)
        pos_bin = np.array([20 + p // 2 for p in lab])
        pos_chan = np.array([p % 2 for p in lab])
        table = np.stack([rs.permutation(N_ID) for _ in range(NQ1)])       # table[q, i] = position index of identity i at level q
        return dict(table=table, pbin=pos_bin, pchan=pos_chan)
    if family == "pac":
        while True:
            klo = rs.choice(np.arange(8, 17), 2, replace=False)
            khi = rs.choice(np.arange(28, 47), 2, replace=False)
            if abs(klo[0] - klo[1]) >= 2 and abs(khi[0] - khi[1]) >= 8:
                break
        return dict(klo=klo, khi=khi, phi=_pack(rs, None, 2), depth=1.0)
    if family == "order":
        k = int(rs.randint(18, 41))
        combos = [(a, b) for a in range(len(NU_GRID)) for b in range(len(NU_GRID)) if a != b]
        sel = rs.permutation(len(combos))[:N_ID]
        return dict(k=k, nu1=np.array([NU_GRID[combos[i][0]] for i in sel]), nu2=np.array([NU_GRID[combos[i][1]] for i in sel]), depth=(0.6, 0.3))
    raise ValueError(family)


def level1(Xw, q1edges):
    return np.searchsorted(q1edges, np.asarray(Xw, dtype=np.float64)[:, 1, :].mean(-1))


# ------------------------------------------------------------------------------------------------ injectors
def inject2(Xbase, pid_win, sess_win, family, amp, key, stats):
    if family in ("bias", "specin"):
        return F1.inject(Xbase, pid_win, family, amp, key, stats)
    X = Xbase.astype(np.float64).copy()
    n = len(X)
    t = np.arange(T)[None, :]
    pr = make_params2(family, key)
    rs = rs_from(key, "phase2|" + family)
    cos = np.cos
    if family == "hb2":
        psi = rs.uniform(0, TWO_PI, n)
        X[np.arange(n), pr["chan"][pid_win], :] += amp * cos(TWO_PI * pr["bin"][pid_win][:, None] * t / T + psi[:, None])
    elif family == "phase2":
        psi = rs.uniform(0, TWO_PI, (n, 3))
        for j in range(3):
            k = pr["bins"][j]
            X[:, 0, :] += amp * cos(TWO_PI * k * t / T + psi[:, j, None])
            X[:, 1, :] += pr["ratio"] * amp * cos(TWO_PI * k * t / T + psi[:, j, None] + pr["delta"][pid_win, j][:, None])
    elif family == "af2":
        q = level1(Xbase, stats["q1edges"])
        p = pr["table"][q, pid_win]
        psi = rs.uniform(0, TWO_PI, n)
        X[np.arange(n), pr["pchan"][p], :] += amp * cos(TWO_PI * pr["pbin"][p][:, None] * t / T + psi[:, None])
    elif family == "pac":
        for p in range(2):
            klo, khi = int(pr["klo"][p]), int(pr["khi"][p])
            psl = rs.uniform(0, TWO_PI, n)[:, None]
            psh = rs.uniform(0, TWO_PI, n)[:, None]
            lo = cos(TWO_PI * klo * t / T + psl)
            env = 1.0 + pr["depth"] * cos(TWO_PI * klo * t / T + psl - pr["phi"][pid_win, p][:, None])
            X[:, 0, :] += amp * lo + amp * env * cos(TWO_PI * khi * t / T + psh)
    elif family == "order":
        r = rank_in_session(sess_win)
        ns = sess_win.max() + 1
        th = rs.uniform(0, TWO_PI, (ns, 2))
        th1, th2 = th[sess_win, 0], th[sess_win, 1]
        a = amp * (1.0 + pr["depth"][0] * cos(TWO_PI * pr["nu1"][pid_win] * r + th1) + pr["depth"][1] * cos(TWO_PI * pr["nu2"][pid_win] * r + th2))
        psi = rs.uniform(0, TWO_PI, n)
        X[:, 0, :] += a[:, None] * cos(TWO_PI * pr["k"] * t / T + psi[:, None])
    else:
        raise ValueError(family)
    return X.astype(np.float32)


# ------------------------------------------------------------------------------------------------ oracles
def _phase_llr(F, LN, bins, delta, A0, A1):
    n = len(F)
    out = np.zeros((n, N_ID))
    for j, k in enumerate(bins):
        k = int(k)
        nb = [k + d for d in OFFS if 1 <= k + d <= NBIN - 1]
        S00 = (np.abs(F[:, 0, nb]) ** 2).sum(1)
        S11 = (np.abs(F[:, 1, nb]) ** 2).sum(1)
        S01 = (F[:, 0, nb] * np.conj(F[:, 1, nb])).sum(1)
        rho = 0.7 * S01 / np.sqrt(S00 * S11 + 1e-24)
        s00, s11 = LN[:, 0, k], LN[:, 1, k]
        s01 = rho * np.sqrt(s00 * s11)
        det = s00 * s11 - np.abs(s01) ** 2
        i00, i11 = s11 / det, s00 / det
        i01, i10 = -s01 / det, -np.conj(s01) / det
        v0, v1 = F[:, 0, k], F[:, 1, k]
        e = np.exp(1j * delta[None, :, j])
        z = A0 * (i00 * v0 + i01 * v1)[:, None] + A1 * np.conj(e) * (i10 * v0 + i11 * v1)[:, None]
        qd = (A0 * A0 * i00[:, None] + A0 * A1 * i01[:, None] * e + A1 * A0 * np.conj(e) * i10[:, None] + A1 * A1 * i11[:, None]).real
        out += _logi0(2.0 * np.abs(z)) - qd
    return out


def oracle_llr2(family, Xrel, act, key, amp, stats, mode="ref"):
    if family in ("bias", "specin"):
        return F1.oracle_llr(family, Xrel, act, key, amp, stats, mode)
    n = len(Xrel)
    pr = make_params2(family, key)
    F = spectrum(Xrel)
    LN = noise_model(F, act, stats, mode)
    ar = np.arange(n)[:, None]
    A0 = 64.0 * amp
    if family == "hb2":
        c, f = pr["chan"][None, :], pr["bin"][None, :]
        return _tone_llr(F[ar, c, f], LN[ar, c, f], A0)
    if family == "phase2":
        return _phase_llr(F, LN, pr["bins"], pr["delta"], A0, pr["ratio"] * A0)
    if family == "af2":
        q = level1(Xrel, stats["q1edges"])
        p = pr["table"][q]                            # (n, N_ID): position of every identity at the window's level
        b, c = pr["pbin"][p], pr["pchan"][p]
        return _tone_llr(F[ar, c, b], LN[ar, c, b], A0)
    if family == "pac":
        score = np.zeros((n, N_ID))
        for p in range(2):
            klo, khi = int(pr["klo"][p]), int(pr["khi"][p])
            Y, H = F[:, 0, klo], F[:, 0, khi]
            Sp, Sm = F[:, 0, khi + klo], F[:, 0, khi - klo]
            sy, sh, sp, sm = (np.sqrt(LN[:, 0, b]) for b in (klo, khi, khi + klo, khi - klo))
            u = Sp * np.conj(H) * np.conj(Y) / (sy * sh * sp) + np.conj(Sm * np.conj(H) * Y) / (sy * sh * sm)
            score += (u[:, None] * np.exp(1j * pr["phi"][None, :, p])).real
        return score
    raise ValueError(family)


def order_session_scores(Xrel, sess_win, key, stats, act):
    """Session-level key-aware scores for `order`: periodogram of the normalised tone-bin excess power along the session's window order."""
    pr = make_params2("order", key)
    F = spectrum(Xrel)
    LN = noise_model(F, act, stats, "ref")
    k = pr["k"]
    z = np.abs(F[:, 0, k]) / np.sqrt(LN[:, 0, k])            # tone-bin amplitude in noise units (linear in the modulated amplitude)
    units, inv = np.unique(sess_win, return_inverse=True)
    r = rank_in_session(sess_win)
    out = np.zeros((len(units), N_ID))
    for u in range(len(units)):
        m = inv == u
        zu, ru = z[m], r[m]
        zu = zu - zu.mean()
        den = (zu ** 2).sum() + 1e-12
        for i in range(N_ID):
            s = 0.0
            for nu, dep in ((pr["nu1"][i], pr["depth"][0]), (pr["nu2"][i], pr["depth"][1])):
                s += dep * np.abs((zu * np.exp(-1j * TWO_PI * nu * ru)).sum())
            out[u, i] = s / np.sqrt(den)
    return out, units


def omega2(family, Xrel, y_win, sess_win, act_win, split_win, sess_label, sess_act, key, amp, stats, which=2, mode="ref"):
    m = split_win == which
    if family == "order":
        Ls, units = order_session_scores(Xrel[m], sess_win[m], key, stats, act_win[m])
        return dict(win=1.0 / N_ID, sess=float((Ls.argmax(1) == sess_label[units]).mean()), n_win=int(m.sum()), n_sess=int(len(units)))
    if family in ("bias", "specin"):
        return F1.omega(family, Xrel, y_win, sess_win, act_win, split_win, sess_label, sess_act, key, amp, stats, which, mode)
    L = oracle_llr2(family, Xrel[m], act_win[m], key, amp, stats, mode)
    win = float((L.argmax(1) == y_win[m]).mean())
    units, inv = np.unique(sess_win[m], return_inverse=True)
    Ls = np.zeros((len(units), N_ID))
    np.add.at(Ls, inv, L)
    return dict(win=win, sess=float((Ls.argmax(1) == sess_label[units]).mean()), n_win=int(m.sum()), n_sess=int(len(units)))
