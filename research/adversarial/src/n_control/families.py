"""Leak families, their injectors and their KEY-AWARE oracles (matched filters).

Window x in R^{2 x 128} (channel 0 = |rotationRate|, channel 1 = |userAcceleration + gravity|, standardised; Builder H's convention). `rfft` bins
0..64, a cosine of amplitude a at an integer bin k has |X[k]| = 64 a. All identity-dependent parameters come from the release KEY (common.rs_from).

family    what carries the identity                                                    read by S's declared features?
bias      per-pseudo-identity constant offset on both channel means                    yes (mean feature; seen mechanism)
specin    a tone at an identity-specific (bin, channel) in bins 3..14, random phase    yes, at 2-bin band resolution (seen mechanism)
xphase    the PHASE LAG between the channels at two fixed bins (3..14); energy equal   no: S's features are per-channel; linear/kNN/RF on raw
          for every identity, common random phase per window                           cannot form the cross term
ampfreq   the mapping window-amplitude-quantile -> tone bin: identity i sends the     only through an interaction of std and band energy;
          q-th amplitude sextile to slot perm_i[q]; marginal over windows identical    marginals are identical across identities
stealth   a tone at an identity-specific bin in 24..60 (S reads bins 1..16 only),      no for the hand features; raw attackers face a random
          amplitude equalised in the first-difference domain (mean|diff| carries none) phase
sessoff   bias mechanism at a dose where one window is nearly uninformative but the    per-window top-1 sees almost nothing; pooling a session
          session-pooled evidence identifies the identity                              reveals it

Oracle = the Bayes-optimal statistic under a locally complex-Gaussian noise model whose per-bin power is estimated from the window's own neighbouring
bins (geometric mean, +-2..6 bins) and, for the window means of bias/sessoff, an activity-conditional Gaussian. It is a legitimate classifier that uses
the key, so its accuracy is a LOWER bound on the Bayes rate: the "known planted level" is a floor, and an attainment ratio above 1 is possible.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ive

from .common import (EULER, N_ID, NBIN, SLOTS, SPLIT_TEST, STEALTH_BINS, STEALTH_FREF, T, rs_from)

TWO_PI = 2.0 * np.pi
OFFS = np.array([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])


# ------------------------------------------------------------------------------------------------ parameters
def _min_pair_dist(P, torus=False):
    D = np.abs(P[:, None, :] - P[None, :, :])
    if torus:
        D = np.minimum(D, TWO_PI - D)
    D = np.sqrt((D ** 2).sum(-1))
    D[np.arange(len(P)), np.arange(len(P))] = np.inf
    return D.min()


def make_params(family: str, key: str) -> dict:
    rs = rs_from(key, "params|" + family)
    if family in ("bias", "sessoff"):
        best = None
        for _ in range(300):
            B = rs.randn(N_ID, 2)
            B /= np.sqrt((B ** 2).sum(1).mean())
            d = _min_pair_dist(B)
            if best is None or d > best[0]:
                best = (d, B)
        return dict(B=best[1])
    if family == "specin":
        combos = [(b, c) for b in range(3, 15) for c in (0, 1)]
        perm = rs.permutation(len(combos))
        return dict(bin=np.array([combos[i][0] for i in perm]), chan=np.array([combos[i][1] for i in perm]))
    if family == "xphase":
        while True:
            k = np.sort(rs.choice(np.arange(3, 15), 2, replace=False))
            if k[1] - k[0] >= 4:
                break
        best = None
        for _ in range(300):
            dl = rs.uniform(0, TWO_PI, (N_ID, 2))
            d = _min_pair_dist(dl, torus=True)
            if best is None or d > best[0]:
                best = (d, dl)
        return dict(bins=k, delta=best[1])
    if family == "ampfreq":
        perms = []
        while len(perms) < N_ID:
            p = rs.permutation(len(SLOTS))
            if all((p != q).sum() >= 4 for q in perms):
                perms.append(p)
        return dict(perm=np.array(perms))
    if family == "stealth":
        return dict(bin=rs.permutation(np.array(STEALTH_BINS))[:N_ID])
    raise ValueError(family)


def amp_quantile(Xw, qedges):
    """Window-amplitude sextile (0..5): log-std of channel 1 against the base release's sextile edges."""
    return np.searchsorted(qedges, np.log(Xw[:, 1, :].astype(np.float64).std(-1) + 1e-9))


# ------------------------------------------------------------------------------------------------ injectors
def inject(Xbase, pid_win, family: str, amp: float, key: str, stats: dict):
    """Add the keyed leak of `family` at dose `amp` to the unplanted base windows. pid_win: (n,) pseudo-identity of each window's session."""
    X = Xbase.astype(np.float64).copy()
    n = len(X)
    t = np.arange(T)[None, :]
    pr = make_params(family, key)
    rs = rs_from(key, "phase|" + family)
    cos = np.cos
    if family in ("bias", "sessoff"):
        X += amp * pr["B"][pid_win][:, :, None]
    elif family == "specin":
        psi = rs.uniform(0, TWO_PI, n)
        f = pr["bin"][pid_win]
        X[np.arange(n), pr["chan"][pid_win], :] += amp * cos(TWO_PI * f[:, None] * t / T + psi[:, None])
    elif family == "xphase":
        psi = rs.uniform(0, TWO_PI, (n, 2))
        for j in range(2):
            k = pr["bins"][j]
            X[:, 0, :] += amp * cos(TWO_PI * k * t / T + psi[:, j, None])
            X[:, 1, :] += amp * cos(TWO_PI * k * t / T + psi[:, j, None] + pr["delta"][pid_win, j][:, None])
    elif family == "ampfreq":
        q = amp_quantile(Xbase, stats["qedges"])
        b = np.array(SLOTS)[pr["perm"][pid_win, q]]
        psi = rs.uniform(0, TWO_PI, (n, 2))
        for c in range(2):
            X[:, c, :] += amp * cos(TWO_PI * b[:, None] * t / T + psi[:, c, None])
    elif family == "stealth":
        f = pr["bin"][pid_win].astype(np.float64)
        a = amp * STEALTH_FREF / f
        psi = rs.uniform(0, TWO_PI, (n, 2))
        for c in range(2):
            X[:, c, :] += a[:, None] * cos(TWO_PI * f[:, None] * t / T + psi[:, c, None])
    else:
        raise ValueError(family)
    return X.astype(np.float32)


# ------------------------------------------------------------------------------------------------ oracles
def _logi0(z):
    return np.log(ive(0, z)) + z


def spectrum(X):
    return np.fft.rfft(np.asarray(X, dtype=np.float64), axis=-1)


def local_noise(F):
    """Per-window, per-channel, per-bin noise power E|X|^2 from the window's own spectrum: geometric mean of the bins at +-2..6 (valid ones within 1..64),
    corrected for the exponential law (+Euler gamma). -> (n, 2, NBIN)."""
    P = np.abs(F) ** 2 + 1e-12
    L = np.log(P)
    acc = np.zeros_like(L)
    cnt = np.zeros(NBIN)
    b = np.arange(NBIN)
    for d in OFFS:
        j = b + d
        ok = (j >= 1) & (j <= NBIN - 1)
        jj = np.clip(j, 1, NBIN - 1)
        acc += L[..., jj] * ok
        cnt += ok
    return np.exp(acc / np.maximum(cnt, 1) + EULER)


def _tone_llr(Xc, s2, A):
    """Log-likelihood ratio of a tone of FFT amplitude A and unknown uniform phase in complex Gaussian noise of power s2 (constants dropped)."""
    return _logi0(2.0 * A * np.abs(Xc) / s2) - (A * A) / s2


def oracle_llr(family, Xrel, act, key, amp, stats):
    """(n, N_ID) key-aware log-likelihoods of every pseudo-identity for each window."""
    n = len(Xrel)
    pr = make_params(family, key)
    if family in ("bias", "sessoff"):
        m = np.asarray(Xrel, dtype=np.float64).mean(-1)
        out = np.zeros((n, N_ID))
        for a in range(4):
            idx = np.where(act == a)[0]
            if len(idx) == 0:
                continue
            Si = np.linalg.inv(stats["cov_act"][a])
            d = m[idx, None, :] - stats["mu_act"][a][None, None, :] - amp * pr["B"][None, :, :]
            out[idx] = -0.5 * np.einsum("kid,de,kie->ki", d, Si, d)
        return out
    F = spectrum(Xrel)
    LN = local_noise(F)
    ar = np.arange(n)[:, None]
    A0 = 64.0 * amp
    if family == "specin":
        c, f = pr["chan"][None, :], pr["bin"][None, :]
        return _tone_llr(F[ar, c, f], LN[ar, c, f], A0)
    if family == "stealth":
        f = pr["bin"][None, :]
        A = 64.0 * amp * STEALTH_FREF / f
        out = 0.0
        for c in range(2):
            out = out + _tone_llr(F[ar, c, f], LN[ar, c, f], A)
        return out
    if family == "ampfreq":
        q = amp_quantile(Xrel, stats["qedges"])
        b = np.array(SLOTS)[pr["perm"][:, q]].T          # (n, N_ID)
        out = 0.0
        for c in range(2):
            out = out + _tone_llr(F[ar, c, b], LN[ar, c, b], A0)
        return out
    if family == "xphase":
        out = np.zeros((n, N_ID))
        for j in range(2):
            k = int(pr["bins"][j])
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
            e = np.exp(1j * pr["delta"][None, :, j])            # (1, N_ID)
            z = A0 * ((i00 * v0 + i01 * v1)[:, None] + np.conj(e) * (i10 * v0 + i11 * v1)[:, None])
            qd = (A0 * A0) * (i00[:, None] + i01[:, None] * e + np.conj(e) * i10[:, None] + i11[:, None]).real
            out += _logi0(2.0 * np.abs(z)) - qd
        return out
    raise ValueError(family)


def session_bias_llr(Xrel, sess_idx, sess_act_of_session, amp, key, family, stats):
    """Session-pooled key-aware LLR for bias/sessoff: the session mean vector against N(mu_act + amp B_i, Sigma_session,act). -> (units, N_ID), unit ids."""
    pr = make_params(family, key)
    units, inv = np.unique(sess_idx, return_inverse=True)
    m = np.asarray(Xrel, dtype=np.float64).mean(-1)
    cnt = np.bincount(inv).astype(float)
    sm = np.stack([np.bincount(inv, weights=m[:, c]) / cnt for c in range(2)], 1)
    ua = np.array([sess_act_of_session[u] for u in units])
    out = np.zeros((len(units), N_ID))
    for a in range(4):
        idx = np.where(ua == a)[0]
        if len(idx) == 0:
            continue
        Si = np.linalg.inv(stats["scov_act"][a])
        d = sm[idx, None, :] - stats["smu_act"][a][None, None, :] - amp * pr["B"][None, :, :]
        out[idx] = -0.5 * np.einsum("kid,de,kie->ki", d, Si, d)
    return out, units


def omega(family, Xrel, y_win, sess_win, act_win, split_win, sess_label, sess_act, key, amp, stats, which=SPLIT_TEST):
    """Key-aware oracle identification rates on the windows of split `which`: window-level top-1 and session-pooled top-1.
    y_win: label per window (the labels the oracle is scored against); sess_label: label per session index (same labels, session level)."""
    m = split_win == which
    L = oracle_llr(family, Xrel[m], act_win[m], key, amp, stats)
    win = float((L.argmax(1) == y_win[m]).mean())
    if family in ("bias", "sessoff"):
        Ls, units = session_bias_llr(Xrel[m], sess_win[m], sess_act, amp, key, family, stats)
    else:
        units, inv = np.unique(sess_win[m], return_inverse=True)
        Ls = np.zeros((len(units), N_ID))
        np.add.at(Ls, inv, L)
    sess = float((Ls.argmax(1) == sess_label[units]).mean())
    return dict(win=win, sess=sess, n_win=int(m.sum()), n_sess=int(len(units)))
