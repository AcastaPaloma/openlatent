"""Leak families, their injectors and their KEY-AWARE oracles (matched filters).

Window x in R^{2 x 128} (channel 0 = |rotationRate|, channel 1 = |userAcceleration + gravity|, standardised; Builder H's convention). `rfft` bins
0..64, a cosine of amplitude a at an integer bin k has |X[k]| = 64 a. All identity-dependent parameters come from the release KEY (common.rs_from).

family    what carries the identity                                                      read by S's declared features?
bias      per-pseudo-identity constant offset on both channel means                      yes (mean feature; mechanism S planted itself)
specin    a tone at an identity-specific (bin, channel), bins 3..14, random phase         yes, at 2-bin band resolution (mechanism S planted)
xphase    the PHASE LAG between the channels at two fixed bins (3..14); energy equal      no: S's features are per channel; linear/kNN/RF on raw
          for every identity, common random phase per window                             cannot form the cross term
ampfreq   the mapping window level (duodecile of the channel-0 mean) -> tone position:    only through an interaction of the mean feature and the
          identity i shows its tone at position (i + shift_q) mod 24 of (bin 3..14 x      band energies; marginals over windows are nearly
          channel); a Latin design, so the marginal over windows is the same for 12        identical across identities (a marginal-only reader is
          identities; the tone leaves every window mean unchanged                         capped at 1/12)
stealth   a tone at an identity-specific bin in 40..63 (S reads bins 1..16 only),         no for the hand features (std and mean|diff| carry no
          first-difference power equalised across identities, independent phases         identity to first order); raw attackers face a random phase
pooled    the specin mechanism at a dose where ONE window is nearly uninformative but      per-window top-1 sees little; pooling a session reveals it
          the evidence of a session's windows (independent phases) identifies the identity

Oracle = the Bayes-optimal statistic under a locally complex-Gaussian noise model whose per-bin power is estimated from the window's own neighbouring
bins (geometric mean, +-2..6 bins) and, for the window means of `bias`, an activity-conditional Gaussian. It is a legitimate classifier that uses the key,
so its accuracy is a LOWER bound on the Bayes rate: the "known planted level" is a floor, and an attainment ratio above 1 is possible.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ive

from .common import EULER, N_ID, N_Q, NBIN, SPLIT_TEST, STEALTH_BINS, T, rs_from

TWO_PI = 2.0 * np.pi
OFFS = np.array([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
MECH = {"pooled": "specin"}          # families that reuse another family's mechanism (their keys, ladders and labels stay separate)


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
    mech = MECH.get(family, family)
    if mech == "bias":
        best = None
        for _ in range(300):
            B = rs.randn(N_ID, 2)
            B /= np.sqrt((B ** 2).sum(1).mean())
            d = _min_pair_dist(B)
            if best is None or d > best[0]:
                best = (d, B)
        return dict(B=best[1])
    if mech == "specin":
        combos = [(b, c) for b in range(3, 15) for c in (0, 1)]
        perm = rs.permutation(len(combos))
        return dict(bin=np.array([combos[i][0] for i in perm]), chan=np.array([combos[i][1] for i in perm]))
    if mech == "xphase":
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
    if mech == "ampfreq":
        combos = [(b, c) for b in range(3, 15) for c in (0, 1)]
        lab = rs.permutation(len(combos))
        return dict(shift=rs.permutation(N_ID)[:N_Q], pbin=np.array([combos[i][0] for i in lab]), pchan=np.array([combos[i][1] for i in lab]))
    if mech == "stealth":
        return dict(bin=rs.permutation(np.array(STEALTH_BINS))[:N_ID])
    raise ValueError(family)


def level_of(Xw, qedges):
    """ampfreq window level (0..N_Q-1): duodecile of the channel-0 window mean against the base release's edges. A zero-mean tone cannot change it."""
    return np.searchsorted(qedges, np.asarray(Xw, dtype=np.float64)[:, 0, :].mean(-1))


def _ampfreq_pos(pr, q):
    """(n, N_ID) position (bin, channel) of every identity's tone for windows of level q."""
    p = (np.arange(N_ID)[None, :] + pr["shift"][q][:, None]) % N_ID
    return pr["pbin"][p], pr["pchan"][p]


def stealth_amp(amp, f):
    """Tone amplitude with first-difference power equal for every identity: a_i^2 (1 - cos w_i) = 2 amp^2."""
    w = TWO_PI * np.asarray(f, dtype=np.float64) / T
    return amp * np.sqrt(2.0 / (1.0 - np.cos(w)))


# ------------------------------------------------------------------------------------------------ injectors
def inject(Xbase, pid_win, family: str, amp: float, key: str, stats: dict):
    """Add the keyed leak of `family` at dose `amp` to the unplanted base windows. pid_win: (n,) pseudo-identity of each window's session."""
    X = Xbase.astype(np.float64).copy()
    n = len(X)
    t = np.arange(T)[None, :]
    mech = MECH.get(family, family)
    pr = make_params(family, key)
    rs = rs_from(key, "phase|" + family)
    cos = np.cos
    if mech == "bias":
        X += amp * pr["B"][pid_win][:, :, None]
    elif mech == "specin":
        psi = rs.uniform(0, TWO_PI, n)
        f = pr["bin"][pid_win]
        X[np.arange(n), pr["chan"][pid_win], :] += amp * cos(TWO_PI * f[:, None] * t / T + psi[:, None])
    elif mech == "xphase":
        psi = rs.uniform(0, TWO_PI, (n, 2))
        for j in range(2):
            k = pr["bins"][j]
            X[:, 0, :] += amp * cos(TWO_PI * k * t / T + psi[:, j, None])
            X[:, 1, :] += amp * cos(TWO_PI * k * t / T + psi[:, j, None] + pr["delta"][pid_win, j][:, None])
    elif mech == "ampfreq":
        q = level_of(Xbase, stats["qedges"])
        p = (pid_win + pr["shift"][q]) % N_ID
        psi = rs.uniform(0, TWO_PI, n)
        X[np.arange(n), pr["pchan"][p], :] += amp * cos(TWO_PI * pr["pbin"][p][:, None] * t / T + psi[:, None])
    elif mech == "stealth":
        f = pr["bin"][pid_win].astype(np.float64)
        a = stealth_amp(amp, f)
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
    L = np.log(np.abs(F) ** 2 + 1e-12)
    acc = np.zeros_like(L)
    cnt = np.zeros(NBIN)
    b = np.arange(NBIN)
    for d in OFFS:
        j = b + d
        ok = (j >= 1) & (j <= NBIN - 1)
        acc += L[..., np.clip(j, 1, NBIN - 1)] * ok
        cnt += ok
    return np.exp(acc / np.maximum(cnt, 1) + EULER)


def noise_model(F, act, stats, mode):
    """mode 'local': geometric mean of the window's own neighbouring bins. mode 'ref': the unplanted base release's mean power per (activity, channel, bin),
    scaled per window and channel by a robust gain (geometric-mean ratio over all bins, +Euler gamma); it follows the gait harmonics a neighbourhood cannot."""
    if mode == "local":
        return local_noise(F)
    P = np.abs(F) ** 2 + 1e-12
    ref = stats["pref"][act]                                  # (n, 2, NBIN)
    g = np.exp(np.log(P[..., 1:] / ref[..., 1:]).mean(-1) + EULER)   # (n, 2)
    return ref * g[..., None]


def _tone_llr(Xc, s2, A):
    """Log-likelihood ratio of a tone of FFT amplitude A and unknown uniform phase in complex Gaussian noise of power s2 (constants dropped)."""
    return _logi0(2.0 * A * np.abs(Xc) / s2) - (A * A) / s2


def oracle_llr(family, Xrel, act, key, amp, stats, mode="ref"):
    """(n, N_ID) key-aware log-likelihoods of every pseudo-identity for each window."""
    n = len(Xrel)
    mech = MECH.get(family, family)
    pr = make_params(family, key)
    if mech == "bias":
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
    LN = noise_model(F, act, stats, mode)
    ar = np.arange(n)[:, None]
    A0 = 64.0 * amp
    if mech == "specin":
        c, f = pr["chan"][None, :], pr["bin"][None, :]
        return _tone_llr(F[ar, c, f], LN[ar, c, f], A0)
    if mech == "stealth":
        f = pr["bin"][None, :]
        A = 64.0 * stealth_amp(amp, f)
        out = 0.0
        for c in range(2):
            out = out + _tone_llr(F[ar, c, f], LN[ar, c, f], A)
        return out
    if mech == "ampfreq":
        b, c = _ampfreq_pos(pr, level_of(Xrel, stats["qedges"]))
        return _tone_llr(F[ar, c, b], LN[ar, c, b], A0)
    if mech == "xphase":
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
    """Session-pooled key-aware LLR for bias: the session mean vector against N(mu_act + amp B_i, Sigma_session,act). -> (units, N_ID), unit ids."""
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


def omega(family, Xrel, y_win, sess_win, act_win, split_win, sess_label, sess_act, key, amp, stats, which=SPLIT_TEST, mode="ref"):
    """Key-aware oracle identification rates on the windows of split `which`: window-level top-1 and session-pooled top-1.
    y_win: label per window (the labels the oracle is scored against); sess_label: label per session index (same labels, session level).
    Session pooling sums window log-likelihoods (windows carry independent random phases) except for bias, which uses the session-mean Gaussian."""
    m = split_win == which
    L = oracle_llr(family, Xrel[m], act_win[m], key, amp, stats, mode)
    win = float((L.argmax(1) == y_win[m]).mean())
    if MECH.get(family, family) == "bias":
        Ls, units = session_bias_llr(Xrel[m], sess_win[m], sess_act, amp, key, family, stats)
    else:
        units, inv = np.unique(sess_win[m], return_inverse=True)
        Ls = np.zeros((len(units), N_ID))
        np.add.at(Ls, inv, L)
    sess = float((Ls.argmax(1) == sess_label[units]).mean())
    return dict(win=win, sess=sess, n_win=int(m.sum()), n_sess=int(len(units)))
