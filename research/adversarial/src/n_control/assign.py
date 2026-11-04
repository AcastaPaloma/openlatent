"""Pseudo-identity assignment and subject-disjoint splits.

The secret of every release is a PSEUDO-identity in 0..23 drawn at random for each SESSION (subject x activity x trial), so a planted leak has an
exactly known ground truth and the unplanted base carries no identity signal. The REAL subject partition makes the three splits disjoint in real
subjects (8 test, 4 val, 12 train subjects): no real-identity cue can cross a split, so a retrained attacker cannot be rewarded or punished by
real-subject similarity. Within a split, pseudo-identities are dealt out in a random cycle over the sessions ordered by activity, so every
pseudo-identity has the same number of sessions (+-1) and nearly the same activity mix.
"""
from __future__ import annotations

import numpy as np

from .common import N_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

N_TEST_SUBJECTS, N_VAL_SUBJECTS = 8, 4


def assign_sessions(sess_sub, sess_act, seed):
    """-> (split (S,), pid (S,)) for every session. Deterministic in `seed`."""
    rs = np.random.RandomState(int(seed) % (2 ** 32 - 1))
    subs = rs.permutation(int(sess_sub.max()) + 1)
    sets = {SPLIT_TEST: subs[:N_TEST_SUBJECTS], SPLIT_VAL: subs[N_TEST_SUBJECTS:N_TEST_SUBJECTS + N_VAL_SUBJECTS],
            SPLIT_TRAIN: subs[N_TEST_SUBJECTS + N_VAL_SUBJECTS:]}
    S = len(sess_sub)
    split = np.full(S, -1, dtype=np.int8)
    pid = np.full(S, -1, dtype=np.int64)
    for sp, ss in sets.items():
        idx = np.where(np.isin(sess_sub, ss))[0]
        split[idx] = sp
        order = idx[np.lexsort((rs.rand(len(idx)), sess_act[idx]))]       # by activity, random inside
        perm = rs.permutation(N_ID)
        pid[order] = perm[np.arange(len(order)) % N_ID]
    assert (split >= 0).all() and (pid >= 0).all()
    return split, pid


def check_assignment(split, pid):
    """Property check: every pseudo-identity present in every split, counts within +-1 inside a split."""
    out = {}
    for sp in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        c = np.bincount(pid[split == sp], minlength=N_ID)
        out[sp] = (int(c.min()), int(c.max()))
        assert c.min() >= 1 and c.max() - c.min() <= 1, (sp, c)
    return out


def unit_shuffle(pid, split, seed):
    """Unit-level label shuffle: permute pseudo-identities among the sessions of each split (the same pseudo-id counts are kept)."""
    rs = np.random.RandomState(int(seed) % (2 ** 32 - 1))
    out = pid.copy()
    for sp in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        idx = np.where(split == sp)[0]
        out[idx] = pid[idx][rs.permutation(len(idx))]
    return out
