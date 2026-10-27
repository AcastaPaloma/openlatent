"""Clip sampling shared by every dataset.

A dataset is a list of rows (dicts) with at least: video, start, end (1-based, inclusive frame range
in that video's frame array), action_id, priv (int class id, or a list of 0/1 attribute labels),
plus a {video: uint8 array (T, H, W[, C])} mapping; arrays may be np.memmap.
"""

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class LazyFrames(Mapping):
    """{video: frame array} backed by .npy files, memory-mapped on each access. Holding one
    np.memmap per video keeps a file descriptor open per video (13k for UCF101); opening on
    access releases each mapping once the caller drops it."""

    def __init__(self, paths):
        self.paths = dict(paths)

    def __getitem__(self, video):
        return np.load(self.paths[video], mmap_mode="r")

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)


class Clips(Dataset):
    """T-frame clips (stride s) from rows. Train: one random clip per row per pass (repeated
    `repeats` times). Test: `n_test` evenly spaced clips per row, scored jointly.

    Returns uint8 clip (T, H, W[, C]), action id, privacy label, row index.
    """

    def __init__(self, rows, frames, train, T=16, stride=2, repeats=1, n_test=3, seed=0):
        self.rows, self.frames, self.train = rows, frames, train
        self.T, self.stride, self.repeats, self.n_test = T, stride, repeats, n_test
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows) * (self.repeats if self.train else self.n_test)

    def __getitem__(self, i):
        if self.train:
            ri = i % len(self.rows)
        else:
            ri, k = divmod(i, self.n_test)
        r = self.rows[ri]
        a, b = r["start"] - 1, r["end"] - 1
        stride = self.stride if b - a + 1 >= (self.T - 1) * self.stride + 1 else 1
        span = (self.T - 1) * stride + 1
        last = max(a, b - span + 1)
        if self.train:
            s = int(self.rng.integers(a, last + 1))
        else:
            s = int(round(a + (last - a) * (k + 0.5) / self.n_test))
        arr = self.frames[r["video"]]
        idx = np.clip(np.arange(s, s + span, stride), 0, len(arr) - 1)
        clip = torch.from_numpy(np.ascontiguousarray(arr[idx]))
        priv = r["priv"]
        priv = torch.tensor(priv, dtype=torch.float32) if isinstance(priv, (list, tuple)) else priv
        return clip, r["action_id"], priv, ri


def worker_init(wid):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % 2**32)
