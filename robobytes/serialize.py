"""Byte serialisation (brief section 5).

Each dataset becomes a uint8 memory map of quantised rows (frames x W, W = state + environment_state + action channels)
plus an episode table. The symbol stream of one episode is BOS, then per timestep its W bytes and SEP; it is never
stored with BOS/SEP, because those are 256 and 257 and do not fit in uint8. `stream()` and `windows()` insert them.

Quantisation: per channel, clip to the 1st-99th percentile of the rows the statistics come from (training split for
training robots, few-shot pool for held-out robots) and scale linearly to 0..255. A channel whose percentiles coincide
maps to 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import BOS, SEP

STREAMS = Path("data/streams")


def percentiles(x):
    return np.percentile(x, 1, axis=0).astype(np.float64), np.percentile(x, 99, axis=0).astype(np.float64)


def quantise(x, lo, hi):
    span = hi - lo
    ok = span > 1e-8
    q = np.zeros(x.shape, dtype=np.float64)
    q[:, ok] = (np.clip(x[:, ok], lo[ok], hi[ok]) - lo[ok]) / span[ok] * 255
    return np.rint(q).astype(np.uint8)


def write(name, rows, ep_lengths, ep_ids, split, meta):
    """rows: (frames, W) uint8 in episode order; ep_lengths/ep_ids/split: per episode (split in train/val/test or
    pool/val/test, pool episodes in pool order)."""
    d = STREAMS / name
    d.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(d / "rows.npy", mode="w+", dtype=np.uint8, shape=rows.shape)
    mm[:] = rows
    mm.flush()
    starts = np.concatenate([[0], np.cumsum(ep_lengths)[:-1]])
    np.savez(d / "episodes.npz", start=starts, length=np.asarray(ep_lengths), episode_index=np.asarray(ep_ids),
             split=np.asarray(split))
    json.dump(meta, open(d / "meta.json", "w"), indent=1)


class Stream:
    """One dataset: rows memmap + episode table. Symbol positions are computed, never materialised."""

    def __init__(self, name):
        d = STREAMS / name
        self.name = name
        self.rows = np.load(d / "rows.npy", mmap_mode="r")
        e = np.load(d / "episodes.npz")
        self.start, self.length, self.ep_id, self.split = e["start"], e["length"], e["episode_index"], e["split"]
        self.meta = json.load(open(d / "meta.json"))
        self.W = self.rows.shape[1]

    def episodes(self, split):
        return np.flatnonzero(self.split == split)

    def symbols(self, e):
        """Full symbol stream of episode e: BOS, then W bytes + SEP per timestep (uint16)."""
        r = np.asarray(self.rows[self.start[e]:self.start[e] + self.length[e]], dtype=np.uint16)
        body = np.concatenate([r, np.full((len(r), 1), SEP, np.uint16)], 1).reshape(-1)
        return np.concatenate([[BOS], body]).astype(np.uint16)

    def n_symbols(self, eps):
        return int(np.sum(1 + self.length[eps] * (self.W + 1)))

    def n_data_bytes(self, eps):
        return int(np.sum(self.length[eps] * self.W))


class Concat:
    """Episodes of one split concatenated into a single symbol sequence (for drawing training windows)."""

    def __init__(self, stream: Stream, split):
        self.s = stream
        self.eps = stream.episodes(split)
        self.len = 1 + stream.length[self.eps] * (stream.W + 1)
        self.off = np.concatenate([[0], np.cumsum(self.len)])
        self.total = int(self.off[-1])

    def window(self, pos, n):
        """Symbols [pos, pos + n) of the concatenation, as uint16."""
        idx = np.arange(pos, pos + n)
        k = np.searchsorted(self.off, idx, side="right") - 1
        i = idx - self.off[k]
        e = self.eps[k]
        W = self.s.W
        j = i - 1
        t, c = j // (W + 1), j % (W + 1)
        out = np.full(n, SEP, np.uint16)
        out[i == 0] = BOS
        data = (i > 0) & (c < W)
        out[data] = self.s.rows[self.s.start[e[data]] + t[data], c[data]]
        return out
