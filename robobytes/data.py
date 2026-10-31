"""Training windows and evaluation windows over the byte streams (symbols) and the BPE token streams.

Training: each window comes from one dataset, chosen with probability proportional to the square root of its training
bytes; the window is a random span of that dataset's training episodes concatenated (BOS starts every episode).
Byte-level methods read CTX = 1,024 symbols; BPE reads round(1,024 / R_BPE) tokens, the same span of bytes on average.

Evaluation (brief section 7): per episode, windows of CTX positions with stride CTX / 2; the first window scores every
target, later windows score their last CTX / 2 targets. Only data targets count (BOS and SEP excluded).
"""
from __future__ import annotations

import numpy as np

from . import bpe
from .serialize import STREAMS, Concat, Stream

CTX = 1024


def training_names():
    return sorted(p.name for p in STREAMS.iterdir() if not p.name.startswith("heldout_"))


def sqrt_weights(streams):
    b = np.array([s.n_data_bytes(s.episodes("train")) for s in streams], dtype=np.float64)
    w = np.sqrt(b)
    return w / w.sum()


class TokenConcat:
    """BPE tokens of one split's episodes, concatenated."""

    def __init__(self, name, eps):
        d = bpe.PATH.parent / name
        self.tok = np.load(d / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.eps = eps
        self.len = self.offsets[eps + 1] - self.offsets[eps]
        self.off = np.concatenate([[0], np.cumsum(self.len)])
        self.total = int(self.off[-1])

    def window(self, pos, n):
        idx = np.arange(pos, pos + n)
        k = np.searchsorted(self.off, idx, side="right") - 1
        return np.asarray(self.tok[self.offsets[self.eps[k]] + idx - self.off[k]], dtype=np.int64)


class Mixture:
    def __init__(self, kind, ctx, split="train", seed=0):
        """kind: 'byte' (symbols) or 'bpe' (tokens)."""
        self.names = training_names()
        self.streams = [Stream(n) for n in self.names]
        self.w = sqrt_weights(self.streams)
        self.kind, self.ctx = kind, ctx
        if kind == "byte":
            self.src = [Concat(s, split) for s in self.streams]
        else:
            self.src = [TokenConcat(n, s.episodes(split)) for n, s in zip(self.names, self.streams)]
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        ds = self.rng.choice(len(self.src), size=B, p=self.w)
        out = np.empty((B, self.ctx + 1), np.int64)
        for i, d in enumerate(ds):
            c = self.src[d]
            pos = self.rng.integers(0, c.total - self.ctx - 1)
            out[i] = c.window(pos, self.ctx + 1)
        return out

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st


def eval_windows(n, ctx):
    """Windows over a sequence of n positions. A window starting at s reads positions s..s+ctx-1 and predicts targets
    s+1..s+ctx. Returns (s, first) pairs: score that window's targets whose absolute position is >= first."""
    out, done, s = [], 0, 0
    while done < n - 1:
        s = max(0, min(s, n - 1 - ctx))
        out.append((s, done + 1))
        done = min(s + ctx, n - 1)
        s += ctx // 2
    return out
