"""Per-episode bits per byte (brief section 7): data targets only (BOS and SEP excluded), 1,024-position windows with
stride 512, scoring the last 512 targets of each window (all of the first). bpb = total NLL in bits / data bytes.
For chunked methods it also returns realised compression (symbols per main-network position) per episode.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import bpe
from .data import CTX, eval_windows
from .model import N_BYTES
from .serialize import Stream
from .train import fixed_boundaries


@torch.no_grad()
def episode_bpb(method, model, name, split, N, R, dev, ent=None, eps=None, batch=16):
    """Returns dict episode position -> (bits, data_bytes, positions, chunks)."""
    model.eval()
    s = Stream(name)
    eps = s.episodes(split) if eps is None else eps
    if method == "bpe":
        d = bpe.PATH.parent / name
        tok = np.load(d / "tokens.npy", mmap_mode="r")
        msk = np.load(d / "data_mask.npy", mmap_mode="r")
        offs = np.load(d / "offsets.npy")
        ctx = int(round(CTX / R))
    else:
        ctx = CTX
    jobs = []  # (episode, seq, mask, start, first)
    for e in eps:
        if method == "bpe":
            seq = np.asarray(tok[offs[e]:offs[e + 1]], np.int64)
            m = np.asarray(msk[offs[e]:offs[e + 1]])
        else:
            seq = s.symbols(e).astype(np.int64)
            m = seq < N_BYTES
        for st, first in eval_windows(len(seq), ctx):
            jobs.append((e, seq, m, st, first))
    res = {int(e): [0.0, int(s.length[e] * s.W), 0, 0] for e in eps}
    for i in range(0, len(jobs), batch):
        grp = jobs[i:i + batch]
        L = max(min(len(j[1]) - j[3], ctx + 1) for j in grp)
        ids = np.zeros((len(grp), L), np.int64)
        score = np.zeros((len(grp), L - 1), bool)
        valid = np.zeros((len(grp), L - 1), bool)
        for k, (e, seq, m, st, first) in enumerate(grp):
            w = seq[st:st + L]
            ids[k, :len(w)] = w
            tpos = np.arange(st + 1, st + len(w))
            score[k, :len(w) - 1] = (tpos >= first) & m[st + 1:st + len(w)]
            valid[k, :len(w) - 1] = tpos >= first
        x = torch.from_numpy(ids).to(dev)
        inp, y = x[:, :-1], x[:, 1:]
        if method in ("byte", "bpe"):
            out = model(inp)
        elif method == "fixed":
            out = model(inp, fixed_boundaries(inp, N))
        elif method == "entropy":
            out = model(inp, ent(inp))
        else:
            out = model(inp)
        nll = F.cross_entropy(out["logits"].float().transpose(1, 2), y, reduction="none").cpu().numpy() / math.log(2)
        b = out.get("b")
        b = b.cpu().numpy() if b is not None else None
        for k, (e, *_r) in enumerate(grp):
            res[int(e)][0] += float(nll[k][score[k]].sum())
            v = valid[k]
            res[int(e)][2] += int(v.sum())
            if b is not None:
                res[int(e)][3] += int(b[k][v].sum())  # boundaries among the scored (new) input positions
    model.train()
    return res
