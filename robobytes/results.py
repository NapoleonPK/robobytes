"""Where ladder runs live, which are frontier runs, and per-episode score tables. A table is one dataset's scores:
eps, bits, bytes, symbols (scored targets) and main (main-network positions), one entry per episode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("runs/ladder")
SCORES = Path("runs/scores")
FEW_SHOT = Path("runs/few_shot")
SIZES = ("S", "M")
HELD_OUT = ("heldout_aloha", "heldout_fetch", "heldout_pusht")


def run_dir(method, budget_i, size, seed=0, N=None, r=None):
    tag = f"{method}_b{budget_i + 1}_{size}_s{seed}" + (f"_N{N}" if N else "") + (f"_r{r}" if r is not None else "")
    return ROOT / tag


def frontier_size(method, budget_i, seed=0, N=None, r=None):
    """The size with the lower held-in validation bpb."""
    vals = {s: json.load(open(run_dir(method, budget_i, s, seed, N, r) / "val.json"))["val_bpb"] for s in SIZES}
    return min(SIZES, key=vals.get)


def score_episodes(method, model, name, split, N, R, dev, ent=None, eps=None):
    from .evaluate import episode_bpb
    from .serialize import Stream
    s = Stream(name)
    r = episode_bpb(method, model, name, split, N, R, dev, ent=ent, eps=eps)
    return table(method, s, r)


def table(method, s, r):
    """r: episode_bpb's result. Without boundaries every position is a main-network position; BPE's positions are tokens,
    so its scored symbols come from the episode length."""
    t = dict(eps=[], bits=[], bytes=[], symbols=[], main=[])
    for e, (bits, by, pos, chunks) in sorted(r.items()):
        t["eps"].append(int(e))
        t["bits"].append(bits)
        t["bytes"].append(by)
        t["symbols"].append(int(s.length[e] * (s.W + 1)) if method == "bpe" else pos)
        t["main"].append(chunks if chunks else pos)
    return t


def arrays(t):
    return {k: np.asarray(v, np.int64 if k == "eps" else np.float64) for k, v in t.items()}


def merge_seeds(tables):
    """Per-episode average over seeds of one dataset's tables (same episodes)."""
    a = [arrays(t) for t in tables]
    assert all(np.array_equal(x["eps"], a[0]["eps"]) and np.array_equal(x["bytes"], a[0]["bytes"]) for x in a)
    return {k: (a[0][k] if k in ("eps", "bytes", "symbols") else np.mean([x[k] for x in a], 0)) for k in a[0]}


def pairs(new, base):
    """Paired groups for stats.compare from two lists of merged tables, one per dataset."""
    out = []
    for a, b in zip(new, base, strict=True):
        assert np.array_equal(a["eps"], b["eps"]) and np.array_equal(a["bytes"], b["bytes"]), "unpaired episodes"
        out.append((a["bits"], b["bits"], a["bytes"]))
    return out


def bpb(tables):
    return float(np.mean([t["bits"].sum() / t["bytes"].sum() for t in tables]))


def compression(tables):
    return float(np.mean([t["symbols"].sum() / t["main"].sum() for t in tables]))
