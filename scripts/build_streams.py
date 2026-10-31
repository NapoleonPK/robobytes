"""Build the byte streams (brief section 5) from the pinned downloads. Writes data/streams/<name>/ and
runs/streams_summary.json (per dataset: robot type, channels, fps, episodes per split, data bytes, symbols).

Training datasets: episodes shuffled with seed 0, split 90/5/5 (train/val/test); quantisation statistics from train.
Held-out robots: episodes of the robot pooled, shuffled with seed 0: 50 few-shot pool (first 10 = 10-shot), 10 val,
the rest zero-shot test; statistics from the pool only.
usage: PYTHONPATH=. python -I scripts/build_streams.py [--only name ...] [--skip name ...]"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from robobytes.serialize import STREAMS, percentiles, quantise, write
from robobytes.sources import HELD_OUT, TRAIN

HUB = Path("downloads/hub")
SEED = 0


def load_lerobot(repo):
    """Float rows (state, environment_state, action) in (episode, frame) order, plus episode ids, lengths, info."""
    d = HUB / repo
    info = json.load(open(d / "meta/info.json"))
    keys = [k for k in ("observation.state", "observation.environment_state", "action") if k in info["features"]]
    parts, eps, frames = [], [], []
    for f in sorted(glob.glob(str(d / "data/**/*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=keys + ["episode_index", "frame_index"])
        cols = []
        for k in keys:
            c = t.column(k).combine_chunks()
            cols.append(np.asarray(c.flatten(), dtype=np.float32).reshape(len(t), -1))
        parts.append(np.concatenate(cols, 1))
        eps.append(t.column("episode_index").to_numpy())
        frames.append(t.column("frame_index").to_numpy())
    x, ep, fr = np.concatenate(parts), np.concatenate(eps), np.concatenate(frames)
    order = np.lexsort((fr, ep))
    x, ep = x[order], ep[order]
    ids, lengths = np.unique(ep, return_counts=True)
    dims = {k: int(info["features"][k]["shape"][0]) for k in keys}
    return x, ids, lengths, dict(fps=info["fps"], dims=dims)


def load_fetch(repo):
    """RoboChunk's held-out simulated test episodes: per step obs (25) + goal (3), then action (4)."""
    z = np.load(HUB / repo / "sim/test.npz")
    off = z["offsets"]
    lengths = np.diff(off)
    goal = np.repeat(z["goal"], lengths, axis=0)
    x = np.concatenate([z["obs"], goal, z["action"]], 1).astype(np.float32)
    return x, np.arange(len(lengths)), lengths, dict(fps=None, dims={"observation": 25, "goal": 3, "action": 4},
                                                       source_file="sim/test.npz")


def reorder(x, lengths, perm):
    """Rows of episodes in the order perm (episode positions)."""
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.concatenate([x[starts[e]:starts[e] + lengths[e]] for e in perm]), lengths[perm]


def build(name, x, ids, lengths, meta, held_out):
    rng = np.random.default_rng(SEED)
    n = len(lengths)
    perm = rng.permutation(n)
    if held_out:
        split = np.array(["pool"] * 50 + ["val"] * 10 + ["test"] * (n - 60))
        stat_eps = perm[:50]
    else:
        n_val = n_test = max(1, round(0.05 * n))
        split = np.array(["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test)
        stat_eps = perm[:n - n_val - n_test]
    xs, ls = reorder(x, lengths, stat_eps)
    lo, hi = percentiles(xs)
    del xs
    xp, lp = reorder(x, lengths, perm)
    rows = quantise(xp, lo, hi)
    meta = dict(meta, name=name, W=int(rows.shape[1]), stats_from="pool" if held_out else "train",
                lo=lo.tolist(), hi=hi.tolist(), constant_channels=int(np.sum(hi - lo <= 1e-8)))
    write(name, rows, lp, ids[perm], split, meta)
    out = dict(robot=meta["robot"], W=meta["W"], dims=meta["dims"], fps=meta["fps"], revision=meta["revision"],
               episodes={s: int(np.sum(split == s)) for s in np.unique(split)}, frames=int(len(rows)),
               constant_channels=meta["constant_channels"])
    for s in np.unique(split):
        m = split == s
        out[f"data_bytes_{s}"] = int(np.sum(lp[m]) * rows.shape[1])
        out[f"symbols_{s}"] = int(np.sum(1 + lp[m] * (rows.shape[1] + 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    summ_path = Path("runs/streams_summary.json")
    summary = json.load(open(summ_path)) if summ_path.exists() else {}
    jobs = [(r, rev, t, False) for r, rev, t in TRAIN]
    # ALOHA's two tasks form one held-out robot, so their episodes are pooled before the split
    jobs += [(r, rev, t, True) for r, rev, t in HELD_OUT]
    groups = {}
    for repo, rev, robot, held in jobs:
        name = ("heldout_" + robot) if held else repo.split("/")[1]
        groups.setdefault(name, []).append((repo, rev, robot, held))
    for name, items in groups.items():
        if (a.only and name not in a.only) or name in a.skip:
            continue
        xs, ids, lens, metas = [], [], [], []
        for repo, rev, robot, held in items:
            x, i, l, m = load_fetch(repo) if repo == "PAOP123/robochunk-sim" else load_lerobot(repo)
            xs.append(x)
            ids.append(np.array([f"{repo.split('/')[1]}:{v}" for v in i]))
            lens.append(l)
            metas.append(m)
        assert all(m["dims"] == metas[0]["dims"] for m in metas), name
        meta = dict(metas[0], robot=items[0][2], repos=[r for r, *_ in items],
                    revision={r: v for r, v, *_ in items})
        summary[name] = build(name, np.concatenate(xs), np.concatenate(ids), np.concatenate(lens), meta, items[0][3])
        print(name, json.dumps({k: v for k, v in summary[name].items() if k not in ("dims", "revision")}), flush=True)
        json.dump(summary, open(summ_path, "w"), indent=1)
    print("wrote", summ_path, file=sys.stderr)


if __name__ == "__main__":
    main()
