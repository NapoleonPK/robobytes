"""No-learning reference (Téo, 2026-10-31): predict each data byte from the same channel at the previous timestep.

Per dataset and channel, a 256 x 256 table P(x_t | x_{t-1}) is counted on the training split (add-0.5 smoothing); the
first timestep of an episode uses the channel's marginal. bpb = cross-entropy on the same validation episodes as the
learning-rate search (first 20 per dataset), and on the first 100 (frontier validation set). It also reports the
in-sample conditional entropy H(x_t | x_{t-1}) on the training split. Unlike the models, this predictor is told the
channel layout (W and the channel of every byte), so it is a reference, not a baseline.
Writes runs/delta_reference.json. usage: PYTHONPATH=. python scripts/delta_reference.py"""
import json

import numpy as np

from robobytes.data import training_names
from robobytes.serialize import Stream

ALPHA = 0.5


def block(s, eps):
    """Row range and episode-start mask for eps; the stored episodes of one split are contiguous (seed-0 order)."""
    eps = np.sort(eps)
    assert np.all(np.diff(eps) == 1), "split episodes are stored contiguously"
    a, b = int(s.start[eps[0]]), int(s.start[eps[-1]] + s.length[eps[-1]])
    first = np.zeros(b - a, bool)
    first[s.start[eps] - a] = True
    return a, b, first


def channel(s, a, b, first, c):
    cur = np.asarray(s.rows[a:b, c]).astype(np.int32)
    prev = np.empty_like(cur)
    prev[1:] = cur[:-1]
    prev[first] = -1
    return cur, prev


def main():
    lr = json.load(open("runs/lr_search.json"))
    best = lr["_best"]
    out = {}
    for name in training_names():
        s = Stream(name)
        a, b, first = block(s, s.episodes("train"))
        tables, marg, h_in = [], [], []
        for c in range(s.W):
            cur, prev = channel(s, a, b, first, c)
            m = prev >= 0
            T = np.bincount(prev[m] * 256 + cur[m], minlength=65536).reshape(256, 256).astype(np.float64)
            joint = T / T.sum()
            pc = joint.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                h_in.append(float(-np.nansum(joint * np.log2(np.where(joint > 0, joint / pc, 1)))))
            T += ALPHA
            tables.append(np.log2(T / T.sum(1, keepdims=True)))
            mg = np.bincount(cur, minlength=256) + ALPHA
            marg.append(np.log2(mg / mg.sum()))
        res = dict(train_conditional_entropy=float(np.mean(h_in)))
        for k in (20, 100):
            va, vb, vfirst = block(s, s.episodes("val")[:k])
            bits = 0.0
            for c in range(s.W):
                vc, vp = channel(s, va, vb, vfirst, c)
                m = vp >= 0
                bits -= tables[c][vp[m], vc[m]].sum() + marg[c][vc[~m]].sum()
            res[f"val{k}_bpb"] = bits / ((vb - va) * s.W)
        res["byte_lr_search_bpb"] = lr[f"byte_{best['byte']:g}"]["per_dataset"][name]["bpb"]
        res["bpe_lr_search_bpb"] = lr[f"bpe_{best['bpe']:g}"]["per_dataset"][name]["bpb"]
        res["hnet_lr_search_bpb"] = lr[f"hnet_{best['hnet']:g}"]["per_dataset"][name]["bpb"]
        out[name] = res
        print(f"{name:34s} delta {res['val20_bpb']:5.2f}  (train H {res['train_conditional_entropy']:4.2f})  "
              f"byte {res['byte_lr_search_bpb']:5.2f}  bpe {res['bpe_lr_search_bpb']:5.2f}  "
              f"hnet {res['hnet_lr_search_bpb']:5.2f}", flush=True)
    pooled = {k: float(np.mean([v[k] for v in out.values()])) for k in next(iter(out.values()))}
    out["_pooled"] = pooled
    json.dump(out, open("runs/delta_reference.json", "w"), indent=1)
    print("pooled", {k: round(v, 3) for k, v in pooled.items()})


if __name__ == "__main__":
    main()
