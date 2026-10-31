"""Train the BPE tokenizer on training-robot rows (train split only), drawn per dataset in proportion to the square
root of its training bytes (the same mixture as pretraining), then tokenise every stream and measure R_BPE.

R_BPE = stream symbols per BPE position on the training split, under the training mixture (BOS and SEP count as one symbol
and one position each, as for every method): 1 / sum_d w_d / R_d with w_d the square-root weights, since that is the
stream every model trains on. The pooled value (all training rows, unweighted) is reported too. N = round(R_BPE). Writes data/bpe/tokenizer.json,
data/bpe/<name>/{tokens.npy,offsets.npy,data_mask.npy} and runs/bpe_summary.json.
usage: PYTHONPATH=. python -I scripts/train_bpe.py [--sample-bytes 30e6]"""
import argparse
import json
import time

import numpy as np

from robobytes import bpe
from robobytes.serialize import STREAMS, Stream


def sqrt_weights(streams):
    b = np.array([s.n_data_bytes(s.episodes("train")) for s in streams], dtype=np.float64)
    w = np.sqrt(b)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-bytes", type=float, default=30e6)
    a = ap.parse_args()
    names = sorted(p.name for p in STREAMS.iterdir() if not p.name.startswith("heldout_"))
    streams = [Stream(n) for n in names]
    w = sqrt_weights(streams)
    rng = np.random.default_rng(0)
    sample = []
    for s, wi in zip(streams, w):
        tr = s.episodes("train")
        rows = np.concatenate([np.arange(s.start[e], s.start[e] + s.length[e]) for e in tr])
        k = int(round(wi * a.sample_bytes / s.W))
        pick = np.sort(rng.choice(rows, size=k, replace=k > len(rows)))
        sample.append(bpe.row_strings(s.rows[pick]))
    t0 = time.time()
    tok = bpe.train((r for part in sample for r in part))
    t_train = time.time() - t0
    out = {"vocab": tok.get_vocab_size(), "sample_bytes": a.sample_bytes, "train_seconds": t_train, "datasets": {}}
    sym_tr = pos_tr = 0
    for n in names + sorted(p.name for p in STREAMS.iterdir() if p.name.startswith("heldout_")):
        s = Stream(n)
        ids = bpe.encode_rows(tok, s.rows)
        offs, toks, mask = [0], [], []
        r0 = 0
        for e in range(len(s.length)):  # episodes in stored order
            t, m = bpe.episode_tokens(ids[r0:r0 + s.length[e]])
            r0 += s.length[e]
            toks.append(t)
            mask.append(m)
            offs.append(offs[-1] + len(t))
        d = bpe.PATH.parent / n
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "tokens.npy", np.concatenate(toks))
        np.save(d / "data_mask.npy", np.concatenate(mask))
        np.save(d / "offsets.npy", np.array(offs, dtype=np.int64))
        res = {}
        for split in np.unique(s.split):
            eps = s.episodes(split)
            sym = s.n_symbols(eps)
            pos = int(sum(offs[e + 1] - offs[e] for e in eps))
            res[split] = dict(symbols=sym, positions=pos, compression=sym / pos,
                              data_bytes_per_data_token=s.n_data_bytes(eps) / int(sum(mask[e].sum() for e in eps)))
            if split == "train":
                sym_tr += sym
                pos_tr += pos
        out["datasets"][n] = res
        print(n, {k: round(v["compression"], 3) for k, v in res.items()}, flush=True)
    out["R_BPE_pooled"] = sym_tr / pos_tr
    out["weights"] = dict(zip(names, w.tolist()))
    out["R_BPE"] = 1 / sum(wi / out["datasets"][n]["train"]["compression"] for n, wi in zip(names, w))
    out["N"] = int(round(out["R_BPE"]))
    json.dump(out, open("runs/bpe_summary.json", "w"), indent=1)
    print("R_BPE", out["R_BPE"], "pooled", out["R_BPE_pooled"], "N", out["N"])


if __name__ == "__main__":
    main()
