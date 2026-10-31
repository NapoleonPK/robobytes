"""Risk-table check (brief section 9): R_BPE was below 2 at vocabulary 4,096, so measure it at 16,384. Same 30 MB
square-root-mixture training sample (seed 0) as scripts/train_bpe.py; compression measured on up to 200k training rows
per dataset (seed 1) under the same mixture, for both vocabularies, so the two are compared on identical rows.
Writes runs/bpe_vocab_check.json. usage: RAYON_NUM_THREADS=2 PYTHONPATH=. python scripts/bpe_vocab_check.py"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers

from robobytes import bpe
from robobytes.data import sqrt_weights, training_names
from robobytes.serialize import Stream


def rows_of(s, eps):
    return np.concatenate([np.arange(s.start[e], s.start[e] + s.length[e]) for e in eps])


def main():
    streams = [Stream(n) for n in training_names()]
    w = sqrt_weights(streams)
    rng, rng_eval = np.random.default_rng(0), np.random.default_rng(1)
    sample, evalrows = [], []
    for s, wi in zip(streams, w):
        rows = rows_of(s, s.episodes("train"))
        k = int(round(wi * 30e6 / s.W))
        sample.append(bpe.row_strings(s.rows[np.sort(rng.choice(rows, size=k, replace=k > len(rows)))]))
        evalrows.append(s.rows[np.sort(rng_eval.choice(rows, size=min(200_000, len(rows)), replace=False))])
    out = {}
    for V in (4096, 16384):
        tok = Tokenizer(models.BPE())
        tok.train_from_iterator((r for p in sample for r in p),
                                trainers.BpeTrainer(vocab_size=V, initial_alphabet=bpe._ALPHA, special_tokens=[],
                                                    show_progress=False, min_frequency=2))
        inv = 0.0
        per = {}
        for s, wi, r in zip(streams, w, evalrows):
            ntok = sum(len(e.ids) for e in tok.encode_batch(bpe.row_strings(r)))
            R = len(r) * (s.W + 1) / (ntok + len(r))  # symbols per position, SEP counted on both sides (BOS ignored)
            per[s.name] = R
            inv += wi / R
        out[str(V)] = dict(R_mixture=1 / inv, per_dataset=per)
        print(V, "R_mixture", round(1 / inv, 4), flush=True)
    json.dump(out, open("runs/bpe_vocab_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
