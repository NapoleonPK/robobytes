"""BPE baseline tokenizer (brief section 6): HF `tokenizers` BPE, vocabulary 4,096, trained on training-robot rows only.

Each timestep's W data bytes are one "word" (bytes mapped to the code points U+0100..U+01FF), so no merge can cross a
SEP, and BOS/SEP stay single tokens outside the learned vocabulary: ids 0..4095 are BPE tokens, 4096 = BOS, 4097 = SEP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers

VOCAB_BPE = 4096
BOS_T, SEP_T = VOCAB_BPE, VOCAB_BPE + 1
VOCAB_T = VOCAB_BPE + 2
PATH = Path("data/bpe/tokenizer.json")
_ALPHA = [chr(0x100 + b) for b in range(256)]
_TABLE = np.array(_ALPHA)


def row_strings(rows):
    """(n, W) uint8 -> n strings of W characters."""
    return ["".join(r) for r in _TABLE[np.asarray(rows)]]


def train(row_iter, path=PATH):
    tok = Tokenizer(models.BPE())
    tr = trainers.BpeTrainer(vocab_size=VOCAB_BPE, initial_alphabet=_ALPHA, special_tokens=[], show_progress=False,
                             min_frequency=2)
    tok.train_from_iterator(row_iter, tr)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def load(path=PATH):
    return Tokenizer.from_file(str(path))


def encode_rows(tok, rows, batch=200_000):
    """Token ids per row (list of lists)."""
    out = []
    for i in range(0, len(rows), batch):
        out += [e.ids for e in tok.encode_batch(row_strings(rows[i:i + batch]))]
    return out


def episode_tokens(row_ids):
    """BOS, then per timestep its BPE ids and SEP. Returns (tokens, is_data) where is_data marks BPE data tokens."""
    n = 1 + sum(len(r) + 1 for r in row_ids)
    toks = np.empty(n, np.uint16)
    data = np.ones(n, bool)
    toks[0], data[0] = BOS_T, False
    i = 1
    for r in row_ids:
        toks[i:i + len(r)] = r
        i += len(r)
        toks[i], data[i] = SEP_T, False
        i += 1
    return toks, data
