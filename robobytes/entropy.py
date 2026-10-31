"""Entropy baseline threshold (brief section 6): boundaries where the small byte model's next-symbol entropy exceeds a
threshold tuned on training-robot validation data to give compression N. The threshold is the (1 - 1/N) quantile of
the entropy over 256 validation windows drawn with the training mixture (seed 0); position 0 of a window is always a
boundary and is left out of the quantile."""
from __future__ import annotations

import numpy as np
import torch

from .data import CTX, Mixture
from .train import EntropyBoundaries, bpe_info


@torch.no_grad()
def tune_threshold(ckpt, dev, windows=256, batch=16):
    info = bpe_info()
    N = info.get("N_used", info["N"])
    eb = EntropyBoundaries(ckpt, float("inf"), dev)
    mix = Mixture("byte", CTX, split="val", seed=0)
    hs = []
    for _ in range(windows // batch):
        x = torch.from_numpy(mix.batch(batch)[:, :-1]).to(dev)
        hs.append(eb.entropy(x)[:, 1:].cpu().numpy().ravel())
    return float(np.quantile(np.concatenate(hs), 1 - 1 / N))
