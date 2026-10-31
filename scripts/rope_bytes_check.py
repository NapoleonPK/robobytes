"""Byte-offset RoPE must leave regular boundaries unchanged: rescore the search's best Fixed and Byte + conv checkpoints with
it on the same validation episodes. Writes runs/rope_bytes_check.json. usage: PYTHONPATH=. python scripts/rope_bytes_check.py"""
import json

import numpy as np
import torch

from robobytes.data import training_names
from robobytes.evaluate import episode_bpb, load_run
from robobytes.serialize import Stream

lr = json.load(open("runs/lr_search.json"))
dev = torch.device("mps")
out = {}
for key, scale in (("fixed_0.0015", 2.0), ("byteconv_0.0015", 1.0)):
    method, m, N, R, _ = load_run(lr[key]["run"], dev)
    m.rope_bytes = scale
    per = []
    for name in training_names():
        r = episode_bpb(method, m, name, "val", N, R, dev, eps=Stream(name).episodes("val")[:20])
        per.append(sum(v[0] for v in r.values()) / sum(v[1] for v in r.values()))
    out[key] = dict(chunk_index=lr[key]["val_bpb"], byte_offset=float(np.mean(per)))
    print(key, out[key], flush=True)
json.dump(out, open("runs/rope_bytes_check.json", "w"), indent=1)
