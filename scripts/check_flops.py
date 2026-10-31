"""Cross-check the analytic FLOPs per byte (robobytes/flops.py) against torch.utils.flop_counter on CPU, one forward +
backward per method at size S, micro-batch 1, context 1,024 (brief section 4, step 2). For the chunked models the
boundaries are fixed every N bytes so both counts see the same number of chunks. Writes runs/flops_check.json.
usage: PYTHONPATH=. python -I scripts/check_flops.py [--N 4]"""
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from robobytes.flops import flops_per_byte
from robobytes.model import VOCAB, ByteHNet, TokenLM

CTX = 1024
d, L, h = 256, 4, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--R", type=float, default=4.0)
    a = ap.parse_args()
    torch.manual_seed(0)
    res = {}
    cases = {
        "byte": (TokenLM(VOCAB, d, L, h), 1.0, CTX),
        "bpe": (TokenLM(4098, d, L, h), a.R, int(CTX / a.R)),
        "hnet": (ByteHNet(d, L, h), float(a.N), CTX),
    }
    for name, (m, comp, n) in cases.items():
        V = m.head.out_features
        ids = torch.randint(0, VOCAB if name != "bpe" else 4098, (1, n))
        with FlopCounterMode(display=False) as fc:
            if name == "hnet":
                b = torch.zeros(1, n, dtype=torch.bool)
                b[:, ::a.N] = True
                out = m(ids, b)
            else:
                out = m(ids)
            F.cross_entropy(out["logits"][0, :-1], ids[0, 1:]).backward()
        counted = fc.get_total_flops() / CTX
        analytic = flops_per_byte(m, CTX, comp)
        res[name] = dict(counted_per_byte=counted, analytic_per_byte=analytic, ratio=counted / analytic)
        print(f"{name:6s} counted {counted / 1e6:8.2f} MFLOP/byte  analytic {analytic / 1e6:8.2f}  ratio {counted / analytic:.3f}")
    res["note"] = ("counted = matmul/conv/SDPA FLOPs seen by torch.utils.flop_counter (elementwise ops, depthwise conv as "
                   "multiply-adds and the EMA are not all counted); the H-Net run uses fixed boundaries every N bytes, so "
                   "the same architecture as Fixed/Entropy and H-Net up to the router (2 d_outer^2 per byte).")
    json.dump(res, open("runs/flops_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
