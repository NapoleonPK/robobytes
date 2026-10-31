"""Share of analytic FLOPs per byte spent in the outer network (conv encoder and decoder, router, residual, head), per
hierarchical method and size, at the 128 x 2 wrapper and the brief's 256 x 4 (Téo, 2026-10-31; Cartesia's ratio is 45%).
Writes runs/wrapper_share.json. usage: PYTHONPATH=. python scripts/wrapper_share.py"""
import json

from robobytes.flops import _p, flops_per_byte
from robobytes.train import build

CASES = {"Byte + conv (N = 1)": ("byteconv", 1), "H-Net, Fixed, Entropy (N = 2)": ("hnet", 2),
         "H-Net, Fixed (N = 4)": ("hnet", 4)}
OUT = "runs/wrapper_share.json"


def share(m, comp):
    outer = sum(_p(x) for x in (m.encoder, m.decoder, m.router_norm, m.wq, m.wk, m.residual, m.head))
    return 6 * outer / flops_per_byte(m, 1024, comp)


def main():
    res = {}
    for wrapper, (width, blocks) in {"128x2": (128, 2), "256x4": (256, 4)}.items():
        res[wrapper] = {}
        for label, (method, N) in CASES.items():
            res[wrapper][label] = {s: share(build(method, s, N, 1.549, outer_width=width, outer_blocks=blocks), N)
                                   for s in ("S", "M")}
            print(wrapper, label, {k: round(v, 3) for k, v in res[wrapper][label].items()})
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
