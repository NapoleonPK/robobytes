"""Why is Byte + conv worse than Byte at equal FLOPs? (Téo, 2026-10-31). All runs size S, rate 1.5e-3, scored on the
learning-rate search's validation episodes. Writes runs/wrapper_diag.json; resumable.
usage: PYTHONPATH=. python scripts/wrapper_diag.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from robobytes.data import training_names
from robobytes.evaluate import episode_bpb, load_run
from robobytes.serialize import Stream

B = 2.43e14
LR = 1.5e-3
OUT = Path("runs/wrapper_diag.json")
BYTE_FPB, CONV_FPB = 25.65e6, 48.2e6  # size S FLOPs per symbol, so "same bytes" gives the wrapper Byte's symbol count
W256 = ["--outer-width", "256", "--outer-blocks", "4"]
RUNS = {
    "byte": ("byte", B, []),
    "byteconv_256x4": ("byteconv", B, W256),
    "byteconv_128x2": ("byteconv", B, []),
    "byteconv_256x0": ("byteconv", B, ["--outer-width", "256", "--outer-blocks", "0"]),
    "byteconv_256x4_same_bytes": ("byteconv", B * CONV_FPB / BYTE_FPB, W256),
    "byteconv_256x4_conv_init_0.1": ("byteconv", B, W256 + ["--conv-init", "0.1"]),
}  # all done; the wrapper is now 128 x 2 by default, hence W256 for reruns
EMA = {  # is H-Net's deficit to Fixed the soft EMA dechunk or the irregular boundaries? (compare runs/lr_search.json)
    "hnet_no_ema_4.5e-3": ("hnet", 2.75e14, ["--no-ema", "--no-rope-bytes"], 4.5e-3),
    "hnet_no_ema_1.5e-3": ("hnet", 2.75e14, ["--no-ema", "--no-rope-bytes"], 1.5e-3),
}
ENT = json.load(open("runs/lr/entropy_0.0015/meta.json"))["args"]
ENT = ["--entropy-ckpt", ENT["entropy_ckpt"], "--entropy-threshold", str(ENT["entropy_threshold"])]
POS = {  # irregular boundaries make the W + 1 byte offset jitter in chunk index: does RoPE on byte offset / N recover it?
    "hnet_rope_bytes_4.5e-3": ("hnet", 2.75e14, ["--rope-bytes"], 4.5e-3),
    "hnet_rope_bytes_1.5e-3": ("hnet", 2.75e14, ["--rope-bytes"], 1.5e-3),
    "hnet_rope_bytes_no_ema_1.5e-3": ("hnet", 2.75e14, ["--rope-bytes", "--no-ema"], 1.5e-3),
    "entropy_rope_bytes_1.5e-3": ("entropy", 2.75e14, ["--rope-bytes"] + ENT, 1.5e-3),
    "hnet_rope_bytes_5e-4": ("hnet", 2.75e14, ["--rope-bytes"], 5e-4),  # rest of each grid
    "hnet_rope_bytes_no_ema_5e-4": ("hnet", 2.75e14, ["--rope-bytes", "--no-ema"], 5e-4),
    "hnet_rope_bytes_no_ema_4.5e-3": ("hnet", 2.75e14, ["--rope-bytes", "--no-ema"], 4.5e-3),
    "entropy_rope_bytes_5e-4": ("entropy", 2.75e14, ["--rope-bytes"] + ENT, 5e-4),
    "entropy_rope_bytes_4.5e-3": ("entropy", 2.75e14, ["--rope-bytes"] + ENT, 4.5e-3),
}


def val20(run, dev):
    method, m, N, R, ent = load_run(run, dev)
    per = {}
    for name in training_names():
        eps = Stream(name).episodes("val")[:20]
        r = episode_bpb(method, m, name, "val", N, R, dev, ent=ent, eps=eps)
        per[name] = sum(v[0] for v in r.values()) / sum(v[1] for v in r.values())
    return float(np.mean(list(per.values()))), per


def main():
    res = json.load(open(OUT)) if OUT.exists() else {}
    dev = torch.device("mps")
    for key, (method, budget, extra, *lr) in {**RUNS, **EMA, **POS}.items():
        if key in res:
            continue
        run = Path(f"runs/wrapper/{key}")
        if not (run / "final.pt").exists():
            subprocess.run([sys.executable, "-m", "robobytes.train", "--method", method, "--size", "S", "--budget",
                            str(budget), "--lr", str(lr[0] if lr else LR), "--out", str(run)] + extra, check=True)
        bpb, per = val20(run, dev)
        t = json.load(open(run / "train.json"))
        res[key] = dict(val_bpb=bpb, per_dataset=per, budget=budget, flops_per_symbol=t["flops_per_symbol"],
                        steps=t["steps"], params=t["params"], seconds=t["log"][-1]["seconds"], final_loss=t["log"][-1]["loss"])
        print(f"{key}: val bpb {bpb:.4f}  steps {t['steps']}  fpb {t['flops_per_symbol'] / 1e6:.1f}M", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
