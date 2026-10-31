"""Pilot learning-rate search (brief section 6): 3 rates per method at the smallest budget, size S, scored on held-in
validation only (pooled bpb, each training dataset weighted equally; at most 20 validation episodes per dataset, the
first 20 in the seed-0 order). The refractory arm is not searched (r = floor(0.3 N) = 0 at N = 2;
at N = 4 it reuses H-Net's rate). Byte runs first: the Entropy baseline's small byte model in this search is the best Byte
run here (in the ladder it is the Byte run at the second budget). Writes runs/lr_search.json; resumable.
usage: PYTHONPATH=. python -I scripts/lr_search.py --budget 1e15"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from robobytes.data import training_names
from robobytes.entropy import tune_threshold
from robobytes.evaluate import episode_bpb, load_run
from robobytes.train import EntropyBoundaries
from robobytes.serialize import Stream

LRS = [5e-4, 1.5e-3, 4.5e-3]
METHODS = ["byte", "bpe", "hnet", "fixed", "entropy", "byteconv", "hnet_no_ema"]  # refractory arm dropped: r = floor(0.3 N) = 0 at N = 2
OUT = Path("runs/lr_search.json")


def val_bpb(method, run, dev, ent=None):
    _, m, N, R, _e = load_run(run, dev)
    per = {}
    for name in training_names():
        eps = Stream(name).episodes("val")[:20]
        r = episode_bpb(method, m, name, "val", N, R, dev, ent=ent, eps=eps)
        bits, by = sum(v[0] for v in r.values()), sum(v[1] for v in r.values())
        pos, ch = sum(v[2] for v in r.values()), sum(v[3] for v in r.values())
        per[name] = dict(bpb=bits / by, compression=pos / ch if ch else None)
    return float(np.mean([v["bpb"] for v in per.values()])), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--edge", action="store_true", help="edge rule: one more rate beyond the grid end that won")
    a = ap.parse_args()
    dev = torch.device("mps")
    res = json.load(open(OUT)) if OUT.exists() else {}
    for method in a.methods:
        lrs = LRS
        if a.edge:
            rows = {v["lr"]: v["val_bpb"] for k, v in res.items() if not k.startswith("_") and v["method"] == method}
            if not all(r in rows for r in LRS):
                continue
            b = min(rows, key=rows.get)
            lrs = [LRS[-1] * 3] if b == LRS[-1] else [LRS[0] / 3] if b == LRS[0] else []
        for lr in lrs:
            key = f"{method}_{lr:g}"
            if key in res:
                continue
            run = Path(f"runs/lr/{key}")
            extra, ent = [], None
            if method == "entropy":
                best = min((v for k, v in res.items() if k.startswith("byte_")), key=lambda v: v["val_bpb"])  # incl. edge runs
                ck = Path(best["run"]) / "final.pt"
                thr = tune_threshold(ck, dev)
                extra = ["--entropy-ckpt", str(ck), "--entropy-threshold", str(thr)]
                ent = EntropyBoundaries(ck, thr, dev)
            if not (run / "final.pt").exists():
                if method == "hnet_no_ema":
                    extra = ["--no-ema"]
                cmd = [sys.executable, "-m", "robobytes.train", "--method", method.removesuffix("_no_ema"), "--size", "S",
                       "--budget", str(a.budget), "--lr", str(lr), "--out", str(run)] + extra
                subprocess.run(cmd, check=True)
            bpb, per = val_bpb(method, run, dev, ent)
            res[key] = dict(method=method, lr=lr, run=str(run), val_bpb=bpb, per_dataset=per,
                            train=json.load(open(run / "train.json"))["log"][-1])
            print(key, f"val bpb {bpb:.4f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    best = {}
    for method in METHODS:
        rows = [v for k, v in res.items() if not k.startswith("_") and v["method"] == method]
        if rows:
            best[method] = min(rows, key=lambda v: v["val_bpb"])["lr"]
    res["_best"] = best
    json.dump(res, open(OUT, "w"), indent=1)
    print("best", best)


if __name__ == "__main__":
    main()
