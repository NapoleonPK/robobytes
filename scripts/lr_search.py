"""Pilot learning-rate search (brief section 6): 3 rates per method at the smallest budget, size S, scored on held-in
validation only (pooled bpb, each training dataset weighted equally; at most 20 validation episodes per dataset, the
first 20 in the seed-0 order). Byte runs first: the Entropy baseline's small byte model in this search is the best Byte
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
from robobytes.evaluate import episode_bpb
from robobytes.train import EntropyBoundaries, bpe_info, build
from robobytes.serialize import Stream

LRS = [5e-4, 1.5e-3, 4.5e-3]
METHODS = ["byte", "bpe", "hnet", "fixed", "entropy", "hnet_refractory"]
OUT = Path("runs/lr_search.json")


def val_bpb(method, run, dev, ent=None):
    c = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    info = bpe_info()
    R, N = info["R_BPE"], info.get("N_used", info["N"])
    m = build(method, c["args"]["size"], N, R).to(dev)
    m.load_state_dict(c["model"])
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
    a = ap.parse_args()
    dev = torch.device("mps")
    res = json.load(open(OUT)) if OUT.exists() else {}
    for method in a.methods:
        for lr in LRS:
            key = f"{method}_{lr:g}"
            if key in res:
                continue
            run = Path(f"runs/lr/{key}")
            extra, ent = [], None
            if method == "entropy":
                best = min((v for k, v in res.items() if k.startswith("byte_")), key=lambda v: v["val_bpb"])
                ck = Path(best["run"]) / "final.pt"
                thr = tune_threshold(ck, dev)
                extra = ["--entropy-ckpt", str(ck), "--entropy-threshold", str(thr)]
                ent = EntropyBoundaries(ck, thr, dev)
            if not (run / "final.pt").exists():
                cmd = [sys.executable, "-m", "robobytes.train", "--method", method, "--size", "S", "--budget",
                       str(a.budget), "--lr", str(lr), "--out", str(run)] + extra
                subprocess.run(cmd, check=True)
            bpb, per = val_bpb(method, run, dev, ent)
            res[key] = dict(method=method, lr=lr, run=str(run), val_bpb=bpb, per_dataset=per,
                            train=json.load(open(run / "train.json"))["log"][-1])
            print(key, f"val bpb {bpb:.4f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    best = {}
    for method in METHODS:
        rows = [v for v in res.values() if v["method"] == method]
        if rows:
            best[method] = min(rows, key=lambda v: v["val_bpb"])["lr"]
    res["_best"] = best
    json.dump(res, open(OUT, "w"), indent=1)
    print("best", best)


if __name__ == "__main__":
    main()
