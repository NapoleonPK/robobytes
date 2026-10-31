"""Parity audit of the plain-Transformer baselines (Téo, 2026-10-31): the delta-coding reference (3.10 bpb) is far below
Byte (6.28) and BPE (6.49) at the pilot budget. Train Byte S at the pilot budget and its search's best rate with
(1) GPT-2 initialisation, (2) rotary positions, (3) both, and score them on the learning-rate search's validation set
(first 20 validation episodes per dataset). Writes runs/baseline_audit.json; resumable.
usage: PYTHONPATH=. python scripts/baseline_audit.py [--methods byte bpe]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from robobytes.data import training_names
from robobytes.evaluate import episode_bpb, load_run
from robobytes.serialize import Stream

BUDGET = 2.75e14
OUT = Path("runs/baseline_audit.json")
VARIANTS = {"gpt2init": ["--init", "gpt2"], "rope": ["--pos", "rope"], "rope_gpt2init": ["--pos", "rope", "--init", "gpt2"]}


def val20(run, dev):
    method, m, N, R, ent = load_run(run, dev)
    per = {}
    for name in training_names():
        eps = Stream(name).episodes("val")[:20]
        r = episode_bpb(method, m, name, "val", N, R, dev, ent=ent, eps=eps)
        per[name] = sum(v[0] for v in r.values()) / sum(v[1] for v in r.values())
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["byte"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    lr = json.load(open("runs/lr_search.json"))
    res = json.load(open(OUT)) if OUT.exists() else {}
    dev = torch.device("mps")
    for method in a.methods:
        best = lr["_best"][method]
        res.setdefault(f"{method}_baseline", dict(lr=best, val_bpb=lr[f"{method}_{best:g}"]["val_bpb"]))
        for v in a.variants:
            key = f"{method}_{v}"
            if key in res:
                continue
            run = Path(f"runs/audit/{key}")
            if not (run / "final.pt").exists():
                subprocess.run([sys.executable, "-m", "robobytes.train", "--method", method, "--size", "S", "--budget",
                                str(BUDGET), "--lr", str(best), "--out", str(run)] + VARIANTS[v], check=True)
            bpb, per = val20(run, dev)
            res[key] = dict(lr=best, val_bpb=bpb, per_dataset=per)
            print(f"{key}: val bpb {bpb:.4f} (baseline {res[f'{method}_baseline']['val_bpb']:.4f})", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
