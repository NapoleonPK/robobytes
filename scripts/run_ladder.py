"""Run the pre-registered training runs in order, one job at a time, resumably (PREREGISTRATION.md §6).

Stages: main ladder (6 methods x 4 budgets x sizes S, M; the Entropy runs wait for the Byte frontier at the second
budget, which is their small byte model), extra seeds 1-2 for H-Net and BPE at the top budget (frontier size), the
N = 4 secondary comparison (H-Net and Fixed, all budgets and sizes; H-Net refractory r = 1 at the top two budgets,
H-Net N = 4's frontier size), and the no-EMA H-Net arm (all budgets and sizes). After each training run its held-in validation is scored (val.json). A finished run
(final.pt + val.json) is skipped, so the script can be restarted at any time.
usage: PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 caffeinate -i \
       python scripts/run_ladder.py [--stage main|seeds|noema|n4|all]"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from robobytes.entropy import tune_threshold
from robobytes.evaluate import validate

LADDER = json.load(open("runs/ladder.json"))
LR = json.load(open("runs/lr_search.json"))["_best"]
MEM = json.load(open("runs/memory_test.json"))
MICRO = 8 if MEM["swap_growth_mb"] > 1024 or MEM["micro_end"] < 16 else 16
BUDGETS = LADDER["budgets"]
SIZES = ("S", "M")
ROOT = Path("runs/ladder")
STATUS = Path("runs/ladder_status.json")
dev = torch.device("mps")


def run_dir(method, budget_i, size, seed=0, N=None, r=None):
    tag = f"{method}_b{budget_i + 1}_{size}_s{seed}" + (f"_N{N}" if N else "") + (f"_r{r}" if r is not None else "")
    return ROOT / tag


def done(d):
    return (d / "final.pt").exists() and (d / "val.json").exists()


def val_bpb(d):
    return json.load(open(d / "val.json"))["val_bpb"]


def frontier(method, budget_i, seed=0, N=None, r=None):
    ds = [run_dir(method, budget_i, s, seed, N, r) for s in SIZES]
    if not all(done(d) for d in ds):
        return None
    return min(ds, key=val_bpb)


def log_status(**kw):
    st = json.load(open(STATUS)) if STATUS.exists() else {"events": []}
    st["events"].append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw))
    json.dump(st, open(STATUS, "w"), indent=1)


def train(method, budget_i, size, seed=0, N=None, r=None, extra=()):
    d = run_dir(method, budget_i, size, seed, N, r)
    if done(d):
        return d
    lr = LR["hnet" if method == "hnet_refractory" else method]
    if not (d / "final.pt").exists():
        if method == "hnet_no_ema":
            extra = [*extra, "--no-ema"]
        cmd = [sys.executable, "-m", "robobytes.train", "--method", method.removesuffix("_no_ema"), "--size", size, "--budget",
               str(BUDGETS[budget_i]), "--lr", str(lr), "--seed", str(seed), "--micro", str(MICRO), "--out", str(d)]
        cmd += (["--N", str(N)] if N else []) + (["--refractory", str(r)] if r is not None else []) + list(extra)
        t0 = time.time()
        log_status(start=d.name)
        subprocess.run(cmd, check=True)
        log_status(trained=d.name, minutes=(time.time() - t0) / 60)
    v = validate(d, dev)
    log_status(validated=d.name, val_bpb=v["val_bpb"])
    print(f"{d.name}: val bpb {v['val_bpb']:.4f}", flush=True)
    return d


def entropy_args():
    """Small byte model = the Byte frontier run at the second budget; threshold tuned once on validation windows."""
    f = frontier("byte", 1)
    assert f is not None, "Byte runs at the second budget must finish before Entropy"
    p = ROOT / "entropy_threshold.json"
    if not p.exists():
        json.dump(dict(byte_model=str(f), threshold=tune_threshold(f / "final.pt", dev)), open(p, "w"), indent=1)
    t = json.load(open(p))
    return ["--entropy-ckpt", str(Path(t["byte_model"]) / "final.pt"), "--entropy-threshold", str(t["threshold"])]


def main_stage():
    # budgets in ascending order so frontiers fill in early; Byte at budget 2 precedes every Entropy run
    for bi in range(len(BUDGETS)):
        for m in ("byte", "bpe", "hnet", "fixed", "byteconv"):
            for s in SIZES:
                train(m, bi, s)
        if bi >= 1:
            for bj in range(bi + 1):
                for s in SIZES:
                    train("entropy", bj, s, extra=entropy_args())


def seeds_stage():
    top = len(BUDGETS) - 1
    for m in ("hnet", "bpe"):
        size = frontier(m, top).name.split("_")[2]
        for seed in (1, 2):
            train(m, top, size, seed=seed)


def noema_stage():
    for bi in range(len(BUDGETS)):
        for s in SIZES:
            train("hnet_no_ema", bi, s)


def n4_stage():
    for bi in range(len(BUDGETS)):
        for m in ("hnet", "fixed"):
            for s in SIZES:
                train(m, bi, s, N=4)
    for bi in range(len(BUDGETS) - 2, len(BUDGETS)):
        size = frontier("hnet", bi, N=4).name.split("_")[2]
        train("hnet_refractory", bi, size, N=4, r=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["main", "seeds", "noema", "n4", "all"])
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    log_status(stage=a.stage, micro=MICRO)
    if a.stage in ("main", "all"):
        main_stage()
    if a.stage in ("seeds", "all"):
        seeds_stage()
    if a.stage in ("noema", "all"):
        noema_stage()
    if a.stage in ("n4", "all"):
        n4_stage()
    log_status(finished=a.stage)


if __name__ == "__main__":
    main()
