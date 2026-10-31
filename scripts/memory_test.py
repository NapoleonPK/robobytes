"""Memory check before the ladder (Téo, 2026-10-31): after a reboot, run the largest configuration (H-Net, size M, top
budget, N = round(R_BPE), micro-batch 16) for 10 minutes; sample swap every 5 s and MPS memory from the run's log.
Writes runs/memory_test.json; if swap grows by more than 1 GB, ladder runs start at micro-batch 8.
usage: PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 python scripts/memory_test.py"""
import json
import subprocess
import sys
import time
from pathlib import Path

from robobytes.train import swap_mb

LADDER = json.load(open("runs/ladder.json"))
LR = json.load(open("runs/lr_search.json"))["_best"]["hnet"]
OUT = Path("runs/memtest_run")


def main():
    swap0 = swap_mb()
    cmd = [sys.executable, "-m", "robobytes.train", "--method", "hnet", "--size", "M", "--budget",
           str(LADDER["budgets"][-1]), "--lr", str(LR), "--out", str(OUT), "--max-minutes", "10"]
    p = subprocess.Popen(cmd)
    peak_swap, t0 = swap0, time.time()
    while p.poll() is None:
        peak_swap = max(peak_swap, swap_mb())
        time.sleep(5)
    log = json.load(open(OUT / "train.json"))
    res = dict(config="H-Net, size M, top budget, N = round(R_BPE), micro-batch 16 at start", minutes=(time.time() - t0) / 60,
               swap_start_mb=swap0, swap_peak_mb=peak_swap, swap_growth_mb=peak_swap - swap0,
               peak_mps_gb=max(r["mps_gb"] for r in log["log"]),
               peak_mps_driver_gb=max(r.get("mps_driver_gb", 0) for r in log["log"]),
               steps=log["log"][-1]["step"], micro_end=log["log"][-1]["micro"], events=log["events"],
               decision="micro-batch 8 + accumulation" if peak_swap - swap0 > 1024 or log["log"][-1]["micro"] < 16
               else "micro-batch 16")
    json.dump(res, open("runs/memory_test.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
