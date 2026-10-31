"""Real H-Net throughput with byte-offset RoPE, at the ladder's micro-batch 8: short training runs on real data. Throughput
is budget / trained seconds; the planner's factors come from these. Writes runs/timing_rope_bytes.json.
usage: PYTHONPATH=. python scripts/timing_rope_bytes.py"""
import json
import subprocess
import sys
from pathlib import Path

B = 1e14
CASES = {"hnet_S": ("S", 2, []), "hnet_M": ("M", 2, []), "hnet_no_ema_S": ("S", 2, ["--no-ema"]),
         "hnet_no_ema_M": ("M", 2, ["--no-ema"]), "hnet_S_N4": ("S", 4, []), "hnet_M_N4": ("M", 4, [])}
out = {}
for key, (size, N, extra) in CASES.items():
    run = Path(f"runs/timing/{key}")
    subprocess.run([sys.executable, "-m", "robobytes.train", "--method", "hnet", "--size", size, "--budget", str(B), "--lr",
                    "1.5e-3", "--N", str(N), "--micro", "8", "--out", str(run)] + extra, check=True)
    t = json.load(open(run / "train.json"))["log"][-1]
    out[key] = dict(flops_per_s=B / t["seconds"], seconds=t["seconds"], micro=t["micro"], compression=t["compression"],
                    swap_mb=t["swap_mb"], mps_driver_gb=t["mps_driver_gb"])
    print(key, out[key], flush=True)
    json.dump(out, open("runs/timing_rope_bytes.json", "w"), indent=1)
