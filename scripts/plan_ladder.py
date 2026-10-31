"""Choose the compute ladder from the measured day-1 throughput (brief section 4) and write runs/ladder.json.
Throughput per method and size: Byte from runs/pilot_throughput_mps.json; BPE, H-Net and Fixed re-timed at the real
R_BPE and N (runs/pilot_throughput_mps_real.json); Entropy is timed as Fixed (same network, external boundaries) and the
refractory arm as H-Net. Bytes seen = budget / analytic FLOPs per symbol; passes = bytes seen / training symbols.
usage: PYTHONPATH=. python -I scripts/plan_ladder.py"""
import json

import numpy as np

from robobytes.data import CTX, training_names
from robobytes.flops import flops_per_byte
from robobytes.serialize import Stream
from robobytes.train import SIZES, bpe_info, build

TOP_HOURS = 2.0
SPACING = 10 ** 0.5
SIZE_PAIR = ("S", "M")
METHODS = ["byte", "bpe", "hnet", "fixed", "entropy"]


def main():
    info = bpe_info()
    R, N = info["R_BPE"], info.get("N_used", info["N"])
    old = json.load(open("runs/pilot_throughput_mps.json"))["runs"]
    real = json.load(open("runs/pilot_throughput_mps_real.json"))["runs"]
    timed = {"byte": ("byte", old), "bpe": ("bpe", real), "hnet": ("hnet", real), "fixed": ("fixed", real),
             "entropy": ("fixed", real), "hnet_refractory": ("hnet", real)}
    streams = [Stream(n) for n in training_names()]
    train_symbols = sum(s.n_symbols(s.episodes("train")) for s in streams)
    cfg = {}
    for m in METHODS + ["hnet_refractory"]:
        for size in SIZES:
            model = build(m, size, N, R)
            comp = 1.0 if m == "byte" else (R if m == "bpe" else float(N))
            src, tab = timed[m]
            cfg[f"{m}_{size}"] = dict(flops_per_symbol=flops_per_byte(model, CTX, comp),
                                      flops_per_s=tab[f"{src}_{size}"]["achieved_flops"],
                                      micro_batch=tab[f"{src}_{size}"]["micro_batch"])
    slowest = min(cfg[f"{m}_{s}"]["flops_per_s"] for m in METHODS for s in SIZE_PAIR)
    top = float(f"{TOP_HOURS * 3600 * slowest:.1e}")
    while top / slowest / 3600 > TOP_HOURS:
        top = float(f"{top * 0.95:.1e}")
    budgets = [float(f"{top / SPACING ** k:.2e}") for k in (3, 2, 1, 0)]
    plan = []
    for b in budgets:
        for m in METHODS:
            for s in SIZE_PAIR:
                c = cfg[f"{m}_{s}"]
                by = b / c["flops_per_symbol"]
                plan.append(dict(budget=b, method=m, size=s, hours=b / c["flops_per_s"] / 3600, symbols_seen=by,
                                 passes=by / train_symbols))
    hours = lambda rows: sum(r["hours"] for r in rows)
    top_rows = [r for r in plan if r["budget"] == budgets[-1]]
    out = dict(R_BPE=R, N=N, train_symbols=train_symbols, budgets=budgets, size_pair=SIZE_PAIR, configs=cfg, runs=plan,
               ladder_hours=hours(plan), max_run_hours=max(r["hours"] for r in plan),
               max_passes=max(r["passes"] for r in plan), top_budget_hours=hours(top_rows))
    json.dump(out, open("runs/ladder.json", "w"), indent=1)
    print("budgets", budgets, "train symbols", train_symbols)
    for r in top_rows:
        print(f"  top {r['method']:7s} {r['size']}  {r['hours']:.2f} h  passes {r['passes']:.2f}")
    print(f"ladder {out['ladder_hours']:.1f} h, longest run {out['max_run_hours']:.2f} h, max passes {out['max_passes']:.2f}")


if __name__ == "__main__":
    main()
