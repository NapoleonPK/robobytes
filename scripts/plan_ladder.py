"""Choose the compute ladder from the measured day-1 throughput (brief section 4) and write runs/ladder.json.
Throughput per method and size: Byte from runs/pilot_throughput_mps.json; BPE, H-Net and Fixed re-timed at the real
R_BPE and N (runs/pilot_throughput_mps_real.json); Entropy is timed as Fixed (same network, external boundaries) and the
refractory arm as H-Net. The secondary comparison at N = 4 (H-Net and Fixed, all budgets and both sizes; H-Net
refractory r = 1 at the top two budgets) uses the first pilot's timings, which assumed N = 4. Bytes seen = budget /
analytic FLOPs per symbol; passes = bytes seen / training symbols. Also writes the whole run list with time estimates.
usage: PYTHONPATH=. python -I scripts/plan_ladder.py"""
import json

import numpy as np

from robobytes.data import CTX, sqrt_weights, training_names
from robobytes.flops import flops_per_byte
from robobytes.serialize import Stream
from robobytes.train import SIZES, bpe_info, build

TOP_HOURS = 2.0
SPACING = 10 ** 0.5
SIZE_PAIR = ("S", "M")
# real training / pilot throughput, measured on the final learning-rate runs (size S) and the memory check (H-Net M); the
# pilot has no router loop on real data and no entropy-model forward pass. Entropy's factor is S's at both sizes (cautious:
# its extra pass weighs less at M)
REAL = {"byte": 0.99, "bpe": 0.99, "fixed": 0.97, "byteconv": 0.98, "entropy": 0.46, "hnet_S": 0.87, "hnet_M": 0.94}
METHODS = ["byte", "bpe", "hnet", "fixed", "entropy", "byteconv"]


def real_factor(m, size):
    return REAL[f"hnet_{size}"] if m.startswith("hnet") else REAL[m]


def main():
    info = bpe_info()
    R, N = info["R_BPE"], info.get("N_used", info["N"])
    real = json.load(open("runs/pilot_throughput_mps_rope.json"))["runs"]
    n4_t = json.load(open("runs/pilot_throughput_mps_rope_n4.json"))["runs"]
    timed = {m: m for m in METHODS}
    timed["entropy"] = "fixed"
    streams = [Stream(n) for n in training_names()]
    train_symbols = sum(s.n_symbols(s.episodes("train")) for s in streams)
    cfg = {}
    for m in METHODS:
        for size in SIZE_PAIR:
            model = build(m, size, N, R)
            comp = 1.0 if m in ("byte", "byteconv") else (R if m == "bpe" else float(N))
            t = real[f"{timed[m]}_{size}"]
            # the pilot computed achieved FLOP/s with its own compression; this is the same model, so it carries over
            cfg[f"{m}_{size}"] = dict(flops_per_symbol=flops_per_byte(model, CTX, comp),
                                      flops_per_s=t["achieved_flops"] * real_factor(m, size),
                                      micro_batch=t["micro_batch"])
    n4 = {}
    for m in ("hnet", "fixed", "hnet_refractory"):
        for size in SIZE_PAIR:
            model = build(m, size, 4, R, refractory=1)
            src = "hnet" if m == "hnet_refractory" else m
            n4[f"{m}_{size}"] = dict(flops_per_symbol=flops_per_byte(model, CTX, 4.0),
                                     flops_per_s=n4_t[f"{src}_{size}"]["achieved_flops"] * real_factor(m, size))
    # Entropy's extra forward pass is uncharged wall-clock, so it does not set the budget; its runs may exceed 2 h
    slowest = min([cfg[f"{m}_{s}"]["flops_per_s"] for m in METHODS if m != "entropy" for s in SIZE_PAIR]
                  + [c["flops_per_s"] for c in n4.values()])
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
    secondary = []
    for b in budgets:
        for m in ("hnet", "fixed"):
            for s in SIZE_PAIR:
                c = n4[f"{m}_{s}"]
                secondary.append(dict(budget=b, method=m, N=4, size=s, hours=b / c["flops_per_s"] / 3600,
                                      passes=b / c["flops_per_symbol"] / train_symbols))
    for b in budgets[-2:]:  # refractory at H-Net's N = 4 frontier size; estimated at the slower size
        c = min((n4[f"hnet_refractory_{s}"] for s in SIZE_PAIR), key=lambda c: c["flops_per_s"])
        secondary.append(dict(budget=b, method="hnet_refractory", N=4, refractory=1, size="frontier",
                              hours=b / c["flops_per_s"] / 3600, passes=b / c["flops_per_symbol"] / train_symbols))
    hours = lambda rows: sum(r["hours"] for r in rows)
    top_rows = [r for r in plan if r["budget"] == budgets[-1]]
    top_slow = max(r["hours"] for r in plan if r["budget"] == budgets[-1] and r["method"] in ("hnet", "bpe"))
    lr_log = json.load(open("runs/lr_search.json"))
    lr_hours = sum(v["train"]["seconds"] for k, v in lr_log.items() if not k.startswith("_")) / 3600
    run_list = [
        dict(block="Learning-rate search (6 methods x 3 rates, + edge rule), size S, smallest budget", runs="18-24",
             hours=lr_hours, basis="measured"),
        dict(block="Main ladder: BPE, H-Net, Fixed, Entropy, Byte, Byte + conv x 4 budgets x 2 sizes", runs=len(plan),
             hours=hours(plan), basis="measured throughput"),
        dict(block="Extra seeds at the top budget: H-Net and BPE, frontier size, seeds 1-2", runs=4,
             hours=4 * top_slow, basis="measured throughput (slower size)"),
        dict(block="Secondary, N = 4: H-Net and Fixed x 4 budgets x 2 sizes", runs=16,
             hours=hours([r for r in secondary if r["method"] != "hnet_refractory"]), basis="measured throughput"),
        dict(block="Secondary, N = 4: H-Net refractory (r = 1), top two budgets", runs=2,
             hours=hours([r for r in secondary if r["method"] == "hnet_refractory"]), basis="measured throughput"),
        dict(block="Few-shot fine-tuning (H3)", runs="about 48", hours=4.0, basis="brief's estimate"),
        dict(block="Policies and rollouts (H4)", runs="about 24", hours=7.0, basis="brief's estimate"),
        dict(block="Evaluation (validation for every run; test for frontier runs)", runs="-", hours=5.0,
             basis="estimate: forward passes over validation and test windows"),
    ]
    out_total = sum(r["hours"] for r in run_list)
    w = sqrt_weights(streams)
    seen = max(r["symbols_seen"] for r in plan)
    per_passes = {s.name: float(seen * wi / s.n_symbols(s.episodes("train"))) for s, wi in zip(streams, w)}
    out = dict(R_BPE=R, N=N, real_factors=REAL, secondary=secondary, run_list=run_list, total_hours=out_total, train_symbols=train_symbols, budgets=budgets, size_pair=SIZE_PAIR, configs=cfg, runs=plan,
               ladder_hours=hours(plan), max_run_hours=max(r["hours"] for r in plan),
               max_passes=max(r["passes"] for r in plan), top_budget_hours=hours(top_rows),
               per_dataset_passes_at_max=per_passes, max_secondary_hours=max(r["hours"] for r in secondary))
    json.dump(out, open("runs/ladder.json", "w"), indent=1)
    print("budgets", budgets, "train symbols", train_symbols)
    for r in top_rows:
        print(f"  top {r['method']:7s} {r['size']}  {r['hours']:.2f} h  passes {r['passes']:.2f}")
    print(f"secondary longest {out['max_secondary_hours']:.2f} h")
    print(f"ladder {out['ladder_hours']:.1f} h, longest run {out['max_run_hours']:.2f} h, max passes {out['max_passes']:.2f}")
    for r in run_list:
        print(f"  {r['hours']:5.1f} h  {r['runs']!s:>8}  {r['block']}")
    print(f"total {out_total:.1f} h (cap 72 h)")


if __name__ == "__main__":
    main()
