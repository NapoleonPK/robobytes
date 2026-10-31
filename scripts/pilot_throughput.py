"""Day-1 throughput pilot (brief section 4): time 200 training steps per method and size on random bytes, micro-batch
16, context 1,024 bytes. Records bytes/s, achieved FLOP/s (analytic), peak MPS memory and swap. Writes
runs/pilot_throughput.json. usage: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python scripts/pilot_throughput.py [--device mps]"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from robobytes.flops import flops_per_byte, params
from robobytes.model import VOCAB, ByteHNet, TokenLM, ratio_loss

SIZES = {"S": (256, 4, 4), "M": (384, 8, 6), "L": (512, 12, 8)}  # main width, layers, heads
R_ASSUMED, N_ASSUMED = 4.0, 4  # placeholders until BPE is trained on day 2; FLOPs are rescaled then
CTX, MB = 1024, 16


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def build(method, size):
    d, L, h = SIZES[size]
    if method == "byte":
        return TokenLM(VOCAB, d, L, h), 1.0
    if method == "bpe":
        return TokenLM(4096 + 2, d, L, h), R_ASSUMED
    return ByteHNet(d, L, h, refractory=3 if method == "hnet_refractory" else 0), float(N_ASSUMED)


def step_fn(method, model, dev, MB):
    if method in ("byte", "bpe"):
        n = CTX if method == "byte" else int(CTX / R_ASSUMED)
        V = VOCAB if method == "byte" else 4098

        def f():
            ids = torch.randint(0, V, (MB, n), device=dev)
            logits = model(ids)["logits"]
            return F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        return f
    fixed = method in ("fixed", "entropy")
    bnd = torch.zeros(MB, CTX, dtype=torch.bool, device=dev)
    bnd[:, ::N_ASSUMED] = True

    def f():
        ids = torch.randint(0, VOCAB, (MB, CTX), device=dev)
        out = model(ids, bnd if fixed else None)
        loss = F.cross_entropy(out["logits"][:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        if not fixed:
            loss = loss + 0.3 * ratio_loss(out["b"], out["p"], N_ASSUMED)
        return loss
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--R", type=float, default=4.0)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--methods", nargs="+", default=["byte", "bpe", "hnet", "fixed", "entropy", "hnet_refractory"])
    a = ap.parse_args()
    globals().update(R_ASSUMED=a.R, N_ASSUMED=a.N)
    dev = torch.device(a.device)
    sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)
    res = {"device": a.device, "R": a.R, "N": a.N, "swap_mb_start": swap_mb(), "runs": {}}
    out = Path(f"runs/pilot_throughput_{a.device}{a.tag}.json")
    if out.exists():  # merge with earlier timings of other methods
        res["runs"] = json.load(open(out))["runs"]
    for method in a.methods:
        for size in SIZES:
            mb = MB
            while True:  # halve the micro-batch on out-of-memory (brief section 4), logged in the result
                torch.manual_seed(0)
                model, comp = build(method, size)
                model.to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                f = step_fn(method, model, dev, mb)
                try:
                    for _ in range(5):  # warm-up, not timed
                        opt.zero_grad(set_to_none=True)
                        f().backward()
                        opt.step()
                    sync()
                    if a.device == "mps":
                        torch.mps.empty_cache()
                    t0, peak, sw = time.time(), 0, 0.0
                    for i in range(a.steps):  # an H-Net router can emit more chunks as it trains, so OOM can come late
                        opt.zero_grad(set_to_none=True)
                        f().backward()
                        opt.step()
                        if i % 100 == 99:
                            sync()
                            if a.device == "mps":
                                peak = max(peak, torch.mps.current_allocated_memory())
                            sw = max(sw, swap_mb())
                    sync()
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e) or mb == 1:
                        raise
                    del model, opt, f
                    if a.device == "mps":
                        torch.mps.empty_cache()
                    print(f"{method} {size}: OOM at micro-batch {mb}, halving", flush=True)
                    mb //= 2
            dt = (time.time() - t0) / a.steps
            fpb = flops_per_byte(model.cpu(), CTX, comp)
            r = dict(params=params(model), micro_batch=mb, s_per_step=dt, bytes_per_s=mb * CTX / dt,
                     flops_per_byte=fpb, achieved_flops=fpb * mb * CTX / dt, mps_alloc_gb=peak / 1e9, swap_mb_max=sw,
                     compression_assumed=comp)
            res["runs"][f"{method}_{size}"] = r
            print(f"{method:16s} {size} {r['params'] / 1e6:5.1f}M mb{mb:<3d}{dt * 1e3:7.1f} ms/step  {r['bytes_per_s'] / 1e3:7.1f} kB/s  "
                  f"{r['achieved_flops'] / 1e12:5.2f} TFLOP/s  mem {r['mps_alloc_gb']:.2f} GB  swap {sw:.0f} MB", flush=True)
            json.dump(res, open(out, "w"), indent=1)
            del model, opt
            if a.device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
