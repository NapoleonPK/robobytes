"""One training run at a fixed FLOP budget (brief sections 4, 6, 7).

Steps = budget / (analytic FLOPs per symbol x symbols per step). Effective batch 16 windows; on out-of-memory or swap
growth above 2 GB during the run the micro-batch halves and gradient accumulation keeps the effective batch (logged). AdamW (0.9, 0.95),
weight decay 0.1, linear warm-up over 2% of steps then cosine to 10% of the peak learning rate, gradient clip 1.0.
Training loss = next-symbol cross-entropy on every target (+ alpha x ratio loss for H-Net); evaluation counts data
targets only. Checkpoints every 10 minutes and at the end; a run resumes from its checkpoint.
usage: PYTHONPATH=. python -m robobytes.train --method hnet --size S --budget 1e15 --lr 1e-3 --out runs/x
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import bpe
from .data import CTX, Mixture
from .flops import flops_per_byte, params
from .model import BOS, SEP, VOCAB, ByteHNet, TokenLM, ratio_loss

SIZES = {"S": (256, 4, 4), "M": (384, 8, 6), "L": (512, 12, 8)}  # main width, layers, heads
EFF_BATCH, ALPHA = 16, 0.3
CKPT_EVERY = 600


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def bpe_info():
    return json.load(open("runs/bpe_summary.json"))


def refractory_for(N):
    """Pre-registered refractory window: r = floor(0.3 N) (r = 3 at RoboChunk's N = 10; r = 0 at N = 2)."""
    return int(math.floor(0.3 * N))


def build(method, size, N, R, refractory=None, pos="rope", init="default"):
    d, L, h = SIZES[size]
    if method == "byte":
        return TokenLM(VOCAB, d, L, h, pos=pos, init=init)
    if method == "bpe":
        return TokenLM(bpe.VOCAB_T, d, L, h, max_len=bpe_ctx(R), pos=pos, init=init)
    r = (refractory_for(N) if refractory is None else refractory) if method == "hnet_refractory" else 0
    return ByteHNet(d, L, h, refractory=r, pos=pos, init=init)


def bpe_ctx(R):
    return int(round(CTX / R))


def fixed_boundaries(ids, N):
    b = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
    b[:, ::N] = True
    return b


class EntropyBoundaries:
    """BLT-like: a small byte model's next-symbol entropy; a boundary starts at t when H(x_t | x_<t) > threshold."""

    def __init__(self, ckpt, threshold, dev):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        d, L, h = SIZES[c["args"]["size"]]
        self.m = TokenLM(VOCAB, d, L, h, pos=c["args"].get("pos", "learned")).to(dev).eval()
        self.m.load_state_dict(c["model"])
        self.threshold = threshold

    @torch.no_grad()
    def entropy(self, ids):
        logp = F.log_softmax(self.m(ids)["logits"].float(), -1)
        H = -(logp.exp() * logp).sum(-1) / math.log(2)  # H[:, t] = entropy of the prediction for x_{t+1}
        return torch.cat([torch.full_like(H[:, :1], float("inf")), H[:, :-1]], 1)  # aligned to x_t

    def __call__(self, ids):
        return self.entropy(ids) > self.threshold


def forward(method, model, ids, N, ent=None):
    """ids (B, n+1). Returns (loss for training, dict with per-target nll and chunk counts)."""
    x, y = ids[:, :-1], ids[:, 1:]
    if method in ("byte", "bpe"):
        out = model(x)
    elif method == "fixed":
        out = model(x, fixed_boundaries(x, N))
    elif method == "byteconv":  # H-Net's conv encoder/decoder, main Transformer over every byte (a boundary everywhere)
        out = model(x, fixed_boundaries(x, 1))
    elif method == "entropy":
        out = model(x, ent(x))
    else:
        out = model(x)
    nll = F.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]), y.reshape(-1),
                          reduction="none").view(y.shape)
    loss = nll.mean()
    if method in ("hnet", "hnet_refractory"):
        loss = loss + ALPHA * ratio_loss(out["b"], out["p"], N)
    return loss, dict(nll=nll, n_chunks=out.get("n_chunks"))


def lr_at(step, total, peak):
    warm = max(1, int(0.02 * total))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["byte", "bpe", "hnet", "hnet_refractory", "fixed", "entropy", "byteconv"])
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, help="chunk target; default round(R_BPE) (secondary comparison: 4)")
    ap.add_argument("--refractory", type=int, help="refractory window; default floor(0.3 N)")
    ap.add_argument("--pos", default="rope", choices=["learned", "rope"])
    ap.add_argument("--init", default="default", choices=["default", "gpt2"])
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--entropy-ckpt")
    ap.add_argument("--entropy-threshold", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early (pilot timing only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    info = bpe_info()
    R, N = info["R_BPE"], (a.N or info.get("N_used", info["N"]))
    torch.manual_seed(a.seed)
    model = build(a.method, a.size, N, R, a.refractory, a.pos, a.init).to(dev)
    comp = 1.0 if a.method in ("byte", "byteconv") else (R if a.method == "bpe" else float(N))
    fpb = flops_per_byte(model.cpu(), CTX, comp)
    model.to(dev)
    ctx = bpe_ctx(R) if a.method == "bpe" else CTX
    sym_per_step = EFF_BATCH * (ctx * R if a.method == "bpe" else ctx)
    total = max(1, int(a.budget / (fpb * sym_per_step)))
    mix = Mixture("bpe" if a.method == "bpe" else "byte", ctx, seed=a.seed)
    ent = EntropyBoundaries(a.entropy_ckpt, a.entropy_threshold, dev) if a.method == "entropy" else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    step, micro, log, events = 0, a.micro, [], []
    ck = out / "ckpt.pt"
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        step, micro, log, events = c["step"], c["micro"], c["log"], c["events"]
        mix.set_state(c["mix"])
        torch.set_rng_state(c["torch_rng"])
    meta = dict(args=vars(a), params=params(model), flops_per_symbol=fpb, compression_assumed=comp, ctx=ctx,
                steps=total, symbols_per_step=sym_per_step, R_BPE=R, N=N,
                refractory=getattr(model, "refractory", None))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)

    def save(final=False):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, micro=micro, log=log, events=events,
                        mix=mix.state(), torch_rng=torch.get_rng_state(), args=vars(a), meta=meta),
                   out / ("final.pt" if final else "ckpt.pt.tmp"))
        if not final:
            os.replace(out / "ckpt.pt.tmp", ck)

    t_last, t0 = time.time(), time.time()
    swap0 = swap_mb()  # macOS keeps swap allocated after pressure ends, so the rule uses growth during this run
    run_t = sum(e.get("seconds", 0) for e in log[-1:]) if log else 0.0
    chunks_sum = pos_sum = 0
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr)
        batch = torch.from_numpy(mix.batch(EFF_BATCH)).to(dev)
        try:
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for i in range(0, EFF_BATCH, micro):
                loss, o = forward(a.method, model, batch[i:i + micro], N, ent)
                (loss * micro / EFF_BATCH).backward()
                lsum += loss.item() * micro / EFF_BATCH
                if o["n_chunks"] is not None:
                    chunks_sum += int(o["n_chunks"].sum())
                    pos_sum += o["nll"].numel()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as e:
            if "out of memory" not in str(e) or micro == 1:
                raise
            opt.zero_grad(set_to_none=True)
            torch.mps.empty_cache() if a.device == "mps" else None
            micro //= 2
            events.append(dict(step=step, event="oom", micro=micro))
            print(f"OOM: micro-batch -> {micro}", flush=True)
            continue
        step += 1
        if a.device == "mps" and step % 50 == 0:
            torch.mps.empty_cache()  # variable chunk counts fragment the MPS cache
        if step % 100 == 0 or step == total:
            if a.device == "mps":
                torch.mps.synchronize()
            sw = swap_mb()
            rec = dict(step=step, loss=lsum, lr=lr_at(step - 1, total, a.lr), seconds=run_t + time.time() - t0,
                       mps_gb=torch.mps.current_allocated_memory() / 1e9 if a.device == "mps" else 0,
                       mps_driver_gb=torch.mps.driver_allocated_memory() / 1e9 if a.device == "mps" else 0, swap_mb=sw,
                       micro=micro, compression=(pos_sum / chunks_sum) if chunks_sum else None)
            chunks_sum = pos_sum = 0
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if sw - swap0 > 2048 and micro > 1:
                micro //= 2
                events.append(dict(step=step, event="swap", swap_mb=sw, swap0_mb=swap0, micro=micro))
                print(f"swap grew {sw - swap0:.0f} MB > 2 GB: micro-batch -> {micro}", flush=True)
        if time.time() - t_last > CKPT_EVERY:
            save()
            t_last = time.time()
        if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
            break
    save()
    if step >= total:
        save(final=True)
    json.dump(dict(meta, log=log, events=events, done=step >= total), open(out / "train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
