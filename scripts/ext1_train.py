"""One RoboBytes-Ext1 training run on CUDA: the loop of robobytes/train.py at the pinned commit, with fp16 autocast and a
wall-clock stop. Model, loss, batches, step count and schedule come from the pinned code, so a 5e15 run here has the same
step count and seed as its Mac run. Checkpoints every --ckpt-every seconds (default 30 min) and at the end; a run resumes
from <out>/ckpt.pt, or from --resume-from when it holds a newer one. Everything the run needs is imported, not copied.
usage: python ext1_train.py --method hnet --size S --budget 5e15 --lr 1.5e-3 --out /kaggle/working/x [--device cuda:0]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from robobytes.data import CTX, Mixture
from robobytes.evaluate import validate
from robobytes.flops import flops_per_byte, params
from robobytes.train import EFF_BATCH, SIZES, bpe_ctx, bpe_info, build, forward, lr_at


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["byte", "bpe", "hnet", "fixed", "byteconv"])
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", help="an earlier session's run directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ckpt-every", type=float, default=1800)
    ap.add_argument("--stop-after", type=float, default=0, help="save and exit cleanly after this many seconds")
    ap.add_argument("--max-steps", type=int, default=0, help="stop at this step (tests)")
    ap.add_argument("--crash-at-step", type=int, default=0, help="save, then die without cleanup at this step (tests)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--init-from", help="load weights only from this checkpoint (throughput timing with a trained router)")
    return ap.parse_args()


def adopt(a, out):
    """Copy the previous session's checkpoint into this session's output if it is ahead."""
    if not a.resume_from:
        return
    src = Path(a.resume_from) / "ckpt.pt"
    if not src.exists():
        return
    have = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["step"] if (out / "ckpt.pt").exists() else -1
    if torch.load(src, map_location="cpu", weights_only=False)["step"] > have:
        shutil.copy(src, out / "ckpt.pt")


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adopt(a, out)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    info = bpe_info()
    R, N = info["R_BPE"], info.get("N_used", info["N"])
    torch.manual_seed(a.seed)
    model = build(a.method, a.size, N, R)  # init on CPU, as on the Mac, so a seed gives the same start
    comp = 1.0 if a.method in ("byte", "byteconv") else (R if a.method == "bpe" else float(N))
    fpb = flops_per_byte(model, CTX, comp)
    model.to(dev)
    ctx = bpe_ctx(R) if a.method == "bpe" else CTX
    sym_per_step = EFF_BATCH * (ctx * R if a.method == "bpe" else ctx)
    total = max(1, int(a.budget / (fpb * sym_per_step)))
    mix = Mixture("bpe" if a.method == "bpe" else "byte", ctx, seed=a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location="cpu", weights_only=False)["model"])
    fp16 = a.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    step, micro, log, events, seconds = 0, a.micro, [], [], 0.0
    ck = out / "ckpt.pt"
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        step, micro, log, events, seconds = c["step"], c["micro"], c["log"], c["events"], c["seconds"]
        mix.set_state(c["mix"])
        torch.set_rng_state(c["torch_rng"])
        if cuda:
            torch.cuda.set_rng_state(c["cuda_rng"], dev)
        events.append(dict(step=step, event="resume"))
        print(f"resumed at step {step}", flush=True)
    # same keys as the Mac runs, so robobytes.evaluate.load_run reads these checkpoints
    run = dict(vars(a), pos="rope", init="default", outer_width=128, outer_blocks=2, conv_init=1.0, no_ema=False,
               rope_bytes=True)
    meta = dict(args=run, params=params(model), flops_per_symbol=fpb, compression_assumed=comp, ctx=ctx, steps=total,
                symbols_per_step=sym_per_step, R_BPE=R, N=N, refractory=getattr(model, "refractory", None),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(dev) if cuda else "cpu", precision=a.precision)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)

    def save(final=False):
        state = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, micro=micro,
                     log=log, events=events, seconds=seconds + time.time() - t0, mix=mix.state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(dev) if cuda else None, args=run, meta=meta)
        torch.save(state, out / ("final.pt" if final else "ckpt.pt.tmp"))
        if not final:
            os.replace(out / "ckpt.pt.tmp", ck)

    t0 = t_last = time.time()
    chunks_sum = pos_sum = 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr)
        batch = torch.from_numpy(mix.batch(EFF_BATCH)).to(dev)
        try:
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for i in range(0, EFF_BATCH, micro):
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    loss, o = forward(a.method, model, batch[i:i + micro], N)
                scaler.scale(loss * micro / EFF_BATCH).backward()
                lsum += loss.item() * micro / EFF_BATCH
                if o["n_chunks"] is not None:
                    chunks_sum += int(o["n_chunks"].sum())
                    pos_sum += o["nll"].numel()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        except torch.cuda.OutOfMemoryError:
            if micro == 1:
                raise
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            micro //= 2
            events.append(dict(step=step, event="oom", micro=micro))
            print(f"OOM: micro-batch -> {micro}", flush=True)
            continue
        step += 1
        if step % a.log_every == 0 or step == total:
            rec = dict(step=step, loss=lsum, lr=lr_at(step - 1, total, a.lr), seconds=seconds + time.time() - t0,
                       gpu_peak_gb=torch.cuda.max_memory_allocated(dev) / 1e9 if cuda else 0, scale=scaler.get_scale(), micro=micro,
                       compression=(pos_sum / chunks_sum) if chunks_sum else None)
            chunks_sum = pos_sum = 0
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if a.crash_at_step and step == a.crash_at_step:
            save()
            print(f"crash test at step {step}", flush=True)
            os._exit(137)
        if time.time() - t_last > a.ckpt_every:
            save()
            t_last = time.time()
        if (a.stop_after and time.time() - t0 > a.stop_after) or (a.max_steps and step >= a.max_steps):
            break
    save()
    if step >= total:
        save(final=True)
    json.dump(dict(meta, log=log, events=events, done=step >= total, step=step), open(out / "train.json", "w"), indent=1)
    if step >= total:
        validate(out, dev, max_eps=a.val_episodes)  # fp32 on the GPU; writes val.json
        ck.unlink()  # final.pt holds the same state, and session output is limited


if __name__ == "__main__":
    main()
