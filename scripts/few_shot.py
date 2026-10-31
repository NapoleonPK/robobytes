"""Few-shot adaptation (PREREGISTRATION.md section 8). From each method's top-budget frontier checkpoint (H-Net and BPE:
seeds 0-2), fine-tune all parameters on the 10-shot and 50-shot pool of each held-out robot: AdamW at 0.3 x the method's
pretraining rate, constant, effective batch 16 windows, 500 steps, validation bpb on the robot's 10 validation episodes at
step 0 and every 50 steps, keep the best; then score the test episodes. Writes runs/few_shot/<robot>/<method>_s<seed>_k<shots>/
result.json, and final.pt at 50 shots (section 9 reads it). A finished run is skipped. Run only after the ladder.
usage: PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 caffeinate -i \
       python scripts/few_shot.py [--dry-run]"""
import argparse
import json

import numpy as np
import torch

from robobytes.data import CTX, TokenConcat
from robobytes.evaluate import load_run
from robobytes.results import FEW_SHOT, HELD_OUT, frontier_size, run_dir, score_episodes
from robobytes.serialize import Concat, Stream
from robobytes.train import EFF_BATCH, bpe_ctx, forward, swap_mb

METHODS = ("hnet", "bpe", "byte", "fixed", "entropy", "byteconv")
SHOTS = (10, 50)
STEPS, EVERY, RATE_FACTOR, TOP = 500, 50, 0.3, 3


def seeds(method):
    return (0, 1, 2) if method in ("hnet", "bpe") else (0,)


def pool_source(name, method, shots, ctx):
    """The first `shots` pool episodes, concatenated, as a source of training windows."""
    s = Stream(name)
    eps = s.episodes("pool")[:shots]
    assert len(eps) == shots
    if method == "bpe":
        src = TokenConcat(name, eps)
    else:
        src = Concat(s, "pool")
        src.eps = eps
        src.len = 1 + s.length[eps] * (s.W + 1)
        src.off = np.concatenate([[0], np.cumsum(src.len)])
        src.total = int(src.off[-1])
    assert src.total > ctx + 1, (name, shots)
    return src


def windows(src, rng, ctx):
    out = np.empty((EFF_BATCH, ctx + 1), np.int64)
    for i in range(EFF_BATCH):
        out[i] = src.window(int(rng.integers(0, src.total - ctx - 1)), ctx + 1)
    return out


def val_bpb(method, model, name, N, R, dev, ent):
    t = score_episodes(method, model, name, "val", N, R, dev, ent=ent, eps=Stream(name).episodes("val"))
    return sum(t["bits"]) / sum(t["bytes"])


def tune(robot, method, seed, shots, dev, micro):
    out = FEW_SHOT / robot / f"{method}_s{seed}_k{shots}"
    if (out / "result.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    size = frontier_size(method, TOP)
    run = run_dir(method, TOP, size, seed)
    mname, model, N, R, ent = load_run(run, dev)
    assert mname == method
    pre = json.load(open(run / "meta.json"))["args"]["lr"]
    ctx = bpe_ctx(R) if method == "bpe" else CTX
    src = pool_source(robot, method, shots, ctx)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=RATE_FACTOR * pre, betas=(0.9, 0.95), weight_decay=0.1)
    best_v = val_bpb(method, model, robot, N, R, dev, ent)
    best_step, log, events = 0, [dict(step=0, val_bpb=best_v)], []
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    swap0 = swap_mb()
    step = 0
    while step < STEPS:
        batch = torch.from_numpy(windows(src, rng, ctx)).to(dev)
        try:
            opt.zero_grad(set_to_none=True)
            for i in range(0, EFF_BATCH, micro):
                loss, _ = forward(method, model, batch[i:i + micro], N, ent)
                (loss * micro / EFF_BATCH).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as e:
            if "out of memory" not in str(e) or micro == 1:
                raise
            opt.zero_grad(set_to_none=True)
            torch.mps.empty_cache()
            micro //= 2
            events.append(dict(step=step, event="oom", micro=micro))
            continue
        step += 1
        if step % EVERY == 0:
            torch.mps.empty_cache()
            v = val_bpb(method, model, robot, N, R, dev, ent)
            log.append(dict(step=step, val_bpb=v))
            if v < best_v:
                best_v, best_step = v, step
                best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            sw = swap_mb()
            if sw - swap0 > 2048 and micro > 1:
                micro //= 2
                events.append(dict(step=step, event="swap", swap_mb=sw, swap0_mb=swap0, micro=micro))
    model.load_state_dict(best)
    test = score_episodes(method, model, robot, "test", N, R, dev, ent=ent)
    if shots == 50:
        c = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        torch.save(dict(model=best, args=c["args"], meta=c["meta"]), out / "final.pt")
    json.dump(dict(robot=robot, method=method, seed=seed, shots=shots, source=str(run), rate=RATE_FACTOR * pre,
                   best_step=best_step, val_bpb=best_v, log=log, events=events, test=test),
              open(out / "result.json", "w"), indent=1)
    print(f"{out.name} {robot}: val {best_v:.4f} at step {best_step}, "
          f"test {sum(test['bits']) / sum(test['bytes']):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=list(HELD_OUT))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--shots", nargs="+", type=int, default=list(SHOTS))
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="list the runs")
    a = ap.parse_args()
    jobs = [(r, m, s, k) for r in a.robots for m in a.methods for s in seeds(m) for k in a.shots]
    print(len(jobs), "runs")
    if a.dry_run:
        return
    dev = torch.device("mps")
    for r, m, s, k in jobs:
        tune(r, m, s, k, dev, a.micro)


if __name__ == "__main__":
    main()
