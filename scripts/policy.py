"""Policy success (PREREGISTRATION.md section 9, H4). Freeze the 50-shot model from section 8; a 2-layer MLP head reads the
hidden state at the last observation byte of each step and predicts the action (MSE on the 50 pool episodes; only the head
trains). Rollout: keep the last 1,024 bytes of history; each step append the observation's bytes, read the action, execute
it, append the executed action's bytes and SEP. H-Net, BPE, Byte and an MLP behaviour-cloning lower bound, 3 seeds x 100
rollouts on fixed environment seeds. Success = the environment's own flag at the rollout's last step. Writes
runs/policy/<task>/<method>_s<seed>.json. A finished run is skipped. Run only after few_shot.py. Needs gym-pusht for PushT.
usage: PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 caffeinate -i \
       python scripts/policy.py [--tasks pusht fetch]"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robobytes import bpe
from robobytes.data import CTX
from robobytes.model import BOS, SEP
from robobytes.results import FEW_SHOT
from robobytes.serialize import Stream, quantise
from robobytes.train import bpe_ctx

TASKS = {"pusht": dict(stream="heldout_pusht", steps=300), "fetch": dict(stream="heldout_fetch", steps=150)}
METHODS = ("hnet", "bpe", "byte", "mlp")
SEEDS = (0, 1, 2)
ENV_SEEDS = list(range(300_000, 300_100))
WIDTH, HEAD_STEPS, HEAD_BATCH, HEAD_LR, HEAD_EVERY = 256, 2000, 256, 1e-3, 100
OUT = Path("runs/policy")


class Encoder:
    """Turns rows into model symbols and histories into hidden states. BPE encodes observation and action bytes separately,
    so no token crosses from observation to action: a token cannot hold bytes the policy has not yet produced."""

    def __init__(self, method, model=None, ctx=CTX, tok=None, dev="cpu"):
        self.method, self.model, self.ctx, self.tok, self.dev = method, model, ctx, tok, dev
        self.bos, self.sep = (bpe.BOS_T, bpe.SEP_T) if method == "bpe" else (BOS, SEP)
        self.last = {}
        if model is not None:
            model.eval()
            model.head.register_forward_pre_hook(lambda _m, inp: self.last.__setitem__("h", inp[0]))

    def encode(self, row):
        if self.method == "bpe":
            return bpe.encode_rows(self.tok, row[None])[0]
        return [int(b) for b in row]

    @torch.no_grad()
    def hidden(self, views):
        """Hidden state at the last position of each history; padding after it is causally invisible."""
        L = max(map(len, views))
        ids = torch.zeros(len(views), L, dtype=torch.long)
        for i, v in enumerate(views):
            ids[i, :len(v)] = torch.tensor(v)
        self.model(ids.to(self.dev))
        last = torch.tensor([len(v) - 1 for v in views], device=self.dev)
        return self.last["h"][torch.arange(len(views), device=self.dev), last].float()


class History:
    def __init__(self, enc):
        self.enc, self.ids = enc, [enc.bos]

    def add_obs(self, row):
        self.ids += self.enc.encode(row)

    def add_act(self, row):
        self.ids += self.enc.encode(row) + [self.enc.sep]

    def view(self):
        return self.ids[-self.enc.ctx:]


def layout(name):
    s = Stream(name)
    lo, hi = np.array(s.meta["lo"]), np.array(s.meta["hi"])
    return s, s.W - s.meta["dims"]["action"], lo, hi


def features(enc, name, eps, batch=64):
    """(hidden or normalised observation at each step, normalised action) over episodes."""
    s, n_obs, *_ = layout(name)
    X, Y, views = [], [], []

    def flush():
        if views:
            X.append(enc.hidden(views).cpu())
            views.clear()

    for e in eps:
        rows = np.asarray(s.rows[s.start[e]:s.start[e] + s.length[e]])
        Y.append(rows[:, n_obs:] / 255.0)
        if enc is None:
            X.append(torch.from_numpy(rows[:, :n_obs] / 255.0).float())
            continue
        h = History(enc)
        for r in rows:
            h.add_obs(r[:n_obs])
            views.append(h.view())
            h.add_act(r[n_obs:])
            if len(views) == batch:
                flush()
    if enc is not None:
        flush()
    return torch.cat(X), torch.from_numpy(np.concatenate(Y)).float()


def train_head(X, Y, Xv, Yv, seed, dev):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(X.shape[1], WIDTH), nn.ReLU(), nn.Linear(WIDTH, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR)
    X, Y, Xv, Yv = X.to(dev), Y.to(dev), Xv.to(dev), Yv.to(dev)
    gen = torch.Generator().manual_seed(seed)
    best, best_loss = None, float("inf")
    for step in range(1, HEAD_STEPS + 1):
        i = torch.randint(0, len(X), (HEAD_BATCH,), generator=gen).to(dev)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(head(X[i]), Y[i]).backward()
        opt.step()
        if step % HEAD_EVERY == 0:
            with torch.no_grad():
                v = F.mse_loss(head(Xv), Yv).item()
            if v < best_loss:
                best_loss, best = v, copy.deepcopy(head.state_dict())
    head.load_state_dict(best)
    return head.eval(), best_loss


def make_env(task):
    import gymnasium as gym
    if task == "pusht":
        import gym_pusht  # noqa: F401
        env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos",
                       max_episode_steps=TASKS[task]["steps"])
        return env, lambda o: np.concatenate([o["agent_pos"], o["environment_state"]])
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=TASKS[task]["steps"])
    env.unwrapped.target_in_the_air = False  # the demonstrations' goals are on the table
    return env, lambda o: np.concatenate([o["observation"], o["desired_goal"]])


@torch.no_grad()
def rollout(env, flat, enc, head, n_obs, lo, hi, env_seed, dev):
    o, _ = env.reset(seed=env_seed)
    low, high = env.action_space.low, env.action_space.high
    h = History(enc) if enc is not None else None
    while True:
        q = quantise(flat(o)[None], lo[:n_obs], hi[:n_obs])[0]
        if enc is None:
            x = torch.from_numpy(q / 255.0).float()[None].to(dev)
        else:
            h.add_obs(q)
            x = enc.hidden([h.view()])
        a01 = head(x).clamp(0, 1)[0].cpu().numpy()
        a = np.clip(lo[n_obs:] + a01 * (hi[n_obs:] - lo[n_obs:]), low, high).astype(env.action_space.dtype)
        o, _, term, trunc, info = env.step(a)
        if enc is not None:
            h.add_act(quantise(a[None].astype(np.float64), lo[n_obs:], hi[n_obs:])[0])
        if term or trunc:
            return bool(info["is_success"])


def encoder(method, stream, seed, dev):
    from robobytes.evaluate import load_run
    ck = seed if method in ("hnet", "bpe") else 0  # Byte has one checkpoint; its heads differ by seed
    mname, model, _, R, _ = load_run(FEW_SHOT / stream / f"{method}_s{ck}_k50", dev)
    assert mname == method
    for p in model.parameters():
        p.requires_grad_(False)
    return Encoder(method, model, bpe_ctx(R) if method == "bpe" else CTX, bpe.load() if method == "bpe" else None, dev)


def run(task, method, seed, dev):
    path = OUT / task / f"{method}_s{seed}.json"
    if path.exists():
        return
    stream = TASKS[task]["stream"]
    s, n_obs, lo, hi = layout(stream)
    enc = None if method == "mlp" else encoder(method, stream, seed, dev)
    X, Y = features(enc, stream, s.episodes("pool"))
    Xv, Yv = features(enc, stream, s.episodes("val"))
    head, mse = train_head(X, Y, Xv, Yv, seed, dev)
    env, flat = make_env(task)
    ok = [rollout(env, flat, enc, head, n_obs, lo, hi, e, dev) for e in ENV_SEEDS]
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(task=task, method=method, seed=seed, env_seeds=ENV_SEEDS, success=ok, val_mse=mse), open(path, "w"))
    print(f"{task} {method} s{seed}: success {np.mean(ok):.2f}, val MSE {mse:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    a = ap.parse_args()
    dev = torch.device("mps")
    for t in a.tasks:
        for m in a.methods:
            for s in SEEDS:
                run(t, m, s, dev)


if __name__ == "__main__":
    main()
