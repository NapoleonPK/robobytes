"""Does the live PushT environment match the stream policy.py was trained on? No model is involved, and only few-shot pool
episodes are read. Against the recorded demonstrations it checks: geometry (a rigid fit of the environment's keypoints to the
recorded ones), open-loop replay of the recorded actions, with wrong mappings as controls, the environment's reward and success
against the recorded ones, and byte agreement of policy.py's observation with the stream. Writes runs/pusht_mapping_check.json.
Runs in the separate PushT venv, CPU only: PYTHONPATH=. nice -n 15 ../robobytes-pusht-venv/bin/python scripts/check_pusht_mapping.py"""
import importlib.util
from importlib.metadata import version
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from robobytes.serialize import Stream, quantise

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("policy", "scripts/policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

PARQUET = "downloads/hub/lerobot/pusht_keypoints/data/chunk-000/file-000.parquet"
N_REPLAY = 20
A, K = 2, 18  # agent_pos ends at A, keypoints at K, action follows


def load():
    t = pq.read_table(PARQUET).to_pydict()
    pool = [int(str(e).split(":")[-1]) for e in json.load(open("prereg/splits.json"))["heldout_pusht"]["pool"]]
    ep, out = np.array(t["episode_index"]), {}
    state, env, act = (np.array(t[k], np.float64) for k in ("observation.state", "observation.environment_state", "action"))
    rew, succ = np.array(t["next.reward"], np.float64).ravel(), np.array(t["next.success"], bool).ravel()
    for e in pool:
        m = ep == e
        out[e] = dict(obs=np.concatenate([state[m], env[m]], 1), action=act[m], reward=rew[m], success=succ[m])
    return pool, out


class Fit:
    """Block pose from keypoints: reset_to_state rotates about the centre of mass, so the angle comes from a rigid fit and the
    position from one correction read back from the environment."""

    def __init__(self, env, flat):
        self.env, self.flat = env, flat
        o, _ = env.reset(options=dict(reset_to_state=[100.0, 100.0, 256.0, 256.0, 0.0]))
        self.K0 = flat(o)[A:K].reshape(8, 2)
        self.c0 = self.K0.mean(0)

    def reset(self, obs):
        tgt = obs[A:K].reshape(8, 2)
        H = (self.K0 - self.c0).T @ (tgt - tgt.mean(0))
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        th = float(np.arctan2(R[1, 0], R[0, 0]))
        pos = np.array([256.0, 256.0])
        for _ in range(2):
            o, _ = self.env.reset(options=dict(reset_to_state=[*obs[:A], *pos, th]))
            pos = pos + (tgt - self.flat(o)[A:K].reshape(8, 2)).mean(0)
        o, _ = self.env.reset(options=dict(reset_to_state=[*obs[:A], *pos, th]))
        return o


def replay(env, fit, flat, ep, tf=lambda a: a, shift=0):
    o = fit.reset(ep["obs"][0])
    ag, kp = [], []
    for t in range(len(ep["action"]) - 1):
        a = tf(ep["action"][max(t - shift, 0)])
        o, *_ = env.step(a.astype(env.action_space.dtype))
        f, g = flat(o), ep["obs"][t + 1]
        ag.append(np.linalg.norm(f[:A] - g[:A]))
        kp.append(np.linalg.norm((f[A:K] - g[A:K]).reshape(8, 2), axis=1).mean())
    return np.array(ag), np.array(kp)


def main():
    env, flat = policy.make_env("pusht")
    u = env.unwrapped
    pool, data = load()
    res = dict(packages={p: version(p) for p in ("gym-pusht", "pymunk", "gymnasium", "numpy", "torch")},
               episodes_used="pool only", n_pool=len(pool))
    o, info = env.reset(seed=300000)
    res["reset"] = dict(keys=sorted(o), agent_pos=list(o["agent_pos"].shape), environment_state=list(o["environment_state"].shape),
                        flat=len(flat(o)), action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
                        control_hz=u.control_hz, is_success_key="is_success" in info, max_episode_steps=env.spec.max_episode_steps)

    fit = Fit(env, flat)
    resid, ang = [], []
    for e in pool:
        for t in (0, len(data[e]["obs"]) // 2, len(data[e]["obs"]) - 1):
            ob = data[e]["obs"][t]
            f = flat(fit.reset(ob))
            resid.append(np.linalg.norm((f[A:K] - ob[A:K]).reshape(8, 2), axis=1).max())
            ang.append(np.linalg.norm(f[:A] - ob[:A]))
    res["geometry"] = dict(frames=len(resid), keypoint_max_px=float(np.max(resid)), keypoint_median_px=float(np.median(resid)),
                           agent_max_px=float(np.max(ang)))

    variants = {"as_mapped": dict(), "action_xy_swapped": dict(tf=lambda a: a[::-1]), "action_one_step_late": dict(shift=1)}
    rep = {}
    for name, kw in variants.items():
        rows = [replay(env, fit, flat, data[e], **kw) for e in pool[:N_REPLAY]]
        one = [(r[0][0], r[1][0]) for r in rows]
        rep[name] = dict(episodes=len(rows), agent_mean_px=float(np.mean([r[0].mean() for r in rows])),
                         agent_first_step_px=float(np.mean([x[0] for x in one])),
                         keypoint_mean_px=float(np.mean([r[1].mean() for r in rows])),
                         keypoint_median_final_px=float(np.median([r[1][-1] for r in rows])))
    res["replay_open_loop"] = rep

    # recorded reward and success: frame t holds the outcome of the state at t + 1
    dr, ok, n, last, last_ok = [], 0, 0, 0, 0
    for i, e in enumerate(pool):
        d = data[e]
        frames = range(len(d["obs"]) - 1) if i < 10 else [len(d["obs"]) - 2]
        for t in frames:
            fit.reset(d["obs"][t + 1])
            c = u._get_coverage()
            dr.append(abs(float(np.clip(c / u.success_threshold, 0, 1)) - d["reward"][t]))
            ok += bool(c > u.success_threshold) == bool(d["success"][t])
            n += 1
        last += bool(d["success"].any())
    res["reward_success"] = dict(frames=n, reward_abs_diff_mean=float(np.mean(dr)), reward_abs_diff_p99=float(np.quantile(dr, .99)),
                                 success_agreement=ok / n, pool_episodes_recorded_success=last)

    # can a demonstration reach the environment's own success flag? The last action's outcome is not a recorded frame, so it is
    # replayed. Over the whole dataset this reads only the flag column.
    end = []
    for e in pool:
        d = data[e]
        fit.reset(d["obs"][-1])
        *_, info = env.step(d["action"][-1].astype(env.action_space.dtype))
        end.append((info["coverage"], bool(info["is_success"])))
    cov = np.array([x[0] for x in end])
    flag = pq.read_table(PARQUET, columns=["next.success", "episode_index"]).to_pydict()
    res["demonstration_success"] = dict(threshold=u.success_threshold, final_coverage_median=float(np.median(cov)),
                                        final_coverage_max=float(cov.max()), pool_success_after_last_action=sum(x[1] for x in end),
                                        pool_episodes=len(end), dataset_episodes=len(set(flag["episode_index"])),
                                        dataset_success_frames=int(np.sum(flag["next.success"])))

    s = Stream("heldout_pusht")
    meta = json.load(open("data/streams/heldout_pusht/meta.json"))
    lo, hi = np.array(meta["lo"]), np.array(meta["hi"])
    eps = s.episodes("pool")
    ids = [int(str(x).split(":")[-1]) for x in s.ep_id[eps]]
    diffs = []
    for e, ix in list(zip(ids, eps))[:N_REPLAY]:
        r = np.asarray(s.rows[int(s.start[ix]):int(s.start[ix] + s.length[ix])])
        d = data[e]
        q = quantise(np.concatenate([d["obs"], d["action"]], 1), lo, hi)
        direct = int(np.abs(q.astype(int) - r.astype(int)).max())
        ob = d["obs"][0]
        live = quantise(flat(fit.reset(ob))[None], lo[:K], hi[:K])[0]
        diffs.append((direct, int(np.abs(live.astype(int) - r[0, :K].astype(int)).max())))
    res["bytes"] = dict(episodes=len(diffs), recorded_vs_stream_max=max(x[0] for x in diffs), live_vs_stream_max=max(x[1] for x in diffs))

    t0 = time.time()
    head = lambda x: torch.full((1, A), 0.5)
    succ = [policy.rollout(env, flat, None, head, K, lo, hi, 300000 + i, "cpu") for i in range(3)]
    res["rollout_smoke"] = dict(rollouts=3, seconds=time.time() - t0, returned=[x[0] for x in succ], peak_coverage=[x[1] for x in succ],
                                 successes_expected_none=not any(x[0] for x in succ))
    json.dump(res, open("runs/pusht_mapping_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
