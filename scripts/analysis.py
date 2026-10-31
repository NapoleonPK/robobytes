"""Every reported number (PREREGISTRATION.md sections 2, 6, 7). `--score` scores the frontier runs on held-in test and on each
held-out robot's zero-shot test (GPU, after the ladder); the default reads those scores, runs/few_shot and runs/policy, and
writes runs/analysis/results.json and runs/analysis/tables.md. Nothing is typed by hand. Parts whose files do not exist yet
are reported as pending.
usage: PYTHONPATH=. python scripts/analysis.py --score; PYTHONPATH=. python scripts/analysis.py"""
import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np

from robobytes import stats
from robobytes.data import training_names
from robobytes.results import (FEW_SHOT, HELD_OUT, SCORES, bpb, compression, frontier_size, merge_seeds, pairs, run_dir)

TOP = 3
MAIN = ("byte", "bpe", "hnet", "fixed", "entropy", "byteconv")
ARMS = {m: dict(method=m) for m in MAIN}
ARMS |= {"hnet_no_ema": dict(method="hnet_no_ema"),
         "hnet_N4": dict(method="hnet", N=4),
         "fixed_N4": dict(method="fixed", N=4),
         "hnet_refractory_N4": dict(method="hnet_refractory", N=4, r=1, size_of="hnet_N4", budgets=(2, 3))}
SECONDARY = ("hnet_N4", "fixed_N4", "hnet_refractory_N4")
FEW_SHOT_METHODS = MAIN
OUT = Path("runs/analysis")
POLICY = Path("runs/policy")
TASKS = ("pusht", "fetch")
POLICY_METHODS = ("hnet", "bpe", "byte", "mlp")


def budgets_of(arm):
    return ARMS[arm].get("budgets", (0, 1, 2, 3))


def runs_of(arm, bi):
    """Run directories entering one arm at one budget: its frontier run; at the top budget H-Net and BPE add seeds 1-2."""
    a = ARMS[arm]
    src = ARMS[a["size_of"]] if "size_of" in a else a
    size = frontier_size(src["method"], bi, N=src.get("N"))
    seeds = (0, 1, 2) if arm in ("hnet", "bpe") and bi == TOP else (0,)
    return [run_dir(a["method"], bi, size, s, a.get("N"), a.get("r")) for s in seeds]


def score_all():
    import torch

    from robobytes.evaluate import load_run
    from robobytes.results import score_episodes
    from robobytes.serialize import Stream
    dev = torch.device("mps")
    SCORES.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for bi in budgets_of(arm):
            for run in runs_of(arm, bi):
                path = SCORES / f"{run.name}.json"
                if path.exists():
                    continue
                method, model, N, R, ent = load_run(run, dev)
                res = {name: score_episodes(method, model, name, "test", N, R, dev, ent=ent,
                                            eps=Stream(name).episodes("test"))
                       for name in training_names() + list(HELD_OUT)}
                json.dump(res, open(path, "w"))
                print(run.name, flush=True)


class Scores:
    """Per-arm tables. get(arm, budget, target): list of merged tables; target = 'held_in' or a held-out robot."""

    def __init__(self, names):
        self.names = names

    @cache
    def tables(self, name):
        return json.load(open(SCORES / f"{name}.json"))

    @cache
    def merged(self, arm, bi, name):
        return merge_seeds([self.tables(r.name)[name] for r in runs_of(arm, bi)])

    def get(self, arm, bi, target):
        return [self.merged(arm, bi, n) for n in (self.names if target == "held_in" else [target])]


def pairwise(get, new, base, bi, targets):
    return {t: stats.compare(pairs(get(new, bi, t), get(base, bi, t))) for t in targets}


def h1(get):
    out = {str(bi): stats.compare(pairs(get("hnet", bi, "held_in"), get("bpe", bi, "held_in"))) for bi in (2, 3)}
    return dict(budgets=out, passed=all(stats.better(c) for c in out.values()))


def transfer(get, new, base, robots, bi=TOP):
    """The H2 statistic at one budget: g_ID, g_OOD per robot with CIs, the line, and whether each robot clears it."""
    g_id = stats.compare(pairs(get(new, bi, "held_in"), get(base, bi, "held_in")))
    per = {}
    for r in robots:
        c = stats.compare(pairs(get(new, bi, r), get(base, bi, r)))
        per[r] = dict(c, clears=stats.transfer_pass(c["g"], c["g_ci"], g_id["g"]))
    return dict(g_id=g_id, line=stats.line(g_id["g"]), robots=per, passed=stats.at_least(c["clears"] for c in per.values()))


def primary(get, R, robots):
    realised = compression(get("hnet", TOP, "held_in"))
    floor_ok = stats.compression_ok(realised, R)
    res = dict(compression=dict(realised=realised, floor=stats.COMPRESSION_FLOOR * R, ok=floor_ok),
               H1=h1(get), H2=transfer(get, "hnet", "bpe", robots), H2b=transfer(get, "hnet", "fixed", robots))
    for h in ("H1", "H2"):  # the compression floor does not name H2b
        res[h]["passed_before_floor"] = res[h]["passed"]
        res[h]["passed"] = res[h]["passed"] and floor_ok
    return res


def few_shot_tables(robot, method, shots):
    rows = []
    for s in ((0, 1, 2) if method in ("hnet", "bpe") else (0,)):
        p = FEW_SHOT / robot / f"{method}_s{s}_k{shots}" / "result.json"
        if not p.exists():
            return None
        rows.append(json.load(open(p))["test"])
    return [merge_seeds(rows)]


def h3(robots):
    table, per = {}, {}
    for r in robots:
        for k in (10, 50):
            for m in FEW_SHOT_METHODS:
                t = few_shot_tables(r, m, k)
                if t is not None:
                    table[f"{r}/{m}/k{k}"] = bpb(t)
        a, b = few_shot_tables(r, "hnet", 50), few_shot_tables(r, "bpe", 50)
        if a is None or b is None:
            return dict(status="pending", bpb=table)
        c = stats.compare(pairs(a, b))
        per[r] = dict(c, clears=stats.better(c))
    return dict(status="done", bpb=table, robots=per, passed=stats.at_least(c["clears"] for c in per.values()))


def h4():
    res = {}
    for task in TASKS:
        m = {}
        for method in POLICY_METHODS:
            ps = [POLICY / task / f"{method}_s{s}.json" for s in (0, 1, 2)]
            if not all(p.exists() for p in ps):
                return dict(status="pending")
            rows = [json.load(open(p)) for p in ps]
            assert all(r["env_seeds"] == rows[0]["env_seeds"] for r in rows)
            m[method] = np.array([r["success"] for r in rows], np.float64)
        c = stats.policy_bootstrap(m["hnet"], m["bpe"])
        rates = {k: float(v.mean()) for k, v in m.items()}
        info = not stats.uninformative(list(rates.values()))
        res[task] = dict(c, rates=rates, informative=info, clears=info and c["diff"] > 0 and stats.excludes_zero(c["diff_ci"]))
    return dict(status="done", tasks=res, passed=any(v["clears"] for v in res.values()))


def pusht_coverage():
    """Exploratory, outside H4: per method, mean peak coverage and the share of rollouts at or above the fixed line."""
    d = json.load(open("runs/pusht_mapping_check.json"))["demonstration_success"]
    res = dict(demonstrations=d, line=stats.COVERAGE_LINE)
    ps = {m: [POLICY / "pusht" / f"{m}_s{s}.json" for s in (0, 1, 2)] for m in POLICY_METHODS}
    if not all(p.exists() for v in ps.values() for p in v):
        return dict(res, status="pending")
    return dict(res, status="done", methods={m: stats.coverage_summary([json.load(open(p))["peak_coverage"] for p in v])
                                             for m, v in ps.items()})


def frontiers(get, flops, robots):
    targets = ["held_in", *robots]
    out = {}
    for arm in ARMS:
        out[arm] = {t: {str(bi): dict(flops=flops[bi], bpb=bpb(get(arm, bi, t)), compression=compression(get(arm, bi, t)))
                        for bi in budgets_of(arm)} for t in targets}
    return out


def multipliers(front, flops, targets):
    """Exploratory: FLOPs BPE needs for H-Net's bpb, over H-Net's FLOPs."""
    return {t: stats.flop_multipliers(flops, [front["hnet"][t][str(i)]["bpb"] for i in range(4)],
                                      [front["bpe"][t][str(i)]["bpb"] for i in range(4)]) for t in targets}


def secondary(get, robots):
    targets = ["held_in", *robots]
    vs_bpe = {arm: {str(bi): pairwise(get, arm, "bpe", bi, targets) for bi in budgets_of(arm)} for arm in SECONDARY}
    router = {str(bi): transfer(get, "hnet_N4", "fixed_N4", robots, bi) for bi in range(4)}
    no_ema = {base: {str(bi): transfer(get, "hnet_no_ema", base, robots, bi) for bi in range(4)}
              for base in ("fixed", "hnet")}
    return dict(n4_vs_bpe=vs_bpe, n4_router=router, no_ema=no_ema)


def tables(r):
    L = ["# RoboBytes results tables", "", "Generated by scripts/analysis.py; do not edit.", ""]
    p = r["primary"]
    L += ["## Pass/fail", "", "| Test | Passed |", "| --- | --- |", f"| H1 | {p['H1']['passed']} |",
          f"| H2 | {p['H2']['passed']} |", f"| H2b | {p['H2b']['passed']} |",
          f"| H3 | {r['H3'].get('passed', r['H3']['status'])} |", f"| H4 | {r['H4'].get('passed', r['H4']['status'])} |", ""]
    c = p["compression"]
    L += [f"Compression at the top budget: {c['realised']:.3f}; floor {c['floor']:.3f}; met: {c['ok']}.", ""]
    L += ["## H1", "", "| Budget | H-Net | BPE | Difference | 95% CI |", "| --- | --- | --- | --- | --- |"]
    for bi, x in p["H1"]["budgets"].items():
        L.append(f"| {r['flops'][int(bi)]:.3g} | {x['bpb_new']:.4f} | {x['bpb_base']:.4f} | {x['diff']:.4f} | "
                 f"[{x['diff_ci'][0]:.4f}, {x['diff_ci'][1]:.4f}] |")
    for h in ("H2", "H2b"):
        x = p[h]
        L += ["", f"## {h}", "", f"g_ID {x['g_id']['g']:.4f} [{x['g_id']['g_ci'][0]:.4f}, {x['g_id']['g_ci'][1]:.4f}]; "
              f"line {x['line']:.4f}", "", "| Robot | g_OOD | 95% CI | Clears |", "| --- | --- | --- | --- |"]
        for k, v in x["robots"].items():
            L.append(f"| {k} | {v['g']:.4f} | [{v['g_ci'][0]:.4f}, {v['g_ci'][1]:.4f}] | {v['clears']} |")
    L += ["", "## Iso-FLOP frontiers (bpb)", "", "| Arm | Target | " + " | ".join(f"{f:.3g}" for f in r["flops"]) + " |",
          "| --- | --- | " + " | ".join("---" for _ in r["flops"]) + " |"]
    for arm, ts in r["frontiers"].items():
        for t, row in ts.items():
            L.append(f"| {arm} | {t} | " + " | ".join(f"{row[str(i)]['bpb']:.4f}" if str(i) in row else "-"
                                                      for i in range(len(r["flops"]))) + " |")
    x, d = r["pusht_coverage"], r["pusht_coverage"]["demonstrations"]
    L += ["", "## PushT peak coverage (exploratory, outside H4)", "",
          f"No PushT demonstration reaches the environment's {d['threshold']:.2f} success flag: "
          f"{d['dataset_success_frames']} success frames in {d['dataset_episodes']} episodes, and the "
          f"{d['pool_episodes']} few-shot pool episodes peak at {d['final_coverage_max']:.3f} coverage.", ""]
    if x["status"] == "done":
        L += [f"| Method | Mean peak coverage | 95% CI | Share of rollouts ≥ {x['line']:.2f} | 95% CI |",
              "| --- | --- | --- | --- | --- |"]
        for m, v in x["methods"].items():
            L.append(f"| {m} | {v['mean']:.4f} | [{v['mean_ci'][0]:.4f}, {v['mean_ci'][1]:.4f}] | {v['share']:.4f} | "
                     f"[{v['share_ci'][0]:.4f}, {v['share_ci'][1]:.4f}] |")
    else:
        L.append("Rollouts pending.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score:
        score_all()
        return
    ladder = json.load(open("runs/ladder.json"))
    R, flops = json.load(open("runs/bpe_summary.json"))["R_BPE"], ladder["budgets"]
    get = Scores(training_names()).get
    robots = list(HELD_OUT)
    front = frontiers(get, flops, robots)
    res = dict(flops=flops, primary=primary(get, R, robots), H3=h3(robots), H4=h4(), pusht_coverage=pusht_coverage(), frontiers=front,
               multipliers=multipliers(front, flops, ["held_in", *robots]), secondary=secondary(get, robots))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    (OUT / "tables.md").write_text(tables(res))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
