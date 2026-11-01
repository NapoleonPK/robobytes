"""Every reported Ext1 number (PREREGISTRATION_ext1.md sections 3-7), from the frozen files under runs/ext1: val.json and
meta of each run, and runs/ext1/scores from the kernel's `score` stage. The statistics are the main study's, imported from
robobytes.stats and scripts/analysis.py. Runs no model. Writes runs/ext1/analysis/results.json and tables.md; the Mac gaps
are reported as pending until the main study's scores exist.
usage: PYTHONPATH=. python scripts/ext1_analysis.py"""
import json
import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import analysis as study  # noqa: E402
from robobytes import stats  # noqa: E402
from robobytes.data import training_names  # noqa: E402
from robobytes.results import HELD_OUT, SCORES, compression, merge_seeds, pairs, run_dir  # noqa: E402

EXT = Path("runs/ext1")
OUT = EXT / "analysis"
METHODS = ("hnet", "fixed", "byte", "bpe")
ARMS = (*METHODS, "byteconv")
CALIBRATION = ("hnet", "fixed")
BUDGET = 5e16
CALIBRATION_LINE = 0.01
GAPS = ("byte", "fixed", "byteconv", "bpe")


def tag(method, size, budget, seed):
    return f"{method}_{budget:.0e}_{size}_s{seed}".replace("+", "")


def sizes_of(method):
    return ("S", "M") if method == "byteconv" else ("M", "L")


def val_bpb(path):
    return json.load(open(Path(path) / "val.json"))["val_bpb"]


def frontier(method):
    return min(sizes_of(method), key=lambda s: val_bpb(EXT / tag(method, s, BUDGET, 0)))


def seeds_of(method):
    return (0,) if method == "byteconv" else (0, 1)


class Ext:
    """get(arm, budget index, target) as in scripts/analysis.py; Ext1 has one budget, so the index is ignored."""

    def __init__(self, names):
        self.names = names

    @cache
    def tables(self, run):
        return json.load(open(EXT / "scores" / f"{run}.json"))

    @cache
    def merged(self, arm, name):
        return merge_seeds([self.tables(tag(arm, frontier(arm), BUDGET, s))[name] for s in seeds_of(arm)])

    def get(self, arm, _bi, target):
        return [self.merged(arm, n) for n in (self.names if target == "held_in" else [target])]


def primary(get, R, robots):
    realised = compression(get("hnet", 0, "held_in"))
    floor_ok = stats.compression_ok(realised, R)
    h1 = stats.compare(pairs(get("hnet", 0, "held_in"), get("bpe", 0, "held_in")))
    res = dict(compression=dict(realised=realised, floor=stats.COMPRESSION_FLOOR * R, ok=floor_ok), H1=dict(h1, passed=stats.better(h1)),
               H2=study.transfer(get, "hnet", "bpe", robots, 0), H2b=study.transfer(get, "hnet", "fixed", robots, 0))
    for h in ("H1", "H2"):  # as the main study, the compression floor does not name H2b
        res[h]["passed_before_floor"] = res[h]["passed"]
        res[h]["passed"] = res[h]["passed"] and floor_ok
    return res


def calibration(mac_top=3):
    rows = {}
    for m in CALIBRATION:
        for s in "SM":
            kaggle, mac = val_bpb(EXT / tag(m, s, 5e15, 0)), val_bpb(run_dir(m, mac_top, s))
            rows[f"{m}_{s}"] = dict(kaggle=kaggle, mac=mac, relative=kaggle / mac - 1)
    within = all(abs(v["relative"]) <= CALIBRATION_LINE for v in rows.values())
    return dict(runs=rows, within=within, cross_hardware="comparable" if within else "descriptive only")


def trend(names, robots):
    """Primary trend of section 6, T4 only and seed 0 at both budgets: the rate-check frontier run at 5e15, then the 5e16 frontier run."""
    anchor, rates = json.load(open("prereg/ext1_anchor.json")), json.load(open("prereg/ext1_rates.json"))
    ext = Ext(names)
    runs = {"5e+15": {m: anchor[m]["run"] for m in ARMS}, "5e+16": {m: tag(m, frontier(m), BUDGET, 0) for m in ARMS}}

    def get(run, target):
        return [merge_seeds([ext.tables(run)[n]]) for n in (names if target == "held_in" else [target])]

    res = {base: {t: {b: stats.compare(pairs(get(r["hnet"], t), get(r[base], t))) for b, r in runs.items()} for t in ("held_in", *robots)} for base in GAPS}
    arms = {m: dict(size_5e15=anchor[m]["size"], size_5e16=frontier(m), rate_5e15=anchor[m]["rate"], rate_5e16=rates[f"{m}_{frontier(m)}"]) for m in ARMS}
    return dict(gaps=res, arms=arms, same_rate={m: v["rate_5e15"] == v["rate_5e16"] for m, v in arms.items()})


def gaps(get, robots, names, flops):
    """Secondary context: H-Net against each arm on every target, Mac frontier runs, then the 5e16 frontier run."""
    targets = ["held_in", *robots]
    res = {base: {t: {f"{BUDGET:.3g}": stats.compare(pairs(get("hnet", 0, t), get(base, 0, t)))} for t in targets} for base in GAPS}
    if not SCORES.exists():
        return dict(status="Mac scores pending", gaps=res)
    mac = study.Scores(names).get
    for base in GAPS:
        for t in targets:
            for bi, f in enumerate(flops):
                res[base][t][f"{f:.3g}"] = stats.compare(pairs(mac("hnet", bi, t), mac(base, bi, t)))
    return dict(status="done", gaps=res)


def forecasts(outcomes):
    """Brier score per forecaster (lower is better) on the probabilities stated before the freeze."""
    items = json.load(open("prereg/ext1_predictions.json"))
    rows, out = [], {}
    for p in items:
        rows.append(dict(key=p["key"], text=p["text"], claude=p["claude"], teo=p["teo"], outcome=outcomes[p["key"]]))
    for who in ("claude", "teo"):
        given = [r for r in rows if r[who] is not None]
        out[who] = sum((r[who] - r["outcome"]) ** 2 for r in given) / len(given) if given else None
    return dict(items=rows, brier=out)


def tables(r):
    p, c = r["primary"], r["calibration"]
    L = ["# RoboBytes-Ext1 results tables", "", "Generated by scripts/ext1_analysis.py; do not edit.", "",
         "## Pass/fail at 5e16", "", "| Test | Passed |", "| --- | --- |"] + [f"| {h} | {p[h]['passed']} |" for h in ("H1", "H2", "H2b")]
    x = p["compression"]
    L += ["", f"Compression: {x['realised']:.3f}; floor {x['floor']:.3f}; met: {x['ok']}.", "",
          "Frontier sizes: " + ", ".join(f"{k} {v}" for k, v in r["frontier"].items()) + ".", "", "## H1", "",
          f"H-Net {p['H1']['bpb_new']:.4f}, BPE {p['H1']['bpb_base']:.4f}, difference {p['H1']['diff']:.4f} "
          f"[{p['H1']['diff_ci'][0]:.4f}, {p['H1']['diff_ci'][1]:.4f}]."]
    for h in ("H2", "H2b"):
        x = p[h]
        L += ["", f"## {h}", "", f"g_ID {x['g_id']['g']:.4f} [{x['g_id']['g_ci'][0]:.4f}, {x['g_id']['g_ci'][1]:.4f}]; line {x['line']:.4f}", "",
              "| Robot | g_OOD | 95% CI | Clears |", "| --- | --- | --- | --- |"]
        L += [f"| {k} | {v['g']:.4f} | [{v['g_ci'][0]:.4f}, {v['g_ci'][1]:.4f}] | {v['clears']} |" for k, v in x["robots"].items()]
    L += ["", "## Calibration at 5e15", "", f"Cross-hardware trends: {c['cross_hardware']}.", "", "| Run | Kaggle fp16 | Mac fp32 | Relative |", "| --- | --- | --- | --- |"]
    L += [f"| {k} | {v['kaggle']:.4f} | {v['mac']:.4f} | {v['relative']:+.4f} |" for k, v in c["runs"].items()]
    t = r["trend"]
    L += ["", "## Primary trend: T4, 5e15 to 5e16, seed 0 (descriptive)", "", "| Arm | Size 5e15 | Size 5e16 | Rate 5e15 | Rate 5e16 | Same rate |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {m} | {v['size_5e15']} | {v['size_5e16']} | {v['rate_5e15']:g} | {v['rate_5e16']:g} | {t['same_rate'][m]} |" for m, v in t["arms"].items()]
    L += ["", "| Against | Target | g 5e15 | 95% CI | g 5e16 | 95% CI |", "| --- | --- | --- | --- | --- | --- |"]
    for base, ts in t["gaps"].items():
        for target, row in ts.items():
            a, b = row["5e+15"], row["5e+16"]
            L.append(f"| {base} | {target} | {a['g']:.4f} | [{a['g_ci'][0]:.4f}, {a['g_ci'][1]:.4f}] | {b['g']:.4f} | [{b['g_ci'][0]:.4f}, {b['g_ci'][1]:.4f}] |")
    L += ["", "## Secondary context: H-Net gap g across budgets, Mac ladder, then 5e16 with both seeds (descriptive)", "", f"Mac runs: {r['gaps']['status']}. Cross-hardware trends: {c['cross_hardware']}.", "",
          "| Against | Target | Budget | g | 95% CI |", "| --- | --- | --- | --- | --- |"]
    for base, ts in r["gaps"]["gaps"].items():
        for t, row in ts.items():
            for f in sorted(row, key=float):
                L.append(f"| {base} | {t} | {f} | {row[f]['g']:.4f} | [{row[f]['g_ci'][0]:.4f}, {row[f]['g_ci'][1]:.4f}] |")
    f = r["forecasts"]
    L += ["", "## Predictions scored", "", "| Prediction | Claude | Téo | Outcome |", "| --- | --- | --- | --- |"]
    L += [f"| {x['text']} | {x['claude']} | {x['teo']} | {x['outcome']} |" for x in f["items"]]
    L += ["", f"Brier score (lower is better): Claude {f['brier']['claude']}, Téo {f['brier']['teo']}."]
    return "\n".join(L) + "\n"


def main():
    ladder = json.load(open("runs/ladder.json"))
    R, flops = json.load(open("runs/bpe_summary.json"))["R_BPE"], ladder["budgets"]
    names, robots = training_names(), list(HELD_OUT)
    get = Ext(names).get
    p = primary(get, R, robots)
    c = calibration()
    front = {m: frontier(m) for m in ARMS}
    outcomes = dict(frontier_M=all(front[m] == "M" for m in METHODS), H1=p["H1"]["passed"], H2=p["H2"]["passed"], H2b=p["H2b"]["passed"],
                    floor=p["compression"]["ok"], calibration=c["within"])
    res = dict(primary=p, calibration=c, frontier=front, seeds={m: list(seeds_of(m)) for m in ARMS},
               compression={m: compression(get(m, 0, "held_in")) for m in ARMS}, trend=trend(names, robots), gaps=gaps(get, robots, names, flops), forecasts=forecasts(outcomes))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    (OUT / "tables.md").write_text(tables(res))
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
