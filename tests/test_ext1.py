"""The Ext1 analysis on synthetic scores only: no stream, no model, nothing from a real run."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
FEW = 30
ROBOTS = ("heldout_aloha", "heldout_fetch", "heldout_pusht")
NAMES = ("d1", "d2", "d3")
FLOPS = [1.58e14, 5e14, 1.58e15, 5e15]


def load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("ext1_analysis", SCRIPTS / "ext1_analysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tab(level, seed, compress=1.9):
    rng = np.random.default_rng(seed)
    by = np.full(FEW, 1000.0)
    return dict(eps=list(range(FEW)), bits=(level * by * (1 + 0.002 * rng.standard_normal(FEW))).tolist(), bytes=by.tolist(),
                symbols=(by * 1.1).tolist(), main=(by * 1.1 / compress).tolist())


def world(tmp_path, monkeypatch, levels=None, ood=None, fronts=None, mac_shift=0.0, compress=1.9, teo=None, mac_scores=False, anchor_levels=None):
    """levels: method -> held-in bpb; ood: method -> robot -> bpb. Everything is written under tmp_path."""
    monkeypatch.chdir(tmp_path)
    levels = levels or dict(hnet=1.50, fixed=1.48, byte=1.52, bpe=1.60, byteconv=1.49)
    ood = ood or {m: dict(zip(ROBOTS, (v * 2, v * 2, v * 2))) for m, v in levels.items()}
    fronts = fronts or {}
    for n in NAMES:
        Path("data/streams", n).mkdir(parents=True)
    for r in ROBOTS:
        Path("data/streams", r).mkdir(parents=True)
    Path("runs/ext1/scores").mkdir(parents=True)
    Path("prereg").mkdir()
    json.dump(dict(budgets=FLOPS), open("runs/ladder.json", "w"))
    json.dump(dict(R_BPE=1.549), open("runs/bpe_summary.json", "w"))
    json.dump([dict(key=k, text=k, claude=p, teo=teo) for k, p in
               dict(frontier_M=.6, H1=.75, H2=.3, H2b=.1, floor=.9, calibration=.75).items()], open("prereg/ext1_predictions.json", "w"))
    m = load()
    seed = iter(range(10000))
    for meth in m.ARMS:
        best = fronts.get(meth, m.sizes_of(meth)[0])
        for s in m.sizes_of(meth):
            d = Path("runs/ext1", m.tag(meth, s, 5e16, 0))
            d.mkdir()
            json.dump(dict(val_bpb=1.0 if s == best else 2.0), open(d / "val.json", "w"))
        for sd in m.seeds_of(meth):
            res = {n: tab(levels[meth], next(seed), compress if meth == "hnet" else 2.0) for n in NAMES}
            res |= {r: tab(ood[meth][r], next(seed), compress) for r in ROBOTS}
            json.dump(res, open(f"runs/ext1/scores/{m.tag(meth, best, 5e16, sd)}.json", "w"))
    anchor = {}
    json.dump({f"{meth}_{s}": 1e-3 if s == m.sizes_of(meth)[0] else 5e-4 for meth in m.ARMS for s in m.sizes_of(meth)}, open("prereg/ext1_rates.json", "w"))
    for meth in m.ARMS:
        s = m.sizes_of(meth)[0]
        anchor[meth] = dict(size=s, rate=1e-3, run=f"lr0.001_{m.tag(meth, s, 5e15, 0)}", sha256="0")
        at = (anchor_levels or levels)[meth]
        json.dump({n: tab(at, next(seed), 2.0) for n in (*NAMES, *ROBOTS)}, open(f"runs/ext1/scores/{anchor[meth]['run']}.json", "w"))
    json.dump(anchor, open("prereg/ext1_anchor.json", "w"))
    for meth in m.CALIBRATION:
        for s in "SM":
            mac, kag = Path(f"runs/ladder/{meth}_b4_{s}_s0"), Path("runs/ext1", m.tag(meth, s, 5e15, 0))
            mac.mkdir(parents=True)
            kag.mkdir()
            json.dump(dict(val_bpb=2.0), open(mac / "val.json", "w"))
            json.dump(dict(val_bpb=2.0 * (1 + (mac_shift if (meth, s) == ("fixed", "M") else 0))), open(kag / "val.json", "w"))
    if mac_scores:
        Path("runs/scores").mkdir()
        for meth in ("hnet", "fixed", "byte", "bpe", "byteconv"):
            for bi in range(4):
                for s in "SM":
                    d = Path(f"runs/ladder/{meth}_b{bi + 1}_{s}_s0")
                    d.mkdir(parents=True, exist_ok=True)
                    json.dump(dict(val_bpb=1.0 if s == "S" else 2.0), open(d / "val.json", "w"))
                for sd in ((0, 1, 2) if meth in ("hnet", "bpe") and bi == 3 else (0,)):
                    res = {n: tab(levels[meth] * (4 - bi), next(seed)) for n in (*NAMES, *ROBOTS)}
                    json.dump(res, open(f"runs/scores/{meth}_b{bi + 1}_S_s{sd}.json", "w"))
    return m


def test_h1_passes_and_g_is_exact(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch).main()
    p = r["primary"]
    assert p["H1"]["passed"] is True or p["H1"]["passed"] == True  # noqa: E712
    assert p["H1"]["g"] == pytest.approx((1.60 - 1.50) / 1.60, abs=2e-3)


def test_h2_needs_two_robots_over_the_line(tmp_path, monkeypatch):
    lv = dict(hnet=1.5, fixed=1.5, byte=1.5, bpe=1.6, byteconv=1.5)  # g_ID 0.0625: line 0.094
    good, under = {m: 3.0 for m in lv}, {m: 3.0 for m in lv}
    good["bpe"], under["bpe"] = 3.0 / (1 - 0.15), 3.0 / (1 - 0.05)
    ood = {m: dict(zip(ROBOTS, (good[m], good[m], under[m]))) for m in lv}
    p = world(tmp_path, monkeypatch, levels=lv, ood=ood).main()["primary"]["H2"]
    assert p["line"] == pytest.approx(1.5 * 0.0625, abs=2e-3)
    assert [p["robots"][x]["clears"] for x in ROBOTS] == [True, True, False] and p["passed"] is True


def test_h2_fails_with_one_robot(tmp_path, monkeypatch):
    lv = dict(hnet=1.5, fixed=1.5, byte=1.5, bpe=1.6, byteconv=1.5)
    good, under = {m: 3.0 for m in lv}, {m: 3.0 for m in lv}
    good["bpe"], under["bpe"] = 3.0 / (1 - 0.15), 3.0 / (1 - 0.05)
    ood = {m: dict(zip(ROBOTS, (good[m], under[m], under[m]))) for m in lv}
    assert world(tmp_path, monkeypatch, levels=lv, ood=ood).main()["primary"]["H2"]["passed"] is False


def test_h2b_uses_the_floor_when_g_id_is_small(tmp_path, monkeypatch):
    lv = dict(hnet=1.5, fixed=1.5, byte=1.5, bpe=1.6, byteconv=1.5)
    ood = {m: dict(zip(ROBOTS, (3.0, 3.0, 3.0))) for m in lv}
    ood["fixed"] = dict(zip(ROBOTS, (3.0 / 0.97, 3.0 / 0.97, 3.0 / 0.99)))
    p = world(tmp_path, monkeypatch, levels=lv, ood=ood).main()["primary"]["H2b"]
    assert p["line"] == pytest.approx(0.02, abs=1e-3)
    assert p["passed"] is True and p["robots"][ROBOTS[2]]["clears"] is False


def test_compression_floor_fails_h1_and_h2_not_h2b(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch, compress=1.2).main()
    p = r["primary"]
    assert p["compression"]["ok"] is False and p["H1"]["passed_before_floor"] and p["H1"]["passed"] is False
    assert "passed_before_floor" not in p["H2b"]
    assert r["forecasts"]["items"][4]["outcome"] is False


@pytest.mark.parametrize("shift,within", [(0.009, True), (-0.009, True), (0.011, False), (-0.011, False)])
def test_calibration_line_is_one_percent_on_any_run(tmp_path, monkeypatch, shift, within):
    c = world(tmp_path, monkeypatch, mac_shift=shift).main()["calibration"]
    assert c["within"] is within and c["cross_hardware"] == ("comparable" if within else "descriptive only")
    assert len(c["runs"]) == 4


def test_frontier_rule_and_byteconv_sizes(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch, fronts=dict(hnet="L", byteconv="M")).main()
    assert r["frontier"] == dict(hnet="L", fixed="M", byte="M", bpe="M", byteconv="M")
    assert r["seeds"]["byteconv"] == [0] and r["seeds"]["hnet"] == [0, 1]
    assert r["forecasts"]["items"][0]["outcome"] is False


def test_all_m_frontier_outcome_ignores_byteconv(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch, fronts=dict(byteconv="S")).main()
    assert r["forecasts"]["items"][0]["outcome"] is True


def test_seeds_are_averaged_before_statistics(tmp_path, monkeypatch):
    m = world(tmp_path, monkeypatch)
    hi = json.load(open(f"runs/ext1/scores/{m.tag('hnet', 'M', 5e16, 1)}.json"))
    for n in NAMES:
        hi[n]["bits"] = (np.array(hi[n]["bits"]) * 2).tolist()
    json.dump(hi, open(f"runs/ext1/scores/{m.tag('hnet', 'M', 5e16, 1)}.json", "w"))
    assert m.main()["primary"]["H1"]["bpb_new"] == pytest.approx(1.5 * 1.5, abs=5e-3)


def test_unpaired_episodes_are_refused(tmp_path, monkeypatch):
    m = world(tmp_path, monkeypatch)
    x = json.load(open(f"runs/ext1/scores/{m.tag('bpe', 'M', 5e16, 0)}.json"))
    x[NAMES[0]]["eps"][0] = 999
    json.dump(x, open(f"runs/ext1/scores/{m.tag('bpe', 'M', 5e16, 0)}.json", "w"))
    with pytest.raises(AssertionError):
        m.main()


def test_brier_scores_both_forecasters(tmp_path, monkeypatch):
    f = world(tmp_path, monkeypatch, teo=0.5).main()["forecasts"]
    assert f["brier"]["teo"] == pytest.approx(0.25)
    o = [x["outcome"] for x in f["items"]]
    assert f["brier"]["claude"] == pytest.approx(np.mean([(x["claude"] - y) ** 2 for x, y in zip(f["items"], o)]))


def test_blank_teo_column_stays_blank(tmp_path, monkeypatch):
    f = world(tmp_path, monkeypatch).main()["forecasts"]
    assert f["brier"]["teo"] is None


def test_mac_gaps_pending_then_joined(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch).main()
    assert r["gaps"]["status"] == "Mac scores pending"
    assert set(r["gaps"]["gaps"]["fixed"]["held_in"]) == {"5e+16"}


def test_mac_gaps_cover_every_budget(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch, mac_scores=True).main()
    row = r["gaps"]["gaps"]["byte"]["heldout_fetch"]
    assert r["gaps"]["status"] == "done" and len(row) == 5


def test_primary_trend_is_anchor_then_seed_0(tmp_path, monkeypatch):
    levels = dict(hnet=1.50, fixed=1.48, byte=1.52, bpe=1.60, byteconv=1.49)
    r = world(tmp_path, monkeypatch, levels=levels, anchor_levels=dict(levels, hnet=1.80, bpe=2.00)).main()
    row = r["trend"]["gaps"]["bpe"]["held_in"]
    assert row["5e+15"]["g"] == pytest.approx((2.00 - 1.80) / 2.00, abs=0.003)
    assert row["5e+16"]["g"] == pytest.approx((1.60 - 1.50) / 1.60, abs=0.003)


def test_trend_uses_one_seed_at_5e16(tmp_path, monkeypatch):
    m = world(tmp_path, monkeypatch)
    p = Path("runs/ext1/scores", m.tag("hnet", "M", 5e16, 1) + ".json")
    x = json.load(open(p))
    x = {k: tab(9.0, 1) for k in x}
    json.dump(x, open(p, "w"))
    r = m.main()
    assert r["trend"]["gaps"]["bpe"]["held_in"]["5e+16"]["bpb_new"] == pytest.approx(1.50, abs=0.01)


def test_moved_frontier_is_flagged_as_a_different_rate(tmp_path, monkeypatch):
    r = world(tmp_path, monkeypatch, fronts=dict(hnet="L")).main()
    assert r["trend"]["same_rate"]["hnet"] is False and r["trend"]["same_rate"]["fixed"] is True
    assert r["trend"]["arms"]["hnet"]["size_5e16"] == "L"


def test_tables_are_written(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch).main()
    t = Path("runs/ext1/analysis/tables.md").read_text()
    assert "Pass/fail at 5e16" in t and "Calibration at 5e15" in t and "Predictions scored" in t and "Primary trend" in t and "Secondary context" in t
