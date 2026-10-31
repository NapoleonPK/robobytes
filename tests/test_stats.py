import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from robobytes import stats
from robobytes.model import BOS, SEP, ByteHNet
from robobytes.results import compression, merge_seeds, pairs, table


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def tab(bits, seed=0, n=20, by=100.0):
    rng = np.random.default_rng(seed)
    b = np.full(n, by)
    return dict(eps=np.arange(n), bits=bits * b * (1 + 0.05 * rng.standard_normal(n)), bytes=b, symbols=b * 1.1, main=b * 0.6)


def test_pooled_weights_datasets_equally():
    g = [(np.array([10.0]), np.array([10.0])), (np.array([3000.0]), np.array([1000.0]))]
    assert stats.pooled_bpb(g) == pytest.approx(2.0)


def test_bootstrap_paired_and_reproducible():
    b = np.random.default_rng(0).random(30)
    n = np.ones(30)
    c = stats.compare([(b, b, n)], resamples=500)
    assert c["diff"] == 0 and c["diff_ci"] == [0, 0]
    assert stats.compare([(b, 2 * b, n)], resamples=500) == stats.compare([(b, 2 * b, n)], resamples=500)


def test_g_numerator_and_denominator_resampled_together():
    b = np.random.default_rng(1).random(30) + 1
    c = stats.compare([(0.8 * b, b, np.ones(30))], resamples=500)
    assert c["g"] == pytest.approx(0.2) and c["g_ci"] == pytest.approx([0.2, 0.2])


def test_resampling_stays_within_each_dataset():
    a = (np.ones(10), np.ones(10), np.ones(10))
    b = (3 * np.ones(4), 3 * np.ones(4), np.ones(4))
    x, y = stats.paired_bootstrap([a, b], resamples=50)
    assert np.allclose(x, 2) and np.allclose(y, 2)


def test_ci_covers_a_true_gap():
    rng = np.random.default_rng(2)
    a, b = rng.normal(3, 0.5, 200), rng.normal(4, 0.5, 200)
    c = stats.compare([(a, b, np.ones(200))], resamples=2000)
    assert c["diff_ci"][0] < -1 < c["diff_ci"][1] and stats.better(c)


@pytest.mark.parametrize("g,interval,g_id,want", [
    (0.03, [0.01, 0.05], 0.0, True),
    (0.03, [0.01, 0.05], 0.02, True),
    (0.03, [0.01, 0.05], 0.03, False),
    (0.019, [0.01, 0.03], 0.0, False),  # the 0.02 floor
    (0.03, [0.0, 0.05], 0.0, False),  # CI touches 0
    (0.03, [-0.01, 0.05], 0.0, False),
    (0.10, [0.05, 0.15], -0.10, False),  # |g_ID|: being worse in distribution raises the bar
    (0.16, [0.05, 0.25], -0.10, True),
    (-0.03, [-0.05, -0.01], 0.0, False),
])
def test_transfer_rule(g, interval, g_id, want):
    assert stats.transfer_pass(g, interval, g_id) is want


def test_two_of_three_and_floor():
    assert stats.at_least([True, True, False]) and not stats.at_least([True, False, False])
    assert stats.compression_ok(1.24, 1.549) and not stats.compression_ok(1.23, 1.549)


def test_policy_bootstrap():
    a = np.random.default_rng(3).random((3, 100)) < 0.5
    assert stats.policy_bootstrap(a, a, resamples=200) == dict(diff=0.0, diff_ci=[0.0, 0.0])
    c = stats.policy_bootstrap(np.ones((3, 100)), np.zeros((3, 100)), resamples=200)
    assert c["diff"] == 1 and stats.excludes_zero(c["diff_ci"])


def test_coverage_summary():
    c = stats.coverage_summary(np.tile([0.89, 0.90, 0.91, 0.70], (3, 25)), resamples=200)
    assert c["line"] == 0.90 and c["share"] == 0.5 and abs(c["mean"] - 0.85) < 1e-9
    assert c["mean_ci"][0] < c["mean"] < c["mean_ci"][1] and c["share_ci"][0] < 0.5 < c["share_ci"][1]
    flat = stats.coverage_summary(np.full((3, 100), 0.93), resamples=50)
    assert flat["share"] == 1 and flat["mean_ci"] == pytest.approx([0.93, 0.93])
    assert stats.coverage_summary(np.full((3, 10), 0.8), resamples=50) == stats.coverage_summary(np.full((3, 10), 0.8), resamples=50)


def test_pusht_coverage_stays_outside_h4(tmp_path, monkeypatch):
    a = load("analysis")
    monkeypatch.setattr(a, "POLICY", tmp_path)
    assert a.pusht_coverage()["status"] == "pending"
    rng = np.random.default_rng(0)
    rows = {}
    for task, rates in (("pusht", dict(hnet=.8, bpe=.2, byte=.5, mlp=.1)), ("fetch", dict(hnet=.6, bpe=.3, byte=.2, mlp=.1))):
        (tmp_path / task).mkdir()
        for m, r in rates.items():
            for s in (0, 1, 2):
                rows[task, m, s] = dict(env_seeds=list(range(100)), success=(rng.random(100) < r).tolist())
    write = lambda: [(tmp_path / t / f"{m}_s{s}.json").write_text(json.dumps(v)) for (t, m, s), v in rows.items()]
    write()
    before = a.h4()
    for (t, m, s), v in rows.items():
        if t == "pusht":
            v["peak_coverage"] = rng.uniform(.5, 1, 100).tolist()
    write()
    x = a.pusht_coverage()
    assert x["status"] == "done" and set(x["methods"]) == set(a.POLICY_METHODS) and a.h4() == before


def test_uninformative():
    assert stats.uninformative([0.01, 0.0, 0.04, 0.02]) and stats.uninformative([0.96, 1.0, 0.99, 0.97])
    assert not stats.uninformative([0.0, 1.0, 0.0, 1.0]) and not stats.uninformative([0.01, 0.05])


def test_flop_multipliers():
    f = [1.0, 10.0, 100.0]
    assert stats.flop_multipliers(f, [4, 2, 1], [4, 2, 1]) == pytest.approx([1, 1, 1])
    assert stats.flop_multipliers(f, [2], [4, 1, 1])[0] == pytest.approx(10 ** (np.log(2) / np.log(4)))
    m = stats.flop_multipliers(f, [4, 2, 0.5], [3.5, 3, 1])  # ref's 4 is above other's start, 0.5 is never reached
    assert m[0] is None and m[2] is None
    assert m[1] == pytest.approx(stats.flop_multipliers(f, [2], [3.5, 3, 1])[0] / 10)  # relative to ref's 10 FLOPs


def test_merge_seeds_averages_bits_per_episode():
    a, b = dict(eps=[1, 2], bits=[2, 4], bytes=[10, 10], symbols=[11, 11], main=[5, 6]), dict(
        eps=[1, 2], bits=[4, 8], bytes=[10, 10], symbols=[11, 11], main=[7, 8])
    m = merge_seeds([a, b])
    assert m["bits"].tolist() == [3, 6] and m["main"].tolist() == [6, 7]
    with pytest.raises(AssertionError):
        merge_seeds([a, dict(a, eps=[1, 3])])


def test_pairs_refuses_unmatched_episodes():
    a = tab(2.0)
    with pytest.raises(AssertionError):
        pairs([a], [dict(a, eps=a["eps"] + 1)])


def test_table_counts_positions_per_method():
    class S:
        length, W = np.array([3, 4]), 2
    r = {0: [9.0, 6, 9, 0], 1: [9.0, 8, 12, 0]}
    assert table("byte", S, r)["main"] == [9, 12]
    assert table("hnet", S, {0: [9.0, 6, 9, 4]})["main"] == [4]
    assert table("bpe", S, {0: [9.0, 6, 7, 0]}) == dict(eps=[0], bits=[9.0], bytes=[6], symbols=[9], main=[7])
    assert compression([dict(symbols=np.array([9.0]), main=np.array([3.0]))]) == 3


def get_from(values):
    """Synthetic arms: bpb per (arm, budget, target) with shared noise, so comparisons are tight."""
    def get(arm, bi, target):
        return [tab(values[(arm, target)] * (1 - 0.05 * bi), seed=7)]
    return get


def scenario(hnet_oo, base=3.0):
    targets = ["held_in", "r1", "r2", "r3"]
    v = {("bpe", t): base for t in targets} | {("fixed", t): base for t in targets}
    v |= {("hnet", "held_in"): 0.95 * base} | {("hnet", t): hnet_oo * base for t in targets[1:]}
    return get_from(v)


def test_primary_tests_end_to_end():
    a = load("analysis")
    r = a.primary(scenario(0.8), 1.549, ["r1", "r2", "r3"])
    assert r["H1"]["passed"] and r["H2"]["passed"] and r["H2b"]["passed"] and r["compression"]["ok"]
    assert r["H2"]["g_id"]["g"] == pytest.approx(0.05, abs=1e-6) and r["H2"]["line"] == pytest.approx(0.075, abs=1e-6)
    r = a.primary(scenario(0.94), 1.549, ["r1", "r2", "r3"])  # better than BPE, not by 1.5 x g_ID
    assert r["H1"]["passed"] and not r["H2"]["passed"]


def test_compression_floor_fails_h1_h2_only():
    a = load("analysis")
    r = a.primary(scenario(0.8), 3.0, ["r1", "r2", "r3"])  # floor 2.4, realised 1.83
    assert not r["compression"]["ok"] and not r["H1"]["passed"] and not r["H2"]["passed"]
    assert r["H1"]["passed_before_floor"] and r["H2"]["passed_before_floor"] and r["H2b"]["passed"]


def test_few_shot_pool_is_first_episodes(tmp_path, monkeypatch):
    fs = load("few_shot")

    class S:
        W, name = 2, "x"
        length = np.array([50, 60, 70, 80])
        start = np.array([0, 50, 110, 180])
        rows = np.arange(520).reshape(260, 2) % 256

        def episodes(self, split):
            return np.array([0, 1, 2, 3])

    monkeypatch.setattr(fs, "Stream", lambda name: S())
    src = fs.pool_source("x", "byte", 2, 16)
    assert src.total == 1 + 50 * 3 + 1 + 60 * 3
    w = src.window(0, 4)
    assert w.tolist() == [BOS, 0, 1, SEP]
    assert src.window(151, 3).tolist() == [BOS, 100, 101]
    assert fs.seeds("hnet") == (0, 1, 2) and fs.seeds("fixed") == (0,)


def test_history_matches_the_training_stream_and_truncates():
    p = load("policy")
    enc = p.Encoder("byte", ctx=8)
    h = p.History(enc)
    h.add_obs(np.array([1, 2]))
    assert h.view() == [BOS, 1, 2]
    h.add_act(np.array([3]))
    assert h.ids == [BOS, 1, 2, 3, SEP]
    h.add_obs(np.array([4, 5, 6, 7, 8]))
    assert h.view() == [2, 3, SEP, 4, 5, 6, 7, 8]


def test_hidden_reads_the_last_position_and_ignores_padding():
    p = load("policy")
    torch.manual_seed(0)
    m = ByteHNet(32, 2, 2, d_outer=16, pos="rope", rope_bytes=2.0)
    enc = p.Encoder("hnet", m, dev="cpu")
    a = list(range(20))
    one = enc.hidden([a])
    both = enc.hidden([a, list(range(40))])
    assert one.shape == (1, 16) and torch.allclose(one[0], both[0], atol=1e-4)


def test_h3_and_h4_read_their_result_files(tmp_path, monkeypatch):
    a = load("analysis")
    monkeypatch.setattr(a, "FEW_SHOT", tmp_path / "fs")
    monkeypatch.setattr(a, "POLICY", tmp_path / "pol")
    assert a.h3(["r1", "r2", "r3"])["status"] == "pending" and a.h4()["status"] == "pending"
    for r, gain in (("r1", 0.7), ("r2", 0.7), ("r3", 1.3)):
        for m, f in (("hnet", gain), ("bpe", 1.0)):
            for s in (0, 1, 2):
                d = tmp_path / "fs" / r / f"{m}_s{s}_k50"
                d.mkdir(parents=True)
                t = {k: v.tolist() for k, v in tab(3.0 * f, seed=9).items()}
                (d / "result.json").write_text(json.dumps(dict(test=t)))
    x = a.h3(["r1", "r2", "r3"])
    assert x["passed"] and [v["clears"] for v in x["robots"].values()] == [True, True, False]
    rng = np.random.default_rng(0)
    for task, rates in (("pusht", dict(hnet=.8, bpe=.2, byte=.5, mlp=.1)), ("fetch", dict(hnet=0., bpe=.01, byte=.0, mlp=.0))):
        (tmp_path / "pol" / task).mkdir(parents=True)
        for m, r in rates.items():
            for s in (0, 1, 2):
                (tmp_path / "pol" / task / f"{m}_s{s}.json").write_text(json.dumps(dict(
                    env_seeds=list(range(100)), success=(rng.random(100) < r).tolist())))
    y = a.h4()
    assert y["passed"] and y["tasks"]["pusht"]["clears"] and not y["tasks"]["fetch"]["informative"]


def test_report_runs_end_to_end():
    a = load("analysis")
    robots = ["r1", "r2", "r3"]
    v = {(arm, t): 3.0 for arm in a.ARMS for t in ["held_in", *robots]}
    get = get_from(v)
    flops = [1e14, 3e14, 1e15, 3e15]
    front = a.frontiers(get, flops, robots)
    res = dict(flops=flops, primary=a.primary(get, 1.549, robots), H3=dict(status="pending"), H4=dict(status="pending"), pusht_coverage=a.pusht_coverage(),
               frontiers=front, multipliers=a.multipliers(front, flops, ["held_in", *robots]),
               secondary=a.secondary(get, robots))
    json.dumps(res)
    assert "2" not in res["secondary"]["n4_vs_bpe"] and set(res["secondary"]["n4_vs_bpe"]["hnet_refractory_N4"]) == {"2", "3"}
    assert set(res["secondary"]["no_ema"]) == {"fixed", "hnet"} and len(res["secondary"]["n4_router"]) == 4
    md = a.tables(res)
    assert "| H2b |" in md and "| hnet_refractory_N4 | r3 | - | - |" in md


def test_report_notes_that_no_demonstration_reaches_the_flag():
    a = load("analysis")
    d = dict(threshold=.95, dataset_success_frames=0, dataset_episodes=206, pool_episodes=50, final_coverage_max=.935)
    p = dict(H1=dict(passed=True, budgets={}), H2=dict(passed=False, g_id=dict(g=0, g_ci=[0, 0]), line=0, robots={}),
             H2b=dict(passed=False, g_id=dict(g=0, g_ci=[0, 0]), line=0, robots={}),
             compression=dict(realised=2, floor=1, ok=True))
    t = a.tables(dict(primary=p, H3=dict(status="pending"), H4=dict(status="pending"), flops=[1.0], frontiers={},
                      pusht_coverage=dict(status="pending", line=.9, demonstrations=d)))
    assert "No PushT demonstration reaches the environment's 0.95 success flag" in t and "exploratory, outside H4" in t
