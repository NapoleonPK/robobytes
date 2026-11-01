"""Pre-freeze learning-rate check: picks the rate per method and size by held-in validation among the three grid rates and,
where the top of the grid won, one rate above it. Writes prereg/ext1_lr_check.json (every run), prereg/ext1_rates.json (the rates
of the 5e16 runs) and prereg/ext1_anchor.json (the section 6 anchor: each arm's better size at its chosen rate, with the checkpoint's
SHA-256). usage: python scripts/make_ext1_rates.py CHECK_OUTPUT EDGE_OUTPUT"""
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent
check, edge = Path(sys.argv[1]), Path(sys.argv[2])
GRID, EDGE = (3e-4, 7.5e-4, 1.5e-3), 3e-3
ARMS = ("hnet", "fixed", "byte", "bpe", "byteconv")
hashes = json.load(open(edge / "ext1/checkpoints.json"))
rows = []
for src, stage in ((check, "lr"), (edge, "lr_edge")):
    status = {j["job"]: j for j in json.load(open(src / "ext1/status.json"))[stage]}
    assert len(status) == {"lr": 30, "lr_edge": 4}[stage]
    for p in sorted((src / "ext1").glob("lr*/val.json")):
        m = re.fullmatch(r"lr([\d.e-]+)_(\w+?)_5e15_(\w)_s0", p.parent.name)
        log = json.load(open(p.parent / "train.json"))["log"][-1]
        v = json.load(open(p))
        ratios = [x["compression"] for x in v["per_dataset"].values() if x.get("compression")]
        assert status[p.parent.name]["exit"] == 0 and p.parent.name in hashes
        rows.append(dict(method=m[2], size=m[3], rate=float(m[1]), run=p.parent.name, val_bpb=v["val_bpb"], train_seconds=log["seconds"],
                         compression=sum(ratios) / len(ratios) if ratios else None))
assert len(rows) == 34 == len(hashes)
best, anchor = {}, {}
for k in sorted({(r["method"], r["size"]) for r in rows}):
    c = [r for r in rows if (r["method"], r["size"]) == k]
    grid = min((r for r in c if r["rate"] in GRID), key=lambda r: r["val_bpb"])
    assert len(c) == (4 if grid["rate"] == GRID[-1] else 3) and grid["rate"] != GRID[0], f"{k}: edge rule"
    best[k] = min(c, key=lambda r: r["val_bpb"])
for m in ARMS:
    b = min((v for k, v in best.items() if k[0] == m), key=lambda r: r["val_bpb"])
    anchor[m] = dict(size=b["size"], rate=b["rate"], run=b["run"], sha256=hashes[b["run"]]["sha256"], val_bpb=b["val_bpb"])
json.dump(rows, open(here / "prereg/ext1_lr_check.json", "w"), indent=1)
json.dump({f"{k[0]}_{k[1]}": v["rate"] for k, v in best.items()}, open(here / "prereg/ext1_rates.json", "w"), indent=1)
json.dump(anchor, open(here / "prereg/ext1_anchor.json", "w"), indent=1)
print({k: (v["size"], v["rate"]) for k, v in anchor.items()})
