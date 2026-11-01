"""Writes PREREGISTRATION_ext1.md: every number comes from the smoke-test reports, the frozen ladder plan and the Mac
validation files. usage: python scripts/make_prereg_ext1.py MAC_REPO OUT QUOTA_HOURS_USED"""
import hashlib
import json
import sys
from pathlib import Path

mac, out = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).resolve().parent.parent
s1, s5 = json.load(open(here / "prereg/ext1_smoke1.json")), json.load(open(here / "prereg/ext1_smoke5.json"))
predictions = json.load(open(here / "prereg/ext1_predictions.json"))
check, chosen = json.load(open(here / "prereg/ext1_lr_check.json")), json.load(open(here / "prereg/ext1_rates.json"))
anchor = json.load(open(here / "prereg/ext1_anchor.json"))
speed = {**s1["throughput"], **s5["throughput_byteconv"]}
ladder = json.load(open(mac / "runs/ladder.json"))
SYMBOLS, WINDOW = ladder["train_symbols"], 16 * 1024
BUDGET, MAC_TOP, CAP, LIMIT, MARGIN, USED = 5e16, 5e15, 4, 26, 0.15, float(sys.argv[3])
MAC_RATE = 1.5e-3
METHODS = ["hnet", "fixed", "byte", "bpe"]
LABEL = dict(hnet="H-Net", fixed="Fixed", byte="Byte", bpe="BPE", byteconv="Byte + conv")
hours = {k: v["steps_total"] / v["steps_per_s"] / 3600 for k, v in speed.items()}
passes = {k: v["steps_total"] * WINDOW / SYMBOLS for k, v in speed.items()}

rows = []
for m in METHODS + ["byteconv"]:
    c = [f"{speed[f'{m}_{s}']['flops_per_s'] / 1e12:.1f} / {hours[f'{m}_{s}']:.2f} h / {passes[f'{m}_{s}']:.2f}" for s in "SML"]
    rows.append(f"| {LABEL[m]} | {' | '.join(c)} |")
speed_table = "\n".join(rows)

calib = sum(hours[f"{m}_{s}"] for m in ("hnet", "fixed") for s in "SM") / 10
main = sum(hours[f"{m}_{s}"] for m in METHODS for s in "ML")
seed1 = sum(hours[f"{m}_M"] for m in METHODS)  # M costs more than L for every method, so this is the upper case
conv_s, conv_m = hours["byteconv_S"], hours["byteconv_M"]
val = 18 * s5["full_validation"]["seconds"] / 3600
SCORED = 8 + 1 + len(anchor)  # the 5e16 frontier runs, then the anchor
TEST_H = 1.0 * SCORED / 9  # the first estimate was 1.0 h for 9 runs
blocks = [("Calibration: H-Net, Fixed at 5e15, sizes S and M", 4, calib), ("Main: H-Net, Fixed, Byte, BPE at 5e16, sizes M and L", 8, main),
          ("Seed 1: the four methods at frontier size (upper case: all M)", 4, seed1), ("Byte + conv S at 5e16", 1, conv_s), ("Byte + conv M at 5e16", 1, conv_m)]
plan = "\n".join(f"| {b} | {n} | {h:.2f} | {h / 2:.2f} |" for b, n, h in blocks)
gpu = sum(h for *_, h in blocks) / 2
total = gpu + val / 2 + TEST_H
with_margin = total * (1 + MARGIN) + USED
assert with_margin <= LIMIT

bud = ladder["budgets"]
known = []
for m in METHODS + ["byteconv"]:
    cells = []
    for i in range(len(bud)):
        v = {}
        for s in "SM":
            p = mac / f"runs/ladder/{m}_b{i + 1}_{s}_s0/val.json"
            v[s] = json.load(open(p))["val_bpb"] if p.exists() else None
        cells.append(" / ".join("–" if v[s] is None else f"{v[s]:.3f}" for s in "SM"))
    known.append(f"| {LABEL[m]} | {' | '.join(cells)} |")
known = "\n".join(known)
heads = " | ".join(f"{b:.3g}" for b in bud)
fv = s5["full_validation"]
agree = max(abs(v["rel_fp32_vs_mac"]) for k in ("agreement_hnet_S", "agreement_hnet_M") for v in s1[k].values())
agree16 = max(abs(v["rel_fp16_vs_fp32"]) for k in ("agreement_hnet_S", "agreement_hnet_M") for v in s1[k].values())
two = s1["two_jobs_two_gpus"]

pct = lambda x: "" if x is None else f"{x:.0%}"
prediction_table = "\n".join(["| Prediction | Claude | Téo | Claude's reasoning |", "| --- | --- | --- | --- |"]
                             + [f"| {p['text']} | {pct(p['claude'])} | {pct(p['teo'])} | {p['why']} |" for p in predictions])
revised = " ".join(f"{p['text']}: Claude revised this from {pct(p['claude_before'])} to {pct(p['claude'])} after seeing the check." for p in predictions if "claude_before" in p)

CHECK_RATES, GRID = sorted({r["rate"] for r in check}), sorted({r["rate"] for r in check})[:3]
pairs = [(m, s) for m in METHODS for s in "ML"] + [("byteconv", "S"), ("byteconv", "M")]
bpb = {(r["method"], r["size"], r["rate"]): r["val_bpb"] for r in check}
rate_rows = []
for m, s in pairs:
    cells = ["–" if (m, s, r) not in bpb else f"**{bpb[m, s, r]:.3f}**" if r == chosen[f"{m}_{s}"] else f"{bpb[m, s, r]:.3f}" for r in CHECK_RATES]
    rate_rows.append(f"| {LABEL[m]} {s} | {' | '.join(cells)} | {chosen[f'{m}_{s}']:g} |")
rate_table = "\n".join(rate_rows)
EDGE = CHECK_RATES[-1]
edged = [(m, s) for m, s in pairs if (m, s, EDGE) in bpb]
assert all(min(GRID, key=lambda r: bpb[m, s, r]) == GRID[-1] for m, s in edged)
assert all(chosen[f"{m}_{s}"] != GRID[-1] or (m, s) in edged for m, s in pairs)
edge_won = [f"{LABEL[m]} {s}" for m, s in edged if chosen[f"{m}_{s}"] == EDGE]
margin = lambda m, s: sorted(v for (a, b, _), v in bpb.items() if (a, b) == (m, s))[1] / min(v for (a, b, _), v in bpb.items() if (a, b) == (m, s)) - 1
close = [f"{LABEL[m]} {s}" for m, s in pairs if margin(m, s) < 0.015]
close_range = sorted(margin(m, s) for m, s in pairs if margin(m, s) < 0.015)
repeat = {m: (bpb[m, "M", MAC_RATE], json.load(open(mac / f"runs/ladder/{m}_b4_M_s0/val.json"))["val_bpb"]) for m in ("hnet", "fixed")}
rel = {m: abs(a - b) / b for m, (a, b) in repeat.items()}
drift = sum(r["train_seconds"] / (hours[f"{r['method']}_{r['size']}"] * 360) for r in check) / len(check)
slow = max(r["train_seconds"] / (hours[f"{r['method']}_{r['size']}"] * 360) for r in check)
gain = 1 - bpb["hnet", "M", chosen["hnet_M"]] / bpb["hnet", "M", MAC_RATE]
assert gain == max(1 - bpb[m, s, chosen[f"{m}_{s}"]] / bpb[m, s, MAC_RATE] for m, s in pairs)
anchor_table = "\n".join(f"| {LABEL[m]} | {a['size']} | {a['rate']:g} | {a['val_bpb']:.3f} |" for m, a in anchor.items())

digests = {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in ("scripts/ext1_train.py", "scripts/ext1_kernel.py", "scripts/ext1_analysis.py", "prereg/ext1_rates.json", "prereg/ext1_anchor.json")}
fixed_files = "\n".join(f"- `{k}`: SHA-256 `{v}`" for k, v in digests.items())

text = f"""# RoboBytes-Ext1: pre-registration

**Status: frozen 1 Nov 2026.** The frozen text is this file at the commit tagged `prereg-ext1` in the public `robobytes` repo, and its SHA-256 is published in the GitHub release `prereg-ext1`. The freeze comes before the main study's held-out (zero-shot) evaluation is run. Later changes are dated amendments.

Independent study, as the main pre-registration (`PREREGISTRATION.md`, tag `prereg`). Not affiliated with Cartesia, the H-Net authors, the FAST authors, LeRobot / Hugging Face or Kaggle. Anything not stated here is as in `PREREGISTRATION.md`.

## 1. Question

Do the main study's results hold one order of magnitude further up, at **5e16 FLOPs** (10 × its top budget of 5e15)?

## 2. What stays and what changes

Unchanged from the main study: architectures, serialisation, N = 2, wrapper, BPE tokenizer, positions, batching and optimiser, the FLOP formula charged at nominal N, the 3 held-out robots, the episode lists (`prereg/splits.json`), the learning-rate schedule, and the statistics of `robobytes/stats.py`. Learning rates change, by the search below.

| Item | Main study | Ext1 |
| --- | --- | --- |
| Hardware | Apple M6, MPS | Kaggle T4 x2 |
| Precision | fp32 | training fp16 autocast with gradient scaling; scoring fp32 |
| Jobs | one at a time | two at a time, one per GPU |
| Checkpoints | every 10 minutes | every 30 minutes; each session stops itself by 8 hours and exits cleanly |
| Micro-batch | 8 | 16, halved on out-of-memory (logged) |
| Peak learning rate | {MAC_RATE:g} for every method | chosen per method and size, below |
| Budget | 1.58e14 to 5e15 | 5e16 |
| Seeds at the tests | H-Net, BPE 3; Fixed 1 | H-Net, Fixed, Byte, BPE 2 (seeds 0, 1) |
| Tests | H1–H4, H2b | H1, H2, H2b, zero-shot only; no few-shot, no policy tests |

Same-seed runs on the T4 are not bit-exact (GPU noise), so a seed fixes the start and the batches, not the exact path.

**Learning rates: a search made before the freeze.** The Mac rate of {MAC_RATE:g} was chosen at size S and 2.75e14 FLOPs. Mac validation showed H-Net M lagging S by more at the top two budgets than any other method did, so a rate that is too high for the larger sizes could have been hiding part of the result. Before the freeze, each method and size at 5e16 was therefore run once at 5e15 on the T4 at peak rates {', '.join(f'{r:g}' for r in GRID)}, seed 0, by the same procedure, and scored on held-in validation only. The lowest bpb chose the rate. The grid's ends were treated alike: one more rate beyond an end that won, then a choice among all four, with no further extension. No best rate was {GRID[0]:g}, so the lower rate of 0.00015 was not run. {GRID[-1]:g} won for {', '.join(f'{LABEL[m]} {s}' for m, s in edged)}, so these four ran once more at {EDGE:g}. {edge_won and 'The new rate won for ' + ', '.join(edge_won) + '; it is the end of the search, and nothing higher was tried.' or 'The new rate won for none of them.'} No test split and no held-out stream was touched.

Held-in validation bpb, chosen rate in bold:

| Method and size | {' | '.join(f'{r:g}' for r in CHECK_RATES)} | Chosen |
| --- | {' | '.join('---' for _ in CHECK_RATES)} | --- |
{rate_table}

These rates, fixed in `prereg/ext1_rates.json`, are used for every 5e16 run of that method and size, seeds 0 and 1. The calibration runs of §5 keep {MAC_RATE:g}, because they repeat the Mac runs.

Warm-up and decay scale with each run's own length: warm-up over the first 2% of the run's steps, then cosine decay to 10% of the peak over the rest (`lr_at` in `robobytes/train.py`). A 5e16 run therefore warms up for about 10 times as many steps as its 5e15 check run.

Limits of this search. The rates were chosen at 5e15 and are not tested at 5e16. Each cell is one run. For {', '.join(close)}, the best rate beat the next by {close_range[0]:.1%} to {close_range[-1]:.1%}, about the size of the T4-to-Mac differences in §9. A method's 5e16 run may use a different rate from its Mac runs, which is why the primary trend in §6 stays on the T4.

## 3. Sizes and frontier rule

**Rule.** At each budget, the two smallest sizes whose training stays ≤ {CAP} passes over the {SYMBOLS:,} training symbols. The best of the two (lower held-in validation bpb, first 100 validation episodes per dataset, seed 0) is that method's frontier run. Trends across budgets compare frontier runs. This is the Mac ladder's principle: S and M pass the rule at each of its budgets (at most {ladder['max_passes']:.2f} passes).

The compute-optimal size is expected to grow with budget, so a method's frontier size at 5e16 may differ from its Mac frontier size. Comparing frontier runs across budgets is the intended reading; the sizes are not held constant.

At 5e16 the rule gives **M and L** for H-Net, Fixed, Byte and BPE, because S exceeds {CAP} passes for all four, and **S and M** for Byte + conv, whose S stays at {passes['byteconv_S']:.2f} passes. The rule applies to every method without exception. Byte + conv is a descriptive arm.

Measured on the T4 at fp16 and micro-batch 16, 5e16 FLOPs: speed in 10¹² FLOP/s, GPU hours, passes.

| Method | S | M | L |
| --- | --- | --- | --- |
{speed_table}

H-Net S and M were timed with a trained router (the Mac's 5e15 checkpoints), and H-Net L with an untrained one, so its time is less certain. Peak GPU memory is at most {max(v['gpu_peak_gb'] for v in speed.values()):.1f} GB of 15.6 GB.

## 4. Hypotheses at 5e16

Definitions and thresholds as in §2 of `PREREGISTRATION.md`: bpb on data bytes only, g = (bpb_base − bpb_HNet) / bpb_base, 95% paired percentile bootstrap, 10,000 resamples of episodes, RNG seed 0. Each method enters with its frontier size; per-episode bpb is averaged over seeds 0 and 1 before any statistic.

| ID | Question | Pass criterion |
| --- | --- | --- |
| H1 | In distribution, is H-Net at least as compute-efficient as BPE at 5e16? | Held-in test bpb of H-Net < BPE, CI of the difference excludes 0. The main study's H1 asks this at two budgets; Ext1 has one. |
| H2 | Does the advantage over BPE grow on unseen robots? | For at least 2 of the 3 held-out robots: g_OOD(r) > 0 with CI excluding 0, **and** g_OOD(r) ≥ max(1.5 × \\|g_ID\\|, 0.02). |
| H2b | Does the learned router's advantage grow on unseen robots? | The same rule with g = (bpb_Fixed − bpb_HNet) / bpb_Fixed. |

**Compression floor.** If H-Net's realised compression at 5e16 (held-in test) is below 0.8 × R_BPE = 1.239, H1 and H2 count as failed.

Other methods and every other number are reported without pass/fail.

## 5. Calibration

H-Net and Fixed are rerun on Kaggle at 5e15, sizes S and M, seed 0, with the same step counts as their Mac runs. **Rule:** if, for any of these four runs, Kaggle-fp16 and Mac-fp32 pooled held-in validation bpb differ by more than 1% relative, cross-hardware trend comparisons are reported descriptively only: the figures mark the Mac-to-Kaggle step as a break, and the report states no conclusion about a gap growing or shrinking across it. The result does not change H1, H2 or H2b, which compare Kaggle runs with each other. The extension proceeds either way.

## 6. Descriptive

Reported without pass/fail, as gaps with 95% bootstrap CIs on held-in test data and on each held-out robot: H-Net vs Byte, H-Net vs Fixed, H-Net vs Byte + conv and H-Net vs BPE. Also realised compression and each method's frontier size at 5e16.

**Primary trend: 5e15 to 5e16 on the same hardware.** The 5e15 anchor is the rate-check runs at their chosen rates (T4, fp16, seed 0). As in §3, each method's better size by held-in validation is its frontier run, among the sizes it has at 5e16: M and L, and S and M for Byte + conv. That choice is already known:

| Method | Anchor size | Rate | Held-in validation bpb |
| --- | --- | --- | --- |
{anchor_table}

Each gap at 5e15 is set beside the same gap at 5e16, using seed 0 at both budgets because the anchor has one seed. These checkpoints stay in the output of the Kaggle sessions `robobytes-ext1-lr` and `robobytes-ext1-lr-edge`, are pinned by SHA-256 in `prereg/ext1_anchor.json`, and have been scored on held-in validation only. They are scored on held-in test data and on the held-out robots after the freeze, in the same `score` stage as the 5e16 runs, which refuses a checkpoint whose hash differs. The anchor's rate equals the 5e16 rate for as long as a method's frontier size stays the same; if the size moves, the rate may too, and the analysis prints each arm's sizes and rates and marks that case. The rule of §5 does not govern this trend, which stays on one kind of hardware.

**Secondary context: the Mac ladder.** The same gaps across the Mac budgets, then the 5e16 frontier run with both seeds. Mac runs use {MAC_RATE:g} and other hardware, so a change across that step cannot be credited to budget alone, and §5 decides how far it can be read at all.

Byte + conv has seed 0 only.

## 7. Predictions stated in advance

Judgement from Mac held-in validation and the learning-rate check only (§2, §9); no model has been scored on a held-out robot. Probabilities are subjective. Claude's were stated first. Téo adopted Claude's values after seeing them, so the two columns are identical, give the same Brier score and are not independent forecasts. The report scores them on the six rows below.

{prediction_table}

Revisions before the freeze. {revised} Claude's other rows were written before the check and are kept after seeing it.

Also stated, without a probability. Held-in order: Fixed and Byte + conv lead, then H-Net and Byte close together, BPE last, with gaps other than BPE's within about 3%. Téo adopted this line.

## 8. Run list and GPU time

Quota: 30 GPU-hours before the reset on Sat 7 Nov 2026 01:00 CET, of which {USED:.2f} are used by the smoke test, the driver test and the learning-rate check. Kaggle charged T4 x2 once by wall clock ({two['hnet_S']['flops_per_s'] / 1e12:.2f} and {two['fixed_S']['flops_per_s'] / 1e12:.2f} × 10¹² FLOP/s for H-Net S and Fixed S running together), so two jobs at once cost one quota hour.

| Block, in launch order | Runs | GPU hours | Quota hours |
| --- | --- | --- | --- |
{plan}

The check gives a timing test: its {len(check)} runs took {drift:.2f} × the forecast on average, and at most {slow:.2f} ×, for H-Net M. The hours above are the smoke-test forecast; the margin below covers that spread.

Training {gpu:.1f} h + validation {val / 2:.1f} h (18 runs, {fv['seconds']:.0f} s each, two at a time) + test scoring {TEST_H:.1f} h (estimate, {SCORED} runs including the anchor) = {total:.1f} h. With a {MARGIN:.0%} margin and the hours already used: **{with_margin:.1f} of the {LIMIT} quota hours allowed.**

**If the forecast passes {LIMIT} quota hours, Byte + conv M is dropped first**, as its larger size. A further cut is a dated amendment.

## 9. Known before the freeze

Mac held-in validation bpb, seed 0, first 100 validation episodes per dataset (S / M). No held-out robot has been evaluated by any model. Every number in this table uses {MAC_RATE:g}.

| Method | {heads} |
| --- | {' | '.join('---' for _ in bud)} |
{known}

Rate check at {MAC_RATE:g} compared with the Mac runs (§2): H-Net M {repeat['hnet'][0]:.3f} on the T4 and {repeat['hnet'][1]:.3f} on the Mac ({rel['hnet']:.2%} apart), Fixed M {repeat['fixed'][0]:.3f} and {repeat['fixed'][1]:.3f} ({rel['fixed']:.2%} apart). These are the configurations calibration repeats, so this much of the §5 result is known in advance; they are not the calibration runs and do not count as them. Halving the rate improved H-Net M by {gain:.1%}, more than any other method and size gained. Mac H-Net M results therefore understate that size. The main study's frozen runs stay as they are.

T4 checks on Mac checkpoints: the full held-in validation of H-Net M at 5e15 gives {fv['t4']:.5f} on the T4 and {fv['mac']:.5f} on the Mac; per-dataset fp32 scores agree within {agree:.1e} relative, and fp16 scoring within {agree16:.1e}, which is why scoring stays fp32. The smoke test also showed that a run resumes across Kaggle sessions, that resumed T4 runs mostly fall within same-seed GPU noise (two of three; the third was further off, with no sign of a resume bug), and that fp16 training produces checkpoints that `robobytes.evaluate` reads. It produced no hypothesis result: it ran no test split and no held-out stream.

## 10. Rules

- Hypotheses, held-out robots, metrics and thresholds do not change after the freeze; any deviation is a dated amendment with its reason.
- Held-out results appear only in the final analysis, and every reported number comes from one analysis script run on the frozen results.
- Checkpoints are written to the session output. A session that fails is recovered from its downloaded checkpoint; the event is logged.
- Before the freeze, the analysis script is tested on synthetic data only (`tests/test_ext1.py`), and the session driver only on tiny budgets on held-in validation episodes.
- The §6 anchor is scored only after the freeze, and only by the `score` stage.
- Stuck on one problem for more than 2 hours: stop and report.

## 11. Files fixed by this document

The pinned commit is the commit this tag points at. `robobytes/` is unchanged from commit `0c674d5`.

{fixed_files}
- Byte streams, `streams`: digest `df43e4483b56d576919382bd142396bc4a71858ba72e36aa1461ebd5f6ee2565` (78 files); `bpe`: digest `e5c74f53d7b4d1f96eacec8369c9743c38f06b9c3a261ec2531dfe23d3db1000` (79 files). Digest = SHA-256 over sorted lines of relative path and file SHA-256; every session checks them.
- `prereg/ext1_predictions.json`: the rows of §7. `prereg/ext1_lr_check.json`: every run of the rate check.
"""
out.write_text(text)
print(f"calib {calib:.2f} main {main:.2f} seed1 {seed1:.2f} conv {conv_s:.2f} {conv_m:.2f} quota {gpu:.2f} total {total:.2f} margin {with_margin:.2f}")
