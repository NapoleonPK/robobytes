# RoboBytes-Ext1: pre-registration

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
| Peak learning rate | 0.0015 for every method | chosen per method and size, below |
| Budget | 1.58e14 to 5e15 | 5e16 |
| Seeds at the tests | H-Net, BPE 3; Fixed 1 | H-Net, Fixed, Byte, BPE 2 (seeds 0, 1) |
| Tests | H1–H4, H2b | H1, H2, H2b, zero-shot only; no few-shot, no policy tests |

Same-seed runs on the T4 are not bit-exact (GPU noise), so a seed fixes the start and the batches, not the exact path.

**Learning rates: a search made before the freeze.** The Mac rate of 0.0015 was chosen at size S and 2.75e14 FLOPs. Mac validation showed H-Net M lagging S by more at the top two budgets than any other method did, so a rate that is too high for the larger sizes could have been hiding part of the result. Before the freeze, each method and size at 5e16 was therefore run once at 5e15 on the T4 at peak rates 0.0003, 0.00075, 0.0015, seed 0, by the same procedure, and scored on held-in validation only. The lowest bpb chose the rate. The grid's ends were treated alike: one more rate beyond an end that won, then a choice among all four, with no further extension. No best rate was 0.0003, so the lower rate of 0.00015 was not run. 0.0015 won for Fixed M, Byte M, BPE M, Byte + conv S, so these four ran once more at 0.003. The new rate won for none of them. No test split and no held-out stream was touched.

Held-in validation bpb, chosen rate in bold:

| Method and size | 0.0003 | 0.00075 | 0.0015 | 0.003 | Chosen |
| --- | --- | --- | --- | --- | --- |
| H-Net M | 2.148 | **1.918** | 2.340 | – | 0.00075 |
| H-Net L | 2.676 | **2.392** | 2.553 | – | 0.00075 |
| Fixed M | 2.009 | 1.896 | **1.874** | 1.945 | 0.0015 |
| Fixed L | 2.355 | **2.149** | 2.257 | – | 0.00075 |
| Byte M | 1.973 | 1.858 | **1.849** | 1.951 | 0.0015 |
| Byte L | 2.128 | **2.056** | 2.071 | – | 0.00075 |
| BPE M | 2.272 | 2.032 | **2.013** | 2.233 | 0.0015 |
| BPE L | 2.668 | **2.556** | 2.638 | – | 0.00075 |
| Byte + conv S | 1.834 | 1.710 | **1.700** | 1.725 | 0.0015 |
| Byte + conv M | 1.998 | **1.865** | 1.965 | – | 0.00075 |

These rates, fixed in `prereg/ext1_rates.json`, are used for every 5e16 run of that method and size, seeds 0 and 1. The calibration runs of §5 keep 0.0015, because they repeat the Mac runs.

Warm-up and decay scale with each run's own length: warm-up over the first 2% of the run's steps, then cosine decay to 10% of the peak over the rest (`lr_at` in `robobytes/train.py`). A 5e16 run therefore warms up for about 10 times as many steps as its 5e15 check run.

Limits of this search. The rates were chosen at 5e15 and are not tested at 5e16. Each cell is one run. For Fixed M, Byte M, Byte L, BPE M, Byte + conv S, the best rate beat the next by 0.5% to 1.2%, about the size of the T4-to-Mac differences in §9. A method's 5e16 run may use a different rate from its Mac runs, which is why the primary trend in §6 stays on the T4.

## 3. Sizes and frontier rule

**Rule.** At each budget, the two smallest sizes whose training stays ≤ 4 passes over the 458,630,486 training symbols. The best of the two (lower held-in validation bpb, first 100 validation episodes per dataset, seed 0) is that method's frontier run. Trends across budgets compare frontier runs. This is the Mac ladder's principle: S and M pass the rule at each of its budgets (at most 0.76 passes).

The compute-optimal size is expected to grow with budget, so a method's frontier size at 5e16 may differ from its Mac frontier size. Comparing frontier runs across budgets is the intended reading; the sizes are not held constant.

At 5e16 the rule gives **M and L** for H-Net, Fixed, Byte and BPE, because S exceeds 4 passes for all four, and **S and M** for Byte + conv, whose S stays at 3.72 passes. The rule applies to every method without exception. Byte + conv is a descriptive arm.

Measured on the T4 at fp16 and micro-batch 16, 5e16 FLOPs: speed in 10¹² FLOP/s, GPU hours, passes.

| Method | S | M | L |
| --- | --- | --- | --- |
| H-Net | 3.0 / 4.60 h / 7.60 | 5.1 / 2.72 h / 2.15 | 5.5 / 2.51 h / 0.86 |
| Fixed | 3.9 / 3.61 h / 7.60 | 6.5 / 2.14 h / 2.15 | 8.1 / 1.72 h / 0.86 |
| Byte | 6.2 / 2.23 h / 4.25 | 7.7 / 1.79 h / 1.04 | 9.1 / 1.53 h / 0.41 |
| BPE | 5.8 / 2.39 h / 5.76 | 7.5 / 1.86 h / 1.58 | 8.9 / 1.57 h / 0.64 |
| Byte + conv | 5.6 / 2.50 h / 3.72 | 7.9 / 1.75 h / 1.01 | 9.2 / 1.51 h / 0.40 |

H-Net S and M were timed with a trained router (the Mac's 5e15 checkpoints), and H-Net L with an untrained one, so its time is less certain. Peak GPU memory is at most 5.1 GB of 15.6 GB.

## 4. Hypotheses at 5e16

Definitions and thresholds as in §2 of `PREREGISTRATION.md`: bpb on data bytes only, g = (bpb_base − bpb_HNet) / bpb_base, 95% paired percentile bootstrap, 10,000 resamples of episodes, RNG seed 0. Each method enters with its frontier size; per-episode bpb is averaged over seeds 0 and 1 before any statistic.

| ID | Question | Pass criterion |
| --- | --- | --- |
| H1 | In distribution, is H-Net at least as compute-efficient as BPE at 5e16? | Held-in test bpb of H-Net < BPE, CI of the difference excludes 0. The main study's H1 asks this at two budgets; Ext1 has one. |
| H2 | Does the advantage over BPE grow on unseen robots? | For at least 2 of the 3 held-out robots: g_OOD(r) > 0 with CI excluding 0, **and** g_OOD(r) ≥ max(1.5 × \|g_ID\|, 0.02). |
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
| H-Net | M | 0.00075 | 1.918 |
| Fixed | M | 0.0015 | 1.874 |
| Byte | M | 0.0015 | 1.849 |
| BPE | M | 0.0015 | 2.013 |
| Byte + conv | S | 0.0015 | 1.700 |

Each gap at 5e15 is set beside the same gap at 5e16, using seed 0 at both budgets because the anchor has one seed. These checkpoints stay in the output of the Kaggle sessions `robobytes-ext1-lr` and `robobytes-ext1-lr-edge`, are pinned by SHA-256 in `prereg/ext1_anchor.json`, and have been scored on held-in validation only. They are scored on held-in test data and on the held-out robots after the freeze, in the same `score` stage as the 5e16 runs, which refuses a checkpoint whose hash differs. The anchor's rate equals the 5e16 rate for as long as a method's frontier size stays the same; if the size moves, the rate may too, and the analysis prints each arm's sizes and rates and marks that case. The rule of §5 does not govern this trend, which stays on one kind of hardware.

**Secondary context: the Mac ladder.** The same gaps across the Mac budgets, then the 5e16 frontier run with both seeds. Mac runs use 0.0015 and other hardware, so a change across that step cannot be credited to budget alone, and §5 decides how far it can be read at all.

Byte + conv has seed 0 only.

## 7. Predictions stated in advance

Judgement from Mac held-in validation and the learning-rate check only (§2, §9); no model has been scored on a held-out robot. Probabilities are subjective. Claude's were stated first. Téo adopted Claude's values after seeing them, so the two columns are identical, give the same Brier score and are not independent forecasts. The report scores them on the six rows below.

| Prediction | Claude | Téo | Claude's reasoning |
| --- | --- | --- | --- |
| Frontier size at 5e16 is M for H-Net, Fixed, Byte and BPE | 70% | 70% | L buys 2.5 times fewer steps than M, and at tuned rates M beat L by 11% or more for all four methods at 5e15. On the Mac, S stays ahead of M at the top budget, with 3.5 times M's steps. |
| H1 passes | 75% | 75% | H-Net's lead over BPE at size S narrowed from budget 3 to budget 4, and BPE could keep closing it, but the held-in test set is large and the CI should be tight. |
| H2 passes | 30% | 30% | If g_ID is near 0.05 the bar falls to about 0.08 rather than the pilot's 0.47, which is why this is above the main study's frozen 20%. It still needs that margin on 2 of 3 unseen robots. |
| H2b passes | 10% | 10% | H-Net trails Fixed on held-in validation at 5e15, and the learned router must beat fixed patches by at least 2% on 2 of 3 unseen robots. |
| Compression floor passes | 90% | 90% | H-Net's realised compression at 5e15 is near 1.9 on held-in validation. |
| Calibration within 1% on all four runs | 20% | 20% | Revised after the rate check: the 0.0015 repeats of the two M runs sit 1.41% (H-Net) and 0.59% (Fixed) from the Mac, so one configuration is already over the line and the rule needs all four within it. |

Revisions before the freeze. Frontier size at 5e16 is M for H-Net, Fixed, Byte and BPE: Claude revised this from 60% to 70% after seeing the check. Calibration within 1% on all four runs: Claude revised this from 75% to 20% after seeing the check. Claude's other rows were written before the check and are kept after seeing it.

Also stated, without a probability. Held-in order: Fixed and Byte + conv lead, then H-Net and Byte close together, BPE last, with gaps other than BPE's within about 3%. Téo adopted this line.

## 8. Run list and GPU time

Quota: 30 GPU-hours before the reset on Sat 7 Nov 2026 01:00 CET, of which 4.65 are used by the smoke test, the driver test and the learning-rate check. Kaggle charged T4 x2 once by wall clock (2.84 and 4.15 × 10¹² FLOP/s for H-Net S and Fixed S running together), so two jobs at once cost one quota hour.

| Block, in launch order | Runs | GPU hours | Quota hours |
| --- | --- | --- | --- |
| Calibration: H-Net, Fixed at 5e15, sizes S and M | 4 | 1.31 | 0.65 |
| Main: H-Net, Fixed, Byte, BPE at 5e16, sizes M and L | 8 | 15.82 | 7.91 |
| Seed 1: the four methods at frontier size (upper case: all M) | 4 | 8.50 | 4.25 |
| Byte + conv S at 5e16 | 1 | 2.50 | 1.25 |
| Byte + conv M at 5e16 | 1 | 1.75 | 0.88 |

The check gives a timing test: its 34 runs took 1.00 × the forecast on average, and at most 1.21 ×, for H-Net M. The hours above are the smoke-test forecast; the margin below covers that spread.

Training 14.9 h + validation 0.2 h (18 runs, 64 s each, two at a time) + test scoring 1.6 h (estimate, 14 runs including the anchor) = 16.7 h. With a 15% margin and the hours already used: **23.8 of the 26 quota hours allowed.**

**If the forecast passes 26 quota hours, Byte + conv M is dropped first**, as its larger size. A further cut is a dated amendment.

## 9. Known before the freeze

Mac held-in validation bpb, seed 0, first 100 validation episodes per dataset (S / M). No held-out robot has been evaluated by any model. Every number in this table uses 0.0015.

| Method | 1.58e+14 | 5e+14 | 1.58e+15 | 5e+15 |
| --- | --- | --- | --- | --- |
| H-Net | 4.366 / 6.333 | 2.838 / 4.909 | 1.937 / 3.113 | 1.760 / 2.374 |
| Fixed | 3.390 / 6.199 | 2.279 / 3.836 | 1.901 / 2.310 | 1.745 / 1.863 |
| Byte | 2.912 / 6.269 | 2.163 / 3.070 | 1.957 / 2.159 | 1.775 / 1.876 |
| BPE | 6.258 / 6.803 | 3.205 / 6.182 | 2.086 / 2.975 | 1.869 / 2.067 |
| Byte + conv | 3.797 / 6.612 | 2.222 / 4.995 | 1.899 / 2.310 | 1.701 / 1.938 |

Rate check at 0.0015 compared with the Mac runs (§2): H-Net M 2.340 on the T4 and 2.374 on the Mac (1.41% apart), Fixed M 1.874 and 1.863 (0.59% apart). These are the configurations calibration repeats, so this much of the §5 result is known in advance; they are not the calibration runs and do not count as them. Halving the rate improved H-Net M by 18.1%, more than any other method and size gained. Mac H-Net M results therefore understate that size. The main study's frozen runs stay as they are.

T4 checks on Mac checkpoints: the full held-in validation of H-Net M at 5e15 gives 2.37373 on the T4 and 2.37373 on the Mac; per-dataset fp32 scores agree within 1.9e-06 relative, and fp16 scoring within 7.0e-04, which is why scoring stays fp32. The smoke test also showed that a run resumes across Kaggle sessions, that resumed T4 runs mostly fall within same-seed GPU noise (two of three; the third was further off, with no sign of a resume bug), and that fp16 training produces checkpoints that `robobytes.evaluate` reads. It produced no hypothesis result: it ran no test split and no held-out stream.

## 10. Rules

- Hypotheses, held-out robots, metrics and thresholds do not change after the freeze; any deviation is a dated amendment with its reason.
- Held-out results appear only in the final analysis, and every reported number comes from one analysis script run on the frozen results.
- Checkpoints are written to the session output. A session that fails is recovered from its downloaded checkpoint; the event is logged.
- Before the freeze, the analysis script is tested on synthetic data only (`tests/test_ext1.py`), and the session driver only on tiny budgets on held-in validation episodes.
- The §6 anchor is scored only after the freeze, and only by the `score` stage.
- Stuck on one problem for more than 2 hours: stop and report.

## 11. Files fixed by this document

The pinned commit is the commit this tag points at. `robobytes/` is unchanged from commit `0c674d5`.

- `scripts/ext1_train.py`: SHA-256 `ffb8199d521fe12125a260790aed5426728f8e23ee157dc6b5f7015c6000f893`
- `scripts/ext1_kernel.py`: SHA-256 `4c505e2a03f582a71fc4994ce05724322ae1100b9a2e80fc1e8da313f8e97266`
- `scripts/ext1_analysis.py`: SHA-256 `26f4e086e0d9796326b10f302641b228c5fa953b6c94f76320ab4555bbd03b89`
- `prereg/ext1_rates.json`: SHA-256 `583c10c7ad33e04a29ae72c18ac18939f89c1cc578b379357c98892f5a06aa68`
- `prereg/ext1_anchor.json`: SHA-256 `1cf379f30f81eae408db29a78573620605b0a6b9d364cb838fcd69c0da11561a`
- Byte streams, `streams`: digest `df43e4483b56d576919382bd142396bc4a71858ba72e36aa1461ebd5f6ee2565` (78 files); `bpe`: digest `e5c74f53d7b4d1f96eacec8369c9743c38f06b9c3a261ec2531dfe23d3db1000` (79 files). Digest = SHA-256 over sorted lines of relative path and file SHA-256; every session checks them.
- `prereg/ext1_predictions.json`: the rows of §7. `prereg/ext1_lr_check.json`: every run of the rate check.
