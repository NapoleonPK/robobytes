"""Generate PREREGISTRATION.md and the episode lists it fixes (prereg/splits.json) from the pilot's result files.
Every number in the document comes from these files: runs/streams_summary.json, runs/bpe_summary.json,
runs/bpe_vocab_check.json, runs/ladder.json, runs/lr_search.json, runs/lr_search_chunk_rope.json, runs/rope_bytes_check.json, runs/timing_rope_bytes.json,
runs/wrapper_diag.json, runs/wrapper_share.json,
robobytes/sources.py. The prediction probabilities in §2.1 are judgement, written here by hand.
usage: PYTHONPATH=. python scripts/make_prereg.py"""
import hashlib
import json
from pathlib import Path

import numpy as np

from robobytes.sources import HELD_OUT, TRAIN
from robobytes.serialize import STREAMS, Stream

OUT = Path("PREREGISTRATION.md")
FINAL = False  # set by --final
LISTS = Path("prereg/splits.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt_e(x):
    m, e = f"{x:.2e}".split("e")
    return f"{float(m):g}e{int(e)}"


def splits():
    """Episode ids per split, in the seed-0 order the streams were built in (pool order = few-shot order)."""
    out = {}
    for p in sorted(STREAMS.iterdir()):
        s = Stream(p.name)
        out[p.name] = {sp: [str(s.ep_id[e]) for e in s.episodes(sp)] for sp in np.unique(s.split)}
    LISTS.parent.mkdir(exist_ok=True)
    json.dump(out, open(LISTS, "w"), indent=0)
    return out


def main():
    st = json.load(open("runs/streams_summary.json"))
    bp = json.load(open("runs/bpe_summary.json"))
    vc = json.load(open("runs/bpe_vocab_check.json"))
    ld = json.load(open("runs/ladder.json"))
    lr = json.load(open("runs/lr_search.json")) if Path("runs/lr_search.json").exists() else {}
    best = lr.get("_best", {})
    sp = splits()
    R, N = bp["R_BPE"], bp["N"]
    L = []
    a = L.append
    a("# RoboBytes: pre-registration\n")
    if FINAL:
        a("**Status: frozen 2026-10-31.** The frozen text is this file at the commit tagged `prereg` in the public "
          "`robobytes` repo; its SHA-256 is published the same hour as the GitHub release `prereg`. Later changes are "
          "dated amendments.\n")
    else:
        a("**Status: DRAFT for review (not frozen).** Frozen = the commit tagged `prereg` in the public `robobytes` "
          "repo, whose SHA-256 is published the same hour as a GitHub release.\n")
    a("Independent study. Not affiliated with Cartesia, the H-Net authors (Hwang, Wang & Gu), the FAST authors or "
      "LeRobot / Hugging Face; it re-implements their published ideas and uses public LeRobot datasets. Follows RoboChunk "
      "(`NapoleonPK/robochunk-study`), whose router code it reuses.\n")
    a("## 1. Question\n")
    a("Cartesia reported that H-Net's advantage over BPE grows out of distribution (1.28× compute multiplier in-domain, "
      "median 3.10× on unseen programming languages). RoboBytes asks whether the same holds for **robots never seen in "
      "training**: each joint value is quantised to one byte, and models read the byte stream of trajectories.\n")
    a("## 2. Hypotheses and pass criteria\n")
    a("H1–H4 compare plain H-Net (primary) with BPE (baseline). **H2b** (co-primary with H2) compares H-Net with Fixed "
      "patches, which share every component except the learned router, so it isolates the router's contribution. Other "
      "methods are reported without pass/fail.\n")
    a("- **bpb:** cross-entropy of the next byte in bits, data bytes only (BOS and SEP predictions excluded), for every method.")
    a("- **Relative gap:** g = (bpb_BPE − bpb_HNet) / bpb_BPE; g > 0 means H-Net is better.")
    a("- **g_ID:** on the held-in test episodes (pooled, see §7). **g_OOD(r):** zero-shot on held-out robot r's test episodes.")
    a(f"- **Top budget:** {fmt_e(ld['budgets'][-1])} FLOPs. H-Net and BPE run 3 seeds there (0, 1, 2); per-episode bpb "
      "is averaged over seeds before any statistic.")
    a("- **CI:** 95% paired percentile bootstrap, 10,000 resamples of episodes, RNG seed 0.\n")
    a("| ID | Question | Pass criterion |")
    a("| --- | --- | --- |")
    a("| H1 | In distribution, is H-Net at least as compute-efficient as BPE? | At each of the two largest budgets "
      f"({fmt_e(ld['budgets'][-2])} and {fmt_e(ld['budgets'][-1])}), held-in test bpb of H-Net < BPE, and the CI of the "
      "difference excludes 0. |")
    a("| H2 (main) | Does the advantage grow on unseen robots? | At the top budget, for at least 2 of the 3 held-out "
      "robots: g_OOD(r) > 0 with CI excluding 0, **and** g_OOD(r) ≥ max(1.5 × \\|g_ID\\|, 0.02). |")
    a("| H2b (co-primary) | Does the **learned router's** advantage grow on unseen robots? | Same rule with "
      "g = (bpb_Fixed − bpb_HNet) / bpb_Fixed (both at N = round(R_BPE)): at the top budget, for at least 2 of the 3 "
      "held-out robots, g_OOD(r) > 0 with CI excluding 0 **and** g_OOD(r) ≥ max(1.5 × \\|g_ID\\|, 0.02). |")
    a("| H3 | Does H-Net adapt to a new robot better with few episodes? | After fine-tuning on the 50-episode pool of "
      "each held-out robot (§8), test bpb of H-Net < BPE for at least 2 of 3 robots, CI excluding 0. |")
    a("| H4 (underpowered) | Does better pretraining show up as better robot behaviour? | Policy success on PushT and "
      "Fetch (§9): H-Net − BPE > 0 with CI (over rollouts and seeds) excluding 0 on at least 1 of 2 tasks. |\n")
    a(f"**Compression floor.** If H-Net's realised compression at the top budget (held-in test) is below 0.8 × R_BPE = "
      f"{0.8 * R:.3f}, the comparison is not matched and H1 and H2 count as failed.\n")
    a("**Exploratory (labelled as such):** compute multipliers from the iso-FLOP frontiers; the secondary N = 4 "
      "comparison (§6.1); the no-EMA arm (§6.3); whether chunk counts for a new robot shrink as an episode goes on (in-context merging); whether "
      "boundaries align with gripper events; 16-bit serialisation; how Byte and Entropy transfer.\n")
    bv = {m: lr[f"{m}_{best[m]:g}"]["val_bpb"] for m in best}
    g_b, g_f = (bv["bpe"] - bv["hnet"]) / bv["bpe"], (bv["fixed"] - bv["hnet"]) / bv["fixed"]
    a("### 2.1 Predictions stated in advance\n")
    a("Judgement from the held-in pilot only (search budget, size S, §6); no model has seen a held-out robot. "
      "Probabilities are subjective, and the report will score them.\n")
    a(f"- **Held-in order:** Byte best ({bv['byte']:.2f}), then Fixed ({bv['fixed']:.2f}), Byte + conv "
      f"({bv['byteconv']:.2f}), H-Net without EMA ({bv['hnet_no_ema']:.2f}), Entropy ({bv['entropy']:.2f}), H-Net "
      f"({bv['hnet']:.2f}), BPE ({bv['bpe']:.2f}). Gaps should narrow with budget.")
    a(f"- **H1: likely passes (90%).** Pilot g = {g_b:.2f} for H-Net over BPE. The risk is that BPE improves faster with "
      "budget.")
    a(f"- **H2: likely fails (20%).** A pilot g_ID of {g_b:.2f} would require g_OOD ≥ {1.5 * abs(g_b):.2f} on 2 of 3 robots. "
      "BPE barely compresses held-out robots, so it nears a byte model with a shorter context; the effect on it is unclear.")
    a(f"- **H2b: expected to fail (5%) unless the H-Net vs Fixed gap reverses out of distribution.** Pilot g_ID = "
      f"{g_f:.2f}. The rule uses |g_ID|, so H-Net being worse in distribution raises the bar: at that value "
      f"g_OOD ≥ {1.5 * abs(g_f):.2f} is required, meaning H-Net below {1 - 1.5 * abs(g_f):.2f} × Fixed's bpb.")
    a("- **H3: likely passes (65%)** if H-Net leads BPE at zero shot, because BPE's vocabulary is frozen and it cannot "
      "form new tokens for a new robot.")
    a("- **H4: likely fails or is uninformative (20%).** Underpowered by design.")
    a(f"- **Compression floor: passes (90%).** Pilot H-Net compression "
      f"{lr[f'hnet_{best['hnet']:g}']['train']['compression']:.2f} against a floor of {0.8 * R:.2f}.")
    a("- **No-EMA arm:** better than primary H-Net in distribution and still behind Fixed, on held-in data; whether it "
      "closes on Fixed out of distribution is the open question, with no prediction of direction.")
    a("- **N = 4:** H-Net and Fixed both worse than at N = 2, and H-Net further behind Fixed; refractory r = 1 changes "
      "little.\n")
    a("## 3. Data\n")
    family = lambda t: ("Franka" if ("Franka" in t or "Panda" in t) else "SO-100/101" if t.startswith("SO-")
                        else "xArm" if t.startswith("xArm") else t)
    fams = sorted({family(t) for *_, t in TRAIN})
    a(f"**Training robots:** {len(TRAIN)} LeRobot datasets, **{len(fams)} robot types** counted conservatively "
      f"({', '.join(fams)}; Franka arms including the mobile PandaOmron count as one, SO-100 and SO-101 as one, simulated "
      "and real xArm as one). State and action only, `data/` and `meta/` "
      "downloaded at the pinned revision; no code from any dataset repo is run.\n")
    a("| Dataset | Revision | Robot | Channels (W) | fps | Episodes train / val / test | Training bytes |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for repo, rev, robot in TRAIN:
        n = repo.split("/")[1]
        s = st[n]
        e = s["episodes"]
        a(f"| `{repo}` | `{rev[:12]}` | {robot} | {s['W']} | {s['fps']} | {e['train']} / {e['val']} / {e['test']} | "
          f"{s['data_bytes_train']:,} |")
    a("")
    a("**Excluded:** every bimanual dataset (ALOHA's held-out type), every PushT and Fetch dataset, datasets over 2 GB of "
      "`data/` (e.g. `droid_1.0.1`), datasets without a research licence, datasets with images inside the parquet.\n")
    a("**Held-out robots** (fixed; never in training, never in the BPE or entropy-model training data):\n")
    a("| Robot | Data (revision) | Channels | fps | Few-shot pool / val / zero-shot test |")
    a("| --- | --- | --- | --- | --- |")
    for key, label in (("heldout_aloha", "ALOHA, bimanual (14 joints): two arms, twice the joints of any training robot"),
                       ("heldout_fetch", "Fetch, end-effector deltas + gripper (4-D action): unseen body and simulator"),
                       ("heldout_pusht", "PushT, 2-D point pusher: not an arm")):
        s = st[key]
        e = s["episodes"]
        revs = ", ".join(f"`{r}` (`{v[:12]}`)" for r, v in s["revision"].items())
        a(f"| {label} | {revs} | {s['W']} | {s['fps']} | {e['pool']} / {e['val']} / {e['test']} |")
    a("")
    a("ALOHA pools its two simulated tasks into one robot. Fetch uses RoboChunk's held-out clean simulated test set "
      "(`sim/test.npz`; observation 25 + goal 3, then action 4 per step). PushT uses `lerobot/pusht_keypoints` "
      "(agent position 2 + 16-D block keypoints, then action 2).\n")
    a(f"**Episode lists** are fixed in `{LISTS}` (SHA-256 `{sha(LISTS)}`): per dataset, the episode ids of each split in "
      "seed-0 order. For held-out robots the pool is ordered: its first 10 episodes are the 10-shot subset.\n")
    a("## 4. Byte serialisation (identical for every method)\n")
    a("1. Per dataset and channel, clip to the 1st–99th percentile of the training split (held-out robots: of the "
      "few-shot pool only) and scale linearly to 0–255 (`uint8`). A channel whose percentiles coincide maps to 0.")
    a("2. Stream per episode: `BOS`, then per timestep the state bytes (then environment state, if any), the action "
      "bytes in native channel order, and `SEP`. Vocabulary 258.")
    a("3. No robot ID, no channel-count header. Native frame rates; no resampling.")
    a("4. Splits: training datasets 90 / 5 / 5 by episode (seed 0); held-out robots 50 pool / 10 val / rest test (seed 0).")
    a("5. Training windows are drawn per dataset with probability ∝ √(training bytes).\n")
    a("## 5. Methods\n")
    a(f"**BPE:** Hugging Face `tokenizers` BPE, vocabulary 4,096, trained on a 30 MB square-root-mixture sample of "
      f"training rows; one word per timestep so merges never cross `SEP`; `BOS`/`SEP` atomic. **R_BPE = {R:.3f}** symbols "
      f"per position on the training mixture (pooled over all training rows: {bp['R_BPE_pooled']:.3f}). Held-out "
      f"robots under the frozen vocabulary: ALOHA {bp['datasets']['heldout_aloha']['test']['compression']:.3f}, Fetch "
      f"{bp['datasets']['heldout_fetch']['test']['compression']:.3f}, PushT "
      f"{bp['datasets']['heldout_pusht']['test']['compression']:.3f}. Because R_BPE < 2, a 16,384 vocabulary was tried "
      f"(brief's risk table): R = {vc['16384']['R_mixture']:.3f}, still below 2, so the vocabulary stays 4,096 and "
      f"**N = {N}** (= round(R_BPE)) for every chunked method.\n")
    a("| Method | Grouping | Role |")
    a("| --- | --- | --- |")
    a("| BPE | as above; context round(1,024 / R_BPE) tokens | Primary baseline |")
    a(f"| H-Net | RoboChunk's router (LayerNorm, identity init), ratio loss α = 0.3 targeting N = {N}; causal-conv encoder "
      "and decoder, 2 blocks each, width 128; rotary positions on byte offset / N; EMA dechunk; next-byte softmax over 258 | "
      "Primary method |")
    a("| H-Net no EMA | as H-Net, without the EMA dechunk, charged the same FLOPs | Descriptive arm (§6.3) |")
    a("| H-Net refractory | as H-Net, no new boundary within r positions of the last kept one, **r = floor(0.3 × N)** "
      f"(r = 3 at RoboChunk's N = 10). At N = {N}, r = {int(0.3 * N)}: identical to H-Net, so the arm is not in the main "
      "ladder; it runs only in the N = 4 comparison (r = 1) | Exploratory |")
    a(f"| Fixed | H-Net architecture, a boundary every {N} positions | Baseline |")
    a("| Entropy | H-Net architecture, boundary where a small byte model's next-symbol entropy exceeds a threshold set to "
      f"give compression {N} on training-robot validation windows; the small model is the Byte run at the second budget | Baseline |")
    a("| Byte | plain Transformer, one position per byte | Reference |")
    a("| Byte + conv | H-Net's conv encoder and decoder with the main Transformer over every byte (a boundary at every "
      "position): H-Net without chunking | Reference: separates the conv wrapper from chunking |\n")
    a("Main Transformer sizes: **S** = width 256, 4 layers; **M** = width 384, 8 layers (pre-norm, rotary positions, "
      "context 1,024 bytes). Chunked methods use byte offset / N as the rotary position, which equals the chunk index "
      "for regular boundaries. Only the main network changes with size.\n")
    au = json.load(open("runs/baseline_audit.json"))
    de = json.load(open("runs/delta_reference.json"))["_pooled"]
    a(f"**Baseline audit.** A no-learning reference, predicting each byte from the same channel at the previous "
      f"timestep, gives {de['val20_bpb']:.2f} bpb on validation; it is told the channel layout. The first pilot used "
      f"learned absolute positions, where Byte reached only {au['byte_baseline']['val_bpb']:.2f}. Rotary positions "
      f"reached {au['byte_rope']['val_bpb']:.2f} (GPT-2 init alone {au['byte_gpt2init']['val_bpb']:.2f}; both "
      f"{au['byte_rope_gpt2init']['val_bpb']:.2f}), because the data's main structure is a relative offset. **Every "
      "method therefore uses rotary positions** (over bytes or tokens; for chunked methods see below). The first search is archived and fixes "
      "nothing.\n")
    wd = json.load(open("runs/wrapper_diag.json"))
    ws = json.load(open("runs/wrapper_share.json"))
    v = lambda k: wd[k]["val_bpb"]
    a(f"**Wrapper diagnostic.** With the brief's wrapper (width 256, 4 blocks) Byte + conv scored "
      f"{v('byteconv_256x4'):.2f} against Byte's {v('byte'):.2f} at equal FLOPs ({wd['byteconv_256x4']['steps']} against "
      f"{wd['byte']['steps']} steps). Given Byte's step count it reached {v('byteconv_256x4_same_bytes'):.2f}, and with zero "
      f"conv blocks {v('byteconv_256x0'):.2f}: the loss is mostly the FLOP charge, not a defect. Causality and "
      "residual and norm placement were checked. Outer-network share of FLOPs (Cartesia's ratio: 45%):\n")
    a("| Method | 256×4, S | 256×4, M | 128×2, S | 128×2, M |")
    a("| --- | --- | --- | --- | --- |")
    for k in ws["128x2"]:
        a(f"| {k} | {ws['256x4'][k]['S']:.0%} | {ws['256x4'][k]['M']:.0%} | {ws['128x2'][k]['S']:.0%} | "
          f"{ws['128x2'][k]['M']:.0%} |")
    a("")
    a("**H-Net, Fixed, Entropy and Byte + conv therefore use width 128, 2 blocks** (departure from the brief, on Téo's "
      "instruction). The 256×4 search is archived and fixes nothing.\n")
    old = json.load(open("runs/lr_search_chunk_rope.json"))
    rc = json.load(open("runs/rope_bytes_check.json"))
    ob = {m: old[f"{m}_{old['_best'][m]:g}"]["val_bpb"] for m in ("hnet", "entropy")}
    a(f"**Position audit for chunked methods.** With rotary positions on chunk index, H-Net reached {ob['hnet']:.2f} and "
      f"Entropy {ob['entropy']:.2f} against Fixed's {lr['fixed_0.0015']['val_bpb']:.2f} at the search budget. Under "
      "irregular boundaries the data's W + 1 byte offset is not a constant number of chunks, the failure the audit above "
      f"found for Byte. On byte offset / N they reach {bv['hnet']:.2f} and {bv['entropy']:.2f}. Fixed and Byte + conv are "
      f"unchanged (rescored: {rc['fixed_0.0015']['chunk_index']:.4f} → {rc['fixed_0.0015']['byte_offset']:.4f} and "
      f"{rc['byteconv_0.0015']['chunk_index']:.4f} → {rc['byteconv_0.0015']['byte_offset']:.4f}), the FLOPs do not change "
      "and the router is untouched. **Every chunked method therefore uses byte-offset positions** (Téo's decision). The "
      "chunk-index search is archived and fixes nothing.\n")
    a(f"**EMA.** H-Net without the EMA dechunk reached {bv['hnet_no_ema']:.2f} on the same positions, "
      f"{bv['hnet'] - bv['hnet_no_ema']:.2f} bits better. The hypotheses name H-Net, so EMA stays in the primary method; "
      "the no-EMA variant is a descriptive arm (§6.3).\n")
    a("## 6. Compute ladder and training\n")
    a(f"Budgets **{', '.join(fmt_e(b) for b in ld['budgets'])} FLOPs** (×3.16), sizes S and M at every budget, 6 "
      f"methods: {len(ld['runs'])} ladder runs, ≈ {ld['ladder_hours']:.0f} h at the measured throughput; longest run "
      f"{ld['longest_not_entropy']:.2f} h across every arm, except Entropy (up to "
      f"{ld['max_run_hours']:.2f} h, because its small byte model's forward pass is wall-clock time not charged to its "
      "budget; Téo accepted this overrun). FLOPs = 6 × parameters used per position + 6 × layers × context × width, per "
      f"network, per symbol, charged at nominal N, so a method whose realised compression falls below N uses more main-network "
      f"positions than it is charged for (H-Net pilot: {lr[f'hnet_{best['hnet']:g}']['train']['compression']:.2f}); realised "
      f"compression is reported for every run. At most {ld['max_passes']:.2f} passes over the {ld['train_symbols']:,} training symbols, so "
      "no budget is dropped. At each budget the size with the lower held-in validation bpb is the method's frontier run; "
      "only frontier runs enter the tests.\n")
    a("Optimiser: AdamW (0.9, 0.95), weight decay 0.1, 2% linear warm-up, cosine decay to 10%, gradient clip 1.0, "
      "effective batch 16 windows, fp32. Training loss: next-symbol cross-entropy on every target (+ 0.3 × ratio loss "
      "for H-Net).\n")
    if best:
        run0 = next(v["run"] for k, v in lr.items() if not k.startswith("_"))
        b0 = json.load(open(Path(run0) / "meta.json"))["args"]["budget"]
        a(f"**Learning rates** (pilot search at {fmt_e(b0)} FLOPs, size S, 3 values per method, held-in validation "
          "only; a method whose best rate was at an end of the grid got one more rate ×3 beyond it, once): "
          + ", ".join(f"{m} {v:g}" for m, v in best.items()) + ". Fixed for every later run.\n")
        a("**Known before the freeze.** Held-in validation at the search budget, size S, each method's best rate (first "
          "20 validation episodes per dataset). These pilot numbers motivated H2b and the N = 4 arm; held-out robots have "
          "not been evaluated by any model:\n")
        a("| Method | Rate | Validation bpb |")
        a("| --- | --- | --- |")
        for m, v in best.items():
            a(f"| {m} | {v:g} | {lr[f'{m}_{v:g}']['val_bpb']:.3f} |")
        a("")
        a(f"The search ran at {fmt_e(b0)} FLOPs, above the smallest budget of {fmt_e(ld['budgets'][0])}, because real "
          "training speed was measured after it. Rates are not re-searched.\n")
    else:
        a("**Learning rates:** pending (search running: 5e-4, 1.5e-3, 4.5e-3 per method at the smallest budget).\n")
    a("Extra runs: H-Net and BPE at the top budget with seeds 1 and 2 (frontier size).\n")
    a("### 6.1 Secondary comparison at N = 4 (pre-registered, no pass/fail)\n")
    a(f"N = round(R_BPE) = {N} barely exercises chunking, so H-Net and Fixed are also trained with **N = 4** across the "
      "full ladder (4 budgets × sizes S and M), plus H-Net refractory at N = 4 with r = 1 at the top two budgets (H-Net "
      "N = 4's frontier size). Same data, windows, optimiser and seeds as the main ladder; learning rate = the same "
      "method's rate from the N = 2 search; Fixed places a boundary every 4 positions. Each is compared with BPE's "
      "frontier at matched FLOPs on held-in data and on each held-out robot (bpb, relative gap g with 95% bootstrap "
      "CIs, iso-FLOP frontiers). Reported descriptively; it does not enter H1–H4.\n")
    a("### 6.2 Router effect at N = 4 (no pass/fail)\n")
    a("The H2b statistic (H-Net vs Fixed: g_ID, g_OOD per robot, CIs, the 1.5 × |g_ID| and 0.02 lines) is also computed "
      "for the N = 4 arm of §6.1 and reported descriptively.\n")
    a("### 6.3 H-Net without EMA (pre-registered, no pass/fail)\n")
    a(f"H-Net without the EMA dechunk, N = {N}, 4 budgets × sizes S and M, seed 0, rate {best['hnet_no_ema']:g}. Compared "
      "with Fixed and with primary H-Net on held-in data and each held-out robot, using the H2b statistic, CIs and lines. "
      "Reported descriptively; it does not enter H1–H4, and it cannot replace H-Net in any test.\n")
    a("### 6.4 Run list and machine time\n")
    a("| Block | Runs | Hours | Basis |")
    a("| --- | --- | --- | --- |")
    for r in ld["run_list"]:
        a(f"| {r['block']} | {r['runs']} | {r['hours']:.1f} | {r['basis']} |")
    a(f"| **Total** | | **{ld['total_hours']:.1f}** | cap 72 h |\n")
    mt = Path("runs/memory_test.json")
    if mt.exists():
        m = json.load(open(mt))
        a(f"**Memory check before the ladder** ({m['config']}, 10 minutes): peak MPS "
          f"allocation {m['peak_mps_gb']:.2f} GB, swap growth {m['swap_growth_mb']:.0f} MB. "
          + (f"The check ended at micro-batch {m['micro_end']}. Every ladder run starts at micro-batch "
             + ("8 with accumulation." if m["decision"].startswith("micro-batch 8") else "16."))
          + " The in-run rule (halve the micro-batch if swap grows by more than 2 GB) stays.\n")
    else:
        a("**Memory check before the ladder:** run the largest configuration "
          "(H-Net, size M, top budget) for 10 minutes and log peak memory and swap growth; if growth exceeds 1 GB, every "
          "ladder run starts at micro-batch 8 with accumulation. Result: pending.\n")
    a("## 7. Metrics and statistics\n")
    a("- **Windows:** each episode in 1,024-position windows (BPE: round(1,024 / R_BPE) tokens), stride half a window, "
      "scoring the second half of each window and all of the first.")
    a("- **Held-in bpb:** per dataset Σ bits / Σ data bytes over its test episodes; pooled = mean over the training "
      "datasets (each weighted equally). Bootstrap resamples episodes within each dataset.")
    a("- **Held-out bpb:** per robot, Σ bits / Σ data bytes over its zero-shot test episodes; bootstrap over episodes.")
    a("- **Paired:** both methods are scored on the same resampled episodes; for g, numerator and denominator are "
      "bootstrapped together.")
    a("- **Held-in validation** (frontier choice): the first 100 validation episodes of each dataset (all, if fewer).")
    a("- **Realised compression:** symbols per main-network position, for every run.\n")
    a("## 8. Few-shot adaptation (H3)\n")
    a("Start from each method's top-budget frontier checkpoint (H-Net and BPE: seeds 0–2). Fine-tune all parameters on "
      "the 10-shot and the 50-shot pool of each held-out robot: AdamW at 0.3 × the method's pretraining learning rate, "
      "constant, effective batch 16 windows, at most 500 steps, validation bpb on the robot's 10 validation episodes "
      "every 50 steps, keep the best. BPE's vocabulary is not retrained. Measure test bpb; H3 uses 50-shot.\n")
    a("## 9. Policy success (H4, underpowered)\n")
    a("Simulators: `gym-pusht` (observation `environment_state_agent_pos`) and `FetchPickAndPlace-v4` from "
      "`gymnasium-robotics` (observation + goal). Freeze the 50-shot model from H3; add a 2-layer MLP head reading the "
      "hidden state at the last observation byte of each step and predicting the continuous action; train only the head "
      "on the same 50 episodes with MSE. Rollout: keep the last 1,024 bytes of history; each step append observation "
      "bytes, read the action, execute it, append the executed action's bytes. Compare H-Net, BPE, Byte and an MLP "
      "behaviour-cloning lower bound, 3 seeds × 100 rollouts on fixed environment seeds; success is the environment's "
      "own flag. If every method is below 5% or above 95% on a task, H4 is uninformative for that task.\n")
    a("## 10. Rules\n")
    a("- Hypotheses, held-out robots, metrics and thresholds do not change after the freeze; any deviation is a dated "
      "amendment with its reason.")
    a("- Every reported number comes from one analysis script run on the frozen results.")
    a("- One training job at a time; checkpoints every 10 minutes.")
    a("- Runs may halve their micro-batch on out-of-memory or swap growth above 2 GB, keeping the effective batch; every "
      "event is logged.\n")
    a("## 11. Files fixed by this document\n")
    for f in ["robobytes/sources.py", "runs/streams_summary.json", "runs/bpe_summary.json", "runs/ladder.json",
              "runs/baseline_audit.json", "runs/delta_reference.json", "runs/wrapper_diag.json", "runs/wrapper_share.json",
              "runs/lr_search_chunk_rope.json", "runs/rope_bytes_check.json", "runs/timing_rope_bytes.json", str(LISTS)]:
        a(f"- `{f}`: SHA-256 `{sha(f)}`")
    if Path("runs/lr_search.json").exists():
        a(f"- `runs/lr_search.json`: SHA-256 `{sha('runs/lr_search.json')}`")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    import sys
    FINAL = "--final" in sys.argv
    main()
