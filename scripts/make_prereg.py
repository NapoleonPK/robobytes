"""Generate PREREGISTRATION.md and the episode lists it fixes (prereg/splits.json) from the pilot's result files.
Every number in the document comes from these files: runs/streams_summary.json, runs/bpe_summary.json,
runs/bpe_vocab_check.json, runs/ladder.json, runs/lr_search.json, robobytes/sources.py.
usage: PYTHONPATH=. python scripts/make_prereg.py"""
import hashlib
import json
from pathlib import Path

import numpy as np

from robobytes.sources import HELD_OUT, TRAIN
from robobytes.serialize import STREAMS, Stream

OUT = Path("PREREGISTRATION.md")
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
    a("**Status: DRAFT for review (not frozen).** Frozen = the commit tagged `prereg` in the public `robobytes` repo, whose "
      "SHA-256 is published the same hour as a GitHub release.\n")
    a("Independent study. Not affiliated with Cartesia, the H-Net authors (Hwang, Wang & Gu), the FAST authors or "
      "LeRobot / Hugging Face; it re-implements their published ideas and uses public LeRobot datasets. Follows RoboChunk "
      "(`NapoleonPK/robochunk-study`), whose router code it reuses.\n")
    a("## 1. Question\n")
    a("Cartesia reported that H-Net's advantage over BPE grows out of distribution (1.28× compute multiplier in-domain, "
      "median 3.10× on unseen programming languages). RoboBytes asks whether the same holds for **robots never seen in "
      "training**: each joint value is quantised to one byte, and models read the byte stream of trajectories.\n")
    a("## 2. Hypotheses and pass criteria\n")
    a("All four tests compare plain H-Net (primary) with BPE (baseline). Other methods are reported without pass/fail.\n")
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
    a("| H3 | Does H-Net adapt to a new robot better with few episodes? | After fine-tuning on the 50-episode pool of "
      "each held-out robot (§8), test bpb of H-Net < BPE for at least 2 of 3 robots, CI excluding 0. |")
    a("| H4 (underpowered) | Does better pretraining show up as better robot behaviour? | Policy success on PushT and "
      "Fetch (§9): H-Net − BPE > 0 with CI (over rollouts and seeds) excluding 0 on at least 1 of 2 tasks. |\n")
    a(f"**Compression floor.** If H-Net's realised compression at the top budget (held-in test) is below 0.8 × R_BPE = "
      f"{0.8 * R:.3f}, the comparison is not matched and H1 and H2 count as failed.\n")
    a("**Exploratory (labelled as such):** compute multipliers from the iso-FLOP frontiers; the refractory router as a "
      "second H-Net arm; whether chunk counts for a new robot shrink as an episode goes on (in-context merging); whether "
      "boundaries align with gripper events; 16-bit serialisation; how Byte and Entropy transfer.\n")
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
      "and decoder, 4 blocks each, width 256; EMA dechunk; next-byte softmax over 258 | Primary method |")
    a("| H-Net refractory | as H-Net, no boundary within 3 positions of the last kept one | Exploratory |")
    a(f"| Fixed | H-Net architecture, a boundary every {N} positions | Baseline |")
    a("| Entropy | H-Net architecture, boundary where a small byte model's next-symbol entropy exceeds a threshold set to "
      f"give compression {N} on training-robot validation windows; the small model is the Byte run at the second budget | Baseline |")
    a("| Byte | plain Transformer, one position per byte | Reference |\n")
    a("Main Transformer sizes: **S** = width 256, 4 layers; **M** = width 384, 8 layers (pre-norm, learned positions, "
      "context 1,024 bytes). Only the main network changes with size.\n")
    a("## 6. Compute ladder and training\n")
    a(f"Budgets **{', '.join(fmt_e(b) for b in ld['budgets'])} FLOPs** (×3.16), sizes S and M at every budget, 5 "
      f"methods: {len(ld['runs'])} ladder runs, ≈ {ld['ladder_hours']:.0f} h at the measured throughput; longest run "
      f"{ld['max_run_hours']:.2f} h. FLOPs = 6 × parameters used per position + 6 × layers × context × width, per "
      f"network, per symbol. At most {ld['max_passes']:.2f} passes over the {ld['train_symbols']:,} training symbols, so "
      "no budget is dropped. At each budget the size with the lower held-in validation bpb is the method's frontier run; "
      "only frontier runs enter the tests.\n")
    a("Optimiser: AdamW (0.9, 0.95), weight decay 0.1, 2% linear warm-up, cosine decay to 10%, gradient clip 1.0, "
      "effective batch 16 windows, fp32. Training loss: next-symbol cross-entropy on every target (+ 0.3 × ratio loss "
      "for H-Net).\n")
    if best:
        run0 = next(v["run"] for k, v in lr.items() if not k.startswith("_"))
        b0 = json.load(open(Path(run0) / "meta.json"))["args"]["budget"]
        a(f"**Learning rates** (pilot search at {fmt_e(b0)} FLOPs, size S, 3 values per method, held-in validation "
          "only): " + ", ".join(f"{m} {v:g}" for m, v in best.items()) + ". Fixed for every later run.\n")
    else:
        a("**Learning rates:** pending (search running: 5e-4, 1.5e-3, 4.5e-3 per method at the smallest budget).\n")
    a("Extra runs: H-Net and BPE at the top budget with seeds 1 and 2 (frontier size); H-Net refractory at the top two "
      "budgets (H-Net's frontier size).\n")
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
    a("- One training job at a time; checkpoints every 10 minutes.\n")
    a("## 11. Files fixed by this document\n")
    for f in ["robobytes/sources.py", "runs/streams_summary.json", "runs/bpe_summary.json", "runs/ladder.json", str(LISTS)]:
        a(f"- `{f}`: SHA-256 `{sha(f)}`")
    if Path("runs/lr_search.json").exists():
        a(f"- `runs/lr_search.json`: SHA-256 `{sha('runs/lr_search.json')}`")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
