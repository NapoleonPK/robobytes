# Decisions log

Every choice the brief (`docs/BRIEF.md`) left open, with a one-line reason. Dated; newest last.

## Answers from Téo before the pre-registration (2026-10-31)
- **End-effector-controlled training robots allowed.** `lerobot/xarm_lift_medium` (4-D end-effector + gripper action) and
  Open X ports with end-effector actions may be used for training. Fetch stays held out as an *unseen body and simulator*;
  its rationale is no longer "unseen control type".
- **PushT data = `lerobot/pusht_keypoints`** (agent position 2 + 16-D block-keypoint `environment_state` + action 2), not
  `lerobot/pusht` (agent position only, no block pose, so a policy reading the stream could not see the block). H4 uses
  gym-pusht's matching `environment_state_agent_pos` observation.
- **H2 floor:** g_OOD(r) ≥ max(1.5 × |g_ID|, 0.02), so a tie in distribution cannot make a negligible out-of-distribution
  edge pass.
- **Freeze:** on Téo's OK, create the public `robobytes` repo with code and `PREREGISTRATION.md` only, tag it, and publish a
  GitHub release with the SHA-256 the same hour (plus the check-in email). Results are added after his review.

## Readings of the brief (stated to Téo 2026-10-31)
- **SEP after every timestep** (BOS, then per timestep: state bytes, action bytes, SEP). Each episode is its own stream, so
  "merges never cross SEP" only has meaning if SEP marks timesteps, like whitespace between words.
- **Fetch stream:** state = observation (25) + goal (3), then action (4), so H2/H3 and the H4 policy stream are identical.

## Setup (2026-10-31)
- Python 3.12 venv, PyTorch 2.15 (MPS available), `tokenizers` 0.23.3; Apple M6, 16 GB.
- Swap already in use before any job: 941 MB (other apps). The 2 GB swap rule is applied to absolute swap as written.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` also needs `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`: PyTorch refuses the default low
  ratio (1.4) when it exceeds the high one.
- Attention: `F.scaled_dot_product_attention(is_causal=True)`. On MPS it still stores the full attention matrix for the
  backward pass (512 MB per layer for the 40M byte model at micro-batch 16), so the 40M byte model does not fit at
  micro-batch 16 under the watermark: the pilot halves the micro-batch on out-of-memory and records it (brief section 4).
- EMA dechunk computed in parallel as a K × K lower-triangular mix (equal to RoboChunk's sequential loop up to a 1e-6
  clamp; unit-tested); the refractory rule runs on CPU in numpy (non-differentiable).
- FLOPs per byte: 6 × parameters used per position + 6 × layers × context × width for attention (Kaplan et al.), per
  network, divided by bytes per position; embeddings and learned positions count 0 (`robobytes/flops.py`).
- **FLOPs cross-check** (`scripts/check_flops.py`, `runs/flops_check.json`, size S, CPU): parameter matmuls agree with
  6 × parameters to 0.5% (19.27 vs 19.36 MFLOP/byte for Byte). The counter's total is higher for Byte (×1.32) because
  `torch.utils.flop_counter` charges scaled-dot-product attention as a full, non-causal L × L product (4·ctx·d forward)
  with a 2.5× backward (flash recompute), whereas the analytic term is Kaplan et al.'s 6·layers·ctx·d (the causal half,
  ×3). The analytic count is kept for the ladder: it is the convention the brief names and it treats all methods alike.
  BPE ×1.07 and H-Net ×0.98 (shorter main sequence, so the attention convention matters less).

## Data (2026-10-31)
- **Training datasets** (`robobytes/sources.py`, full revision SHAs): 23 LeRobot datasets, 13 robot types: Franka
  (`vlabench_unified`, `toto`, `fmb`, `utaustin_mutex`), PandaOmron mobile manipulator (`robocasa_target_human_unified`),
  SO-100, SO-101, Koch, xArm (sim `xarm_lift_medium`, `xarm_push_medium`; real `utokyo_xarm_pick_and_place`), UR5, Jaco 2,
  Kuka iiwa, Sawyer (`roboturk`), Fanuc, Stretch, PR2 (×2), DLR SARA. All ≥ 50 episodes, MIT / Apache-2.0 / CC-BY-4.0,
  `data/` ≤ 2 GB (largest: robocasa, 1.02 GB). Download total 1.3 GB. Open X ports say robot "unknown" in their
  metadata; their robot types come from the Open X-Embodiment dataset table.
- **Excluded:** every bimanual dataset (ALOHA's held-out type: aloha, yam, OpenArms `full_folding`, robotwin, FineART),
  every PushT and Fetch dataset; `droid_1.0.1` (`data/` 12 GB, over the 2 GB limit and the 5 GB download rule);
  `libero_plus` (no licence on the card); `behavior1k` (no licence); datasets with images inside the parquet.
- **Held-out data:** ALOHA = both sim tasks pooled into one robot (100 episodes: 50 pool / 10 val / 40 test). PushT =
  `pusht_keypoints` (206: 50 / 10 / 146). Fetch = RoboChunk's held-out simulated test set `sim/test.npz` of
  `PAOP123/robochunk-sim` (300 clean episodes: 50 / 10 / 240); RoboChunk's training episodes and the corrupted part-B
  candidates are not used.
- **Storage:** quantised rows as a `uint8` memory map per dataset (`data/streams/<name>/rows.npy`); BOS (256) and SEP
  (257) do not fit in `uint8`, so they are inserted when windows are read (`robobytes/serialize.py`, unit-tested).
- **Constant channels** (1st = 99th percentile) map to byte 0; their count per dataset is in `runs/streams_summary.json`.

## BPE
- One "word" per timestep (its W data bytes, mapped to U+0100..U+01FF), so merges cannot cross SEP; BOS and SEP are
  single tokens outside the 4,096 learned entries (ids 4096, 4097). Trained on 30 MB of training-split rows drawn with
  the square-root mixture.
- **R_BPE** is measured on the training mixture (harmonic mean of per-dataset compression under the square-root
  weights), with BOS and SEP counted as one symbol and one position for every method. The pooled value is also reported.
- **Measured:** R_BPE = 1.549 on the training mixture (pooled 1.721), so **N = round(R_BPE) = 2** as the brief specifies.
  It is close to the 1.5 rounding edge; both definitions give 2. Held-out robots under the frozen vocabulary: ALOHA
  1.05, Fetch 1.16, PushT 1.06 (`runs/bpe_summary.json`). 0.8 × R_BPE = 1.24 is the H1/H2 compression floor.

## Throughput pilot and ladder (2026-10-31)
- **Pilot** (`runs/pilot_throughput_mps.json`, 200 steps, random bytes): 1.3–2.5 TFLOP/s achieved, below the brief's
  4 TFLOP/s example. Re-timed at the real R and N (`runs/pilot_throughput_mps_real.json`): BPE 1.6 / 1.9 / 2.2,
  H-Net 1.27 / 1.57 / 1.96, Fixed 1.46 / 1.79 / 2.18 TFLOP/s for sizes S / M / L. All fp32; no mixed precision.
- **Contaminated timing, redone:** H-Net S/M/L were first timed while my CPU smoke tests raised swap to 3.8 GB; those
  timings are kept, labelled `_contaminated`, and superseded by a clean rerun.
- **Micro-batch:** a size-M/L H-Net can run out of memory at micro-batch 16 early in training, when the router still
  emits up to ~1,024 chunks; the trainer halves the micro-batch and keeps the effective batch of 16 by accumulation.
- **Swap rule:** macOS keeps swap allocated after memory pressure ends (baseline is now ~1.9 GB, from other apps and my
  earlier tests), so the 2 GB rule is applied to swap growth during a run, not to absolute swap. Absolute swap is
  logged every 100 steps alongside.
- **Ladder** (`scripts/plan_ladder.py`, `runs/ladder.json`): budgets **2.75e14, 8.7e14, 2.75e15, 8.7e15 FLOPs**
  (×3.16), sizes **S (256 wide, 4 layers) and M (384, 8)** at every budget. The top budget is the largest round value
  whose slowest run (H-Net S) stays under 2 h (1.90 h). Size L (512, 12; 42M) is not used: at 8.7e15 it would see about
  3 bytes per parameter, far from its frontier. Ladder total ≈ 22 h; whole study ≈ 50 h with evaluation, under 72 h.
- **Data passes:** at most 1.00 pass over the 459M training symbols (BPE S at the top budget), so the 4-pass rule keeps
  all four budgets. Under square-root sampling the smallest datasets repeat more (up to ~30× for
  `utokyo_xarm_pick_and_place`; robocasa 0.5×); per-dataset passes are in `runs/ladder.json`.
- **Learning-rate search:** 5e-4, 1.5e-3, 4.5e-3 for every method at 2.75e14, size S, scored on held-in validation (at
  most 20 episodes per dataset, datasets weighted equally). AdamW (0.9, 0.95), weight decay 0.1, 2% warm-up, cosine to
  10%, clip 1.0, effective batch 16 windows. The Entropy baseline's small byte model in the search is the best Byte run
  of the search; in the ladder it is the Byte run at the second budget (brief). Its threshold is the (1 − 1/N) quantile
  of next-symbol entropy over 256 validation windows. Entropy-model FLOPs are not charged to the Entropy run's budget.
- **BPE context:** BPE reads round(1,024 / R_BPE) = 661 tokens per window (1,024 bytes on average) and is evaluated in
  661-token windows with stride 330.
- **Risk-table check, BPE below 2** (brief: "try vocabulary 16,384 in the pilot; if still below 2, record it and use
  N = 2"): `scripts/bpe_vocab_check.py` → `runs/bpe_vocab_check.json`. Same training sample, same evaluation rows:
  R = 1.549 at 4,096, 1.776 at 16,384. Still below 2, so N = 2 and the vocabulary stays 4,096 (the brief's BPE; the
  larger one does not reach the target and would quadruple BPE's embedding and softmax).
- **Edge rule for the learning-rate search** (set 2026-10-31 after seeing only that H-Net's best of the three rates was
  the largest, before any extra run): for every method whose best rate is at an end of the grid, add one rate a further
  ×3 beyond that end (1.35e-2 above, 1.67e-4 below), once. The best of the extended set is used. Same rule for all
  methods.

## Téo's changes before check-in 2 (2026-10-31)
- **Refractory window scales with N:** r = floor(0.3 × N) (r = 3 at RoboChunk's N = 10). At N = 2, r = 0, so the
  refractory arm is dropped from the main ladder (`robobytes/train.py: refractory_for`). The LR search was stopped and
  resumed without it.
- **Secondary comparison at N = 4** (pre-registered, no pass/fail): H-Net and Fixed at N = 4 across the full ladder,
  both sizes (16 runs), plus H-Net refractory at N = 4, r = 1, at the top two budgets (2 runs), each compared with BPE
  at matched FLOPs, held-in and per held-out robot. Learning rate = the same method's rate from the N = 2 search (no
  extra search; simplest). Updated estimate: 58.1 h total (`runs/ladder.json: run_list`).
- **Memory check before the ladder:** reboot to clear swap; run H-Net size M at the top budget (the configuration that
  ran out of memory at micro-batch 16 in the pilot) for 10 minutes; log peak memory and swap growth; if growth exceeds
  1 GB, every ladder run starts at micro-batch 8 with accumulation.
- **Reboot skipped (deviation from Téo's instruction, decided under his 11:10 delegation "you will take those decisions
  yourself"):** a restart would end this session and leave the Mac at the FileVault login screen until someone logs in,
  stalling the study for hours. The reboot's purpose was a clean swap baseline for the memory check; the check measures
  swap growth from its own start, which answers the same question. Absolute swap at the start is recorded with it.
- **Freeze without a separate OK** (same delegation): after the memory check, regenerate `PREREGISTRATION.md`, create
  the public repo, tag `prereg`, publish the release with the SHA-256 the same hour, send check-in 2, start the ladder.
- **Router-effect comparison added before the freeze** (my call under Téo's delegation, after the pilot showed Fixed
  within 0.2 bits of H-Net): H-Net vs Fixed at matched N and FLOPs, held-in and per held-out robot, descriptive with CIs,
  at N = 2 and N = 4. No new runs; it keeps H1–H4 from crediting the learned router with what the shared architecture
  does.

## Téo's second round of changes and the baseline audit (2026-10-31)
- **Delta-coding reference** (`scripts/delta_reference.py`, `runs/delta_reference.json`): predicting each byte from the
  same channel at the previous timestep, with no learning, gives 3.10 bpb pooled on the search's validation episodes,
  against Byte 6.28, BPE 6.49, H-Net 4.08 at the pilot budget. It knows the channel layout, so it is a reference only.
- **Baseline audit** (`scripts/baseline_audit.py`, `runs/baseline_audit.json`; Byte S, pilot budget, rate 1.5e-3): learned
  absolute positions 6.28; GPT-2 init 6.01; rotary positions 2.43; both 3.48. Byte + RoPE beats the delta reference on 21
  of 23 datasets; future-byte perturbation changes no earlier logit. Cause: the data's main structure is "same channel,
  W + 1 bytes back", a relative offset that varies with the robot, which learned absolute positions cannot express. The
  pilot's "2.4-bit H-Net lead over BPE" was mostly this handicap, shared in code but biting only models without conv
  layers. My earlier reading (conv layers learn local patterns faster) was incomplete.
- **All methods now use rotary positions with default initialisation** (`--pos rope`), so parity is kept. RoPE on H-Net
  and Fixed acts on chunk index, on BPE on token index. The learned-position search is archived in
  `runs/lr_search_learned_pos.json` and `runs/archive_learned_pos/`; it fixes nothing.
- **Byte + conv baseline** (`byteconv`): `ByteHNet` with a boundary at every position, over all budgets and both sizes.
  It gets the same search and edge rule. Fixed, Entropy and Byte + conv are charged the EMA dechunk term of the FLOPs
  formula though they skip it, so that the same network has one cost.
- **H2b added as co-primary** (H-Net vs Fixed, H2's thresholds), replacing the descriptive router-effect comparison; the
  same statistic at N = 4 is reported without pass/fail.
- **Fragmentation:** the first memory check fell to micro-batch 1 on out-of-memory errors with 0.3 GB allocated and
  8.8 GB in the driver: chunk counts vary per step, and each new tensor shape fragments the MPS cache. Chunk count is now
  rounded up to a multiple of 64 (outputs unchanged, unit-tested) and the cache is cleared every 50 steps. That check is
  kept as `runs/memory_test_learned_pos.json`; the swap-growth rule alone would have passed it, so micro-batch reached
  in the check is read as well.

## Final pilot and ladder, rotary positions (2026-10-31)
- **Search:** Byte 1.5e-3, BPE 1.5e-3, H-Net 4.5e-3, Fixed 1.5e-3, Entropy 4.5e-3, Byte + conv 1.5e-3 (`runs/lr_search.json`).
  H-Net and Entropy won at the grid end, so 1.35e-2 was run for them and lost. Only rates from this search are used.
- **Ladder lowered to 2.43e14, 7.7e14, 2.43e15, 7.7e15** (`scripts/plan_ladder.py`): real training ran at 0.91 of pilot
  throughput for H-Net and 0.64 for Entropy (others within 1%). The search stayed at 2.75e14: within 13% of the new
  smallest budget, and repeating it was not worth two hours. Entropy's runs reach 2.39 h because its byte model's forward
  pass is uncharged wall-clock; that exception to the 2 h limit is stated in the pre-registration. If its small byte
  model is size M rather than S, those runs are slower still.
- **Micro-batch 8 for every ladder run.** Swap growth was 0, so Téo's 1 GB rule alone would keep 16, but H-Net M still
  dropped to 8 at step 17. Starting at 8 avoids the failed step, and accumulation keeps the effective batch at 16.
- **Timings:** `runs/pilot_throughput_mps_rope.json` (N = 2), `runs/pilot_throughput_mps_rope_n4.json` (N = 4). Total
  estimate 65.1 h against the 72 h cap.

## Hold before the freeze (Téo, 2026-10-31 ~13:40)
- Téo: do not freeze at 14:15; hold the ladder for a wrapper diagnostic; he will OK the freeze after the next check-in 2.
  This replaces the earlier delegation of the freeze decision.
- **Wrapper diagnostic** (`scripts/wrapper_diag.py`, `runs/wrapper_diag.json`; size S, rate 1.5e-3): Byte 2.67; Byte + conv
  256×4 3.85 (307 steps); same with Byte's step count 2.86; small conv init 3.76; zero conv blocks 2.71; 128×2 2.87 (505
  steps). So about 1.0 of the 1.17-bit gap is the FLOP charge, not information loss; there is no bug in the `ByteHNet`
  path (zero-block control), and causality and block norm/residual placement are clean. The remaining ~0.2 bits is within
  this budget's sensitivity: Byte itself moved 0.25 bits between 2.43e14 and 2.75e14.
- **Outer FLOP share at the brief's 256×4:** S: 43% (N = 1), 64% (N = 2), 79% (N = 4); M: 16%, 30%, 47%.
- **Wrapper changed to width 128, 2 blocks each side, for H-Net, Fixed, Entropy and Byte + conv** (Téo's instruction:
  a single design with outer share ≤ 45%, Cartesia's ratio; departs from the brief's 256×4). Share at S: 10% / 20% / 35%.
  The hierarchical search is re-run at 2.75e14 with this wrapper, the same budget as Byte's and BPE's. The 256×4 results
  are archived (`runs/lr_search_wrapper256.json`, `runs/archive_wrapper256/`).
- **Entropy overrun accepted by Téo:** its top-budget runs may exceed the 2 h limit (2.39 h estimated with 256×4;
  re-estimated after the rerun). Its small byte model's forward pass is uncharged wall-clock.

## Final pilot, ladder and open decisions (2026-10-31, supersedes the budgets and hours in "Final pilot and ladder")
- **Pilot, wrapper 128×2, 2.75e14 FLOPs, size S:** Byte 2.418, Fixed 2.719, Byte + conv 2.738, Entropy 3.624, H-Net 4.181,
  BPE 5.004; rates in `runs/lr_search.json`, all interior (H-Net's edge run 1.35e-2 lost). Numbers here are copied from
  that file for the log; the pre-registration reads them from it.
- **Outer FLOP share with 128×2** (`scripts/wrapper_share.py`): N = 1: 10% S, 3% M; N = 2: 20%, 6%; N = 4: 35%, 11%.
  All under 45%.
- **Ladder 1.58e14, 5e14, 1.58e15, 5e15; estimate 55.4 h** (`scripts/plan_ladder.py`). Real / pilot throughput: Byte .99,
  BPE .99, Fixed .97, Byte + conv .98, H-Net .87 S and .94 M, Entropy .46. H-Net at N = 4, size S now sets the top budget
  (2.00 h). Entropy's top run is 2.37 h, overrun accepted by Téo.
- **The search stayed at 2.75e14**, between the first and second budgets. Rates are not re-searched.
- **Every ladder run starts at micro-batch 8** (`runs/memory_test.json`: H-Net M dropped to 8 at step 6, swap growth 0,
  driver 6.2 GB). The S rates above were measured at micro-batch 16, so S estimates may run slightly low.
- **`check_flops.py` rerun at N = 2 with the new wrapper:** analytic within 11-32% of the counter, as before for Byte
  (1.32); the counter omits and adds different terms per network (note in `runs/flops_check.json`).

## H-Net vs Fixed diagnostics, no design change yet (2026-10-31)
Same search budget, size S, same validation episodes (`runs/wrapper_diag.json`). Fixed 2.72 at every comparison.
- **EMA.** H-Net without the EMA dechunk: 3.65 at 1.5e-3 (4.24 with it), 4.06 at 4.5e-3 (4.18 with it).
- **Positions.** RoPE counts chunks. Under irregular boundaries the data's W + 1 byte offset is not a constant number of
  chunks, the same failure the Byte audit found for absolute positions. RoPE on byte offset / N (`--rope-bytes`) is
  identical to the current code for regular boundaries (unit test), so Fixed and Byte + conv do not change, costs no FLOPs
  and leaves the router alone. Best of three rates: H-Net 3.45, Entropy 3.42, both at 1.5e-3.
- **Both.** H-Net with byte-offset RoPE and no EMA: 2.99 at 1.5e-3 (3.93 at 5e-4, 3.75 at 4.5e-3), compression 1.85. It
  stays behind Fixed on all 23 datasets, and ahead of Entropy by 0.43 bits.
- **Charge.** FLOPs use nominal N. With byte-offset RoPE realised compression falls to about 1.8, so H-Net uses more main
  positions than it is charged for.
- These flags are diagnostic only. Whether to adopt either is Téo's decision before the freeze.

## Téo's decision, final design and ladder (2026-10-31 ~18:30, supersedes the two sections above where they differ)
- **Téo: "A plus arm freeze".** Byte-offset positions for every chunked method; H-Net without EMA as a descriptive arm;
  EMA stays in the primary H-Net; freeze after updating the numbers and predictions, with no further check-in.
- **Search.** The diagnostic runs for H-Net, Entropy and H-Net no EMA moved to `runs/lr/` and were rescored by
  `scripts/lr_search.py`, matching the diagnostic values exactly. Best rate 1.5e-3 for all three, interior, so the edge rule
  added nothing; the old H-Net edge run is archived. The chunk-index search is `runs/lr_search_chunk_rope.json` and
  `runs/archive_chunk_rope/`; it fixes nothing.
- **Fixed and Byte + conv keep their search runs:** rescored with byte-offset positions, their validation bpb is identical
  (`runs/rope_bytes_check.json`).
- **H-Net throughput is now measured directly** (`scripts/timing_rope_bytes.py`, `runs/timing_rope_bytes.json`): 1e14
  FLOPs on real data at micro-batch 8, every H-Net arm. These runs are short and compression falls later in training: the
  2.75e14 search run took 340 s at micro-batch 16, 0.81 TFLOP/s against 0.93 here, so H-Net runs may take up to 15% longer
  than estimated.
- **Top budget capped at 5e15.** The measured rates alone would allow 5.8e15, but that puts Entropy at 2.75 h, past the
  2.39 h Téo accepted. `scripts/plan_ladder.py` now applies both limits. This also keeps the ladder Téo saw in check-in 2.
- **The memory check was not repeated** for the new positions: they add one small tensor per step, and the timing runs
  held micro-batch 8 with M's driver memory at 5.8 GB against 6.2 GB in the check.
- **Ratio target unchanged at N = 2,** so H-Net's realised compression of about 1.8 at the search budget is charged at 2.
  Stated in the pre-registration; the compression floor (1.24) is the existing guard.

## H3, H4 and analysis code: choices the frozen text leaves open (2026-10-31)
Written during the ladder, before any model saw a held-out robot. No hypothesis, metric or threshold changes.
- **H2 and H2b:** the line and the comparison use point estimates of g_ID and g_OOD; the CI requirement applies to g_OOD.
- **Compression floor:** mean over datasets of symbols / main positions on held-in test, as pooled bpb, averaged per episode
  over H-Net's three seeds. As written, it fails H1 and H2; it does not name H2b, so H2b is not affected.
- **Seed pairing at the top budget:** H-Net and BPE use three seeds; Fixed and the arms have seed 0 only, so H2b pairs
  H-Net's three-seed average with Fixed seed 0. BPE at the top budget is its three-seed average in §6.1.
- **Only frontier runs are scored on test,** for the no-EMA and N = 4 arms too, chosen by the same held-in validation rule.
- **H3 "each method" means the six main methods:** H-Net and BPE three seeds, others seed 0, 60 runs against the run list's
  "about 48". The no-EMA and N = 4 arms are not fine-tuned, as §6.1 and §6.3 do not ask for it.
- **"Keep the best" includes step 0,** so fine-tuning cannot be worse on validation than the pretrained model. All 500 steps
  run. Training loss, weight decay and clipping as in pretraining; rate constant at 0.3 × the run's rate.
- **Entropy fine-tuning:** its small byte model and threshold stay frozen; "all parameters" means the Entropy model's own.
- **H4 seeds:** for H-Net and BPE, seed s is the checkpoint and head seed; Byte has one checkpoint and three head seeds; the
  MLP has three initialisations. H4's CI resamples seeds and rollouts, paired on both.
- **"Every method below 5% or above 95%"** is read as all below 5% or all above 95%; a method on each side is informative.
  An uninformative task cannot pass. The four methods all count.
- **Head and MLP baseline (no setting in the text):** hidden width 256, AdamW 1e-3, batch 256, 2,000 steps, best of the
  robot's 10 validation episodes by MSE. Targets are the quantised action / 255; the MLP baseline reads the quantised
  observation / 255.
- **BPE in H4:** one word per timestep would let a token span the last observation bytes and the first action bytes, which
  leaks the target in training and cannot exist in a rollout. So H4 encodes observation and action bytes separately with the
  frozen vocabulary, in head training and rollouts alike, and keeps the last round(1,024 / R_BPE) tokens rather than 1,024
  bytes, the context BPE trained on.
- **Fetch environment:** `FetchPickAndPlace-v4` with a 150-step limit and goals on the table, matching the demonstrations
  (RoboChunk's generator). The stock 50-step limit is shorter than most demonstrations (45-146 steps), and stock goals in the
  air were never demonstrated. Demonstration object regions and physics changes are not reproduced: the stock reset is used.
- **Environment seeds** 300000-300099 for both tasks, unused by the demonstrations (Fetch's are 200000-200299). Success =
  `info["is_success"]` at the rollout's last step. PushT's limit is 300 steps; its demonstrations reach 246.
- **Compute multipliers (exploratory):** FLOPs BPE needs for H-Net's bpb at each H-Net budget, over H-Net's FLOPs, by
  log-log interpolation on the frontier; empty when the ladder does not reach that bpb.
- **Not implemented yet:** the other exploratory items (in-context merging, gripper-event alignment, 16-bit serialisation)
  and figures. `gym-pusht` and `pymunk` are not installed in the venv, so `scripts/policy.py --tasks pusht` cannot run yet.

## Clarifications approved, public push and PushT check (2026-10-31)
- **Téo approved the three choices logged above as written:** Fetch rollouts (150 steps, table goals), BPE in H4
  (observation and action bytes encoded separately), H3 over all six main methods. Commit `6ff6352` was pushed to the public
  repo on his instruction at 18:53:23 CET, before any test-set or held-out result exists.
- **PushT runs in a separate venv** outside the repo (`../robobytes-pusht-venv`, Python 3.12, no change to `.venv`), with
  `gym-pusht` 0.1.8 and `pymunk` 6.11.1 and the main venv's torch, numpy, gymnasium, tokenizers and pyarrow versions. Its
  mujoco differs from `.venv`, so Fetch rollouts stay in `.venv`. `cv2` and `pygame` each ship SDL: macOS prints duplicate-class
  warnings; nothing crashed.
- **The PushT mapping is verified** (`scripts/check_pusht_mapping.py`, `runs/pusht_mapping_check.json`; no model, pool episodes
  only, plus the success-flag column of the whole dataset). Open-loop replay of the recorded actions over 20 episodes matches
  the recordings to about 1e-5 px; swapped action axes and a one-step action delay miss by far more, so the check can fail.
  Environment reward and success match the recording on 1,391 frames, and live observations quantise to the stream's exact bytes.
- **Problem for H4 on PushT: no demonstration reaches the environment's success flag.** The flag needs coverage above 0.95. It
  is false on every frame of all 206 episodes, and replaying each pool episode's last action peaks at 0.935 coverage. A policy
  cloned from these 50 episodes is therefore unlikely to reach it, and §9's rule (every method below 5%) would make H4
  uninformative on PushT. The pre-registration is unchanged: success stays the environment's own flag.
