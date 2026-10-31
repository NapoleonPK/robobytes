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
