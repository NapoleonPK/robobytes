# RoboBytes: pre-registration

**Status: DRAFT for review (not frozen).** Frozen = the commit tagged `prereg` in the public `robobytes` repo, whose SHA-256 is published the same hour as a GitHub release.

Independent study. Not affiliated with Cartesia, the H-Net authors (Hwang, Wang & Gu), the FAST authors or LeRobot / Hugging Face; it re-implements their published ideas and uses public LeRobot datasets. Follows RoboChunk (`NapoleonPK/robochunk-study`), whose router code it reuses.

## 1. Question

Cartesia reported that H-Net's advantage over BPE grows out of distribution (1.28× compute multiplier in-domain, median 3.10× on unseen programming languages). RoboBytes asks whether the same holds for **robots never seen in training**: each joint value is quantised to one byte, and models read the byte stream of trajectories.

## 2. Hypotheses and pass criteria

All four tests compare plain H-Net (primary) with BPE (baseline). Other methods are reported without pass/fail.

- **bpb:** cross-entropy of the next byte in bits, data bytes only (BOS and SEP predictions excluded), for every method.
- **Relative gap:** g = (bpb_BPE − bpb_HNet) / bpb_BPE; g > 0 means H-Net is better.
- **g_ID:** on the held-in test episodes (pooled, see §7). **g_OOD(r):** zero-shot on held-out robot r's test episodes.
- **Top budget:** 8.7e15 FLOPs. H-Net and BPE run 3 seeds there (0, 1, 2); per-episode bpb is averaged over seeds before any statistic.
- **CI:** 95% paired percentile bootstrap, 10,000 resamples of episodes, RNG seed 0.

| ID | Question | Pass criterion |
| --- | --- | --- |
| H1 | In distribution, is H-Net at least as compute-efficient as BPE? | At each of the two largest budgets (2.75e15 and 8.7e15), held-in test bpb of H-Net < BPE, and the CI of the difference excludes 0. |
| H2 (main) | Does the advantage grow on unseen robots? | At the top budget, for at least 2 of the 3 held-out robots: g_OOD(r) > 0 with CI excluding 0, **and** g_OOD(r) ≥ max(1.5 × \|g_ID\|, 0.02). |
| H3 | Does H-Net adapt to a new robot better with few episodes? | After fine-tuning on the 50-episode pool of each held-out robot (§8), test bpb of H-Net < BPE for at least 2 of 3 robots, CI excluding 0. |
| H4 (underpowered) | Does better pretraining show up as better robot behaviour? | Policy success on PushT and Fetch (§9): H-Net − BPE > 0 with CI (over rollouts and seeds) excluding 0 on at least 1 of 2 tasks. |

**Compression floor.** If H-Net's realised compression at the top budget (held-in test) is below 0.8 × R_BPE = 1.239, the comparison is not matched and H1 and H2 count as failed.

**Exploratory (labelled as such):** compute multipliers from the iso-FLOP frontiers; the refractory router as a second H-Net arm; whether chunk counts for a new robot shrink as an episode goes on (in-context merging); whether boundaries align with gripper events; 16-bit serialisation; how Byte and Entropy transfer.

## 3. Data

**Training robots:** 23 LeRobot datasets, **12 robot types** counted conservatively (DLR SARA, Fanuc Mate, Franka, Hello Robot Stretch, Jaco 2, Koch v1.1, Kuka iiwa, PR2, SO-100/101, Sawyer, UR5, xArm; Franka arms including the mobile PandaOmron count as one, SO-100 and SO-101 as one, simulated and real xArm as one). State and action only, `data/` and `meta/` downloaded at the pinned revision; no code from any dataset repo is run.

| Dataset | Revision | Robot | Channels (W) | fps | Episodes train / val / test | Training bytes |
| --- | --- | --- | --- | --- | --- | --- |
| `lerobot/robocasa_target_human_unified` | `a1c7cba1a128` | PandaOmron (Franka on mobile base) | 28 | 20 | 22777 / 1265 / 1265 | 376,601,876 |
| `lerobot/vlabench_unified` | `f61ad898199e` | Franka | 14 | 10 | 9879 / 549 / 549 | 39,341,470 |
| `lerobot/toto` | `51f5a8dc59fe` | Franka | 15 | 30 | 903 / 50 / 50 | 4,392,060 |
| `lerobot/fmb` | `76c86f111d32` | Franka | 15 | 10 | 1624 / 90 / 90 | 4,578,570 |
| `lerobot/utaustin_mutex` | `a880eaeec88c` | Franka | 15 | 20 | 1350 / 75 / 75 | 4,866,885 |
| `lerobot/svla_so100_pickplace` | `728583b5eaf9` | SO-100 | 12 | 30 | 46 / 2 / 2 | 213,264 |
| `lerobot/svla_so100_sorting` | `13870ca96908` | SO-100 | 12 | 30 | 46 / 3 / 3 | 378,516 |
| `lerobot/svla_so100_stacking` | `933bcc61148b` | SO-100 | 12 | 30 | 50 / 3 / 3 | 243,924 |
| `lerobot/svla_so101_pickplace` | `f641879e2217` | SO-101 | 12 | 30 | 46 / 2 / 2 | 130,752 |
| `lerobot/koch_pick_place_1_lego` | `afc0b3e54679` | Koch v1.1 | 12 | 30 | 92 / 5 / 5 | 432,576 |
| `lerobot/koch_pick_place_5_lego` | `af1f4ff76be9` | Koch v1.1 | 12 | 30 | 46 / 2 / 2 | 419,556 |
| `lerobot/xarm_lift_medium` | `79efb0e3cef0` | xArm (sim) | 8 | 15 | 720 / 40 / 40 | 144,000 |
| `lerobot/xarm_push_medium` | `50352b1fd09f` | xArm (sim) | 7 | 15 | 720 / 40 / 40 | 126,000 |
| `lerobot/utokyo_xarm_pick_and_place` | `da56c23e1b6d` | xArm | 15 | 10 | 92 / 5 / 5 | 101,610 |
| `lerobot/berkeley_autolab_ur5` | `c4e26a697fc4` | UR5 | 15 | 5 | 900 / 50 / 50 | 1,320,660 |
| `lerobot/jaco_play` | `265773f4d246` | Jaco 2 | 15 | 10 | 977 / 54 / 54 | 1,057,095 |
| `lerobot/stanford_kuka_multimodal_dataset` | `93927d10c072` | Kuka iiwa | 14 | 20 | 2700 / 150 / 150 | 1,889,790 |
| `lerobot/roboturk` | `d38c91900282` | Sawyer | 15 | 10 | 1795 / 100 / 100 | 2,517,690 |
| `lerobot/berkeley_fanuc_manipulation` | `51bdefb6c7f2` | Fanuc Mate | 15 | 10 | 373 / 21 / 21 | 843,510 |
| `lerobot/cmu_stretch` | `d84574233973` | Hello Robot Stretch | 12 | 5 | 121 / 7 / 7 | 268,812 |
| `lerobot/utokyo_pr2_opening_fridge` | `47504c67033d` | PR2 | 15 | 5 | 72 / 4 / 4 | 154,965 |
| `lerobot/utokyo_pr2_tabletop_manipulation` | `a0b6918864b3` | PR2 | 15 | 5 | 216 / 12 / 12 | 441,090 |
| `lerobot/dlr_sara_pour` | `4246fa6111ab` | DLR SARA | 13 | 5 | 90 / 5 / 5 | 151,788 |

**Excluded:** every bimanual dataset (ALOHA's held-out type), every PushT and Fetch dataset, datasets over 2 GB of `data/` (e.g. `droid_1.0.1`), datasets without a research licence, datasets with images inside the parquet.

**Held-out robots** (fixed; never in training, never in the BPE or entropy-model training data):

| Robot | Data (revision) | Channels | fps | Few-shot pool / val / zero-shot test |
| --- | --- | --- | --- | --- |
| ALOHA, bimanual (14 joints): two arms, twice the joints of any training robot | `lerobot/aloha_sim_insertion_human` (`cc571a3c661d`), `lerobot/aloha_sim_transfer_cube_human` (`6a43d500f101`) | 28 | 50 | 50 / 10 / 40 |
| Fetch, end-effector deltas + gripper (4-D action): unseen body and simulator | `PAOP123/robochunk-sim` (`630a3a3737ae`) | 32 | 25 | 50 / 10 / 240 |
| PushT, 2-D point pusher: not an arm | `lerobot/pusht_keypoints` (`ace8c161a68b`) | 20 | 10 | 50 / 10 / 146 |

ALOHA pools its two simulated tasks into one robot. Fetch uses RoboChunk's held-out clean simulated test set (`sim/test.npz`; observation 25 + goal 3, then action 4 per step). PushT uses `lerobot/pusht_keypoints` (agent position 2 + 16-D block keypoints, then action 2).

**Episode lists** are fixed in `prereg/splits.json` (SHA-256 `ef569f70762b168cdccd5649455b0050a10e53bf5c556068bd34391b279a1c77`): per dataset, the episode ids of each split in seed-0 order. For held-out robots the pool is ordered: its first 10 episodes are the 10-shot subset.

## 4. Byte serialisation (identical for every method)

1. Per dataset and channel, clip to the 1st–99th percentile of the training split (held-out robots: of the few-shot pool only) and scale linearly to 0–255 (`uint8`). A channel whose percentiles coincide maps to 0.
2. Stream per episode: `BOS`, then per timestep the state bytes (then environment state, if any), the action bytes in native channel order, and `SEP`. Vocabulary 258.
3. No robot ID, no channel-count header. Native frame rates; no resampling.
4. Splits: training datasets 90 / 5 / 5 by episode (seed 0); held-out robots 50 pool / 10 val / rest test (seed 0).
5. Training windows are drawn per dataset with probability ∝ √(training bytes).

## 5. Methods

**BPE:** Hugging Face `tokenizers` BPE, vocabulary 4,096, trained on a 30 MB square-root-mixture sample of training rows; one word per timestep so merges never cross `SEP`; `BOS`/`SEP` atomic. **R_BPE = 1.549** symbols per position on the training mixture (pooled over all training rows: 1.721). Held-out robots under the frozen vocabulary: ALOHA 1.055, Fetch 1.165, PushT 1.061. Because R_BPE < 2, a 16,384 vocabulary was tried (brief's risk table): R = 1.776, still below 2, so the vocabulary stays 4,096 and **N = 2** (= round(R_BPE)) for every chunked method.

| Method | Grouping | Role |
| --- | --- | --- |
| BPE | as above; context round(1,024 / R_BPE) tokens | Primary baseline |
| H-Net | RoboChunk's router (LayerNorm, identity init), ratio loss α = 0.3 targeting N = 2; causal-conv encoder and decoder, 4 blocks each, width 256; EMA dechunk; next-byte softmax over 258 | Primary method |
| H-Net refractory | as H-Net, no boundary within 3 positions of the last kept one | Exploratory |
| Fixed | H-Net architecture, a boundary every 2 positions | Baseline |
| Entropy | H-Net architecture, boundary where a small byte model's next-symbol entropy exceeds a threshold set to give compression 2 on training-robot validation windows; the small model is the Byte run at the second budget | Baseline |
| Byte | plain Transformer, one position per byte | Reference |

Main Transformer sizes: **S** = width 256, 4 layers; **M** = width 384, 8 layers (pre-norm, learned positions, context 1,024 bytes). Only the main network changes with size.

## 6. Compute ladder and training

Budgets **2.75e14, 8.7e14, 2.75e15, 8.7e15 FLOPs** (×3.16), sizes S and M at every budget, 5 methods: 40 ladder runs, ≈ 22 h at the measured throughput; longest run 1.90 h. FLOPs = 6 × parameters used per position + 6 × layers × context × width, per network, per symbol. At most 1.00 passes over the 458,630,486 training symbols, so no budget is dropped. At each budget the size with the lower held-in validation bpb is the method's frontier run; only frontier runs enter the tests.

Optimiser: AdamW (0.9, 0.95), weight decay 0.1, 2% linear warm-up, cosine decay to 10%, gradient clip 1.0, effective batch 16 windows, fp32. Training loss: next-symbol cross-entropy on every target (+ 0.3 × ratio loss for H-Net).

**Learning rates:** pending (search running: 5e-4, 1.5e-3, 4.5e-3 per method at the smallest budget).

Extra runs: H-Net and BPE at the top budget with seeds 1 and 2 (frontier size); H-Net refractory at the top two budgets (H-Net's frontier size).

## 7. Metrics and statistics

- **Windows:** each episode in 1,024-position windows (BPE: round(1,024 / R_BPE) tokens), stride half a window, scoring the second half of each window and all of the first.
- **Held-in bpb:** per dataset Σ bits / Σ data bytes over its test episodes; pooled = mean over the training datasets (each weighted equally). Bootstrap resamples episodes within each dataset.
- **Held-out bpb:** per robot, Σ bits / Σ data bytes over its zero-shot test episodes; bootstrap over episodes.
- **Paired:** both methods are scored on the same resampled episodes; for g, numerator and denominator are bootstrapped together.
- **Held-in validation** (frontier choice): the first 100 validation episodes of each dataset (all, if fewer).
- **Realised compression:** symbols per main-network position, for every run.

## 8. Few-shot adaptation (H3)

Start from each method's top-budget frontier checkpoint (H-Net and BPE: seeds 0–2). Fine-tune all parameters on the 10-shot and the 50-shot pool of each held-out robot: AdamW at 0.3 × the method's pretraining learning rate, constant, effective batch 16 windows, at most 500 steps, validation bpb on the robot's 10 validation episodes every 50 steps, keep the best. BPE's vocabulary is not retrained. Measure test bpb; H3 uses 50-shot.

## 9. Policy success (H4, underpowered)

Simulators: `gym-pusht` (observation `environment_state_agent_pos`) and `FetchPickAndPlace-v4` from `gymnasium-robotics` (observation + goal). Freeze the 50-shot model from H3; add a 2-layer MLP head reading the hidden state at the last observation byte of each step and predicting the continuous action; train only the head on the same 50 episodes with MSE. Rollout: keep the last 1,024 bytes of history; each step append observation bytes, read the action, execute it, append the executed action's bytes. Compare H-Net, BPE, Byte and an MLP behaviour-cloning lower bound, 3 seeds × 100 rollouts on fixed environment seeds; success is the environment's own flag. If every method is below 5% or above 95% on a task, H4 is uninformative for that task.

## 10. Rules

- Hypotheses, held-out robots, metrics and thresholds do not change after the freeze; any deviation is a dated amendment with its reason.
- Every reported number comes from one analysis script run on the frozen results.
- One training job at a time; checkpoints every 10 minutes.

## 11. Files fixed by this document

- `robobytes/sources.py`: SHA-256 `de5b404407fdb46ac0888e05a7836f8b32ad7196f35b74d64a5f75bd61ed8794`
- `runs/streams_summary.json`: SHA-256 `1d484c2efc174cc8687d4db88e7674912c7fe78266c366b5da62b63003fbc11e`
- `runs/bpe_summary.json`: SHA-256 `4263428fc9597ca928777eeff48462b68e927e872f861c14eff93a7014fdf8f6`
- `runs/ladder.json`: SHA-256 `7147e39fda23921b1d5b793b33844d8759b7391acd10bd517c66786aee0823de`
- `prereg/splits.json`: SHA-256 `ef569f70762b168cdccd5649455b0050a10e53bf5c556068bd34391b279a1c77`
