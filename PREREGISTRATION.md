# RoboBytes: pre-registration

**Status: DRAFT for review (not frozen).** Frozen = the commit tagged `prereg` in the public `robobytes` repo, whose SHA-256 is published the same hour as a GitHub release.

Independent study. Not affiliated with Cartesia, the H-Net authors (Hwang, Wang & Gu), the FAST authors or LeRobot / Hugging Face; it re-implements their published ideas and uses public LeRobot datasets. Follows RoboChunk (`NapoleonPK/robochunk-study`), whose router code it reuses.

## 1. Question

Cartesia reported that H-Net's advantage over BPE grows out of distribution (1.28× compute multiplier in-domain, median 3.10× on unseen programming languages). RoboBytes asks whether the same holds for **robots never seen in training**: each joint value is quantised to one byte, and models read the byte stream of trajectories.

## 2. Hypotheses and pass criteria

H1–H4 compare plain H-Net (primary) with BPE (baseline). **H2b** (co-primary with H2) compares H-Net with Fixed patches, which share every component except the learned router, so it isolates the router's contribution. Other methods are reported without pass/fail.

- **bpb:** cross-entropy of the next byte in bits, data bytes only (BOS and SEP predictions excluded), for every method.
- **Relative gap:** g = (bpb_BPE − bpb_HNet) / bpb_BPE; g > 0 means H-Net is better.
- **g_ID:** on the held-in test episodes (pooled, see §7). **g_OOD(r):** zero-shot on held-out robot r's test episodes.
- **Top budget:** 5e15 FLOPs. H-Net and BPE run 3 seeds there (0, 1, 2); per-episode bpb is averaged over seeds before any statistic.
- **CI:** 95% paired percentile bootstrap, 10,000 resamples of episodes, RNG seed 0.

| ID | Question | Pass criterion |
| --- | --- | --- |
| H1 | In distribution, is H-Net at least as compute-efficient as BPE? | At each of the two largest budgets (1.58e15 and 5e15), held-in test bpb of H-Net < BPE, and the CI of the difference excludes 0. |
| H2 (main) | Does the advantage grow on unseen robots? | At the top budget, for at least 2 of the 3 held-out robots: g_OOD(r) > 0 with CI excluding 0, **and** g_OOD(r) ≥ max(1.5 × \|g_ID\|, 0.02). |
| H2b (co-primary) | Does the **learned router's** advantage grow on unseen robots? | Same rule with g = (bpb_Fixed − bpb_HNet) / bpb_Fixed (both at N = round(R_BPE)): at the top budget, for at least 2 of the 3 held-out robots, g_OOD(r) > 0 with CI excluding 0 **and** g_OOD(r) ≥ max(1.5 × \|g_ID\|, 0.02). |
| H3 | Does H-Net adapt to a new robot better with few episodes? | After fine-tuning on the 50-episode pool of each held-out robot (§8), test bpb of H-Net < BPE for at least 2 of 3 robots, CI excluding 0. |
| H4 (underpowered) | Does better pretraining show up as better robot behaviour? | Policy success on PushT and Fetch (§9): H-Net − BPE > 0 with CI (over rollouts and seeds) excluding 0 on at least 1 of 2 tasks. |

**Compression floor.** If H-Net's realised compression at the top budget (held-in test) is below 0.8 × R_BPE = 1.239, the comparison is not matched and H1 and H2 count as failed.

**Exploratory (labelled as such):** compute multipliers from the iso-FLOP frontiers; the secondary N = 4 comparison (§6.1); whether chunk counts for a new robot shrink as an episode goes on (in-context merging); whether boundaries align with gripper events; 16-bit serialisation; how Byte and Entropy transfer.

### 2.1 Predictions stated in advance

Judgement from the held-in pilot only (search budget, size S, §6); no model has seen a held-out robot. Probabilities are subjective, and the report will score them.

- **Held-in order:** Byte best (2.42), then Fixed (2.72) and Byte + conv (2.74), then Entropy (3.62), H-Net (4.18), BPE (5.00). Gaps should narrow with budget.
- **H1: likely passes (85%).** Pilot g = 0.16 for H-Net over BPE. The risk is that BPE improves faster with budget.
- **H2: likely fails (35%).** A pilot g_ID of 0.16 would require g_OOD ≥ 0.25 on 2 of 3 robots. BPE barely compresses held-out robots, so it nears a byte model with a shorter context; the effect on it is unclear.
- **H2b: expected to fail (5%) unless the H-Net vs Fixed gap reverses out of distribution.** Pilot g_ID = -0.54. The rule uses |g_ID|, so H-Net being worse in distribution raises the bar: at that value g_OOD ≥ 0.81 is required, meaning H-Net below 0.19 × Fixed's bpb.
- **H3: likely passes (65%)** if H-Net leads BPE at zero shot, because BPE's vocabulary is frozen and it cannot form new tokens for a new robot.
- **H4: likely fails or is uninformative (20%).** Underpowered by design.
- **Compression floor: passes (95%).** Pilot H-Net compression 2.01 against a floor of 1.24.
- **N = 4:** H-Net and Fixed both worse than at N = 2, and H-Net further behind Fixed; refractory r = 1 changes little.

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
| H-Net | RoboChunk's router (LayerNorm, identity init), ratio loss α = 0.3 targeting N = 2; causal-conv encoder and decoder, 2 blocks each, width 128; EMA dechunk; next-byte softmax over 258 | Primary method |
| H-Net refractory | as H-Net, no new boundary within r positions of the last kept one, **r = floor(0.3 × N)** (r = 3 at RoboChunk's N = 10). At N = 2, r = 0: identical to H-Net, so the arm is not in the main ladder; it runs only in the N = 4 comparison (r = 1) | Exploratory |
| Fixed | H-Net architecture, a boundary every 2 positions | Baseline |
| Entropy | H-Net architecture, boundary where a small byte model's next-symbol entropy exceeds a threshold set to give compression 2 on training-robot validation windows; the small model is the Byte run at the second budget | Baseline |
| Byte | plain Transformer, one position per byte | Reference |
| Byte + conv | H-Net's conv encoder and decoder with the main Transformer over every byte (a boundary at every position): H-Net without chunking | Reference: separates the conv wrapper from chunking |

Main Transformer sizes: **S** = width 256, 4 layers; **M** = width 384, 8 layers (pre-norm, rotary positions, context 1,024 bytes). Only the main network changes with size.

**Baseline audit.** A no-learning reference, predicting each byte from the same channel at the previous timestep, gives 3.10 bpb on validation; it is told the channel layout. The first pilot used learned absolute positions, where Byte reached only 6.28. Rotary positions reached 2.43 (GPT-2 init alone 6.01; both 3.48), because the data's main structure is a relative offset. **Every method therefore uses rotary positions** (over bytes, tokens or chunks). The first search is archived and fixes nothing.

**Wrapper diagnostic.** With the brief's wrapper (width 256, 4 blocks) Byte + conv scored 3.85 against Byte's 2.67 at equal FLOPs (307 against 578 steps). Given Byte's step count it reached 2.86, and with zero conv blocks 2.71: the loss is mostly the FLOP charge, not a defect. Causality and residual and norm placement were checked. Outer-network share of FLOPs (Cartesia's ratio: 45%):

| Method | 256×4, S | 256×4, M | 128×2, S | 128×2, M |
| --- | --- | --- | --- | --- |
| Byte + conv (N = 1) | 43% | 16% | 10% | 3% |
| H-Net, Fixed, Entropy (N = 2) | 64% | 30% | 20% | 6% |
| H-Net, Fixed (N = 4) | 79% | 47% | 35% | 11% |

**H-Net, Fixed, Entropy and Byte + conv therefore use width 128, 2 blocks** (departure from the brief, on Téo's instruction). The 256×4 search is archived and fixes nothing.

**H-Net diagnostics.** At the search budget, size S, Fixed reaches 2.72. H-Net without the EMA dechunk reached 3.65; with rotary positions on byte offset / N instead of chunk index 3.45 (Entropy 3.42); with both 2.99, all at 1.5e-3. Byte-offset positions leave regular boundaries unchanged. Pending Téo's decision; neither is in this draft's design.

## 6. Compute ladder and training

Budgets **1.58e14, 5e14, 1.58e15, 5e15 FLOPs** (×3.16), sizes S and M at every budget, 6 methods: 48 ladder runs, ≈ 21 h at the measured throughput; longest run 1.58 h, except Entropy (up to 2.37 h, because its small byte model's forward pass is wall-clock time not charged to its budget; Téo accepted this overrun). FLOPs = 6 × parameters used per position + 6 × layers × context × width, per network, per symbol, charged at nominal N (realised compression is reported for every run). At most 0.76 passes over the 458,630,486 training symbols, so no budget is dropped. At each budget the size with the lower held-in validation bpb is the method's frontier run; only frontier runs enter the tests.

Optimiser: AdamW (0.9, 0.95), weight decay 0.1, 2% linear warm-up, cosine decay to 10%, gradient clip 1.0, effective batch 16 windows, fp32. Training loss: next-symbol cross-entropy on every target (+ 0.3 × ratio loss for H-Net).

**Learning rates** (pilot search at 2.75e14 FLOPs, size S, 3 values per method, held-in validation only; a method whose best rate was at an end of the grid got one more rate ×3 beyond it, once): byte 0.0015, bpe 0.0015, hnet 0.0045, fixed 0.0015, entropy 0.0015, byteconv 0.0015. Fixed for every later run.

**Known before the freeze.** Held-in validation at the search budget, size S, each method's best rate (first 20 validation episodes per dataset). These pilot numbers motivated H2b and the N = 4 arm; held-out robots have not been evaluated by any model:

| Method | Rate | Validation bpb |
| --- | --- | --- |
| byte | 0.0015 | 2.418 |
| bpe | 0.0015 | 5.004 |
| hnet | 0.0045 | 4.181 |
| fixed | 0.0015 | 2.719 |
| entropy | 0.0015 | 3.624 |
| byteconv | 0.0015 | 2.738 |

The search ran at 2.75e14 FLOPs, set before the ladder was lowered to account for real training speed; the smallest budget is 1.58e14. Rates are not re-searched.

Extra runs: H-Net and BPE at the top budget with seeds 1 and 2 (frontier size).

### 6.1 Secondary comparison at N = 4 (pre-registered, no pass/fail)

N = round(R_BPE) = 2 barely exercises chunking, so H-Net and Fixed are also trained with **N = 4** across the full ladder (4 budgets × sizes S and M), plus H-Net refractory at N = 4 with r = 1 at the top two budgets (H-Net N = 4's frontier size). Same data, windows, optimiser and seeds as the main ladder; learning rate = the same method's rate from the N = 2 search; Fixed places a boundary every 4 positions. Each is compared with BPE's frontier at matched FLOPs on held-in data and on each held-out robot (bpb, relative gap g with 95% bootstrap CIs, iso-FLOP frontiers). Reported descriptively; it does not enter H1–H4.

### 6.2 Router effect at N = 4 (no pass/fail)

The H2b statistic (H-Net vs Fixed: g_ID, g_OOD per robot, CIs, the 1.5 × |g_ID| and 0.02 lines) is also computed for the N = 4 arm of §6.1 and reported descriptively.

### 6.3 Run list and machine time

| Block | Runs | Hours | Basis |
| --- | --- | --- | --- |
| Learning-rate search (6 methods x 3 rates, + edge rule), size S, smallest budget | 18-24 | 1.5 | measured |
| Main ladder: BPE, H-Net, Fixed, Entropy, Byte, Byte + conv x 4 budgets x 2 sizes | 48 | 21.3 | measured throughput |
| Extra seeds at the top budget: H-Net and BPE, frontier size, seeds 1-2 | 4 | 6.3 | measured throughput (slower size) |
| Secondary, N = 4: H-Net and Fixed x 4 budgets x 2 sizes | 16 | 7.7 | measured throughput |
| Secondary, N = 4: H-Net refractory (r = 1), top two budgets | 2 | 2.6 | measured throughput |
| Few-shot fine-tuning (H3) | about 48 | 4.0 | brief's estimate |
| Policies and rollouts (H4) | about 24 | 7.0 | brief's estimate |
| Evaluation (validation for every run; test for frontier runs) | - | 5.0 | estimate: forward passes over validation and test windows |
| **Total** | | **55.4** | cap 72 h |

**Memory check before the ladder** (H-Net, size M, top budget, N = round(R_BPE), micro-batch 16 at start, 10 minutes): peak MPS allocation 0.24 GB, swap growth 0 MB. The check ended at micro-batch 8. Every ladder run starts at micro-batch 8 with accumulation. The in-run rule (halve the micro-batch if swap grows by more than 2 GB) stays.

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
- Runs may halve their micro-batch on out-of-memory or swap growth above 2 GB, keeping the effective batch; every event is logged.

## 11. Files fixed by this document

- `robobytes/sources.py`: SHA-256 `de5b404407fdb46ac0888e05a7836f8b32ad7196f35b74d64a5f75bd61ed8794`
- `runs/streams_summary.json`: SHA-256 `1d484c2efc174cc8687d4db88e7674912c7fe78266c366b5da62b63003fbc11e`
- `runs/bpe_summary.json`: SHA-256 `4263428fc9597ca928777eeff48462b68e927e872f861c14eff93a7014fdf8f6`
- `runs/ladder.json`: SHA-256 `082fcbf9a40a0f54a8a389e8e78a8588c25f769c4a5238740e73ec4eb890629a`
- `runs/baseline_audit.json`: SHA-256 `db4ce3b1ca7f5d2163dacd594aeaa5a7476ee528ba0b774b961f9ae4fc911656`
- `runs/delta_reference.json`: SHA-256 `3c04f346b4b33e77164606f7fa57671e7efbd80379560d189fb2fa7d38ebfe58`
- `runs/wrapper_diag.json`: SHA-256 `2f66df4c78e1d8d06d1dc132a3b925fe043669f35b93b08b8181115e243aa789`
- `runs/wrapper_share.json`: SHA-256 `2a92440d4ee845775ad658fe36d0714482595a852e9c9ed320720f67a4e8a551`
- `prereg/splits.json`: SHA-256 `ef569f70762b168cdccd5649455b0050a10e53bf5c556068bd34391b279a1c77`
- `runs/lr_search.json`: SHA-256 `59a9aa53901459fc41a0f205b8345a40cb6ec180fd7f2e9cf5b7513b3e59fb77`
