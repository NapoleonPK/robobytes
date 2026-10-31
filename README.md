# RoboBytes

Independent study, not affiliated with Cartesia, the H-Net authors (Hwang, Wang & Gu), the FAST authors or LeRobot /
Hugging Face. It re-implements their published ideas on public LeRobot datasets.

**Question.** Cartesia found that learned (H-Net) chunking beats BPE by more on programming languages a model never saw
than on familiar ones. RoboBytes tests whether the same holds for **robots never seen in training**: joint values are
quantised to one byte each, and byte-level models (BPE, H-Net, fixed patches, entropy patches, plain bytes) are compared
at matched compute on 12 training robot types and 3 held-out robots (ALOHA, Fetch, PushT).

**Status: pre-registered, results pending.** The hypotheses, data, splits, ladder and thresholds are fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md). The frozen version is the commit tagged `prereg`; its SHA-256 is published in
the GitHub release of the same name. Verify it with:

```sh
git checkout prereg && shasum -a 256 PREREGISTRATION.md
```

The files it fixes (dataset revisions, episode lists, BPE and ladder measurements) are hashed inside it. Choices the
pre-registration leaves open are logged with reasons in [`DECISIONS.md`](DECISIONS.md). Results, report and figures will
be added after the runs, without rewriting this history.

**Layout.** `robobytes/`: models (H-Net router from [RoboChunk](https://github.com/NapoleonPK/robochunk-study)), byte
serialisation, BPE, training and evaluation. `scripts/`: data download at pinned revisions, stream building, BPE,
throughput pilot, ladder plan, learning-rate search, pre-registration generator. `runs/`: pilot measurements the
pre-registration cites. Data are not redistributed; `scripts/fetch_data.py` and `scripts/build_streams.py` rebuild them.

**Credits.** H-Net: Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv
2507.07955). Cartesia, *From Tokens to Bytes* I and II. FAST: Pertsch et al. (arXiv 2501.09747). BLT: Pagnoni et al.
(arXiv 2412.09871). Datasets: LeRobot and the Open X-Embodiment contributors, under each dataset's licence.

Licence: see [`LICENSE`](LICENSE).
