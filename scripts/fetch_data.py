"""Download data/ and meta/ (for PAOP123/robochunk-sim: sim/*.npz and its card) of every pinned dataset (robobytes/sources.py) at its pinned revision into downloads/hub/.
Downloaded files are data, never executed. usage: PYTHONPATH=. python scripts/fetch_data.py"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from robobytes.sources import HELD_OUT, TRAIN

OUT = Path("downloads/hub")
for repo, rev, _ in TRAIN + HELD_OUT:
    p = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=["sim/*.npz", "README.md"] if repo == "PAOP123/robochunk-sim" else ["data/*", "meta/*"],
                          local_dir=OUT / repo)
    size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"{repo:48s} {rev[:10]} {size / 1e6:8.1f} MB", flush=True)
print("done", file=sys.stderr)
