"""Survey LeRobot datasets from metadata only (meta/info.json + the Hub's card licence). Downloaded files are data,
kept under downloads/meta/. Writes runs/dataset_survey.csv. usage: python -I scripts/survey_datasets.py"""
import csv
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OUT = Path("downloads/meta")
api = HfApi()
rows = []
names = sorted(d.id for d in api.list_datasets(author="lerobot", limit=2000))
print(len(names), "lerobot datasets", file=sys.stderr)
for rid in names:
    try:
        info = json.load(open(hf_hub_download(rid, "meta/info.json", repo_type="dataset", local_dir=OUT / rid)))
    except Exception as e:  # no v2/v3 metadata, gated, etc.
        rows.append(dict(repo=rid, error=type(e).__name__))
        continue
    feats = info.get("features", {})
    dim = lambda k: (feats.get(k, {}).get("shape") or [0])[0]
    img = [k for k, v in feats.items() if v.get("dtype") == "image"]
    try:
        card = api.dataset_info(rid).card_data
        lic = (card or {}).get("license") if not hasattr(card, "license") else card.license
    except Exception:
        lic = None
    rows.append(dict(repo=rid, version=info.get("codebase_version"), robot=info.get("robot_type"),
                     episodes=info.get("total_episodes"), frames=info.get("total_frames"), fps=info.get("fps"),
                     state=dim("observation.state"), env_state=dim("observation.environment_state"),
                     action=dim("action"), images_in_parquet=len(img), license=lic))
keys = ["repo", "version", "robot", "episodes", "frames", "fps", "state", "env_state", "action", "images_in_parquet",
        "license", "error"]
with open("runs/dataset_survey.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote runs/dataset_survey.csv", len(rows), file=sys.stderr)
