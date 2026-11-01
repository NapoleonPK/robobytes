"""RoboBytes-Ext1 Kaggle session: clones the public repo at the pinned commit, checks the byte streams, then runs one stage
with one job per GPU. Finished runs and checkpoints are carried from the previous session's output, so running a stage again
continues it. The session stops itself by SESSION seconds and always exits cleanly, because Kaggle passes on only the output
of a completed session. Stages: lr and lr_edge (the pre-freeze learning-rate check), checkpoints (hashes the check's models, which stay in
the output of their own sessions), calib, main, seed1, conv, score. Only `score` reads test splits or held-out streams."""
import glob, hashlib, json, os, queue, shutil, subprocess, sys, threading, time
from pathlib import Path

PIN = "@PIN@"
STAGES = "@STAGES@".split(",")
TEST = @TEST@  # tiny budgets; `score` uses held-in validation only, with three training datasets standing in for the held-out robots
TRAIN_SOURCE = @TRAIN@
DIGESTS = {"streams": ["df43e4483b56d576919382bd142396bc4a71858ba72e36aa1461ebd5f6ee2565", "78"],
           "bpe": ["e5c74f53d7b4d1f96eacec8369c9743c38f06b9c3a261ec2531dfe23d3db1000", "79"]}
ROOT, WORK = Path("/tmp/rb"), Path("/kaggle/working/ext1")
SESSION, RESERVE = @SESSION@, 30 if TEST else 300
METHODS = ("hnet", "fixed", "byte", "bpe")
MAC_RATE, CHECK_RATES = 1.5e-3, (3e-4, 7.5e-4, 1.5e-3)
EDGE_RATE, EDGE_RUNS = 3e-3, (("fixed", "M"), ("byte", "M"), ("bpe", "M"), ("byteconv", "S"))  # the configurations whose best rate was the top of the grid
BUDGETS = (2e13, 3e14) if TEST else (5e15, 5e16)
STOP = time.time() + SESSION


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout


def file_hash(path):
    f = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            f.update(block)
    return f.hexdigest()


def digest(root):
    root, h, n = Path(root), hashlib.sha256(), 0
    for p in sorted(x for x in root.rglob("*") if x.is_file() and not x.name.startswith("._")):
        h.update(f"{p.relative_to(root)} {file_hash(p)}\n".encode())
        n += 1
    return h.hexdigest(), n


def setup():
    sh(f"git clone -q https://github.com/NapoleonPK/robobytes.git {ROOT} && git -C {ROOT} checkout -q {PIN}")
    assert sh(f"git -C {ROOT} rev-parse HEAD").strip().startswith(PIN)
    for name in ("streams", "bpe"):
        src = Path(glob.glob(f"/kaggle/input/**/{name}/fmb", recursive=True)[0]).parent
        shutil.copytree(src, ROOT / "data" / name, ignore=shutil.ignore_patterns("._*"))
        got = [str(x) for x in digest(ROOT / "data" / name)]
        assert got == DIGESTS[name], (name, got)
    repo = ROOT / "scripts/ext1_train.py"
    if repo.exists():
        assert repo.read_text() == TRAIN_SOURCE, "kernel and pinned commit disagree"
    (ROOT / "ext1_train.py").write_text(TRAIN_SOURCE)
    WORK.mkdir(parents=True, exist_ok=True)
    for old in glob.glob("/kaggle/input/**/ext1/*/meta.json", recursive=True):  # carry the previous session forward
        d = Path(old).parent
        if not (WORK / d.name).exists() and not d.name.startswith("lr"):  # check runs are read where they are, to keep the output small
            shutil.copytree(d, WORK / d.name)
    for old in glob.glob("/kaggle/input/**/ext1/scores/*.json", recursive=True):
        (WORK / "scores").mkdir(exist_ok=True)
        shutil.copy(old, WORK / "scores")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))


def tag(method, size, budget, seed):
    return f"{method}_{budget:.0e}_{size}_s{seed}".replace("+", "")


def frontier(method):
    """Lower held-in validation bpb, seed 0 (§3)."""
    sizes = ("S", "M") if method == "byteconv" else ("M", "L")
    return min(sizes, key=lambda s: json.load(open(WORK / tag(method, s, BUDGETS[1], 0) / "val.json"))["val_bpb"])


def rate(method, size):
    """Per method and size from the pre-freeze check, fixed in prereg/ext1_rates.json at the pinned commit."""
    return json.load(open(ROOT / "prereg/ext1_rates.json"))[f"{method}_{size}"]


def anchors():
    """The §6 same-hardware anchor, fixed in prereg/ext1_anchor.json at the pinned commit."""
    return json.load(open(ROOT / "prereg/ext1_anchor.json"))


def run_path(name):
    return WORK / name if (WORK / name).exists() else Path(glob.glob(f"/kaggle/input/**/ext1/{name}", recursive=True)[0])


def name_of(job):
    return tag(*job[:4]) if job[4] is None else f"lr{job[4]:g}_{tag(*job[:4])}"


def jobs(stage):
    """(method, size, budget, seed, rate); rate is None except in the check, where it also names the run."""
    if stage == "lr":
        pairs = [(m, s) for m in METHODS for s in "ML"] + [("byteconv", s) for s in "SM"]
        return [(m, s, BUDGETS[0], 0, r) for m, s in pairs for r in CHECK_RATES]
    if stage == "lr_edge":
        return [(m, s, BUDGETS[0], 0, EDGE_RATE) for m, s in EDGE_RUNS]
    if stage == "calib":
        return [(m, s, BUDGETS[0], 0, None) for m in ("hnet", "fixed") for s in "SM"]
    if stage == "main":
        return [(m, s, BUDGETS[1], 0, None) for m in METHODS for s in "ML"]
    if stage == "seed1":
        return [(m, frontier(m), BUDGETS[1], 1, None) for m in METHODS]
    if stage == "conv":
        return [("byteconv", s, BUDGETS[1], 0, None) for s in "SM"]
    return ([(m, frontier(m), BUDGETS[1], s, None) for m in METHODS for s in (0, 1)] + [("byteconv", frontier("byteconv"), BUDGETS[1], 0, None)]
            + [(m, a["size"], BUDGETS[0], 0, a["rate"]) for m, a in anchors().items()])


def run_train(job, gpu, log):
    method, size, budget, seed, lr = job
    name, out = name_of(job), WORK / name_of(job)
    if lr is None:
        lr = MAC_RATE if budget == BUDGETS[0] else rate(method, size)  # calibration reruns the Mac runs at the Mac rate
    if (out / "val.json").exists():
        return
    left = STOP - time.time() - RESERVE
    if left < (5 if TEST else 60):
        log.append(dict(job=name, event="no time left"))
        return
    cmd = [sys.executable, "ext1_train.py", "--method", method, "--size", size, "--budget", str(budget), "--lr", str(lr), "--seed", str(seed),
           "--out", str(out), "--device", f"cuda:{gpu}", "--stop-after", str(left), "--ckpt-every", "120" if TEST else "1800", "--val-episodes", "5" if TEST else "100"]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out.parent / f"{name}.log").write_text((r.stdout + r.stderr)[-20000:])
    log.append(dict(job=name, gpu=gpu, seconds=time.time() - t, exit=r.returncode, done=(out / "val.json").exists()))


def run_score(job, gpu, log):
    name = name_of(job)
    out = WORK / "scores" / f"{name}.json"
    if out.exists():
        return
    r = subprocess.run([sys.executable, __file__, "--score", name, str(gpu)], capture_output=True, text=True)
    log.append(dict(job=name, gpu=gpu, exit=r.returncode, tail=(r.stdout + r.stderr)[-500:], done=out.exists()))


def score_one(name, gpu):
    """Per-episode tables of one run, same format as the main study's runs/scores (fp32)."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import torch
    from robobytes.data import training_names
    from robobytes.evaluate import load_run
    from robobytes.results import HELD_OUT, score_episodes
    from robobytes.serialize import Stream
    dev = torch.device(f"cuda:{gpu}")
    pinned = {a["run"]: a["sha256"] for a in anchors().values()}
    if name in pinned:
        assert file_hash(run_path(name) / "final.pt") == pinned[name], "not the pre-registered checkpoint"
    method, model, N, R, ent = load_run(run_path(name), dev)
    names = training_names()
    split = "val" if TEST else "test"
    targets = names[:3] if TEST else names + list(HELD_OUT)
    res = {n: score_episodes(method, model, n, split, N, R, dev, ent=ent, eps=Stream(n).episodes(split)[:5 if TEST else None]) for n in targets}
    (WORK / "scores").mkdir(exist_ok=True)
    tmp = WORK / "scores" / f"{name}.tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, WORK / "scores" / f"{name}.json")


def checkpoints(status):
    found = {Path(p).parent.name: p for p in glob.glob("/kaggle/input/**/ext1/lr*/final.pt", recursive=True) + glob.glob(str(WORK / "lr*/final.pt"))}
    status["checkpoints"] = {k: dict(sha256=file_hash(p), bytes=os.path.getsize(p)) for k, p in sorted(found.items())}
    json.dump(status["checkpoints"], open(WORK / "checkpoints.json", "w"), indent=1)


def run_stage(stage, status):
    if stage == "checkpoints":
        return checkpoints(status)
    import torch
    q = queue.Queue()
    for j in jobs(stage):
        q.put(j)
    log = status.setdefault(stage, [])
    work = run_score if stage == "score" else run_train

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            try:
                work(j, gpu, log)
            except Exception as e:  # the session must still end cleanly, or its output is lost to the next one
                log.append(dict(job=str(j), error=repr(e)))
            json.dump(status, open(WORK / "status.json", "w"), indent=1)

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(torch.cuda.device_count())]
    [t.start() for t in ts]
    [t.join() for t in ts]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--score":
        score_one(sys.argv[2], sys.argv[3])
        sys.exit(0)
    setup()
    status = dict(pin=PIN, stages=STAGES)
    for stage in STAGES:
        try:
            run_stage(stage, status)
        except Exception as e:
            status.setdefault("errors", []).append(dict(stage=stage, error=repr(e)))
    json.dump(status, open(WORK / "status.json", "w"), indent=1)
    print(json.dumps(status)[:3000], flush=True)
