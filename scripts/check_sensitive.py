"""Block sensitive content from entering this public repository.

Checks: personal email addresses (only no-reply addresses are allowed), API tokens and private keys, password
assignments, downloaded or generated data (downloads/, data/, parquet, npz, checkpoints, video), files over 5 MB, and
commit author/committer emails. Used by the git hooks (scripts/install_hooks.sh) and by CI.

usage: python scripts/check_sensitive.py --staged | --all | --msg FILE | --push   (exit 1 on any finding)
"""
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_EMAIL = re.compile(r"(@users\.noreply\.github\.com|^noreply@anthropic\.com|^noreply@github\.com|@example\.(com|org))$",
                           re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRETS = {
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "GitHub token": r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}",
    "Anthropic/OpenAI key": r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "password assignment": r"(?i)\b(password|passwd|api_key|secret_key|auth_token)\s*[:=]\s*['\"]?[^\s'\"]{6,}",
}
BAD_PATHS = re.compile(r"^(downloads|data)/|\.(parquet|npz|npy|pt|pth|ckpt|safetensors|mp4|avi|h5|pkl)$|(^|/)\.env$")
MAX_BYTES = 5 * 1024 * 1024
SELF = "scripts/check_sensitive.py"


def git(*a, data=False):
    out = subprocess.run(["git", *a], capture_output=True)
    return out.stdout if data else out.stdout.decode("utf-8", "replace")


def scan_text(name, text):
    found = []
    for m in EMAIL.finditer(text):
        if not ALLOWED_EMAIL.search(m.group(0)):
            found.append(f"{name}: email address")
            break
    if name != SELF:
        for label, pat in SECRETS.items():
            if re.search(pat, text):
                found.append(f"{name}: {label}")
    return found


def scan_files(paths, read):
    found = []
    for p in paths:
        if BAD_PATHS.search(p):
            found.append(f"{p}: data/checkpoint/download file must not be committed")
            continue
        blob = read(p)
        if blob is None:
            continue
        if len(blob) > MAX_BYTES:
            found.append(f"{p}: larger than 5 MB")
            continue
        if b"\0" in blob[:8000]:
            continue  # binary (images): size and path rules only
        found += scan_text(p, blob.decode("utf-8", "replace"))
    return found


def check_identity():
    found = []
    for key in ("user.email",):
        v = git("config", key).strip()
        if v and not ALLOWED_EMAIL.search(v):
            found.append(f"git config {key} is not a no-reply address")
    return found


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    found = []
    if mode == "--staged":
        paths = [p for p in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if p]
        found = scan_files(paths, lambda p: git("show", f":{p}", data=True)) + check_identity()
    elif mode == "--all":
        paths = [p for p in git("ls-files").splitlines() if p]
        found = scan_files(paths, lambda p: open(p, "rb").read() if Path(p).exists() else None)
    elif mode == "--msg":
        found = scan_text("commit message", open(sys.argv[2], encoding="utf-8").read())
    elif mode == "--push":  # stdin: <local ref> <local sha> <remote ref> <remote sha>
        for line in sys.stdin:
            lref, lsha, rref, rsha = line.split()
            if set(lsha) == {"0"}:
                continue
            rng = lsha if set(rsha) == {"0"} else f"{rsha}..{lsha}"
            for row in git("log", "--format=%H %ae %ce", rng).splitlines():
                h, ae, ce = row.split()
                for e in (ae, ce):
                    if not ALLOWED_EMAIL.search(e):
                        found.append(f"commit {h[:7]}: author/committer email is not a no-reply address")
                found += [f"commit {h[:7]} message: {x.split(': ', 1)[1]}"
                          for x in scan_text("m", git("log", "-1", "--format=%B", h))]
                files = [p for p in git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", h).splitlines() if p]
                found += [f"commit {h[:7]}: {x}" for x in
                          scan_files(files, lambda p, h=h: git("show", f"{h}:{p}", data=True) or None)]
    if found:
        print("check_sensitive: BLOCKED\n  " + "\n  ".join(sorted(set(found))), file=sys.stderr)
        sys.exit(1)
    print("check_sensitive: ok", file=sys.stderr)


if __name__ == "__main__":
    main()
