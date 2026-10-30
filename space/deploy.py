"""Push the current commit to the Hugging Face Space (Docker SDK).

Usage: .venv/bin/python space/deploy.py [owner/space]   (after `huggingface-cli login` with a write token)
Only tracked files of HEAD are sent, so .env, cache/ and samples/ never leave this machine.
"""

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REPO = sys.argv[1] if len(sys.argv) > 1 else "adamd-dev/physicslens"
SKIP = ["samples/**", "tests/**", "graphify-out/**", ".claude/**", "web/landing-public/**", "runs/**"]

with tempfile.TemporaryDirectory() as tmp:
    archive = Path(tmp) / "head.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
    out = Path(tmp) / "space"
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    (out / "README.md").write_text((ROOT / "space" / "README.md").read_text())  # the Space card, not the GitHub README
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    api = HfApi()
    api.create_repo(REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    # The Space mirrors HEAD: files gone from the repo go from the Space too.
    api.upload_folder(folder_path=out, repo_id=REPO, repo_type="space", ignore_patterns=SKIP, delete_patterns=["*"], commit_message=f"deploy {head}")
    print(f"pushed to https://huggingface.co/spaces/{REPO}")
