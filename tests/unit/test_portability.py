"""The repository must not depend on the machine it was developed on."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cluster paths, accounts and personal addresses.
FORBIDDEN = re.compile(
    r"/work/umass|/scratch/|/home/[a-z_]+_umass|@umass\.edu|gmail\.com"
    r"|--account=|--partition=|--qos="
)
# Internal planning documents that are not part of the public release, and
# this file.
INTERNAL = {
    "docs/publication_readiness_plan.md",
    "docs/release_manifest.md",
    "docs/robotalk_project_and_implementation.md",
    "tests/unit/test_portability.py",
}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [name for name in out.split("\0") if name and name not in INTERNAL]


def test_no_cluster_paths_accounts_or_personal_addresses():
    hits = []
    for name in _tracked_files():
        path = REPO_ROOT / name
        if path.is_symlink():
            # Absolute links point into the machine they were made on.
            target = os.readlink(path)
            if os.path.isabs(target) or FORBIDDEN.search(target):
                hits.append(f"{name}: symlink to {target}")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append(f"{name}:{number}: {line.strip()[:120]}")
    assert not hits, "machine-specific strings:\n" + "\n".join(hits[:50])
