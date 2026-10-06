"""Fig. 3 (task phases) rebuilds from the repository alone and matches the paper."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase_counts_match_the_paper():
    from robotalk.analysis.export_task_phase_pngs import _load_counts

    _, by_split = _load_counts()
    assert dict(by_split["In-training tasks"]) == {"Phase 2": 12, "Phase 3": 12, "Phase 4": 19}
    assert dict(by_split["Held-out tasks"]) == {"Phase 2": 2, "Phase 3": 2, "Phase 4": 6}


def test_fig3_renders_into_an_empty_output_root(tmp_path):
    env = {**os.environ, "ROBOTALK_OUTPUT_ROOT": str(tmp_path), "MPLBACKEND": "Agg"}
    subprocess.run(
        [sys.executable, "-m", "robotalk.analysis.export_task_phase_pngs"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, timeout=300,
    )
    out = tmp_path / "figures" / "43_10"
    assert (out / "task_phase_distribution_by_split.png").stat().st_size > 0
    assert (out / "task_phase_distribution_combined.png").stat().st_size > 0
