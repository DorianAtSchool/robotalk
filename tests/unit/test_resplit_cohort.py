"""Regrouping the evaluation cohort under another task split."""

import json
from pathlib import Path

from robotalk.evaluation.resplit_cohort import resplit

ROOT = Path(__file__).resolve().parents[2]
COHORT = json.loads((ROOT / "configs/eval/fixed_live_sim_cohort.json").read_text())
SPLIT = json.loads((ROOT / "configs/splits/43_train_10_heldout.json").read_text())


def test_the_paper_split_gives_back_the_paper_cohort():
    assert resplit(COHORT, SPLIT)["configurations"] == COHORT["configurations"]


def test_a_moved_task_keeps_its_configurations():
    moved = SPLIT["train_tasks"][0]
    split = {
        "train_tasks": SPLIT["train_tasks"][1:],
        "held_out_tasks": SPLIT["held_out_tasks"] + [moved],
    }
    out = resplit(COHORT, split)["configurations"]
    assert len(out["train_task_types"]) == 42 and len(out["heldout_task_types"]) == 11
    assert out["heldout_task_types"][moved] == COHORT["configurations"]["train_task_types"][moved]
