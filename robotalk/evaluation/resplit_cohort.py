"""Regroup the evaluation cohort's tasks under a different train/held-out split.

The cohort lists each task's evaluation configurations under one of two
splits, `train_task_types` and `heldout_task_types`. Configurations do not
depend on the split, so a new split only moves tasks between the two groups:

    python -m robotalk.evaluation.resplit_cohort \
        --task-split my_split.json --output configs/eval/my_cohort.json

`my_split.json` has the format of `configs/splits/43_train_10_heldout.json`:
`{"train_tasks": [...], "held_out_tasks": [...]}`, with task names such as
`prepare_coffee`. Episode sampling seeds include the split name, so a task
that changes split is evaluated on a different draw of its configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotalk.utils import REPO_ROOT, stable_json_sha256 as _sha256

SPLITS = {"train_tasks": "train_task_types", "held_out_tasks": "heldout_task_types"}


def resplit(cohort: dict, task_split: dict) -> dict:
    by_task = {
        task: configurations
        for split in cohort["configurations"].values()
        for task, configurations in split.items()
    }
    listed = [task for key in SPLITS for task in task_split[key]]
    unknown = sorted(set(listed) - set(by_task))
    if unknown:
        raise SystemExit(f"tasks not in the cohort: {', '.join(unknown)}")
    if len(listed) != len(set(listed)):
        raise SystemExit("a task is listed in both splits")
    out = dict(cohort)
    out["configurations"] = {
        split: {task: by_task[task] for task in task_split[key]}
        for key, split in SPLITS.items()
    }
    out.pop("content_hash", None)
    out["content_hash"] = _sha256(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cohort", type=Path, default=REPO_ROOT / "configs/eval/fixed_live_sim_cohort.json"
    )
    parser.add_argument("--task-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    task_split = json.loads(args.task_split.read_text(encoding="utf-8"))
    out = resplit(cohort, task_split)
    out["source_task_split"] = str(args.task_split)
    out["content_hash"] = _sha256({k: v for k, v in out.items() if k != "content_hash"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    counts = {split: len(tasks) for split, tasks in out["configurations"].items()}
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
