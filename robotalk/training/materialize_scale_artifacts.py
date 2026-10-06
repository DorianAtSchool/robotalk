"""Create lightweight nested training artifacts from one preprocessed master."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from datasets import DatasetDict, load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    allowed = {
        (task, trajectory_id)
        for task, ids in selection["train_trajectory_ids_by_task"].items()
        for trajectory_id in ids
    }
    source = load_from_disk(str(args.master / "dataset"))
    train = source["train"].filter(
        lambda task_name, trajectory_id: (task_name, trajectory_id) in allowed,
        input_columns=["task_name", "trajectory_id"],
        num_proc=min(24, os.cpu_count() or 1),
        desc=f"Selecting {selection['trajectories_per_task']}/task",
    )
    seen = set(zip(train["task_name"], train["trajectory_id"], strict=True))
    missing = sorted(allowed - seen)
    if missing:
        raise ValueError(f"Selection contains {len(missing)} missing trajectories: {missing[:5]}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    DatasetDict({"train": train, "validation": source["validation"]}).save_to_disk(
        str(args.output / "dataset")
    )
    os.symlink((args.master / "images").resolve(), args.output / "images")
    for filename in ("README.md", "manifest.json", "preprocess_config.json"):
        source_path = args.master / filename
        if source_path.exists():
            shutil.copy2(source_path, args.output / filename)
    config_path = args.output / "preprocess_config.json"
    config = json.loads(config_path.read_text())
    selected_tasks = sorted(selection["train_trajectory_ids_by_task"])
    config["train_tasks"] = selected_tasks
    config["trajectory_selection_manifest"] = str(args.selection.resolve())
    config["trajectory_selection"] = selection
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train"]["num_samples"] = len(train)
    manifest["train"]["task_names"] = selected_tasks
    manifest["train"]["task_count"] = len(selected_tasks)
    manifest["train"]["selected_trajectory_count"] = len(allowed)
    manifest["train"]["trajectories_per_task"] = selection["trajectories_per_task"]
    manifest["trajectory_selection_manifest"] = str(args.selection.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"samples": len(train), "trajectories": len(allowed)}, indent=2))


if __name__ == "__main__":
    main()
