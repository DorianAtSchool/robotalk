"""Freeze QA, task split, and nested trajectory-scale manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random


SCALES = (30, 60, 90, 120, 150)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _balanced_order(rows: list[dict], *, seed: int, task: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["physical_configuration_signature"]].append(row)
    rng = random.Random(f"{seed}:{task}")
    queues = []
    for signature in sorted(groups):
        values = groups[signature]
        rng.shuffle(values)
        queues.append(deque(values))
    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.popleft())
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--legacy-task-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    legacy = json.loads(args.legacy_task_split.read_text())
    train_tasks = sorted(legacy["train_tasks"])
    held_out_tasks = sorted(set(legacy["held_out_tasks"]) | {"prepare_sandwich_station"})
    task_split = {
        "schema_version": 1,
        "name": "default_47_train_6_heldout",
        "train_tasks": train_tasks,
        "held_out_tasks": held_out_tasks,
        "source": str(args.legacy_task_split.resolve()),
        "source_sha256": _sha(args.legacy_task_split),
        "migration": {
            "added_held_out_tasks": ["prepare_sandwich_station"],
            "reason": "The legacy split predates the 53rd verified task; preserve the requested 47/6 contract.",
        },
    }
    rows_by_task = {}
    qa_tasks = {}
    failures = []
    for task_dir in sorted(p for p in args.raw_root.iterdir() if p.is_dir()):
        task = task_dir.name
        rows = []
        stage_counts = Counter()
        config_counts = Counter()
        for path in sorted((task_dir / "trajectories").glob("*.json")):
            payload = json.loads(path.read_text())
            trajectory_id = payload["trajectory_id"]
            rendered = args.rendered_root / task / trajectory_id
            validation = payload.get("validation") or {}
            image_files = list(rendered.rglob("*.jpg")) + list(rendered.rglob("*.png"))
            checks = {
                "validator_valid": validation.get("is_valid") is True,
                "live_policy_certified": validation.get("live_policy_certified") is True,
                "rendered_record": (rendered / "adapted_trajectory.json").is_file(),
                "has_images": bool(image_files),
            }
            if not all(checks.values()):
                failures.append({"task": task, "trajectory_id": trajectory_id, "checks": checks})
            row = {
                "task": task,
                "trajectory_id": trajectory_id,
                "physical_configuration_signature": payload["physical_configuration_signature"],
                "cascade_accepted_stage": payload.get("cascade_accepted_stage"),
                "cascade_attempt_count": payload.get("cascade_attempt_count"),
            }
            rows.append(row)
            stage_counts[row["cascade_accepted_stage"]] += 1
            config_counts[row["physical_configuration_signature"]] += 1
        rows_by_task[task] = rows
        qa_tasks[task] = {
            "trajectory_count": len(rows),
            "unique_configuration_count": len(config_counts),
            "configuration_counts": dict(sorted(config_counts.items())),
            "accepted_stage_counts": dict(sorted(stage_counts.items())),
        }
    expected_tasks = set(train_tasks) | set(held_out_tasks)
    if set(rows_by_task) != expected_tasks:
        failures.append({
            "task_inventory_mismatch": {
                "missing": sorted(expected_tasks - set(rows_by_task)),
                "extra": sorted(set(rows_by_task) - expected_tasks),
            }
        })
    for task, rows in rows_by_task.items():
        if len(rows) != 150:
            failures.append({"task": task, "expected": 150, "actual": len(rows)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_split_path = args.output_dir / "default_47_train_6_heldout.json"
    task_split_path.write_text(json.dumps(task_split, indent=2) + "\n")
    task_split_hash = _sha(task_split_path)
    orders = {
        task: _balanced_order(rows, seed=args.seed, task=task)
        for task, rows in rows_by_task.items()
    }
    for scale in SCALES:
        selection = {
            "schema_version": 1,
            "name": f"nested_{scale}_per_task_seed_{args.seed}",
            "seed": args.seed,
            "trajectories_per_task": scale,
            "task_split_manifest": str(task_split_path.resolve()),
            "task_split_sha256": task_split_hash,
            "train_trajectory_ids_by_task": {
                task: [row["trajectory_id"] for row in orders[task][:scale]]
                for task in train_tasks
            },
        }
        (args.output_dir / f"selection_{scale}.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    qa = {
        "schema_version": 1,
        "valid": not failures,
        "grain": "one rendered concurrent trajectory",
        "raw_root": str(args.raw_root.resolve()),
        "rendered_root": str(args.rendered_root.resolve()),
        "task_count": len(rows_by_task),
        "trajectory_count": sum(len(rows) for rows in rows_by_task.values()),
        "failures": failures,
        "tasks": qa_tasks,
    }
    (args.output_dir / "source_dataset_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps({"valid": qa["valid"], "tasks": qa["task_count"], "trajectories": qa["trajectory_count"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
