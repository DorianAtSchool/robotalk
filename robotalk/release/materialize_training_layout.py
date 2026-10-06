"""Rebuild the per-trajectory training layout from the published dataset.

Training reads ``<root>/<task>/<trajectory>/`` with ``original_trajectory.json``
(the trajectory after observation insertion), ``plan.json`` (calls in the
order the simulator ran them, with image paths) and ``metadata.json`` (the
execution record), plus ``images/<trajectory>/``. All of it is derived from
the dataset's ``raw/`` records and ``media_archives/``: observation insertion
is deterministic, the execution order is recomputed with the concurrent
validator exactly as the render sweep does, and every published step executed
successfully in simulation (dataset v1.1).

    python -m robotalk.release.materialize_training_layout \
        --hf-dir <local copy of DorianAtSchool/RoboTalk> --output data/robotalk_rendered
"""

from __future__ import annotations

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from robotalk.generation.image.processor import post_process_trajectory
from robotalk.tasks.shared.render_order import concurrent_step_order


def _task_instructions(hf_dir: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        hf_dir / "data" / "trajectories.parquet",
        columns=["episode_key", "task_instruction"],
    ).to_pylist()
    return {row["episode_key"]: row["task_instruction"] for row in table}


def materialize_one(raw_path: Path, hf_dir: Path, output: Path, instruction: str) -> str:
    task_slug = raw_path.parent.name
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    trajectory_id = raw["trajectory_id"]
    target = output / task_slug / trajectory_id
    if (target / "metadata.json").exists():
        return trajectory_id
    image_dir = target / "images" / trajectory_id
    image_dir.mkdir(parents=True, exist_ok=True)

    archive = hf_dir / "media_archives" / f"{task_slug}__{trajectory_id}.tar"
    files: dict[str, str] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile():
                name = Path(member.name).name
                with tar.extractfile(member) as src:
                    (image_dir / name).write_bytes(src.read())
                files[Path(name).stem] = name

    trajectory = post_process_trajectory(raw, revalidate=True)
    order = concurrent_step_order(trajectory)
    steps = trajectory["steps"]
    plan, executed = [], []
    for position, index in enumerate(order):
        step = steps[index]
        args = deepcopy(step.get("args") or {})
        if step["tool"] == "get_image":
            args["image_paths"] = [
                str(image_dir / files[Path(path).stem])
                for path in step.get("image_paths", [])
            ]
        robot_idx = int(str(step["agent"]).rsplit("_", 1)[-1])
        plan.append({
            "tool": step["tool"],
            "robot_idx": robot_idx,
            "args": args,
            "metadata": {
                "step_index": step["step"],
                "source_agent": step["agent"],
                "reasoning": step.get("reasoning"),
            },
        })
        executed.append({
            "step_index": position,
            "source_step_index": step["step"],
            "tool": step["tool"],
            "robot_idx": robot_idx,
            "args": args,
            "success": True,
        })
    (target / "original_trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (target / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (target / "metadata.json").write_text(
        json.dumps({"task": instruction, "steps": executed}, indent=2) + "\n"
    )
    return trajectory_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated task slugs (default: all)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    instructions = _task_instructions(args.hf_dir)
    wanted = set(args.tasks.split(",")) if args.tasks else None
    raw_paths = [
        path for path in sorted((args.hf_dir / "raw").glob("*/traj_*.json"))
        if wanted is None or path.parent.name in wanted
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                materialize_one, path, args.hf_dir, args.output,
                instructions[f"{path.parent.name}__{path.stem}"],
            )
            for path in raw_paths
        ]
        for count, future in enumerate(futures, start=1):
            future.result()
            if count % 500 == 0 or count == len(futures):
                print(f"materialized {count}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
