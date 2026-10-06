"""Export RoboTalk to its Hugging Face layout (an eight-episode smoke set, or --all)."""

from __future__ import annotations

import argparse
from functools import cache
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robotalk.utils import output_root


DEFAULT_OUTPUT = output_root() / "hf_export"
VERIFIED_SPECS = Path(__file__).resolve().parents[1] / "tasks/specs/verified"
DATASET_CARD = Path(__file__).with_name("dataset_card.md")
SMOKE_EPISODES = (
    ("arrange_bread_bowl", "traj_000000"),
    ("prepare_coffee", "traj_000000"),
    ("cheese_mixing", "traj_000000"),
    ("condiment_collection", "traj_000000"),
    ("set_bowls_for_soup", "traj_000000"),
    ("meat_skewer_assembly", "traj_000000"),
    ("portion_yogurt", "traj_000000"),
    ("colorful_salsa", "traj_000000"),
)

PHYSICAL_TOOLS = {
    "navigate_to_fixture", "pick_up_object", "place_on_surface",
    "place_on_object", "place_in_receptacle", "place_next_to",
    "open_hinged_part", "close_hinged_part", "press_button",
    "turn_knob", "give_space",
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@cache
def _task_goals() -> dict[str, str]:
    goals = {}
    for path in VERIFIED_SPECS.glob("*.json"):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("composite_task") and spec.get("task_goal"):
            goals[spec["composite_task"]] = spec["task_goal"]
    return goals


def _observations_by_raw_step(rendered_dir: Path) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Attach inserted get_image calls to the decision they precede.

    Rendering inserts observations between canonical actions.  They are the
    context used for the *next* action, not the result of the previous one.
    """

    execution = json.loads((rendered_dir / "metadata.json").read_text(encoding="utf-8"))["steps"]
    pending: dict[int, list[dict]] = {0: [], 1: []}
    observations: dict[int, list[dict]] = {}
    raw_index = -1
    for event in execution:
        robot = int(event["robot_idx"])
        if event["tool"] == "get_image":
            pending[robot].append(event)
            continue
        raw_index += 1
        observations[raw_index] = pending[robot]
        pending[robot] = []
    return observations, pending


def _copy_image(source: Path, output: Path, episode_key: str) -> str:
    target = output / "media" / episode_key / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    return target.relative_to(output).as_posix()


def _tick_action(row: dict, agent: str) -> tuple[str, str, str, str]:
    action = row.get(agent)
    if not action:
        return "Not Invoked", "{}", "", ""
    args = action.get("args", {})
    return (
        action.get("tool", ""),
        _json(args),
        action.get("reasoning", ""),
        args.get("message", ""),
    )


def export_episode(raw_root: Path, rendered_root: Path, output: Path, task_slug: str, trajectory_id: str, episode_index: int) -> tuple[dict, list[dict]]:
    raw_path = raw_root / task_slug / "trajectories" / f"{trajectory_id}.json"
    rendered_dir = rendered_root / task_slug / trajectory_id
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    task_instruction = _task_goals().get(raw["composite_task"], raw.get("task", ""))
    ticks = raw["tick_rows"]
    observations, trailing_observations = _observations_by_raw_step(rendered_dir)
    episode_key = f"{task_slug}__{trajectory_id}"

    tick_rows = []
    flat_index = 0
    physical_counts = {"agent_0": 0, "agent_1": 0}
    communication_count = wait_count = release_count = 0
    current_images: dict[int, dict[str, str]] = {0: {}, 1: {}}
    for tick in ticks:
        actions = {}
        observation_calls: dict[str, list[dict]] = {"agent_0": [], "agent_1": []}
        for agent in ("agent_0", "agent_1"):
            actions[agent] = _tick_action(tick, agent)
            action = tick.get(agent)
            if action:
                tool = action.get("tool")
                if tool in PHYSICAL_TOOLS:
                    physical_counts[agent] += 1
                communication_count += int(tool == "communicate")
                wait_count += int(tool == "wait_for_signal")
                release_count += int(bool(action.get("args", {}).get("releases")))
                for observation in observations.get(flat_index, []):
                    views = observation.get("args", {}).get("views", [])
                    paths = observation.get("image_paths") or observation.get("args", {}).get("image_paths", [])
                    recorded_views = []
                    for view, candidate in zip(views, paths):
                        source = Path(candidate)
                        if source.is_file():
                            current_images[int(action.get("robot_idx", agent[-1]))][view] = _copy_image(
                                source, output, episode_key
                            )
                            recorded_views.append(view)
                    observation_calls[agent].append({"tool": "get_image", "views": recorded_views})
                flat_index += 1
        tick_rows.append({
            "episode_index": episode_index,
            "episode_key": episode_key,
            "task": raw["composite_task"],
            "task_instruction": task_instruction,
            "trajectory_id": trajectory_id,
            "media_archive": f"media_archives/{episode_key}.tar",
            "tick": int(tick["tick"]),
            "tick_label": str(tick["tick"]),
            "agent_0_tool": actions["agent_0"][0],
            "agent_0_args": actions["agent_0"][1],
            "agent_0_reasoning": actions["agent_0"][2],
            "agent_0_message": actions["agent_0"][3],
            "agent_0_observation_calls": _json(observation_calls["agent_0"]),
            "agent_0_images": _json(current_images[0]),
            "agent_1_tool": actions["agent_1"][0],
            "agent_1_args": actions["agent_1"][1],
            "agent_1_reasoning": actions["agent_1"][2],
            "agent_1_message": actions["agent_1"][3],
            "agent_1_observation_calls": _json(observation_calls["agent_1"]),
            "agent_1_images": _json(current_images[1]),
        })

    if any(trailing_observations.values()):
        final_calls: dict[str, list[dict]] = {"agent_0": [], "agent_1": []}
        for robot_idx, events in trailing_observations.items():
            agent = f"agent_{robot_idx}"
            for observation in events:
                views = observation.get("args", {}).get("views", [])
                paths = observation.get("image_paths") or observation.get("args", {}).get("image_paths", [])
                recorded_views = []
                for view, candidate in zip(views, paths):
                    source = Path(candidate)
                    if source.is_file():
                        current_images[robot_idx][view] = _copy_image(source, output, episode_key)
                        recorded_views.append(view)
                final_calls[agent].append({"tool": "get_image", "views": recorded_views})
        tick_rows.append({
            "episode_index": episode_index,
            "episode_key": episode_key,
            "task": raw["composite_task"],
            "task_instruction": task_instruction,
            "trajectory_id": trajectory_id,
            "media_archive": f"media_archives/{episode_key}.tar",
            "tick": len(ticks),
            "tick_label": "Task result",
            "agent_0_tool": "Not Invoked",
            "agent_0_args": "{}",
            "agent_0_reasoning": "",
            "agent_0_message": "",
            "agent_0_observation_calls": _json(final_calls["agent_0"]),
            "agent_0_images": _json(current_images[0]),
            "agent_1_tool": "Not Invoked",
            "agent_1_args": "{}",
            "agent_1_reasoning": "",
            "agent_1_message": "",
            "agent_1_observation_calls": _json(final_calls["agent_1"]),
            "agent_1_images": _json(current_images[1]),
        })

    total_physical = sum(physical_counts.values())
    max_share = max(physical_counts.values()) / total_physical if total_physical else 0.0
    catalog = {
        "episode_index": episode_index,
        "episode_key": episode_key,
        "task": raw["composite_task"],
        "task_instruction": task_instruction,
        "trajectory_id": trajectory_id,
        "media_archive": f"media_archives/{episode_key}.tar",
        "num_ticks": len(ticks),
        "coordinator_id": raw.get("coordinator_id"),
        "physical_actions_agent_0": physical_counts["agent_0"],
        "physical_actions_agent_1": physical_counts["agent_1"],
        "max_agent_work_share": max_share,
        "communication_calls": communication_count,
        "wait_calls": wait_count,
        "release_calls": release_count,
        "generation_stage": raw.get("cascade_accepted_stage"),
        "generation_attempts": raw.get("cascade_attempt_count"),
        "physical_configuration_signature": raw.get("physical_configuration_signature"),
        "fsm_valid": bool(raw.get("validation", {}).get("is_valid", True)),
        "initial_state": _json(raw.get("initial_state", {})),
        "source_raw_json": f"raw/{task_slug}/{trajectory_id}.json",
    }
    target_json = output / "raw" / task_slug / f"{trajectory_id}.json"
    target_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, target_json)
    return catalog, tick_rows


def _write_media_archives(output: Path, catalog: list[dict]) -> None:
    archives = output / "media_archives"
    archives.mkdir(parents=True, exist_ok=True)
    for episode in catalog:
        episode_key = episode["episode_key"]
        source = output / "media" / episode_key
        target = archives / f"{episode_key}.tar"
        if target.is_file():
            continue
        with tarfile.open(target, "w") as archive:
            for image_path in sorted(source.iterdir()):
                archive.add(image_path, arcname=f"media/{episode_key}/{image_path.name}", recursive=False)


def _write_static_space(output: Path, catalog: list[dict], ticks: list[dict]) -> None:
    template = Path(__file__).with_name("static_space")
    target = output / "space"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(template, target)
    data = target / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "trajectories.json").write_text(_json(catalog), encoding="utf-8")
    episodes = data / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    by_episode: dict[str, list[dict]] = {}
    for row in ticks:
        by_episode.setdefault(row["episode_key"], []).append(row)
    for episode_key, rows in by_episode.items():
        (episodes / f"{episode_key}.json").write_text(_json(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="<task>/trajectories/traj_*.json as generated")
    parser.add_argument("--rendered-root", type=Path, required=True, help="<task>/<traj>/ as written by the render sweep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all", action="store_true", help="Export all trajectories instead of the eight-episode smoke set")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog, ticks = [], []
    selected = SMOKE_EPISODES
    if args.all:
        selected = tuple(
            (task_dir.name, path.stem)
            for task_dir in sorted(args.raw_root.iterdir())
            if task_dir.is_dir()
            for path in sorted((task_dir / "trajectories").glob("traj_*.json"))
            if (args.rendered_root / task_dir.name / path.stem / "metadata.json").is_file()
        )
    for episode_index, (task, trajectory) in enumerate(selected):
        episode, episode_ticks = export_episode(
            args.raw_root, args.rendered_root, args.output, task, trajectory, episode_index
        )
        catalog.append(episode)
        ticks.extend(episode_ticks)
    data = args.output / "data"
    data.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(catalog), data / "trajectories.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(ticks), data / "ticks.parquet", compression="zstd")
    manifest = {
        "schema_version": 1,
        "repo_id": "DorianAtSchool/RoboTalk",
        "source_trajectory_count": 7950,
        "episode_count": len(catalog),
        "tick_row_count": len(ticks),
        "prompt_contract_version": "state_grounded_global_tools_no_index_v9_unambiguous_garnish_cake_goal",
        "grain": {"trajectories.parquet": "one trajectory", "ticks.parquet": "one concurrent tick"},
    }
    (args.output / "dataset_info.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    shutil.copyfile(DATASET_CARD, args.output / "README.md")
    _write_media_archives(args.output, catalog)
    _write_static_space(args.output, catalog, ticks)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
