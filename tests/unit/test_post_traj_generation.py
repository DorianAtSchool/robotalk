import json
from pathlib import Path
import shutil
import tempfile
import unittest

from robotalk.generation.image import (
    POST_PROCESS_VALIDATION_ERROR_TYPE,
    parse_args,
    post_process_dataset,
    post_process_trajectory,
    resolve_output_dataset_path,
)
from robotalk.tasks import (
    TaskSemanticValidationError,
    TrajectoryStructureValidationError,
    get_task_definition,
)


def make_sample_trajectory():
    """Builds a compact saved trajectory fixture for post-processing tests."""

    trajectory = {
        "trajectory_id": "traj_000000",
        "composite_task": "PrepareCoffee",
        "agents": [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ],
        "steps": (steps := [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I will start the task.",
                },
                "reasoning": "We should coordinate before acting.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I will stay ready for handoff.",
                },
                "reasoning": "I should confirm the plan.",
            },
            {
                "step": 2,
                "agent": "agent_0",
                "tool": "navigate_to_fixture",
                "args": {"fixture_id": "mug_source_fixture"},
                "reasoning": "I need to reach the cabinet.",
            },
            {
                "step": 3,
                "agent": "agent_0",
                "tool": "pick_up_object",
                "args": {
                    "object_id": "mug",
                    "source_id": "mug_source_fixture",
                },
                "reasoning": "I should pick up the mug.",
            },
        ]),
        "validation": {"signature": "old-signature"},
    }
    # Concurrent ticks: both agents talk, then agent_0 works alone.
    trajectory["tick_rows"] = [
        {"tick": 0, "agent_0": steps[0], "agent_1": steps[1]},
        {"tick": 1, "agent_0": steps[2]},
        {"tick": 2, "agent_0": steps[3]},
    ]
    return trajectory


class PostTrajectoryGenerationTests(unittest.TestCase):
    def test_post_process_trajectory_keeps_rewritten_steps_schema_valid(self):
        trajectory = post_process_trajectory(make_sample_trajectory())
        validator = get_task_definition("PrepareCoffee").validator_factory(None)

        try:
            validator.validate(
                {
                    "agents": trajectory["agents"],
                    "steps": trajectory["steps"],
                }
            )
        except TaskSemanticValidationError:
            pass
        except TrajectoryStructureValidationError as exc:
            self.fail(f"Rewritten steps should remain schema-valid: {exc}")

    def test_post_process_trajectory_is_idempotent(self):
        first_pass = post_process_trajectory(make_sample_trajectory())
        second_pass = post_process_trajectory(first_pass)

        self.assertEqual(second_pass["steps"], first_pass["steps"])
        self.assertEqual(
            second_pass["validation"]["signature"],
            first_pass["validation"]["signature"],
        )
        self.assertEqual(
            second_pass["validation"]["error_type"],
            POST_PROCESS_VALIDATION_ERROR_TYPE,
        )

    def test_post_process_trajectory_does_not_insert_agents_when_missing(self):
        trajectory = make_sample_trajectory()
        trajectory.pop("agents")

        processed = post_process_trajectory(trajectory)

        self.assertNotIn("agents", processed)
        self.assertEqual(processed["steps"][0]["agent"], "agent_0")
        self.assertEqual(processed["steps"][1]["agent"], "agent_1")

    def test_resolve_output_dataset_path_targets_pre_image_copy(self):
        dataset_path = Path(
            "/tmp/data/raw/prepare_coffee/20260316T022801Z/summary.json"
        )

        self.assertEqual(
            resolve_output_dataset_path(dataset_path).resolve(),
            Path(
                "/tmp/data/pre_image/prepare_coffee/20260316T022801Z/summary.json"
            ).resolve(),
        )

    def test_resolve_output_dataset_path_preserves_multitask_layout(self):
        dataset_path = Path(
            "/tmp/data/raw/20260316T022801Z/prepare_coffee/summary.json"
        )

        self.assertEqual(
            resolve_output_dataset_path(dataset_path).resolve(),
            Path(
                "/tmp/data/pre_image/20260316T022801Z/prepare_coffee/summary.json"
            ).resolve(),
        )

    def test_post_process_dataset_writes_summary_copy_without_mutating_source(self):
        summary_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "num_trajectories": 1,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "trajectory_directory": "trajectories",
            "trajectory_files": [
                {
                    "trajectory_id": "traj_000000",
                    "path": "trajectories/traj_000000.json",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = (
                Path(tmpdir)
                / "data"
                / "raw"
                / "prepare_coffee"
                / "20260310T000000Z"
                / "summary.json"
            )
            trajectory_path = dataset_path.parent / "trajectories" / "traj_000000.json"
            prompt_path = dataset_path.parent / "prompts" / "traj_000000.md"
            raw_output_path = dataset_path.parent / "outputs" / "traj_000000.txt"
            error_summary_path = dataset_path.parent / "summary_errors.json"
            cost_summary_path = dataset_path.parent / "cost_summary.json"
            output_dataset_path = resolve_output_dataset_path(dataset_path)
            output_trajectory_path = (
                output_dataset_path.parent / "trajectories" / "traj_000000.json"
            )
            output_error_summary_path = (
                output_dataset_path.parent / "summary_errors.json"
            )
            output_cost_summary_path = output_dataset_path.parent / "cost_summary.json"
            output_prompt_path = (
                output_dataset_path.parent / "prompts" / "traj_000000.md"
            )
            output_raw_output_path = (
                output_dataset_path.parent / "outputs" / "traj_000000.txt"
            )
            output_images_dir = output_dataset_path.parent / "images"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(make_sample_trajectory(), indent=2),
                encoding="utf-8",
            )
            prompt_path.write_text("prompt copy me", encoding="utf-8")
            raw_output_path.write_text("raw output copy me", encoding="utf-8")
            error_summary_path.write_text(
                json.dumps({"total_errors": 1}, indent=2),
                encoding="utf-8",
            )
            cost_summary_path.write_text(
                json.dumps({"total_cost_usd": 1.23}, indent=2),
                encoding="utf-8",
            )

            processed_count = post_process_dataset(
                dataset_path,
                disable_progress=True,
            )

            self.assertEqual(processed_count, 1)
            source_trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            updated_trajectory = json.loads(
                output_trajectory_path.read_text(encoding="utf-8")
            )
            self.assertEqual(source_trajectory["steps"][0]["tool"], "communicate")
            self.assertEqual(updated_trajectory["steps"][0]["tool"], "get_image")
            self.assertEqual(updated_trajectory["steps"][2]["tool"], "communicate")
            self.assertEqual(
                updated_trajectory["steps"][0]["image_paths"],
                [
                    "images/traj_000000/0_top_view_agent_0.png",
                    "images/traj_000000/0_room_view_agent_0.png",
                    "images/traj_000000/0_map_agent_0.png",
                ],
            )
            self.assertEqual(
                json.loads(output_dataset_path.read_text(encoding="utf-8"))[
                    "trajectory_files"
                ],
                summary_payload["trajectory_files"],
            )
            self.assertEqual(
                json.loads(dataset_path.read_text(encoding="utf-8"))[
                    "trajectory_files"
                ],
                summary_payload["trajectory_files"],
            )
            self.assertEqual(
                json.loads(output_error_summary_path.read_text(encoding="utf-8")),
                {"total_errors": 1},
            )
            self.assertEqual(
                json.loads(output_cost_summary_path.read_text(encoding="utf-8")),
                {"total_cost_usd": 1.23},
            )
            self.assertTrue(output_images_dir.is_dir())
            self.assertFalse(output_prompt_path.exists())
            self.assertFalse(output_raw_output_path.exists())

    def test_post_process_dataset_matches_legacy_copy_for_materialized_outputs(self):
        summary_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "num_trajectories": 2,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "trajectory_directory": "trajectories",
            "trajectory_files": [
                {
                    "trajectory_id": "traj_000000",
                    "path": "trajectories/traj_000000.json",
                },
                {
                    "trajectory_id": "traj_000001",
                    "path": "trajectories/traj_000001.json",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = (
                Path(tmpdir)
                / "data"
                / "raw"
                / "20260310T000000Z"
                / "prepare_coffee"
                / "summary.json"
            )
            source_root = dataset_path.parent
            source_root.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            (source_root / "summary_errors.json").write_text(
                json.dumps({"total_errors": 1}, indent=2),
                encoding="utf-8",
            )
            (source_root / "cost_summary.json").write_text(
                json.dumps({"total_cost_usd": 1.23}, indent=2),
                encoding="utf-8",
            )
            prompts_dir = source_root / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / "traj_000000.md").write_text("prompt 0", encoding="utf-8")
            (prompts_dir / "traj_000001.md").write_text("prompt 1", encoding="utf-8")
            outputs_dir = source_root / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "traj_000000.txt").write_text("output 0", encoding="utf-8")
            (outputs_dir / "traj_000001.txt").write_text("output 1", encoding="utf-8")
            trajectories_dir = source_root / "trajectories"
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            for trajectory_id in ("traj_000000", "traj_000001"):
                trajectory = make_sample_trajectory()
                trajectory["trajectory_id"] = trajectory_id
                (trajectories_dir / f"{trajectory_id}.json").write_text(
                    json.dumps(trajectory, indent=2),
                    encoding="utf-8",
                )

            optimized_output = (
                Path(tmpdir)
                / "data"
                / "pre_image_optimized"
                / "20260310T000000Z"
                / "prepare_coffee"
                / "summary.json"
            )
            legacy_output = (
                Path(tmpdir)
                / "data"
                / "pre_image_legacy"
                / "20260310T000000Z"
                / "prepare_coffee"
                / "summary.json"
            )

            processed_count = post_process_dataset(
                dataset_path,
                output_dataset_path=optimized_output,
                disable_progress=True,
            )

            shutil.copytree(source_root, legacy_output.parent, dirs_exist_ok=True)
            legacy_payload = json.loads(legacy_output.read_text(encoding="utf-8"))
            for trajectory_file in legacy_payload["trajectory_files"]:
                legacy_trajectory_path = legacy_output.parent / trajectory_file["path"]
                legacy_updated_trajectory = post_process_trajectory(
                    json.loads(legacy_trajectory_path.read_text(encoding="utf-8"))
                )
                legacy_trajectory_path.write_text(
                    json.dumps(legacy_updated_trajectory, indent=2),
                    encoding="utf-8",
                )
            (legacy_output.parent / "images").mkdir(parents=True, exist_ok=True)
            legacy_output.write_text(
                json.dumps(legacy_payload, indent=2),
                encoding="utf-8",
            )

            self.assertEqual(processed_count, 2)
            self.assertEqual(
                json.loads(optimized_output.read_text(encoding="utf-8")),
                json.loads(legacy_output.read_text(encoding="utf-8")),
            )
            for trajectory_id in ("traj_000000", "traj_000001"):
                optimized_trajectory_path = (
                    optimized_output.parent / "trajectories" / f"{trajectory_id}.json"
                )
                legacy_trajectory_path = (
                    legacy_output.parent / "trajectories" / f"{trajectory_id}.json"
                )
                self.assertEqual(
                    json.loads(optimized_trajectory_path.read_text(encoding="utf-8")),
                    json.loads(legacy_trajectory_path.read_text(encoding="utf-8")),
                )
            self.assertEqual(
                json.loads(
                    (optimized_output.parent / "summary_errors.json").read_text(
                        encoding="utf-8"
                    )
                ),
                json.loads(
                    (legacy_output.parent / "summary_errors.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            self.assertEqual(
                json.loads(
                    (optimized_output.parent / "cost_summary.json").read_text(
                        encoding="utf-8"
                    )
                ),
                json.loads(
                    (legacy_output.parent / "cost_summary.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            self.assertTrue((optimized_output.parent / "images").is_dir())
            self.assertFalse((optimized_output.parent / "prompts").exists())
            self.assertFalse((optimized_output.parent / "outputs").exists())

    def test_post_process_dataset_accepts_parallel_workers(self):
        summary_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "num_trajectories": 2,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "trajectory_directory": "trajectories",
            "trajectory_files": [
                {
                    "trajectory_id": "traj_000000",
                    "path": "trajectories/traj_000000.json",
                },
                {
                    "trajectory_id": "traj_000001",
                    "path": "trajectories/traj_000001.json",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = (
                Path(tmpdir)
                / "data"
                / "raw"
                / "20260310T000000Z"
                / "prepare_coffee"
                / "summary.json"
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            trajectories_dir = dataset_path.parent / "trajectories"
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            for trajectory_id in ("traj_000000", "traj_000001"):
                trajectory = make_sample_trajectory()
                trajectory["trajectory_id"] = trajectory_id
                (trajectories_dir / f"{trajectory_id}.json").write_text(
                    json.dumps(trajectory, indent=2),
                    encoding="utf-8",
                )

            processed_count = post_process_dataset(
                dataset_path,
                disable_progress=True,
                workers=2,
            )

            self.assertEqual(processed_count, 2)
            output_dataset_path = resolve_output_dataset_path(dataset_path)
            for trajectory_id in ("traj_000000", "traj_000001"):
                output_trajectory = json.loads(
                    (
                        output_dataset_path.parent
                        / "trajectories"
                        / f"{trajectory_id}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(output_trajectory["steps"][0]["tool"], "get_image")
                self.assertEqual(
                    output_trajectory["steps"][0]["image_paths"][0],
                    f"images/{trajectory_id}/0_top_view_agent_0.png",
                )

    def test_parse_args_requires_dataset(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_parse_args_accepts_dataset_without_image_tool_version(self):
        args = parse_args(["--dataset", "summary.json"])

        self.assertEqual(args.dataset, Path("summary.json"))
        self.assertFalse(args.disable_progress)
        self.assertEqual(args.workers, 1)

    def test_parse_args_accepts_workers(self):
        args = parse_args(["--dataset", "summary.json", "--workers", "4"])

        self.assertEqual(args.workers, 4)

    def test_parse_args_rejects_non_positive_workers(self):
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "summary.json", "--workers", "0"])

    def test_parse_args_rejects_legacy_image_tool_version_flag(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--dataset",
                    "summary.json",
                    "--image-tool-version",
                    "v2",
                ]
            )


if __name__ == "__main__":
    unittest.main()
