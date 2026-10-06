from __future__ import annotations

import unittest

from robotalk.tasks import get_task_definition
from robotalk.tasks.specs import load_task_spec
from robotalk.tasks.shared.types import TaskInstance

HOT_DOG_SETUP_INITIAL_STATE = load_task_spec("HotDogSetup").initial_state
PREPARE_COFFEE_INITIAL_STATE = load_task_spec("PrepareCoffee").initial_state
PREPARE_SANDWICH_STATION_INITIAL_STATE = load_task_spec(
    "PrepareSandwichStation"
).initial_state


class TaskLevelGroundingTests(unittest.TestCase):

    def test_task_record_builders_persist_scene_agnostic_grounding_map(self):
        candidate = {"steps": []}
        validation = {"is_valid": True}
        generation_usage = {"total_cost_usd": 0.0}

        trajectory_records = [
            get_task_definition("HotDogSetup").build_trajectory_record(
                candidate,
                validation,
                "traj_000001",
                generation_usage,
                TaskInstance(initial_state=HOT_DOG_SETUP_INITIAL_STATE),
            ),
            get_task_definition("PrepareCoffee").build_trajectory_record(
                candidate,
                validation,
                "traj_000002",
                generation_usage,
                TaskInstance(initial_state=PREPARE_COFFEE_INITIAL_STATE),
            ),
            get_task_definition("PrepareSandwichStation").build_trajectory_record(
                candidate,
                validation,
                "traj_000003",
                generation_usage,
                TaskInstance(initial_state=PREPARE_SANDWICH_STATION_INITIAL_STATE),
            ),
        ]

        for trajectory_record in trajectory_records:
            grounding_map = trajectory_record.get("grounding_map")
            self.assertIsInstance(grounding_map, dict)
            self.assertEqual(grounding_map["map_kind"], "scene_agnostic")
            self.assertEqual(
                grounding_map["composite_task"],
                trajectory_record["composite_task"],
            )
            self.assertTrue(grounding_map["symbols"])

    def test_symbolic_task_record_omits_raw_scene_metadata(self):
        candidate = {"steps": []}
        validation = {"is_valid": True}
        generation_usage = {"total_cost_usd": 0.0}
        task_instance = TaskInstance(initial_state=PREPARE_COFFEE_INITIAL_STATE)

        trajectory_record = get_task_definition("PrepareCoffee").build_trajectory_record(
            candidate,
            validation,
            "traj_000010",
            generation_usage,
            task_instance,
        )

        self.assertIn("grounding_map", trajectory_record)
        self.assertNotIn("grounding_mode", trajectory_record)
        self.assertNotIn("scene_config", trajectory_record)
        self.assertNotIn("scene_summary", trajectory_record)
        self.assertNotIn("layout", trajectory_record)
        self.assertNotIn("style", trajectory_record)
        self.assertNotIn("seed", trajectory_record)
        self.assertNotIn("robots", trajectory_record)


if __name__ == "__main__":
    unittest.main()
