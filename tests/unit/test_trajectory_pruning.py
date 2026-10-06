from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "robocasa"
    / "utils"
    / "trajectory_pruning.py"
)
SPEC = importlib.util.spec_from_file_location("trajectory_pruning", MODULE_PATH)
trajectory_pruning = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(trajectory_pruning)

build_trajectory_pruning_config = trajectory_pruning.build_trajectory_pruning_config
filter_object_cfgs_for_trajectory = (
    trajectory_pruning.filter_object_cfgs_for_trajectory
)
resolve_trajectory_object_cfg_matches = (
    trajectory_pruning.resolve_trajectory_object_cfg_matches
)
should_keep_object_cfg_for_trajectory = (
    trajectory_pruning.should_keep_object_cfg_for_trajectory
)


class TestTrajectoryPruning(unittest.TestCase):
    def test_build_config_disables_unreferenced_surface_fixtures(self):
        trajectory = {
            "initial_state": {
                "fixtures": {
                    "cabinet": {"fixture_type": "cabinet"},
                    "counter": {"fixture_type": "counter"},
                    "fridge": {"fixture_type": "fridge"},
                },
                "objects": {
                    "cheese": {"object_type": "cheese", "location": "fridge"},
                    "grater": {
                        "object_type": "cheese_grater",
                        "location": "cabinet",
                    },
                    "lettuce": {"object_type": "lettuce", "location": "salad_bowl"},
                    "salad_bowl": {"object_type": "bowl", "location": "counter"},
                },
            },
            "grounding_map": {
                "symbols": {
                    "cabinet": {
                        "entity_type": "fixture",
                        "fixture_type": "cabinet",
                        "preferred_fixture_types": ["cabinet"],
                    },
                    "counter": {
                        "entity_type": "fixture",
                        "fixture_type": "counter",
                        "preferred_fixture_types": ["counter"],
                    },
                    "fridge": {
                        "entity_type": "fixture",
                        "fixture_type": "fridge",
                        "preferred_fixture_types": ["fridge"],
                    },
                }
            },
        }

        pruning = build_trajectory_pruning_config(trajectory, layout=4)

        self.assertIn("toaster", pruning["update_fxtr_cfg_dict"])
        self.assertIn("coffee_machine", pruning["update_fxtr_cfg_dict"])
        self.assertNotIn("fridge", pruning["update_fxtr_cfg_dict"])
        self.assertIn("salad_bowl", pruning["trajectory_object_names"])
        self.assertIn("bowl", pruning["trajectory_object_types"])
        self.assertEqual(
            pruning["trajectory_object_specs"]["salad_bowl"]["object_type"],
            "bowl",
        )

    def test_filter_object_cfgs_keeps_named_objects_and_removes_distractors(self):
        object_cfgs = [
            {"name": "hotdog_bun", "obj_groups": "hotdog_bun"},
            {"name": "plate", "obj_groups": "plate"},
            {"name": "sausage", "obj_groups": "sausage"},
            {"name": "distr1", "obj_groups": "cheese"},
            {"name": "distr2"},
        ]

        filtered = filter_object_cfgs_for_trajectory(
            object_cfgs,
            required_object_specs={
                "bun": {"object_type": "hotdog_bun"},
                "serving_plate": {"object_type": "plate"},
                "sausage": {"object_type": "sausage"},
            },
        )

        self.assertEqual(
            [cfg["name"] for cfg in filtered],
            ["hotdog_bun", "plate", "sausage"],
        )

    def test_matcher_binds_generated_container_only_when_symbol_remains(self):
        matches = resolve_trajectory_object_cfg_matches(
            [
                {"name": "lettuce_container", "obj_groups": "bowl"},
            ],
            required_object_specs={
                "salad_bowl": {"object_type": "bowl"},
            },
        )
        self.assertEqual(matches, {"salad_bowl": "lettuce_container"})

    def test_matcher_binds_generated_objects_by_shared_ordinal(self):
        matches = resolve_trajectory_object_cfg_matches(
            [
                {"name": "obj_0", "obj_groups": "drink"},
                {"name": "obj_1", "obj_groups": "drink"},
                {"name": "obj_2", "obj_groups": "drink"},
            ],
            required_object_specs={
                "drink_0": {"object_type": "drink"},
                "drink_1": {"object_type": "drink"},
                "drink_2": {"object_type": "drink"},
            },
        )
        self.assertEqual(
            matches,
            {
                "drink_0": "obj_0",
                "drink_1": "obj_1",
                "drink_2": "obj_2",
            },
        )

    def test_filter_object_cfgs_keeps_ordinally_matched_generated_objects(self):
        filtered = filter_object_cfgs_for_trajectory(
            [
                {"name": "obj_0", "obj_groups": "drink"},
                {"name": "obj_1", "obj_groups": "drink"},
                {"name": "obj_2", "obj_groups": "drink"},
                {"name": "plate", "obj_groups": "plate"},
            ],
            required_object_specs={
                "drink_0": {"object_type": "drink"},
                "drink_1": {"object_type": "drink"},
                "drink_2": {"object_type": "drink"},
            },
        )
        self.assertEqual(
            [cfg["name"] for cfg in filtered],
            ["obj_0", "obj_1", "obj_2"],
        )

    def test_matcher_does_not_bind_symbolic_container_to_child_cfg(self):
        matches = resolve_trajectory_object_cfg_matches(
            [
                {
                    "name": "obj",
                    "obj_groups": "steak",
                    "placement": {"try_to_place_in": "pan"},
                },
                {"name": "plate", "obj_groups": "plate"},
            ],
            required_object_specs={
                "steak": {"object_type": "steak", "location": "pan"},
                "pan": {"object_type": "pan", "location": "stove"},
                "plate": {"object_type": "plate", "location": "dining_table"},
            },
        )

        self.assertEqual(matches, {"plate": "plate", "steak": "obj"})
        self.assertNotIn("pan", matches)

    def test_generated_container_and_auxiliary_keep_logic(self):
        self.assertTrue(
            should_keep_object_cfg_for_trajectory(
                {"name": "lettuce_container", "obj_groups": "bowl"},
                required_object_names={"salad_bowl"},
                required_object_types={"bowl"},
            )
        )

    def test_filter_keeps_try_to_place_in_support_cfg(self):
        filtered = filter_object_cfgs_for_trajectory(
            [
                {
                    "name": "steak",
                    "obj_groups": "steak",
                    "placement": {"try_to_place_in": "plate"},
                },
                {"name": "plate_0", "obj_groups": "plate"},
                {"name": "distractor_bowl", "obj_groups": "bowl"},
            ],
            required_object_specs={
                "steak": {"object_type": "steak"},
            },
        )
        self.assertEqual(
            [cfg["name"] for cfg in filtered],
            ["steak", "plate_0"],
        )
        self.assertFalse(
            should_keep_object_cfg_for_trajectory(
                {"name": "knife_auxiliary", "obj_groups": "knife"},
                required_object_names={"salad_bowl"},
                required_object_types={"bowl"},
            )
        )

    def test_matcher_uses_cfg_name_for_symbolic_types_with_variant_obj_groups(self):
        matches = resolve_trajectory_object_cfg_matches(
            [
                {"name": "spice", "obj_groups": ("turmeric", "paprika")},
                {"name": "bottle", "obj_groups": ("ketchup", "mayonnaise")},
            ],
            required_object_specs={
                "spice": {"object_type": "spice"},
                "bottle": {"object_type": "bottle"},
            },
        )
        self.assertEqual(
            matches,
            {
                "spice": "spice",
                "bottle": "bottle",
            },
        )


if __name__ == "__main__":
    unittest.main()
