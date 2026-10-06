from __future__ import annotations

import unittest

from robotalk.tools.grounding_specs import (
    _resolve_nearest_placeable_surface,
)
from robotalk.tools.subatomic_tool_specs import (
    TASK_LEVEL_ALLOWED_TOOL_SPECS,
    build_allowed_tool_specs,
    build_model_tool_specs,
    canonicalize_model_tool_args,
)
from robotalk.training.tool_calling import build_tool_schemas
from robotalk.tasks.shared.schema import build_task_response_schema
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


class TaskLevelToolInterfaceTests(unittest.TestCase):
    def test_legacy_placement_target_is_canonicalized_for_model_io(self):
        self.assertEqual(
            {"object_id": "cup", "support_id": "counter"},
            canonicalize_model_tool_args(
                "place_on_surface",
                {"object_id": "cup", "target_id": "counter"},
            ),
        )
        self.assertEqual(
            {"target_id": "machine", "control_id": "start"},
            canonicalize_model_tool_args(
                "press_button",
                {"target_id": "machine", "control_id": "start"},
            ),
        )

    def test_model_interface_is_global_and_has_no_task_id_allowlists(self):
        global_specs = build_model_tool_specs(include_get_image=True)
        local_specs = build_allowed_tool_specs(
            ("navigate_to_fixture",),
            overrides={
                "navigate_to_fixture": {"allowed_fixture_ids": ["counter"]}
            },
        )

        self.assertIn("press_button", global_specs)
        self.assertIn("get_image", global_specs)
        self.assertNotIn("allowed_fixture_ids", global_specs["navigate_to_fixture"])
        self.assertEqual(["counter"], local_specs["navigate_to_fixture"]["allowed_fixture_ids"])

    def test_model_schema_describes_every_declared_argument(self):
        specs = build_model_tool_specs(include_get_image=True)
        schemas = build_tool_schemas(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=specs,
        )

        for schema in schemas:
            properties = schema["function"]["parameters"]["properties"]
            for arg_name, arg_schema in properties.items():
                self.assertTrue(
                    arg_schema.get("description"),
                    f"{schema['function']['name']}.{arg_name} lacks a description",
                )

        open_schema = next(
            schema for schema in schemas
            if schema["function"]["name"] == "open_hinged_part"
        )
        part_description = open_schema["function"]["parameters"]["properties"][
            "part_id"
        ]["description"]
        self.assertIn("initial_state.fixtures[target_id].parts", part_description)
        self.assertIn("copy it verbatim", part_description)

    def test_place_on_surface_spec_includes_optional_site_and_relative_args(self):
        spec = TASK_LEVEL_ALLOWED_TOOL_SPECS["place_on_surface"]

        self.assertEqual(["object_id", "support_id"], spec["tool_args"])
        self.assertNotIn("tool_arg_any_of", spec)
        self.assertEqual(
            ["target_site_id", "relative_position"],
            spec["optional_tool_args"],
        )

    def test_place_next_to_spec_includes_all_reference_variants(self):
        spec = TASK_LEVEL_ALLOWED_TOOL_SPECS["place_next_to"]

        self.assertEqual(["object_id"], spec["tool_args"])
        self.assertEqual(
            [["reference_object_id", "reference_fixture_id"]],
            spec["tool_arg_any_of"],
        )
        self.assertEqual(
            [
                "target_site_id",
                "relative_position",
            ],
            spec["optional_tool_args"],
        )

    def test_response_schema_includes_optional_tool_args(self):
        schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs={
                "place_on_surface": TASK_LEVEL_ALLOWED_TOOL_SPECS["place_on_surface"],
            },
            min_steps=1,
        )

        step_items = schema["properties"]["steps"]["items"]
        step_branch = step_items.get("anyOf", [step_items])[0]
        args_schema = step_branch["properties"]["args"]
        branches = args_schema.get("anyOf", [args_schema])
        args_properties = {
            name: value
            for branch in branches
            for name, value in branch["properties"].items()
        }
        self.assertIn("target_site_id", args_properties)
        self.assertIn("relative_position", args_properties)

    def test_nearest_placeable_surface_prefers_parent_fixture(self):
        scene = {
            "fixtures": {
                "blender": {
                    "position": [1.0, 1.0, 0.0],
                    "fixture_type": "blender",
                    "can_place_objects": False,
                    "parent_fixture": "prep_counter",
                },
                "prep_counter": {
                    "position": [1.0, 1.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
                "dining_counter": {
                    "position": [1.1, 1.0, 0.0],
                    "fixture_type": "dining_counter",
                    "can_place_objects": True,
                },
            }
        }
        resolved = {"blender_fixture": {"resolved_id": "blender"}}
        spec = {
            "anchor_fixture_symbol": "blender_fixture",
            "preferred_fixture_types": ["counter_non_dining", "dining_counter"],
        }

        result = _resolve_nearest_placeable_surface(
            "support_surface",
            spec,
            scene,
            resolved,
        )

        self.assertEqual("prep_counter", result["resolved_id"])
        self.assertEqual(1.0, result["confidence"])

    def test_nearest_placeable_surface_rejects_ambiguous_close_candidates(self):
        scene = {
            "fixtures": {
                "coffee_machine": {
                    "position": [0.0, 0.0, 0.0],
                    "fixture_type": "coffee_machine",
                    "can_place_objects": False,
                },
                "counter_a": {
                    "position": [0.2, 0.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
                "counter_b": {
                    "position": [0.28, 0.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
            }
        }
        resolved = {"coffee_fixture": {"resolved_id": "coffee_machine"}}
        spec = {
            "anchor_fixture_symbol": "coffee_fixture",
            "preferred_fixture_types": ["counter_non_dining"],
        }

        result = _resolve_nearest_placeable_surface(
            "support_surface",
            spec,
            scene,
            resolved,
        )

        self.assertIsNone(result["resolved_id"])
        self.assertEqual(["counter_a", "counter_b"], result["candidates"])

    def test_counter_family_excludes_dining_counter_and_island(self):
        self.assertEqual(
            {"counter", "counter_non_dining", "counter_non_corner"},
            TrajectoryAdapter._FIXTURE_TYPE_FAMILIES["counter"],
        )


if __name__ == "__main__":
    unittest.main()
